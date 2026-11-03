"""Behavioral tests for the calibration service and loopback HTTP boundary."""

from __future__ import annotations

import copy
import contextlib
import hashlib
import http.client
import json
import shutil
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from task_skills.calibration.http_server import MAX_JSON_BODY, create_server
from task_skills.calibration.integration import load_validated_calibration
from task_skills.calibration.service import CalibrationService
from task_skills.calibration import table_calibration as table_math
from task_skills.calibration.state import (
    CalibrationError,
    SessionStore,
    arm_content_hashes,
    fit_status,
    hash_json,
    intrinsics_status,
    validate_arm_metadata,
    validate_teach_points,
    validate_verify_rows,
    verification_status,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPO_ROOT / "task_skills" / "calibration" / "web"
ARMS = ("left", "right")


def wait_job(service: CalibrationService, session_id: str, job_id: str,
             timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = service.session_view(session_id)
        job = next(item for item in state["jobs"] if item["id"] == job_id)
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


class CalibrationBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo_tmp = tempfile.TemporaryDirectory(prefix="calibration-demo-")
        cls.demo_root = Path(cls.demo_tmp.name) / "sessions"
        cls.demo_service = CalibrationService(SessionStore(cls.demo_root))
        session = cls.demo_service.create_session({
            "display_name": "behavior-test-demo", "mode": "synthetic"})
        cls.session_id = session["session_id"]
        demo_job = session["jobs"][0]["id"]
        job = wait_job(cls.demo_service, cls.session_id, demo_job)
        if job["status"] != "passed":
            raise AssertionError(f"demo failed: {job}")

    @classmethod
    def tearDownClass(cls):
        cls.demo_service.close()
        cls.demo_tmp.cleanup()

    def copy_demo(self, label: str) -> tuple[CalibrationService, str, tempfile.TemporaryDirectory]:
        temp = tempfile.TemporaryDirectory(prefix=f"calibration-{label}-")
        root = Path(temp.name) / "sessions"
        root.mkdir(parents=True)
        shutil.copytree(self.demo_root / self.session_id, root / self.session_id)
        self.addCleanup(temp.cleanup)
        service = CalibrationService(SessionStore(root))
        self.addCleanup(service.close)
        return service, self.session_id, temp

    def test_001_session_persists_and_demo_gates_are_server_verified(self):
        reopened = CalibrationService(SessionStore(self.demo_root))
        self.addCleanup(reopened.close)
        view = reopened.session_view(self.session_id)
        self.assertTrue(view["synthetic_only"])
        self.assertTrue(view["ui"]["gates"]["intrinsics"]["ok"])
        self.assertTrue(view["ui"]["gates"]["board"]["ok"])
        for arm in ARMS:
            gate = view["ui"]["gates"]["arms"][arm]
            self.assertTrue(gate["fit_ok"], gate)
            self.assertTrue(gate["verification_ok"], gate)
            self.assertEqual(gate["train_count"], 11)
            self.assertEqual(gate["val_count"], 4)
        self.assertEqual(view["ui"]["counts"], {"intrinsics_train": 18, "intrinsics_val": 6})

    def test_002_empty_session_gates_fit_and_verification(self):
        temp = tempfile.TemporaryDirectory(prefix="calibration-empty-")
        self.addCleanup(temp.cleanup)
        service = CalibrationService(SessionStore(Path(temp.name) / "sessions"))
        self.addCleanup(service.close)
        session_id = service.create_session({"display_name": "empty", "mode": "real"})["session_id"]
        initial = service.store.read(session_id)
        for arm in ARMS:
            self.assertEqual(initial["arms"][arm]["metadata_hash"],
                             hash_json(initial["arms"][arm]["metadata"]))
            self.assertEqual(initial["arms"][arm]["points_hash"],
                             hash_json(initial["arms"][arm]["points"]))
        with self.assertRaises(CalibrationError) as fit:
            service.run_fit(session_id, "left")
        self.assertEqual(fit.exception.code, "INTRINSICS_NOT_READY")
        with self.assertRaises(CalibrationError) as verify:
            service.run_verification(session_id, "left")
        self.assertEqual(verify.exception.code, "FIT_NOT_READY")

    def test_003_intrinsics_minimum_count_and_duplicate_image_rejected(self):
        temp = tempfile.TemporaryDirectory(prefix="calibration-intrinsics-")
        self.addCleanup(temp.cleanup)
        service = CalibrationService(SessionStore(Path(temp.name) / "sessions"))
        self.addCleanup(service.close)
        session_id = service.create_session({"display_name": "counts", "mode": "real"})["session_id"]
        ok, encoded = cv2.imencode(".png", np.zeros((960, 1280, 3), dtype=np.uint8))
        self.assertTrue(ok)
        image = encoded.tobytes()
        service.add_intrinsics_image(session_id, image, "image/png", "train", "blank.png")
        with self.assertRaises(CalibrationError) as duplicate:
            service.add_intrinsics_image(session_id, image, "image/png", "val", "again.png")
        self.assertEqual(duplicate.exception.code, "DUPLICATE_IMAGE")
        job = wait_job(service, session_id, service.run_intrinsics(session_id))
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["exit_code"], 1)
        self.assertIn("10 train / 3 val", job["error"])

    def test_004_teach_minimum_counts_fail_with_retained_error(self):
        service, session_id, _temp = self.copy_demo("counts")
        state = service.store.read(session_id)
        points = copy.deepcopy(state["arms"]["left"]["points"])
        train_ids = {point["corner_id"] for point in points if point["split"] == "train"}
        for corner_id in sorted(train_ids)[5:]:
            points = [point for point in points if point["corner_id"] != corner_id]
        service.update_teach_points(session_id, "left", {"points": points})
        job = wait_job(service, session_id, service.run_fit(session_id, "left"))
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["exit_code"], 1)
        self.assertIn("6 train / 3 val", job["error"])
        self.assertFalse(service.session_view(session_id)["ui"]["gates"]["arms"]["left"]["fit_ok"])

    def test_005_left_right_metadata_and_gate_isolation(self):
        service, session_id, _temp = self.copy_demo("isolation")
        metadata = copy.deepcopy(service.store.read(session_id)["arms"]["left"]["metadata"])
        metadata["robot_id"] = "NOVA2_LEFT_CHANGED"
        service.update_arm_metadata(session_id, "left", metadata)
        gates = service.session_view(session_id)["ui"]["gates"]["arms"]
        self.assertFalse(gates["left"]["fit_ok"])
        self.assertFalse(gates["left"]["verification_ok"])
        self.assertTrue(gates["right"]["fit_ok"])
        self.assertTrue(gates["right"]["verification_ok"])

    def test_006_failed_threshold_is_exit_2_with_evidence_and_no_use(self):
        service, session_id, _temp = self.copy_demo("threshold")
        current = service.store.read(session_id)["settings"]
        settings = {
            "camera_id": current["camera_id"],
            "width": current["image_size_wh"][0],
            "height": current["image_size_wh"][1],
            "board": copy.deepcopy(current["board"]),
            "thresholds": copy.deepcopy(current["thresholds"]),
        }
        settings["thresholds"]["fit_max_error_mm"] = 1e-8
        service.update_settings(session_id, settings)
        job = wait_job(service, session_id, service.run_fit(session_id, "left"))
        self.assertEqual(job["status"], "failed_threshold")
        self.assertEqual(job["exit_code"], 2)
        state = service.store.read(session_id)
        fit = state["arms"]["left"]["fit"]
        self.assertFalse(fit["usable"])
        self.assertFalse(service.session_view(session_id)["ui"]["gates"]["arms"]["left"]["fit_ok"])
        session_dir = service.store.session_dir(session_id)
        self.assertTrue((session_dir / fit["artifact_path"]).is_file())
        self.assertTrue((session_dir / fit["chart_path"]).is_file())
        with self.assertRaises(CalibrationError) as export:
            service.create_export(session_id, {"arms": ["left"]})
        self.assertEqual(export.exception.code, "EXPORT_NOT_READY")

    def test_007_fit_edit_during_job_cannot_commit_validated_result(self):
        service, session_id, _temp = self.copy_demo("fit-race")

        def edit_points() -> None:
            current = service.store.read(session_id)
            points = copy.deepcopy(current["arms"]["left"]["points"])
            points[0]["z"] += 1.0
            service.update_teach_points(session_id, "left", {"points": points})

        service._fit_hook = edit_points
        job = wait_job(service, session_id, service.run_fit(session_id, "left"))
        self.assertEqual(job["status"], "invalidated")
        self.assertEqual(job["exit_code"], 1)
        fit = service.store.read(session_id)["arms"]["left"]["fit"]
        self.assertTrue(fit["stale"])
        self.assertFalse(fit["usable"])
        self.assertIn("输入已更改", fit["stale_reason"])

    def test_008_verification_edit_during_job_cannot_commit_validated_result(self):
        service, session_id, _temp = self.copy_demo("verify-race")
        original = service._compute_verification_artifact

        def edit_after_compute(state, arm, job_id):
            computed = original(state, arm, job_id)
            draft = state["arms"][arm]["verification"]
            service.update_verification_draft(session_id, arm, {
                "rows": draft["rows"],
                "method": draft["method"] + "（并发编辑）",
                "independence_confirmed": True,
            })
            return computed

        service._compute_verification_artifact = edit_after_compute
        job = wait_job(service, session_id, service.run_verification(session_id, "left"))
        self.assertEqual(job["status"], "invalidated")
        result = service.store.read(session_id)["arms"]["left"]["verification"]["result"]
        self.assertTrue(result["stale"])
        self.assertFalse(result["usable"])

    def test_009_input_and_artifact_tampering_invalidate_downstream(self):
        service, session_id, _temp = self.copy_demo("tamper")
        with service.store.edit(session_id) as state:
            state["arms"]["left"]["metadata_hash"] = "0" * 64
        gates = service.session_view(session_id)["ui"]["gates"]["arms"]
        self.assertFalse(gates["left"]["fit_ok"])
        self.assertIn("输入已更改", gates["left"]["fit_reason"])
        self.assertTrue(gates["right"]["fit_ok"])

        service2, session_id2, _temp2 = self.copy_demo("artifact-tamper")
        state = service2.store.read(session_id2)
        artifact = service2.store.session_dir(session_id2) / state["arms"]["left"]["fit"]["artifact_path"]
        artifact.write_bytes(artifact.read_bytes() + b" ")
        gates2 = service2.session_view(session_id2)["ui"]["gates"]["arms"]
        self.assertFalse(gates2["left"]["fit_ok"])
        self.assertIn("产物缺失或被修改", gates2["left"]["fit_reason"])
        with self.assertRaises(CalibrationError) as export:
            service2.create_export(session_id2, {"arms": ["left"]})
        self.assertEqual(export.exception.code, "EXPORT_NOT_READY")

    def test_010_settings_change_invalidates_fits_and_verifications(self):
        service, session_id, _temp = self.copy_demo("settings")
        current = service.store.read(session_id)["settings"]
        settings = {
            "camera_id": current["camera_id"],
            "width": current["image_size_wh"][0],
            "height": current["image_size_wh"][1],
            "board": copy.deepcopy(current["board"]),
            "thresholds": copy.deepcopy(current["thresholds"]),
        }
        settings["thresholds"]["verify_max_error_mm"] = 0.1
        service.update_settings(session_id, settings)
        gates = service.session_view(session_id)["ui"]["gates"]["arms"]
        for arm in ARMS:
            self.assertFalse(gates[arm]["fit_ok"])
            self.assertFalse(gates[arm]["verification_ok"])

    def test_011_nonfinite_duplicates_units_and_csv_metadata_rejected(self):
        with self.assertRaises(CalibrationError) as bad_number:
            validate_teach_points([
                {"corner_id": 0, "split": "train", "x": float("nan"), "y": 0, "z": 0},
            ], 88)
        self.assertEqual(bad_number.exception.code, "NONFINITE_NUMBER")
        with self.assertRaises(CalibrationError) as duplicate:
            validate_teach_points([
                {"corner_id": 0, "split": "train", "x": 0, "y": 0, "z": 0},
                {"corner_id": 0, "split": "val", "x": 1, "y": 1, "z": 1},
            ], 88)
        self.assertEqual(duplicate.exception.code, "DUPLICATE_CORNER_ID")
        with self.assertRaises(CalibrationError) as unit:
            validate_arm_metadata({"robot_id": "r", "base_frame": "b", "tcp_name": "t",
                                   "unit": "cm"})
        self.assertEqual(unit.exception.code, "INVALID_UNIT")
        with self.assertRaises(CalibrationError) as verify_duplicate:
            validate_verify_rows([
                {"u_px": 1, "v_px": 2, "board_z_m": 0, "reference_x_m": 0,
                 "reference_y_m": 0, "reference_z_m": 0},
                {"u_px": 1, "v_px": 2, "board_z_m": .1, "reference_x_m": 0,
                 "reference_y_m": 0, "reference_z_m": 0},
            ], (1280, 960))
        self.assertEqual(verify_duplicate.exception.code, "DUPLICATE_VERIFY_PIXEL")

        service, session_id, _temp = self.copy_demo("csv-unit")
        csv_text = "corner_id,split,x,y,z,unit,robot_id,base_frame,tcp_name\n" \
                   "0,train,10,20,30,m,SIM_NOVA2_LEFT,SIM_BASE_LEFT,SIM_TIP\n"
        with self.assertRaises(CalibrationError) as mismatch:
            service.import_teach_csv(session_id, "left", {"csv": csv_text})
        self.assertEqual(mismatch.exception.code, "CSV_METADATA_MISMATCH")
        blank_teach_csv = "corner_id,split,x,y,z\n0,train, ,20,30\n"
        with self.assertRaises(CalibrationError) as blank_teach:
            service.import_teach_csv(session_id, "left", {"csv": blank_teach_csv})
        self.assertEqual(blank_teach.exception.code, "INVALID_CSV")
        zero_teach_csv = "corner_id,split,x,y,z\n0,train,0,0,0\n"
        imported = service.import_teach_csv(session_id, "left", {"csv": zero_teach_csv})
        self.assertEqual(imported["points"][0]["x"], 0.0)
        blank_verify_csv = (
            "u_px,v_px,board_z_m,reference_x_m,reference_y_m,reference_z_m\n"
            "640,480,0, ,0.2,0.3\n"
        )
        with self.assertRaises(CalibrationError) as blank_verify:
            service.import_verify_csv(session_id, "left", {"csv": blank_verify_csv})
        self.assertEqual(blank_verify.exception.code, "INVALID_CSV")

    def test_012_query_maps_pixels_and_depth_stays_disabled(self):
        service, session_id, _temp = self.copy_demo("query")
        result = service.query(session_id, {"u_px": 640, "v_px": 480, "board_z_m": 0})
        self.assertEqual(result["image_size_wh"], [1280, 960])
        for arm in ARMS:
            self.assertTrue(result["arms"][arm]["ok"])
            self.assertEqual(len(result["arms"][arm]["point_base_m"]), 3)
            self.assertEqual(len(result["arms"][arm]["point_base_mm"]), 3)
        with self.assertRaises(CalibrationError) as depth:
            service.query(session_id, {"u_px": 640, "v_px": 480, "depth_z_m": 1})
        self.assertEqual(depth.exception.code, "DEPTH_DISABLED")
        with self.assertRaises(CalibrationError) as outside:
            service.query(session_id, {"u_px": -1, "v_px": 480, "board_z_m": 0})
        self.assertEqual(outside.exception.code, "PIXEL_OUT_OF_BOUNDS")

    def test_013_independent_verification_is_distinct_and_not_autofilled(self):
        state = self.demo_service.store.read(self.session_id)
        for arm in ARMS:
            artifact_path = state["arms"][arm]["verification"]["result"]["artifact_path"]
            artifact = json.loads((self.demo_service.store.session_dir(self.session_id)
                                   / artifact_path).read_text(encoding="utf-8"))
            self.assertEqual(len(artifact["points"]), 4)
            self.assertTrue(all(point["source"] == "operator_provided_independent_reference"
                                for point in artifact["points"]))
            self.assertIn("distinct from fit held-out points", artifact["validation_scope"])
            fit_scope = state["arms"][arm]["fit"]["summary"]
            self.assertNotIn("verification_method", fit_scope)

    def test_014_export_bundle_and_validated_integration_api(self):
        exported = self.demo_service.create_export(self.session_id, {"arms": ["left", "right"]})
        bundle = self.demo_service.store.session_dir(self.session_id) / exported["bundle_path"]
        self.assertTrue(bundle.is_file())
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            for required in (
                    "report.json", "manifest.json", "session_settings.json",
                    "camera/intrinsics.json", "board/observation.json",
                    "board/confirmation.json",
                    "arms/left/extrinsics.json", "arms/left/verification.json",
                    "arms/left/board_axes.png", "arms/left/residuals.png",
                    "arms/right/extrinsics.json", "arms/right/verification.json",
                    "board/source.png", "board/numbered_corners.png"):
                self.assertIn(required, names)
            report = json.loads(archive.read("report.json"))
            manifest = json.loads(archive.read("manifest.json"))
            intrinsics = json.loads(archive.read("camera/intrinsics.json"))
            board_observation = json.loads(archive.read("board/observation.json"))
            board_confirmation = json.loads(archive.read("board/confirmation.json"))
            extrinsics = json.loads(archive.read("arms/left/extrinsics.json"))
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest, name)
            intrinsics_record = self.demo_service.store.read(self.session_id)["intrinsics"]["result"]
            intrinsics_path = self.demo_service.store.session_dir(self.session_id) / intrinsics_record["artifact_path"]
            self.assertEqual(archive.read("camera/intrinsics.json"), intrinsics_path.read_bytes())
            board_confirmation_state = self.demo_service.store.read(self.session_id)["board"]["confirmation"]
            observation_path = self.demo_service.store.session_dir(self.session_id) / board_confirmation_state["observation_path"]
            self.assertEqual(archive.read("board/observation.json"), observation_path.read_bytes())
        self.assertTrue(report["synthetic_only"])
        self.assertFalse(report["hardware_validation_claimed"])
        self.assertEqual(set(report["arms"]), set(ARMS))
        self.assertIn("report.json", manifest["files"])
        self.assertIn("camera/intrinsics.json", manifest["files"])
        self.assertIn("board/observation.json", manifest["files"])
        self.assertIn("board/confirmation.json", manifest["files"])
        self.assertIn("arms/left/extrinsics.json", manifest["files"])
        self.assertEqual(len(manifest["files"]), len(names) - 1)
        self.assertEqual(intrinsics["kind"], "intrinsics")
        self.assertEqual(len(intrinsics["K"]), 3)
        self.assertEqual(len(intrinsics["distortion"]), 5)
        self.assertEqual(intrinsics["camera_id"], report["camera_id"])
        self.assertEqual(report["intrinsics"]["K"], intrinsics["K"])
        self.assertEqual(report["intrinsics"]["distortion"], intrinsics["distortion"])
        self.assertEqual(len(board_observation["corners_uv"]), 88)
        self.assertEqual(board_confirmation["corner_order"], report["board_corner_order"])
        self.assertEqual(board_confirmation["observation_sha256"],
                         manifest["files"]["board/observation.json"])
        saved_left = self.demo_service.store.read(self.session_id)["arms"]["left"]
        actual_hashes = arm_content_hashes(saved_left)
        self.assertEqual(actual_hashes["metadata_hash"], hash_json(saved_left["metadata"]))
        self.assertEqual(actual_hashes["points_hash"], hash_json(saved_left["points"]))
        self.assertEqual(extrinsics["arm_metadata_hash"], actual_hashes["metadata_hash"])
        self.assertEqual(extrinsics["teach_points_hash"], actual_hashes["points_hash"])
        self.assertEqual(report["arms"]["left"]["metadata_hash"], actual_hashes["metadata_hash"])
        self.assertEqual(report["arms"]["left"]["teach_points_hash"], actual_hashes["points_hash"])

        # The archive alone is sufficient to reproduce the pixel-to-base math;
        # no live session directory or browser state is required.
        extracted = tempfile.TemporaryDirectory(prefix="calibration-export-extracted-")
        self.addCleanup(extracted.cleanup)
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(extracted.name)
        extracted_root = Path(extracted.name).resolve()
        original_session_dir = self.demo_service.store.session_dir(self.session_id).resolve()
        self.assertFalse(extracted_root.is_relative_to(original_session_dir))
        extracted_intrinsics = json.loads((extracted_root / "camera/intrinsics.json").read_text(encoding="utf-8"))
        extracted_extrinsics = json.loads((extracted_root / "arms/left/extrinsics.json").read_text(encoding="utf-8"))
        extracted_observation = json.loads((extracted_root / "board/observation.json").read_text(encoding="utf-8"))
        self.assertEqual(len(extracted_observation["corners_uv"]), 88)
        K = np.asarray(extracted_intrinsics["K"], dtype=float)
        D = np.asarray(extracted_intrinsics["distortion"], dtype=float)
        T_base_camera = np.asarray(extracted_extrinsics["T_base_camera"], dtype=float)
        T_base_board = np.asarray(extracted_extrinsics["T_base_board"], dtype=float)
        archived_point, _ = table_math.pixel_to_plane(
            (640.0, 480.0), extracted_intrinsics, K, D, T_base_camera, T_base_board, 0.0)

        metadata = report["arms"]["left"]["metadata"]
        transform = load_validated_calibration(
            self.demo_root, self.session_id, "left",
            expected_camera_id=report["camera_id"],
            expected_robot_id=metadata["robot_id"],
            expected_base_frame=metadata["base_frame"],
            expected_tcp_name=metadata["tcp_name"],
            expected_unit=metadata["unit"],
            allow_synthetic=True,
        )
        point_m = transform.point_base_m(640, 480, 0.0)
        np.testing.assert_allclose(archived_point, point_m, atol=1e-12)
        np.testing.assert_allclose(transform.point_base_mm(640, 480, 0.0), point_m * 1000)
        self.assertTrue(transform.snapshot()["hashes"]["fit_artifact_sha256"])
        with self.assertRaises(CalibrationError) as synthetic:
            load_validated_calibration(
                self.demo_root, self.session_id, "left",
                expected_camera_id=report["camera_id"],
                expected_robot_id=metadata["robot_id"],
                expected_base_frame=metadata["base_frame"],
                expected_tcp_name=metadata["tcp_name"], expected_unit=metadata["unit"])
        self.assertEqual(synthetic.exception.code, "SYNTHETIC_REJECTED")
        with self.assertRaises(CalibrationError) as wrong_robot:
            load_validated_calibration(
                self.demo_root, self.session_id, "left",
                expected_camera_id=report["camera_id"], expected_robot_id="WRONG",
                expected_base_frame=metadata["base_frame"],
                expected_tcp_name=metadata["tcp_name"], expected_unit=metadata["unit"],
                allow_synthetic=True)
        self.assertEqual(wrong_robot.exception.code, "METADATA_MISMATCH")

    def test_021_actual_saved_arm_payload_hash_tampering_rejects_downstream_use(self):
        service, session_id, _temp = self.copy_demo("saved-points-tamper")
        with service.store.edit(session_id) as state:
            state["arms"]["left"]["points"][0]["x"] += 99999
        state = service.store.read(session_id)
        session_dir = service.store.session_dir(session_id)
        fit_ok, fit_reason = fit_status(state, session_dir, "left")
        verify_ok, verify_reason = verification_status(state, session_dir, "left")
        self.assertFalse(fit_ok)
        self.assertFalse(verify_ok)
        self.assertIn("示教点哈希与实际内容不一致", fit_reason)
        self.assertIn("示教点哈希与实际内容不一致", verify_reason)
        query = service.query(session_id, {"u_px": 640, "v_px": 480, "board_z_m": 0})
        self.assertFalse(query["arms"]["left"]["ok"])
        self.assertEqual(query["arms"]["left"]["code"], "FIT_NOT_READY")
        with self.assertRaises(CalibrationError) as export:
            service.create_export(session_id, {"arms": ["left"]})
        self.assertEqual(export.exception.code, "EXPORT_NOT_READY")
        metadata = state["arms"]["left"]["metadata"]
        with self.assertRaises(CalibrationError) as integration:
            load_validated_calibration(
                Path(_temp.name) / "sessions", session_id, "left",
                expected_camera_id=state["settings"]["camera_id"],
                expected_robot_id=metadata["robot_id"],
                expected_base_frame=metadata["base_frame"],
                expected_tcp_name=metadata["tcp_name"], expected_unit=metadata["unit"],
                allow_synthetic=True,
            )
        self.assertEqual(integration.exception.code, "FIT_NOT_VALID")

        metadata_service, metadata_session_id, metadata_temp = self.copy_demo("saved-metadata-tamper")
        with metadata_service.store.edit(metadata_session_id) as current:
            current["arms"]["left"]["metadata"]["base_frame"] = "tampered_base"
        changed = metadata_service.store.read(metadata_session_id)
        changed_dir = metadata_service.store.session_dir(metadata_session_id)
        metadata_fit_ok, metadata_fit_reason = fit_status(changed, changed_dir, "left")
        metadata_verify_ok, metadata_verify_reason = verification_status(changed, changed_dir, "left")
        self.assertFalse(metadata_fit_ok)
        self.assertFalse(metadata_verify_ok)
        self.assertIn("元数据哈希与实际内容不一致", metadata_fit_reason)
        self.assertIn("元数据哈希与实际内容不一致", metadata_verify_reason)
        metadata_query = metadata_service.query(
            metadata_session_id, {"u_px": 640, "v_px": 480, "board_z_m": 0})
        self.assertFalse(metadata_query["arms"]["left"]["ok"])
        with self.assertRaises(CalibrationError) as metadata_export:
            metadata_service.create_export(metadata_session_id, {"arms": ["left"]})
        self.assertEqual(metadata_export.exception.code, "EXPORT_NOT_READY")
        changed_metadata = changed["arms"]["left"]["metadata"]
        with self.assertRaises(CalibrationError) as metadata_integration:
            load_validated_calibration(
                Path(metadata_temp.name) / "sessions", metadata_session_id, "left",
                expected_camera_id=changed["settings"]["camera_id"],
                expected_robot_id=changed_metadata["robot_id"],
                expected_base_frame=changed_metadata["base_frame"],
                expected_tcp_name=changed_metadata["tcp_name"],
                expected_unit=changed_metadata["unit"], allow_synthetic=True,
            )
        self.assertEqual(metadata_integration.exception.code, "FIT_NOT_VALID")

    def test_015_getpose_is_explicit_single_command_and_not_saved(self):
        service, session_id, _temp = self.copy_demo("getpose")
        metadata = copy.deepcopy(service.store.read(session_id)["arms"]["left"]["metadata"])
        metadata["controller_host"] = "controller.invalid"
        service.update_arm_metadata(session_id, "left", metadata)
        sock = MagicMock()
        sock.__enter__.return_value = sock
        sock.recv.return_value = b"0,{100,200,300,0,90,0},GetPose();"
        with patch("task_skills.calibration.nova2_read_pose.socket.create_connection",
                   return_value=sock):
            result = service.get_pose(session_id, "left", {
                "confirm_base": True, "confirm_tcp": True, "confirm_unit": True})
        sock.sendall.assert_called_once_with(b"GetPose()")
        self.assertEqual(result["xyz"], [100.0, 200.0, 300.0])
        self.assertFalse(result["auto_saved"])
        self.assertEqual(result["network_command"], "GetPose()")
        with self.assertRaises(CalibrationError) as confirmation:
            service.get_pose(session_id, "left", {"confirm_base": True})
        self.assertEqual(confirmation.exception.code, "POSE_CONFIRMATION_REQUIRED")

    def test_016_demo_job_cannot_be_submitted_for_real_session(self):
        temp = tempfile.TemporaryDirectory(prefix="calibration-demo-guard-")
        self.addCleanup(temp.cleanup)
        service = CalibrationService(SessionStore(Path(temp.name) / "sessions"))
        self.addCleanup(service.close)
        session_id = service.create_session({"display_name": "real", "mode": "real"})["session_id"]
        with self.assertRaises(CalibrationError) as demo:
            service._submit_job(session_id, "demo", None, service._run_demo)
        self.assertEqual(demo.exception.code, "DEMO_REQUIRES_SYNTHETIC")

    def test_017_board_previews_are_known_servable_assets(self):
        state = self.demo_service.store.read(self.session_id)
        for relative in list(state["board"]["previews"].values()) + list(
                state["board"]["observations"].values()):
            path = self.demo_service.asset(self.session_id, relative)
            self.assertTrue(path.is_file(), relative)

    def test_018_intrinsics_source_bytes_tampering_invalidates_all_downstream(self):
        service, session_id, _temp = self.copy_demo("source-tamper")
        state = service.store.read(session_id)
        image = state["intrinsics"]["images"][0]
        path = service.store.session_dir(session_id) / image["original_path"]
        original_sha256 = image["sha256"]

        # Replace with a different, valid PNG of the same resolution. The task
        # hash list still points at the old bytes, so it must not be accepted.
        ok, encoded = cv2.imencode(".png", np.zeros((960, 1280, 3), dtype=np.uint8))
        self.assertTrue(ok)
        path.write_bytes(encoded.tobytes())
        self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_sha256)

        session_dir = service.store.session_dir(session_id)
        intr_ok, intr_reason = intrinsics_status(state, session_dir)
        self.assertFalse(intr_ok)
        self.assertIn("哈希不匹配", intr_reason)
        for arm in ARMS:
            fit_ok, fit_reason = fit_status(state, session_dir, arm)
            verify_ok, verify_reason = verification_status(state, session_dir, arm)
            self.assertFalse(fit_ok)
            self.assertFalse(verify_ok)
            self.assertIn("内参", fit_reason)
            self.assertIn("内参", verify_reason)
        with self.assertRaises(CalibrationError) as export:
            service.create_export(session_id, {"arms": ["left"]})
        self.assertEqual(export.exception.code, "EXPORT_NOT_READY")

        job_id = service.run_intrinsics(session_id)
        job = wait_job(service, session_id, job_id)
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["exit_code"], 1)
        self.assertIn("INTRINSICS_SOURCE_CHANGED", job["error"])
        self.assertFalse(service.session_view(session_id)["ui"]["gates"]["intrinsics"]["ok"])

    def test_019_completion_faults_leave_terminal_error_and_allow_retry(self):
        temp = tempfile.TemporaryDirectory(prefix="calibration-job-fault-")
        self.addCleanup(temp.cleanup)
        service = CalibrationService(SessionStore(Path(temp.name) / "sessions"), max_workers=1)
        self.addCleanup(service.close)
        session_id = service.create_session({"display_name": "faults", "mode": "real"})["session_id"]

        def failing_commit(ctx):
            ctx.phase("即将提交", 0.5)

            def commit(_state):
                raise OSError("injected commit failure")

            ctx.finish(0, {"unexpected": True}, commit)

        first_id = service._submit_job(session_id, "fit", "left", failing_commit)
        first = wait_job(service, session_id, first_id)
        self.assertEqual(first["status"], "error")
        self.assertEqual(first["exit_code"], 1)
        self.assertIn("injected commit failure", first["error"])
        self.assertTrue(first["finished_utc"])
        current = service.store.read(session_id)
        self.assertIsNone(current["arms"]["left"].get("fit_job_id"))
        self.assertFalse(any(job["id"] == first_id and job["status"] in ("queued", "running")
                             for job in current["jobs"]))

        def retry_target(ctx):
            ctx.finish(0, {"retry": "available"})

        retry_id = service._submit_job(session_id, "fit", "left", retry_target)
        retry = wait_job(service, session_id, retry_id)
        self.assertEqual(retry["status"], "passed")
        self.assertEqual(retry["result_summary"]["retry"], "available")

    def test_020_state_and_log_write_faults_do_not_orphan_running_job(self):
        temp = tempfile.TemporaryDirectory(prefix="calibration-io-fault-")
        self.addCleanup(temp.cleanup)
        service = CalibrationService(SessionStore(Path(temp.name) / "sessions"), max_workers=1)
        self.addCleanup(service.close)
        session_id = service.create_session({"display_name": "io-faults", "mode": "real"})["session_id"]

        def state_failure(ctx):
            original_edit = service.store.edit
            calls = {"count": 0}

            @contextlib.contextmanager
            def flaky_edit(target_session_id):
                if calls["count"] == 0:
                    calls["count"] += 1
                    raise OSError("injected state write failure")
                with original_edit(target_session_id) as state:
                    yield state

            with patch.object(service.store, "edit", side_effect=flaky_edit):
                ctx.finish(0, {"unexpected": True})

        state_job_id = service._submit_job(session_id, "fit", "right", state_failure)
        state_job = wait_job(service, session_id, state_job_id)
        self.assertEqual(state_job["status"], "error")
        self.assertEqual(state_job["exit_code"], 1)
        self.assertIn("injected state write failure", state_job["error"])

        def log_failure(ctx):
            def broken_log(_message):
                raise OSError("injected log write failure")

            ctx._append_log = broken_log
            ctx.finish(0, {"log_failure_isolated": True})

        log_job_id = service._submit_job(session_id, "verify", "left", log_failure)
        log_job = wait_job(service, session_id, log_job_id)
        self.assertEqual(log_job["status"], "error")
        self.assertEqual(log_job["exit_code"], 1)
        self.assertIn("终态日志写入失败", log_job["error"])
        self.assertTrue(log_job["finished_utc"])

        def retry_after_log_failure(ctx):
            ctx.finish(0, {"retry": "available after log failure"})

        retry_id = service._submit_job(session_id, "verify", "left", retry_after_log_failure)
        retry_job = wait_job(service, session_id, retry_id)
        self.assertEqual(retry_job["status"], "passed")
        self.assertEqual(retry_job["result_summary"]["retry"], "available after log failure")


class CalibrationHTTPBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="calibration-http-")
        self.addCleanup(self.temp.cleanup)
        self.server = create_server("127.0.0.1", 0, Path(self.temp.name) / "sessions", WEB_ROOT)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        self.host = f"127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server.camera.close()
        self.server.service.close()
        self.thread.join(timeout=3)

    def request(self, method: str, path: str, *, body: bytes | None = None,
                headers: dict[str, str] | None = None) -> tuple[int, dict, dict]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            data = response.read()
            return response.status, dict(response.getheaders()), json.loads(data)
        finally:
            connection.close()

    def bootstrap(self) -> dict:
        status, _headers, body = self.request("GET", "/api/bootstrap")
        self.assertEqual(status, 200)
        return body

    def test_bootstrap_and_security_headers(self):
        status, headers, body = self.request("GET", "/api/bootstrap")
        self.assertEqual(status, 200)
        self.assertIn("csrf_token", body)
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertNotIn("Access-Control-Allow-Origin", headers)

    def test_cross_origin_host_and_token_rejected(self):
        status, _headers, body = self.request(
            "GET", "/api/bootstrap", headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(body["code"], "CROSS_ORIGIN_REJECTED")
        status, _headers, body = self.request(
            "GET", "/api/bootstrap", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(body["code"], "HOST_REJECTED")
        status, _headers, body = self.request(
            "POST", "/api/sessions", body=b"{}",
            headers={"Content-Type": "application/json"})
        self.assertEqual(status, 403)
        self.assertEqual(body["code"], "CSRF_TOKEN_INVALID")

    def test_write_requires_same_origin_token_and_bounded_json(self):
        token = self.bootstrap()["csrf_token"]
        origin = f"http://{self.host}"
        status, _headers, body = self.request(
            "POST", "/api/sessions",
            body=json.dumps({"display_name": "http-test", "mode": "real"}).encode(),
            headers={"Content-Type": "application/json", "Origin": origin,
                     "X-Calibration-Token": token})
        self.assertEqual(status, 201, body)
        session_id = body["session"]["session_id"]
        self.assertRegex(session_id, r"^cal-[a-z0-9-]+$")
        status, _headers, body = self.request(
            "POST", "/api/sessions", body=b"[]",
            headers={"Content-Type": "application/json", "Origin": origin,
                     "X-Calibration-Token": token})
        self.assertEqual(status, 400)
        self.assertEqual(body["code"], "INVALID_JSON")
        status, _headers, body = self.request(
            "POST", "/api/sessions", body=b"{}",
            headers={"Content-Type": "application/json", "Origin": origin,
                     "X-Calibration-Token": token,
                     "Content-Length": str(MAX_JSON_BODY + 1)})
        self.assertEqual(status, 413)
        self.assertEqual(body["code"], "REQUEST_TOO_LARGE")

    def test_path_traversal_is_not_served(self):
        status, _headers, body = self.request(
            "GET", "/static/%2e%2e/%2e%2e/README.md")
        self.assertEqual(status, 404)
        self.assertIn("STATIC", body["code"])
        status, _headers, body = self.request("GET", "/api/sessions/../session.json")
        self.assertIn(status, (400, 404))
        self.assertNotIn("schema_version", body)
        status, _headers, body = self.request("GET", "/api/sessions/not_a_session/assets/x.png")
        self.assertEqual(status, 400)
        self.assertEqual(body["code"], "INVALID_SESSION_ID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
