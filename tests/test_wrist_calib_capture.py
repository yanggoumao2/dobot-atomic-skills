from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from chem_demo import run_wrist_calib
from chem_demo.wrist_camera import LiveWristCamera, SimWristCamera
from chem_demo.wrist_runtime import TABLE_Z as TABLE_HEIGHTS
from task_skills.run_cartesian_jog import SimArm, snap_down_rpy


HOME = [117.6, -344.8, 415.5]
TABLE_Z = 243.2


class WristCaptureTests(unittest.TestCase):
    def make_rig(self, width: int = 64, height: int = 36):
        arm = SimArm(HOME, snap_down_rpy([180.0, 0.0, 90.0]))
        camera = SimWristCamera(arm, table_z=TABLE_Z, width=width, height=height)
        return arm, camera

    def make_interrupted_sim_dataset(self, root: Path):
        source = root / "interrupted-right"
        arm = SimArm(HOME, snap_down_rpy([180.0, 0.0, 90.0]))
        camera = SimWristCamera(arm, table_z=TABLE_HEIGHTS["right"], width=64, height=36)
        args = run_wrist_calib.build_parser().parse_args(
            ["--sim", "--yes", "--arm", "right", "--table-z", str(TABLE_HEIGHTS["right"]),
             "--capture-only", "--capture-burst", "2"])
        run_wrist_calib.run(arm, camera, args, source, ask=lambda _: "y", out=lambda _: None,
                            stop_pressed=lambda: False)

        acquisition_path = source / "acquisition.json"
        poses_path = source / "poses.json"
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        poses = json.loads(poses_path.read_text(encoding="utf-8"))
        for later_group in ("z-yaw+", "ref_return"):
            group = acquisition["pose_groups"].pop(later_group)
            for capture in group.get("captures", []):
                for name in capture.get("files", {}).values():
                    (source / name).unlink(missing_ok=True)
            representative = group.get("representative_depth_file")
            if representative:
                (source / representative).unlink(missing_ok=True)
            preview = group.get("representative_depth_preview")
            if preview:
                (source / preview).unlink(missing_ok=True)
            poses.pop(later_group, None)

        z_group = acquisition["pose_groups"]["z-"]
        interrupted_capture = z_group["captures"][1]
        interrupted_capture["status"] = "capturing"
        interrupted_capture.pop("robot_after", None)
        interrupted_capture.pop("robot_delta_during_capture", None)
        interrupted_capture.pop("capture_metadata", None)
        interrupted_capture.pop("frame_shape", None)
        for name in interrupted_capture.get("files", {}).values():
            (source / name).unlink(missing_ok=True)
        representative = z_group.get("representative_depth_file")
        if representative:
            (source / representative).unlink(missing_ok=True)
        preview = z_group.get("representative_depth_preview")
        if preview:
            (source / preview).unlink(missing_ok=True)
        z_group["representative_depth_file"] = None
        z_group.pop("representative_depth_preview", None)
        z_group.pop("representative_depth_unit", None)
        z_group.pop("depth_aggregation", None)
        z_group.pop("pose_stability", None)
        acquisition["status"] = "in_progress"
        acquisition.pop("capture_only", None)
        acquisition.pop("finished_at", None)
        acquisition_path.write_text(json.dumps(acquisition, indent=2), encoding="utf-8")
        poses_path.write_text(json.dumps(poses, indent=2), encoding="utf-8")

        snapshot = interrupted_capture["robot_before"]
        arm.pose = list(snapshot["tcp_pose_mm_deg"])
        arm.joints = list(snapshot["joints_deg"])
        arm.sent.clear()
        return arm, camera, source

    def test_capture_burst_is_bounded_and_defaults_to_one(self):
        parser = run_wrist_calib.build_parser()
        self.assertEqual(parser.parse_args(["--sim"]).capture_burst, 1)
        self.assertEqual(parser.parse_args(["--sim", "--capture-burst", "15"]).capture_burst, 15)
        with redirect_stderr(StringIO()):
            for value in ("0", "16", "1.5", "many"):
                with self.subTest(value=value), self.assertRaises(SystemExit):
                    parser.parse_args(["--sim", "--capture-burst", value])

    def test_capture_only_finishes_without_solving_or_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            arm, camera = self.make_rig()
            with (patch.object(run_wrist_calib, "process") as solve,
                  patch.object(run_wrist_calib, "save_calibration") as publish,
                  patch.object(arm, "close", wraps=arm.close) as close_arm,
                  patch.object(camera, "close", wraps=camera.close) as close_camera):
                code = run_wrist_calib.main(
                    ["--sim", "--yes", "--capture-only", "--track-board-route", "--capture-burst", "2", "--out", tmp],
                    arm=arm, camera=camera, ask=lambda _: "y", out=lambda _: None, stop_pressed=lambda: False)
            self.assertEqual(code, 0)
            solve.assert_not_called()
            publish.assert_not_called()
            close_camera.assert_called_once_with()
            close_arm.assert_called_once_with()
            manifest = json.loads(next(Path(tmp).glob("*/acquisition.json")).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["capture_only"])
            self.assertEqual(len(manifest["pose_groups"]["ref_return"]["captures"]), 2)
            self.assertTrue(all(g["gripper_position"] >= 250 for g in manifest["pose_groups"].values()))
            route = manifest["route_observations"]
            self.assertEqual(route[-1]["to"], "ref_return")
            self.assertTrue(all(next(Path(tmp).glob("*/" + item["color"])).is_file() for item in route))

    def test_blocked_gripper_is_not_treated_as_open(self):
        arm, camera = self.make_rig()
        args = run_wrist_calib.build_parser().parse_args(["--sim", "--yes", "--capture-only"])
        with tempfile.TemporaryDirectory() as tmp, patch.object(arm, "gripper_position", return_value=134):
            with self.assertRaisesRegex(run_wrist_calib.Aborted, "134"):
                run_wrist_calib.run(arm, camera, args, Path(tmp), ask=lambda _: "y", out=lambda _: None,
                                    stop_pressed=lambda: False)
            self.assertFalse((Path(tmp) / "ref.png").exists())

    def test_temporal_median_requires_half_and_preserves_fractional_millimeters(self):
        samples = [
            {"depth": np.array([[1.25, 0.0, 7.0]], np.float32)},
            {"depth": np.array([[2.50, 4.25, 0.0]], np.float32)},
            {"depth": np.array([[1.75, 0.0, 8.0]], np.float32)},
        ]

        median, summary = run_wrist_calib._temporal_median_depth(samples)

        np.testing.assert_allclose(median, [[1.75, 0.0, 7.5]])
        self.assertEqual(median.dtype, np.float32)
        self.assertEqual(summary["minimum_valid_observations"], 2)
        self.assertEqual(summary["depth_unit"], "mm")
        self.assertAlmostEqual(summary["valid_pixel_fraction"], 2 / 3, places=6)

    def test_burst_persists_replayable_observations_and_return_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            arm, camera = self.make_rig()
            seen: dict = {}

            def fake_process(frames, poses, K, table_z, out):
                seen["frames"] = frames
                return {"problems": []}

            args = run_wrist_calib.build_parser().parse_args(
                ["--sim", "--yes", "--arm", "left", "--table-z", str(TABLE_Z), "--capture-burst", "3"])
            try:
                with patch("chem_demo.run_wrist_calib.process", side_effect=fake_process):
                    result = run_wrist_calib.run(
                        arm, camera, args, folder, ask=lambda _: "y", out=lambda _: None,
                        stop_pressed=lambda: False)
            finally:
                camera.close()

            acquisition = json.loads((folder / "acquisition.json").read_text(encoding="utf-8"))
            poses = json.loads((folder / "poses.json").read_text(encoding="utf-8"))
            labels = [move[0] for move in run_wrist_calib.MOVES]
            self.assertEqual(list(poses), labels)
            self.assertNotIn("ref_return", poses)
            self.assertEqual(acquisition["status"], "complete")
            self.assertEqual(acquisition["capture_burst"], 3)
            self.assertEqual(acquisition["table_z_reference_mm"], TABLE_Z)
            self.assertIsNone(acquisition["camera"]["serial"])
            self.assertEqual(acquisition["camera"]["depth_output_unit"], "mm")
            self.assertIsNone(acquisition["camera"]["depth_scale_mm_per_raw_unit"])
            self.assertEqual(len(acquisition["pose_groups"]["ref"]["captures"]), 3)
            self.assertEqual(len(acquisition["pose_groups"]["ref_return"]["captures"]), 3)
            self.assertTrue(acquisition["return_home_drift"]["stable"])

            for label in labels:
                group = acquisition["pose_groups"][label]
                self.assertEqual(len(group["captures"]), 3)
                self.assertTrue(group["pose_stability"]["stable"])
                for capture in group["captures"]:
                    self.assertIn("robot_state_source", capture["robot_before"])
                    self.assertEqual(len(capture["robot_before"]["joints_deg"]), 6)
                    self.assertEqual(len(capture["robot_after"]["tcp_pose_mm_deg"]), 6)
                    metadata = capture["capture_metadata"]
                    self.assertEqual(metadata["source"], "simulation")
                    self.assertEqual(metadata["frame_counter_units"], "frames")
                    self.assertTrue((folder / capture["files"]["depth_mm_npy"]).is_file())
                self.assertTrue((folder / f"{label}.png").is_file())
                self.assertTrue((folder / f"{label}_depth.png").is_file())
                self.assertTrue((folder / f"{label}_burst_01.png").is_file())
            self.assertTrue((folder / "ref_return.png").is_file())
            representative_path = folder / "ref_depth_temporal_median_mm.npy"
            representative = np.load(representative_path, allow_pickle=False)
            self.assertEqual(representative.dtype, np.float32)
            self.assertTrue(np.isfinite(representative).all())
            np.testing.assert_array_equal(seen["frames"]["ref"]["depth"], representative)
            self.assertEqual(result["problems"], [])

    def test_partial_burst_keeps_completed_pose_and_failed_sample_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            arm, simulated = self.make_rig()

            class FailsOnSecondFrame:
                K = simulated.K

                def __init__(self):
                    self.calls = 0

                def describe(self):
                    return simulated.describe()

                def grab(self, flush=4):
                    self.calls += 1
                    if self.calls == 2:
                        raise RuntimeError("synthetic camera timeout")
                    return simulated.grab(flush=flush)

            args = run_wrist_calib.build_parser().parse_args(
                ["--sim", "--yes", "--arm", "left", "--table-z", str(TABLE_Z), "--capture-burst", "3"])
            with self.assertRaisesRegex(RuntimeError, "synthetic camera timeout"):
                run_wrist_calib.run(arm, FailsOnSecondFrame(), args, folder, ask=lambda _: "y",
                                    out=lambda _: None, stop_pressed=lambda: False)

            poses = json.loads((folder / "poses.json").read_text(encoding="utf-8"))
            acquisition = json.loads((folder / "acquisition.json").read_text(encoding="utf-8"))
            self.assertEqual(list(poses), ["ref"])
            self.assertEqual(acquisition["status"], "partial")
            captures = acquisition["pose_groups"]["ref"]["captures"]
            self.assertEqual([capture["status"] for capture in captures], ["captured", "failed"])
            self.assertTrue((folder / "ref_depth_mm.npy").is_file())

    def test_resume_cli_requires_capture_only_no_home_and_manual_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "missing-source"
            invalid_argv = [
                ["--sim", "--resume-capture", str(source)],
                ["--sim", "--capture-only", "--resume-capture", str(source)],
                ["--sim", "--capture-only", "--no-home", "--yes", "--resume-capture", str(source)],
            ]
            for argv in invalid_argv:
                with self.subTest(argv=argv), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    run_wrist_calib.main(argv, out=lambda _: None)

    def test_resume_copies_interrupted_data_and_recaptures_z_route_under_original_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm, camera, source = self.make_interrupted_sim_dataset(root)
            original_acquisition = (source / "acquisition.json").read_bytes()
            original_poses = (source / "poses.json").read_bytes()
            original_acquisition_data = json.loads(original_acquisition)
            original_poses_data = json.loads(original_poses)
            resume_output = root / "recovery-output"
            prompts: list[str] = []
            output: list[str] = []

            code = run_wrist_calib.main(
                ["--sim", "--capture-only", "--no-home", "--arm", "right",
                 "--resume-capture", str(source), "--out", str(resume_output)],
                arm=arm, camera=camera, ask=lambda prompt: (prompts.append(prompt) or "y"),
                out=output.append, stop_pressed=lambda: False)

            self.assertEqual(code, 0, "\n".join(output))
            self.assertEqual(len(prompts), 3)  # recapture z-, move to z-yaw+, return to ref
            self.assertIn("full z- burst", prompts[0])
            self.assertIn("z-", prompts[1])
            self.assertIn("ref", prompts[2])
            self.assertEqual((source / "acquisition.json").read_bytes(), original_acquisition)
            self.assertEqual((source / "poses.json").read_bytes(), original_poses)

            recovered_folder = next(resume_output.iterdir())
            recovered_manifest_path = recovered_folder / "acquisition.json"
            recovered = json.loads(recovered_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(recovered["status"], "complete")
            self.assertTrue(recovered["capture_only"])
            self.assertEqual(recovered["recovery"]["source_dataset"], str(source.resolve()))
            self.assertEqual((recovered_folder / "acquisition_interrupted.json").read_bytes(), original_acquisition)
            self.assertEqual((recovered_folder / "poses_interrupted.json").read_bytes(), original_poses)
            for label in ("ref", "yaw+", "yaw-", "x+", "x-", "y+", "y-", "z-", "z-yaw+", "ref_return"):
                self.assertIn(label, recovered["pose_groups"])
            for label in ("z-", "z-yaw+", "ref_return"):
                group = recovered["pose_groups"][label]
                self.assertEqual(len(group["captures"]), 2)
                self.assertTrue(group["pose_stability"]["stable"])
                self.assertTrue(all(capture["status"] == "captured" for capture in group["captures"]))
                self.assertTrue(all(capture["files"]["color_png"].startswith("recovery_")
                                    for capture in group["captures"]))
            self.assertTrue(recovered["return_home_drift"]["stable"])

            recovered_poses = json.loads((recovered_folder / "poses.json").read_text(encoding="utf-8"))
            ref_pose = original_poses_data["ref"]
            self.assertAlmostEqual(recovered_poses["z-"][2], ref_pose[2] - 50.0, delta=0.5)
            self.assertAlmostEqual(recovered_poses["z-yaw+"][2], ref_pose[2] - 50.0, delta=0.5)
            self.assertAlmostEqual(recovered_poses["z-yaw+"][5], ref_pose[5] + 15.0, delta=0.5)
            self.assertAlmostEqual(arm.pose[2], ref_pose[2], delta=0.5)
            self.assertAlmostEqual(arm.pose[5], ref_pose[5], delta=0.5)
            self.assertEqual(arm.sent[0][0], original_poses_data["z-"][:3])
            self.assertEqual(arm.sent[0][1][2], original_poses_data["z-"][5] + 5.0)

    def test_resume_preflight_rejects_pose_joints_open_and_camera_mismatch_before_copy_or_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm, camera, source = self.make_interrupted_sim_dataset(root)
            resume_output = root / "recovery-output"
            source_manifest = source / "acquisition.json"
            saved_manifest = source_manifest.read_bytes()

            def attempt(*, current_arm=arm, current_camera=camera):
                return run_wrist_calib.main(
                    ["--sim", "--capture-only", "--no-home", "--arm", "right",
                     "--resume-capture", str(source), "--out", str(resume_output)],
                    arm=current_arm, camera=current_camera, ask=lambda _: "y", out=lambda _: None,
                    stop_pressed=lambda: False)

            original_pose = list(arm.pose)
            original_joints = list(arm.joints)
            arm.pose[0] += 1.0
            self.assertEqual(attempt(), 1)
            arm.pose = original_pose
            arm.pose[5] += 1.0
            self.assertEqual(attempt(), 1)
            arm.pose = original_pose
            arm.joints[0] += 1.0
            self.assertEqual(attempt(), 1)
            arm.joints = original_joints
            arm.gripper_pos = 134
            self.assertEqual(attempt(), 1)
            arm.gripper_pos = 255

            wrong_camera = SimWristCamera(arm, table_z=TABLE_HEIGHTS["right"], width=80, height=36)
            self.assertEqual(attempt(current_camera=wrong_camera), 1)
            wrong_camera.close()

            class CameraWithChangedDescription:
                def __init__(self, *, distortion=None, undistorted=None):
                    self.distortion = distortion
                    self.undistorted = undistorted

                def describe(self):
                    description = copy.deepcopy(camera.describe())
                    intrinsics = description["intrinsics"]
                    if self.distortion is not None:
                        intrinsics["distortion_coefficients"] = self.distortion
                    if self.undistorted is not None:
                        intrinsics["distortion_removed_from_output"] = self.undistorted
                    return description

                def grab(self, *args, **kwargs):
                    return camera.grab(*args, **kwargs)

                def close(self):
                    pass

            changed_distortion = CameraWithChangedDescription(distortion=[0.01, 0.0, 0.0, 0.0, 0.0])
            self.assertEqual(attempt(current_camera=changed_distortion), 1)
            changed_undistortion = CameraWithChangedDescription(undistorted=False)
            self.assertEqual(attempt(current_camera=changed_undistortion), 1)

            manifest = json.loads(saved_manifest)
            manifest["camera"]["serial"] = "different-serial"
            source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(attempt(), 1)
            source_manifest.write_bytes(saved_manifest)

            self.assertFalse(resume_output.exists())
            self.assertEqual(arm.sent, [])
            self.assertEqual(source_manifest.read_bytes(), saved_manifest)

    def test_resume_rechecks_open_feedback_after_each_manual_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm, camera, source = self.make_interrupted_sim_dataset(root)
            resume_output = root / "recovery-output"
            prompts: list[str] = []

            def confirm(prompt):
                prompts.append(prompt)
                if len(prompts) == 2:
                    arm.gripper_pos = 134
                return "y"

            code = run_wrist_calib.main(
                ["--sim", "--capture-only", "--no-home", "--arm", "right",
                 "--resume-capture", str(source), "--out", str(resume_output)],
                arm=arm, camera=camera, ask=confirm, out=lambda _: None, stop_pressed=lambda: False)

            self.assertEqual(code, 1)
            self.assertEqual(len(prompts), 2)
            self.assertIn("z-yaw+", prompts[1])
            self.assertEqual(arm.sent, [])
            recovered = json.loads(next(resume_output.iterdir()).joinpath("acquisition.json").read_text())
            self.assertEqual(recovered["status"], "partial")
            self.assertNotIn("z-yaw+", recovered["pose_groups"])

    def test_duplicate_simulated_frame_counter_rejects_burst_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            arm, simulated = self.make_rig()

            class RepeatedCounterCamera:
                K = simulated.K

                def describe(self):
                    return simulated.describe()

                def grab(self, flush=4):
                    frame = simulated.grab(flush=flush)
                    frame["capture_metadata"]["frame_counter"] = 1
                    return frame

            args = run_wrist_calib.build_parser().parse_args(
                ["--sim", "--yes", "--arm", "left", "--table-z", str(TABLE_Z), "--capture-burst", "2"])
            try:
                with patch("chem_demo.run_wrist_calib.process", return_value={"problems": []}):
                    result = run_wrist_calib.run(
                        arm, RepeatedCounterCamera(), args, folder, ask=lambda _: "y", out=lambda _: None,
                        stop_pressed=lambda: False)
            finally:
                simulated.close()
            self.assertTrue(result["problems"])
            self.assertIn("frame counters", result["problems"][0])
            self.assertEqual(result["acquisition"]["status"], "rejected")

    def test_live_camera_failure_happens_before_arm_session_can_move(self):
        class ArmSpy:
            def __init__(self):
                self.opened = 0
                self.closed = 0

            def open_session(self, **_kwargs):
                self.opened += 1

            def close(self):
                self.closed += 1

        with tempfile.TemporaryDirectory() as tmp:
            arm = ArmSpy()
            output: list[str] = []
            with patch("chem_demo.run_wrist_calib.camera_serial", return_value="missing-camera"), \
                    patch("chem_demo.wrist_camera.LiveWristCamera", side_effect=RuntimeError("camera missing")):
                code = run_wrist_calib.main(
                    ["--live", "--no-home", "--out", tmp], arm=arm, out=output.append)
        self.assertEqual(code, 4)
        self.assertEqual(arm.opened, 0)
        self.assertEqual(arm.closed, 1)

    def test_live_camera_retains_sdk_timestamps_counters_intrinsics_and_scale(self):
        class FakeFrame:
            def __init__(self, data, timestamp, counter):
                self.data, self.timestamp, self.counter = data, timestamp, counter

            def get_data(self):
                return self.data

            def get_timestamp(self):
                return self.timestamp

            def get_frame_number(self):
                return self.counter

            def get_frame_timestamp_domain(self):
                return "hardware_clock"

        class FakeFrameset:
            def __init__(self, timestamp, color_counter, depth_counter):
                self.color = FakeFrame(np.zeros((6, 8, 3), np.uint8), timestamp, color_counter)
                self.depth = FakeFrame(np.full((6, 8), 1000, np.uint16), timestamp + 0.5, depth_counter)

            def get_color_frame(self):
                return self.color

            def get_depth_frame(self):
                return self.depth

        class FakePipeline:
            def __init__(self):
                self.index = 0
                self.stopped = False
                self.samples = [(1201.0, 1, 11), (1201.5, 1, 12), (1202.0, 2, 13)]

            def start(self, _config):
                return profile

            def wait_for_frames(self, _timeout):
                sample = self.samples[min(self.index, len(self.samples) - 1)]
                self.index += 1
                return FakeFrameset(*sample)

            def stop(self):
                self.stopped = True

        intrinsics = types.SimpleNamespace(fx=5.0, fy=5.5, ppx=3.5, ppy=2.5, coeffs=[0, 0, 0, 0, 0])

        class FakeVideoProfile:
            def as_video_stream_profile(self):
                return self

            def get_intrinsics(self):
                return intrinsics

        class FakeSensor:
            def get_depth_scale(self):
                return 0.001

        class FakeDevice:
            def first_depth_sensor(self):
                return FakeSensor()

        class FakeProfile:
            def get_device(self):
                return FakeDevice()

            def get_stream(self, _stream):
                return FakeVideoProfile()

        profile = FakeProfile()
        rs = types.SimpleNamespace(
            pipeline=FakePipeline,
            config=lambda: types.SimpleNamespace(enable_device=lambda *_: None, enable_stream=lambda *_: None),
            stream=types.SimpleNamespace(color="color", depth="depth"),
            format=types.SimpleNamespace(bgr8="bgr8", z16="z16"),
            align=lambda _stream: types.SimpleNamespace(process=lambda frames: frames),
        )
        with patch.dict(sys.modules, {"pyrealsense2": rs}):
            camera = LiveWristCamera("fake-serial", width=8, height=6, fps=30, warmup=0)
            try:
                frame = camera.grab(flush=0)
                fresh_frame = camera.grab(flush=0, require_fresh_pair=True)
                description = camera.describe()
            finally:
                camera.close()

        sdk = frame["capture_metadata"]["sdk"]
        self.assertEqual(sdk["color_timestamp_ms"], 1201.0)
        self.assertEqual(sdk["depth_timestamp_ms"], 1201.5)
        self.assertEqual(sdk["color_frame_counter"], 1)
        self.assertEqual(sdk["depth_frame_counter"], 11)
        self.assertEqual(sdk["timestamp_units"], "milliseconds")
        self.assertEqual(sdk["frame_counter_units"], "frames")
        self.assertEqual(description["serial"], "fake-serial")
        self.assertEqual(description["intrinsics"]["fx_px"], 5.0)
        self.assertEqual(description["depth_scale_mm_per_raw_unit"], 1.0)
        self.assertTrue(np.all(frame["depth"] == 1000.0))
        fresh_sdk = fresh_frame["capture_metadata"]["sdk"]
        self.assertEqual(fresh_sdk["color_frame_counter"], 2)
        self.assertEqual(fresh_sdk["depth_frame_counter"], 13)
        self.assertEqual(fresh_sdk["freshness_retries"], 1)


if __name__ == "__main__":
    unittest.main()
