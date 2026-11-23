"""Board-free hand-eye calibration of one wrist D405 (no teaching, no chessboard).

The arm parks at its calibrated home (tool down, above the matte, textured tabletop beside its
base), opens the gripper fully and makes nine small moves around that spot: turns of ±15° about the
tool axis, ±30 mm in X and Y, 50 mm down, and down plus a turn. Each transition is planned as small
waypoints and IK/FK-checked in full before confirmation. Horizontal travel and vertical descent are
separate, with the order reversed when returning home. At every stop the wrist camera takes a
colour frame and aligned depth. The tabletop features seen from all stops give the camera mount
(see chem_demo.wrist_geometry.solve_hand_eye); pixels that stay put while the tool turns and sit
close to the lens are the gripper and are saved as a mask.

Every waypoint is checked with the jog program's IK/FK checks. The operator confirms each
transition after seeing its motion. Space stops the arm. On every exit the arm is disabled. The camera height along the tool axis cannot be
seen by these moves; run_bottle_pick measures it on the white platform each time, so the value
saved here is provisional (see the result's notes).

    C:/software/anaconda3/envs/dobot-xtrainer/python.exe -X utf8 -m chem_demo.run_wrist_calib --arm left --live
    C:/software/anaconda3/envs/dobot-xtrainer/python.exe -X utf8 -m chem_demo.run_wrist_calib --arm left --live --capture-burst 15
    ... --sim       (synthetic camera and arm, no hardware)

Each run keeps poses.json in its existing format, the original ref.png/ref_depth.png style files,
and an acquisition.json recovery/replay manifest. --capture-burst N (1..15, default 1) stores N
synchronized color/depth observations at each pose, plus N ref_return observations after returning
home when N > 1. Per-observation float32 depth is saved in millimeters; burst processing uses the
per-pixel median of valid depths where at least half the observations are valid. Color is never
averaged. The manifest records robot state, camera intrinsics/serial/depth scale, SDK counters and
timestamps, and all source and representative depth file names.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from chem_demo.wrist_geometry import (PairObservation, backproject, fit_plane, gripper_mask, jaw_gap,
                                      match_features, pose_matrix, rotation_angle_deg, sample_depth,
                                      solve_hand_eye, transform)
from chem_demo.wrist_runtime import CALIBRATION_DIR, TABLE_Z, camera_serial, save_calibration, save_frame
from task_skills.run_cartesian_jog import (DEFAULT_PROFILE, Jog, JogRefused, JogStopped, Limits, LiveArm, SimArm,
                                           LINE_CHECK_MM, down_tilt_deg, snap_down_rpy, start_refusal)

# (label, dx, dy, dz mm, dyaw deg) relative to the start; every one is compared with "ref".
MOVES = [("ref", 0, 0, 0, 0), ("yaw+", 0, 0, 0, 15), ("yaw-", 0, 0, 0, -15),
         ("x+", 30, 0, 0, 0), ("x-", -30, 0, 0, 0), ("y+", 0, 30, 0, 0), ("y-", 0, -30, 0, 0),
         ("z-", 0, 0, -50, 0), ("z-yaw+", 0, 0, -50, 15)]
ACCEPT = {"median_mm": 1.5, "rotation_spread_deg": 0.5, "min_points": 30, "height_spread_mm": 2.0}
YAW_WAYPOINT_DEG = 5.0
CAPTURE_BURST_MAX = 15
CAPTURE_STABILITY_MM = 0.5
CAPTURE_STABILITY_DEG = 0.5


class Aborted(RuntimeError):
    pass


def _write_json(path: Path, value: Any, *, ensure_ascii: bool = True) -> None:
    """Replace JSON snapshots atomically so interruption leaves the last complete pose record."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=ensure_ascii, indent=2), encoding="utf-8")
    # On Windows a concurrent status reader can briefly deny delete sharing.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 5:
                raise
            time.sleep(0.02 * (attempt + 1))


def _camera_description(camera: Any) -> dict[str, Any]:
    describe = getattr(camera, "describe", None)
    if callable(describe):
        return describe()
    K = getattr(camera, "K", None)
    return {
        "source": "unknown",
        "serial": getattr(camera, "serial", None),
        "intrinsics": None if K is None else {"matrix_px": np.asarray(K, float).tolist()},
        "depth_scale_m_per_raw_unit": None,
        "depth_scale_mm_per_raw_unit": getattr(camera, "scale_mm", None),
        "depth_output_unit": "mm",
    }


def _robot_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    joints = state.get("joints")
    pose = state.get("pose")
    return {
        "joints_deg": None if joints is None else [float(value) for value in joints],
        "tcp_pose_mm_deg": None if pose is None else [float(value) for value in pose],
        "robot_mode": state.get("mode"),
        "robot_state_source": state.get("source"),
    }


def _state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Measure robot state movement during an exposure using mm and degrees."""
    joints_a, joints_b = before.get("joints_deg"), after.get("joints_deg")
    pose_a, pose_b = before.get("tcp_pose_mm_deg"), after.get("tcp_pose_mm_deg")
    if not joints_a or not joints_b or not pose_a or not pose_b:
        return {"available": False, "stable": False, "reason": "robot read lacks joints or TCP pose"}
    joint_delta = max(abs((joints_b[i] - joints_a[i] + 180.0) % 360.0 - 180.0)
                      for i in range(min(len(joints_a), len(joints_b))))
    position_delta = math.dist(pose_a[:3], pose_b[:3])
    orientation_delta = max(abs((pose_b[i] - pose_a[i] + 180.0) % 360.0 - 180.0)
                            for i in range(3, min(len(pose_a), len(pose_b))))
    stable = (position_delta <= CAPTURE_STABILITY_MM and
              orientation_delta <= CAPTURE_STABILITY_DEG and joint_delta <= CAPTURE_STABILITY_DEG)
    return {"available": True, "stable": stable,
            "tcp_translation_mm": round(position_delta, 4),
            "tcp_orientation_max_deg": round(orientation_delta, 4),
            "joint_max_deg": round(joint_delta, 4)}


def _validate_frame(frame: dict[str, Any]) -> None:
    color, depth = frame.get("color"), frame.get("depth")
    if not isinstance(color, np.ndarray) or color.ndim != 3 or color.shape[2] != 3:
        raise RuntimeError("camera acquisition has no valid BGR color image")
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.shape != color.shape[:2]:
        raise RuntimeError("camera acquisition has no depth image aligned to color")
    if not np.isfinite(depth).all():
        raise RuntimeError("camera acquisition contains non-finite depth values")


def _temporal_median_depth(frames: Sequence[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Median positive depth per pixel; require at least half the requested observations."""
    if not frames:
        raise ValueError("at least one depth observation is required")
    stack = np.stack([np.asarray(frame["depth"], dtype=np.float32) for frame in frames])
    valid = np.isfinite(stack) & (stack > 0)
    counts = valid.sum(axis=0)
    required = math.ceil(len(frames) / 2)
    median = np.ma.median(np.ma.array(stack, mask=~valid), axis=0).filled(0).astype(np.float32)
    representative = np.where(counts >= required, median, 0).astype(np.float32)
    summary = {"method": "per-pixel temporal median of positive finite depth",
               "input_observations": len(frames), "minimum_valid_observations": required,
               "depth_unit": "mm", "valid_pixel_fraction": round(float(np.mean(counts >= required)), 6)}
    return representative, summary


def _capture_burst_value(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("capture burst must be an integer from 1 to 15") from error
    if not 1 <= count <= CAPTURE_BURST_MAX:
        raise argparse.ArgumentTypeError("capture burst must be from 1 to 15")
    return count


def _angle_delta_degrees(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _gripper_is_open(position: Any) -> bool:
    try:
        return math.isfinite(float(position)) and float(position) >= 250
    except (TypeError, ValueError):
        return False


def _pose_deltas(actual: Sequence[float], expected: Sequence[float]) -> tuple[float, float]:
    try:
        actual_values = [float(value) for value in actual]
        expected_values = [float(value) for value in expected]
    except (TypeError, ValueError) as error:
        raise Aborted("resume requires numeric saved and current TCP poses") from error
    if len(actual_values) != 6 or len(expected_values) != 6:
        raise Aborted("resume requires saved six-axis TCP poses")
    values = [*actual_values, *expected_values]
    if not np.isfinite(values).all():
        raise Aborted("resume requires finite saved and current TCP poses")
    return math.dist(values[:3], values[6:9]), max(
        _angle_delta_degrees(values[i], values[i + 6]) for i in range(3, 6))


def _validate_resume_state(current: dict[str, Any], saved: dict[str, Any], saved_pose: Sequence[float],
                           *, label: str = "interrupted z-") -> dict[str, float]:
    """Require Cartesian position, orientation, and every joint to match an interrupted snapshot."""
    current_snapshot = _robot_snapshot(current)
    current_pose = current_snapshot.get("tcp_pose_mm_deg")
    saved_snapshot_pose = saved.get("tcp_pose_mm_deg")
    current_joints = current_snapshot.get("joints_deg")
    saved_joints = saved.get("joints_deg")
    if any(value is None for value in (current_pose, saved_snapshot_pose, current_joints, saved_joints)):
        raise Aborted("resume requires saved and current TCP pose plus all six joint readings")
    if len(current_joints) != 6 or len(saved_joints) != 6:
        raise Aborted("resume requires six saved and current joint readings")
    xyz_delta, orientation_delta = _pose_deltas(current_pose, saved_snapshot_pose)
    saved_pose_delta, saved_pose_orientation_delta = _pose_deltas(current_pose, saved_pose)
    try:
        joint_values = [float(value) for value in (*current_joints, *saved_joints)]
    except (TypeError, ValueError) as error:
        raise Aborted("resume requires numeric saved and current joint readings") from error
    if not np.isfinite(joint_values).all():
        raise Aborted("resume requires finite saved and current joint readings")
    joint_delta = max(_angle_delta_degrees(current_joints[i], saved_joints[i]) for i in range(6))
    if (xyz_delta > CAPTURE_STABILITY_MM or saved_pose_delta > CAPTURE_STABILITY_MM or
            orientation_delta > CAPTURE_STABILITY_DEG or saved_pose_orientation_delta > CAPTURE_STABILITY_DEG or
            joint_delta > CAPTURE_STABILITY_DEG):
        raise Aborted(
            f"current arm is not at the saved {label} pose "
            f"(snapshot xyz delta={xyz_delta:.3f} mm, orientation delta={orientation_delta:.3f} deg, "
            f"joint delta={joint_delta:.3f} deg; saved pose xyz delta={saved_pose_delta:.3f} mm, "
            f"orientation delta={saved_pose_orientation_delta:.3f} deg; tolerances "
            f"{CAPTURE_STABILITY_MM:g} mm / {CAPTURE_STABILITY_DEG:g} deg)"
        )
    return {"tcp_translation_delta_mm": xyz_delta, "tcp_orientation_delta_deg": orientation_delta,
            "saved_z_translation_delta_mm": saved_pose_delta,
            "saved_z_orientation_delta_deg": saved_pose_orientation_delta,
            "joint_max_delta_deg": joint_delta}


def _resume_group_files_exist(folder: Path, group: dict[str, Any], *, complete: bool) -> None:
    for capture in group.get("captures", []):
        if capture.get("status") != "captured":
            if complete:
                raise Aborted("resume source has an incomplete pose before z-")
            continue
        files = capture.get("files", {})
        if not isinstance(files, dict):
            raise Aborted("resume source has invalid recorded file metadata")
        for key in ("color_png", "depth_png", "depth_mm_npy"):
            name = files.get(key)
            if not isinstance(name, str) or Path(name).name != name or not (folder / name).is_file():
                raise Aborted(f"resume source is missing a recorded capture file: {name!r}")
    if complete:
        representative = group.get("representative_depth_file")
        if (not isinstance(representative, str) or Path(representative).name != representative or
                not (folder / representative).is_file()):
            raise Aborted("resume source is missing a completed representative depth file")


def _read_resume_source(path: Path, arm_name: str, table_z: float) -> tuple[Path, dict[str, Any], dict[str, Any], int]:
    source = Path(path).expanduser().resolve()
    manifest_path, poses_path = source / "acquisition.json", source / "poses.json"
    if not source.is_dir() or not manifest_path.is_file() or not poses_path.is_file():
        raise Aborted("--resume-capture must point to a dataset with acquisition.json and poses.json")
    try:
        acquisition = json.loads(manifest_path.read_text(encoding="utf-8"))
        poses = json.loads(poses_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Aborted(f"resume source metadata cannot be read: {error}") from error
    if not isinstance(acquisition, dict) or not isinstance(poses, dict):
        raise Aborted("resume source metadata must contain acquisition and pose objects")
    if not isinstance(acquisition.get("camera"), dict):
        raise Aborted("resume source is missing its camera metadata")
    if acquisition.get("schema_version") != 1 or acquisition.get("arm") != arm_name:
        raise Aborted("resume source schema or arm does not match --arm")
    if acquisition.get("status") not in ("in_progress", "partial"):
        raise Aborted("--resume-capture accepts only an interrupted in-progress/partial dataset")
    if acquisition.get("capture_problems"):
        raise Aborted("resume source contains earlier capture quality problems")
    try:
        burst = _capture_burst_value(str(acquisition.get("capture_burst", "")))
    except argparse.ArgumentTypeError as error:
        raise Aborted(f"resume source has an invalid capture burst: {error}") from error
    try:
        saved_table_z = float(acquisition.get("table_z_reference_mm"))
    except (TypeError, ValueError) as error:
        raise Aborted("resume source is missing its original table-Z reference") from error
    if not math.isfinite(saved_table_z) or not math.isclose(saved_table_z, float(table_z), rel_tol=0, abs_tol=1e-6):
        raise Aborted("--table-z must match the interrupted dataset's original reference")
    if not _gripper_is_open(acquisition.get("initial_gripper_position")):
        raise Aborted("resume source was not started with a fully open gripper")

    groups = acquisition.get("pose_groups", {})
    if not isinstance(groups, dict):
        raise Aborted("resume source is missing its pose-group metadata")
    poses = {str(key): value for key, value in poses.items()}
    z_index = next(i for i, move in enumerate(MOVES) if move[0] == "z-")
    labels_before_z = [move[0] for move in MOVES[:z_index]]
    for label in labels_before_z:
        group = groups.get(label)
        if not isinstance(group, dict):
            raise Aborted(f"resume source is missing the completed {label} pose group")
        captures = group.get("captures", [])
        stability = group.get("pose_stability")
        if (not isinstance(captures, list) or any(not isinstance(capture, dict) for capture in captures) or
                len(captures) != burst or
                any(capture.get("status") != "captured" for capture in captures) or
                not isinstance(stability, dict) or stability.get("stable") is not True or
                not _gripper_is_open(group.get("gripper_position"))):
            raise Aborted(f"resume source's {label} pose is not a complete, stable open-gripper group")
        _resume_group_files_exist(source, group, complete=True)
        if label not in poses:
            raise Aborted(f"resume source is missing the saved {label} robot pose")

    interrupted = groups.get("z-")
    interrupted_captures = interrupted.get("captures", []) if isinstance(interrupted, dict) else []
    if (not isinstance(interrupted_captures, list) or
            any(not isinstance(capture, dict) for capture in interrupted_captures) or
            not interrupted_captures or len(interrupted_captures) > burst or
            not _gripper_is_open(interrupted.get("gripper_position")) or "z-" not in poses):
        raise Aborted("resume source has no interrupted open-gripper z- pose to recover")
    if "z-yaw+" in groups or "ref_return" in groups:
        raise Aborted("resume source has already entered z-yaw+ or ref_return; this recovery starts at z-")
    if "z-yaw+" in poses or "ref_return" in poses:
        raise Aborted("resume source pose log has advanced beyond z-")
    if any(capture.get("status") not in ("captured", "capturing", "failed") for capture in interrupted_captures):
        raise Aborted("resume source's z- group has an unknown capture status")
    if not any(capture.get("status") == "captured" for capture in interrupted_captures):
        raise Aborted("resume source's z- group has no completed observation")
    _resume_group_files_exist(source, interrupted, complete=False)
    last_capture = interrupted_captures[-1]
    saved_snapshot = last_capture.get("robot_after") or last_capture.get("robot_before")
    if not isinstance(saved_snapshot, dict):
        raise Aborted("resume source is missing its interrupted z- TCP/joint snapshot")

    def valid_pose(value: Any) -> bool:
        if not isinstance(value, list) or len(value) != 6:
            return False
        try:
            return bool(np.isfinite(np.asarray(value, float)).all())
        except (TypeError, ValueError):
            return False

    if not valid_pose(poses.get("ref")) or not valid_pose(poses.get("z-")):
        raise Aborted("resume source is missing finite six-axis ref/z- poses")
    for label, *_ in MOVES:
        if label in poses and not valid_pose(poses[label]):
            raise Aborted(f"resume source has an invalid saved pose for {label}")
    if (not valid_pose(saved_snapshot.get("joints_deg")) or
            not valid_pose(saved_snapshot.get("tcp_pose_mm_deg"))):
        raise Aborted("resume source lacks finite six-axis interrupted TCP/joint readings")
    saved_snapshot_pose_delta = _pose_deltas(saved_snapshot["tcp_pose_mm_deg"], poses["z-"])
    if (saved_snapshot_pose_delta[0] > CAPTURE_STABILITY_MM or
            saved_snapshot_pose_delta[1] > CAPTURE_STABILITY_DEG):
        raise Aborted("resume source's z- capture snapshot disagrees with poses.json")
    return source, acquisition, poses, burst


def _validate_resume_camera(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    if saved.get("source") != current.get("source"):
        raise Aborted("current camera source differs from the interrupted dataset")
    saved_serial, current_serial = saved.get("serial"), current.get("serial")
    if saved.get("source") != "simulation" and (not saved_serial or not current_serial):
        raise Aborted("resume requires an identified camera serial")
    if saved_serial != current_serial:
        raise Aborted("current camera serial differs from the interrupted dataset")
    saved_intrinsics, current_intrinsics = saved.get("intrinsics") or {}, current.get("intrinsics") or {}
    saved_k = saved_intrinsics.get("matrix_px")
    current_k = current_intrinsics.get("matrix_px")
    try:
        saved_k, current_k = np.asarray(saved_k, float), np.asarray(current_k, float)
    except (TypeError, ValueError) as error:
        raise Aborted("resume requires saved and current camera intrinsics") from error
    if saved_k.shape != (3, 3) or current_k.shape != (3, 3) or not np.isfinite(saved_k).all() or not np.isfinite(current_k).all():
        raise Aborted("resume requires finite 3x3 saved and current camera intrinsics")
    if not np.allclose(saved_k, current_k, rtol=1e-6, atol=1e-5):
        raise Aborted("current camera intrinsics differ from the interrupted dataset")
    saved_distortion = saved_intrinsics.get("distortion_coefficients")
    current_distortion = current_intrinsics.get("distortion_coefficients")
    if saved.get("source") == "realsense" and (saved_distortion is None or current_distortion is None):
        raise Aborted("resume requires saved and current RealSense distortion coefficients")
    if (saved_distortion is None) != (current_distortion is None):
        raise Aborted("saved and current camera distortion metadata differ")
    distortion_delta = 0.0
    if saved_distortion is not None:
        try:
            saved_distortion = np.asarray(saved_distortion, float)
            current_distortion = np.asarray(current_distortion, float)
        except (TypeError, ValueError) as error:
            raise Aborted("resume requires numeric saved and current camera distortion coefficients") from error
        if (saved_distortion.ndim != 1 or current_distortion.shape != saved_distortion.shape or
                not np.isfinite(saved_distortion).all() or not np.isfinite(current_distortion).all()):
            raise Aborted("resume requires matching finite camera distortion coefficient vectors")
        distortion_delta = float(np.max(np.abs(saved_distortion - current_distortion)))
        if not np.allclose(saved_distortion, current_distortion, rtol=1e-6, atol=1e-7):
            raise Aborted("current camera distortion coefficients differ from the interrupted dataset")
    saved_undistorted = saved_intrinsics.get("distortion_removed_from_output")
    current_undistorted = current_intrinsics.get("distortion_removed_from_output")
    if saved.get("source") == "realsense" and (saved_undistorted is None or current_undistorted is None):
        raise Aborted("resume requires saved and current RealSense distortion-removal metadata")
    if saved_undistorted != current_undistorted:
        raise Aborted("current camera distortion-removal setting differs from the interrupted dataset")
    if saved.get("depth_output_unit") != "mm" or current.get("depth_output_unit") != "mm":
        raise Aborted("resume requires millimeter depth in both saved and current camera data")
    saved_stream, current_stream = saved.get("stream") or {}, current.get("stream") or {}
    if saved_stream and current_stream and saved_stream != current_stream:
        raise Aborted("current camera stream resolution/rate differs from the interrupted dataset")
    for key in ("depth_alignment",):
        if saved.get(key) is not None and current.get(key) is not None and saved[key] != current[key]:
            raise Aborted(f"current camera {key} differs from the interrupted dataset")
    return {"intrinsics_max_abs_delta": float(np.max(np.abs(saved_k - current_k))),
            "distortion_max_abs_delta": distortion_delta}


def _new_recovery_folder(source: Path, output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(output_root).expanduser().resolve()
    candidate = root / f"{source.name}-recovery-{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{source.name}-recovery-{stamp}-{suffix:02d}"
        suffix += 1
    return candidate


def _prepare_recovery_copy(source: Path, destination: Path, acquisition: dict[str, Any],
                           poses: dict[str, Any], *, burst: int,
                           pose_match: dict[str, float], camera_match: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = source.resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise Aborted("recovery output folder already exists; choose a fresh --out folder")
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        raise Aborted("recovery output folder cannot be inside the interrupted source dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    shutil.copy2(source / "acquisition.json", destination / "acquisition_interrupted.json")
    shutil.copy2(source / "poses.json", destination / "poses_interrupted.json")
    resumed_acquisition = copy.deepcopy(acquisition)
    resumed_poses = copy.deepcopy(poses)
    resumed_acquisition["status"] = "in_progress"
    resumed_acquisition.pop("finished_at", None)
    resumed_acquisition["recovery"] = {
        "source_dataset": str(source),
        "source_status": acquisition.get("status"),
        "source_started_at": acquisition.get("started_at"),
        "source_interrupted_pose": "z-",
        "archived_partial_metadata": ["acquisition_interrupted.json", "poses_interrupted.json"],
        "capture_burst": burst,
        "pose_preflight": pose_match,
        "camera_preflight": camera_match,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(destination / "poses.json", resumed_poses)
    _write_json(destination / "acquisition.json", resumed_acquisition, ensure_ascii=False)
    return resumed_acquisition, resumed_poses


def _capture_group(arm: Any, camera: Any, camera_info: dict[str, Any], folder: Path,
                   acquisition: dict[str, Any], group_name: str, file_prefix: str, count: int,
                   persist: Callable[[], None], stop_pressed: Callable[[], bool]
                   ) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Capture one pose burst and persist the same replay metadata for normal and recovery runs."""
    group: dict[str, Any] = {"captures": [], "representative_depth_file": None}
    acquisition["pose_groups"][group_name] = group
    group["gripper_position"] = arm.gripper_position()
    if group["gripper_position"] < 250:
        acquisition["status"] = "partial"
        persist()
        raise Aborted(f"{group_name}: gripper feedback {group['gripper_position']} is not fully open")
    observed: list[dict[str, Any]] = []
    for index in range(count):
        if stop_pressed():
            arm.stop()
            acquisition["status"] = "partial"
            persist()
            raise JogStopped("emergency stop: calibration stopped during capture burst")
        stem = file_prefix if index == 0 else f"{file_prefix}_burst_{index:02d}"
        before = _robot_snapshot(arm.read())
        capture: dict[str, Any] = {
            "index": index, "status": "capturing",
            "files": {"color_png": f"{stem}.png", "depth_png": f"{stem}_depth.png",
                      "depth_mm_npy": f"{stem}_depth_mm.npy"},
            "robot_before": before,
        }
        group["captures"].append(capture)
        persist()
        try:
            if camera_info.get("source") == "realsense":
                frame = camera.grab(flush=4 if index == 0 else 0, require_fresh_pair=True)
            else:
                frame = camera.grab() if index == 0 else camera.grab(flush=0)
            stop_after_capture = stop_pressed()
            if stop_after_capture:
                arm.stop()
            _validate_frame(frame)
            after = _robot_snapshot(arm.read())
            paths = save_frame(frame, folder, stem)
            depth_npy = folder / f"{stem}_depth_mm.npy"
            np.save(depth_npy, np.asarray(frame["depth"], dtype=np.float32), allow_pickle=False)
            if not all(Path(path).is_file() and Path(path).stat().st_size for path in paths.values()):
                raise RuntimeError("camera image could not be written to the acquisition folder")
            capture["files"]["depth_mm_npy"] = depth_npy.name
            capture["robot_after"] = after
            capture["robot_delta_during_capture"] = _state_delta(before, after)
            capture["capture_metadata"] = frame.get("capture_metadata", {})
            capture["frame_shape"] = {"height_px": int(frame["depth"].shape[0]),
                                       "width_px": int(frame["depth"].shape[1]),
                                       "color_format": "BGR", "depth_unit": "mm"}
            capture["status"] = "captured"
            observed.append(frame)
        except Exception as error:
            capture["status"] = "failed"
            capture["error"] = f"{type(error).__name__}: {error}"
            acquisition["status"] = "partial"
            persist()
            raise
        persist()
        if stop_after_capture:
            acquisition["status"] = "partial"
            persist()
            raise JogStopped("emergency stop: calibration stopped after a capture")

    baseline = group["captures"][0]["robot_before"]
    stable_group = True
    for capture in group["captures"]:
        before_delta = _state_delta(baseline, capture["robot_before"])
        after_delta = _state_delta(baseline, capture["robot_after"])
        capture["robot_delta_from_group_start_before"] = before_delta
        capture["robot_delta_from_group_start_after"] = after_delta
        if not all(delta.get("stable", False) for delta in
                   (before_delta, after_delta, capture["robot_delta_during_capture"])):
            stable_group = False
    group["pose_stability"] = {
        "stable": stable_group,
        "translation_tolerance_mm": CAPTURE_STABILITY_MM,
        "orientation_tolerance_deg": CAPTURE_STABILITY_DEG,
        "joint_tolerance_deg": CAPTURE_STABILITY_DEG,
    }
    if count > 1 and not stable_group:
        acquisition["capture_problems"].append(
            f"{group_name}: robot pose changed beyond capture stability tolerance")

    source = camera_info.get("source")
    if source == "realsense":
        color_counters, depth_counters, color_timestamps, depth_timestamps = [], [], [], []
        for capture in group["captures"]:
            sdk = capture.get("capture_metadata", {}).get("sdk", {})
            required = ("color_timestamp_ms", "color_frame_counter", "depth_timestamp_ms",
                        "depth_frame_counter", "color_timestamp_domain", "depth_timestamp_domain",
                        "timestamp_units", "frame_counter_units")
            if any(key not in sdk for key in required):
                acquisition["capture_problems"].append(
                    f"{group_name}: missing RealSense timestamp or frame-counter metadata")
                continue
            if sdk["timestamp_units"] != "milliseconds" or sdk["frame_counter_units"] != "frames":
                acquisition["capture_problems"].append(
                    f"{group_name}: RealSense timestamp/counter units are not explicit")
            color_counters.append(sdk["color_frame_counter"])
            depth_counters.append(sdk["depth_frame_counter"])
            color_timestamps.append(sdk["color_timestamp_ms"])
            depth_timestamps.append(sdk["depth_timestamp_ms"])
        if len(set(color_counters)) != len(color_counters) or len(set(depth_counters)) != len(depth_counters):
            acquisition["capture_problems"].append(
                f"{group_name}: duplicate RealSense frame counter in burst")
        if any(not math.isfinite(float(value)) for value in color_timestamps + depth_timestamps):
            acquisition["capture_problems"].append(
                f"{group_name}: RealSense timestamp is not finite")
        for previous, current in zip(group["captures"], group["captures"][1:]):
            previous_sdk = previous.get("capture_metadata", {}).get("sdk", {})
            current_sdk = current.get("capture_metadata", {}).get("sdk", {})
            for stream in ("color", "depth"):
                domain = f"{stream}_timestamp_domain"
                timestamp = f"{stream}_timestamp_ms"
                if (previous_sdk.get(domain) == current_sdk.get(domain) and
                        current_sdk.get(timestamp, -math.inf) <= previous_sdk.get(timestamp, math.inf)):
                    acquisition["capture_problems"].append(
                        f"{group_name}: RealSense {stream} timestamp did not advance in the same clock domain")
                    break
    elif source == "simulation":
        counters = [c.get("capture_metadata", {}).get("frame_counter") for c in group["captures"]]
        if any(counter is None for counter in counters) or len(set(counters)) != len(counters):
            acquisition["capture_problems"].append(
                f"{group_name}: simulated burst does not contain unique frame counters")

    if count == 1:
        representative = np.asarray(observed[0]["depth"], dtype=np.float32)
        summary = {"method": "single observation", "input_observations": 1,
                   "minimum_valid_observations": 1, "depth_unit": "mm",
                   "valid_pixel_fraction": round(float(np.mean(representative > 0)), 6)}
        group["representative_depth_file"] = group["captures"][0]["files"]["depth_mm_npy"]
    else:
        representative, summary = _temporal_median_depth(observed)
        median_npy = folder / f"{file_prefix}_depth_temporal_median_mm.npy"
        median_png = folder / f"{file_prefix}_depth_temporal_median.png"
        np.save(median_npy, representative, allow_pickle=False)
        if not cv2.imwrite(str(median_png), np.clip(representative, 0, 65535).astype(np.uint16)):
            acquisition["capture_problems"].append(
                f"{group_name}: representative median depth preview could not be written")
        group["representative_depth_file"] = median_npy.name
        group["representative_depth_preview"] = median_png.name
        group["representative_depth_unit"] = "mm"
        group["depth_aggregation"] = summary
    persist()
    return observed, representative


def _precheck_transition(jog: Jog, source: str, destination: str, target_xyz: Sequence[float],
                         target_rpy: Sequence[float], stop_pressed: Callable[[], bool]
                         ) -> tuple[list[tuple[str, list[float], list[float]]], list[str]]:
    """Plan and IK/FK-check every small waypoint before a transition can send motion."""
    current_xyz, current_rpy = list(jog.xyz), list(jog.rpy)
    target_xyz, target_rpy = list(target_xyz), list(target_rpy)
    seed = list(jog.joints)
    planned: list[tuple[str, list[float], list[float]]] = []
    descriptions: list[str] = []

    def add(kind: str, xyz: Sequence[float], rpy: Sequence[float]) -> None:
        nonlocal seed
        if stop_pressed():
            jog.arm.stop()
            raise JogStopped("emergency stop: calibration stopped")
        point, orientation = list(xyz), list(rpy)
        previous_xyz, previous_rpy = ((planned[-1][1], planned[-1][2]) if planned else
                                      (jog.xyz, jog.rpy))
        if math.dist(point, previous_xyz) <= 1e-3 and max(
                abs(orientation[i] - previous_rpy[i]) for i in range(3)) <= 1e-4:
            return
        try:
            seed = jog.check(point, orientation, seed)
        except JogRefused as refusal:
            raise JogRefused(
                f"transition {source} -> {destination} rejected before motion at "
                f"{[round(v, 1) for v in point]}: {refusal}"
            ) from refusal
        planned.append((kind, point, orientation))

    def segment(kind: str, start: Sequence[float], end: Sequence[float], rpy: Sequence[float]) -> None:
        distance = math.dist(start, end)
        if distance <= 1e-3:
            return
        start_count = len(planned)
        steps = math.ceil(distance / LINE_CHECK_MM)
        for index in range(1, steps + 1):
            point = [round(start[axis] + (end[axis] - start[axis]) * index / steps, 4)
                     for axis in range(3)]
            add(kind, point, rpy)
        count = len(planned) - start_count
        if count:
            if kind == "horizontal":
                descriptions.append(
                    f"保持 Z={planned[-1][1][2]:.1f} mm，水平移动到 "
                    f"X={planned[-1][1][0]:.1f}, Y={planned[-1][1][1]:.1f} mm "
                    f"（{count} 个不超过 {LINE_CHECK_MM:g} mm 的小步）"
                )
            else:
                direction = "上升" if kind == "vertical ascent" else "下降"
                descriptions.append(
                    f"保持 XY 不变，垂直{direction}到 Z={planned[-1][1][2]:.1f} mm "
                    f"（{count} 个不超过 {LINE_CHECK_MM:g} mm 的小步）"
                )

    # Rotate in place, in modest independent increments, before translating.
    starting_yaw = current_rpy[2]
    yaw_delta = target_rpy[2] - starting_yaw
    yaw_steps = math.ceil(abs(yaw_delta) / YAW_WAYPOINT_DEG - 1e-9)
    yaw_start_count = len(planned)
    for index in range(1, yaw_steps + 1):
        orientation = list(current_rpy)
        orientation[2] = (target_rpy[2] if index == yaw_steps else
                          round(starting_yaw + yaw_delta * index / yaw_steps, 4))
        add("yaw", current_xyz, orientation)
        current_rpy = orientation
    yaw_count = len(planned) - yaw_start_count
    if yaw_count:
        descriptions.append(
            f"原地将 yaw 转到 {current_rpy[2]:.1f} 度，分 {yaw_count} 个小步，"
            f"每步不超过 {YAW_WAYPOINT_DEG:g} 度"
        )

    end_xyz = [round(value, 4) for value in target_xyz]
    horizontal_end = [end_xyz[0], end_xyz[1], current_xyz[2]]
    vertical_end = [current_xyz[0], current_xyz[1], end_xyz[2]]
    dz = end_xyz[2] - current_xyz[2]
    if dz > 1e-6:  # retreat: reverse the descent order (vertical, then horizontal)
        segment("vertical ascent", current_xyz, vertical_end, current_rpy)
        current_xyz = vertical_end
        segment("horizontal", current_xyz, end_xyz, current_rpy)
    else:  # level travel or descent: return horizontally first, then descend vertically
        segment("horizontal", current_xyz, horizontal_end, current_rpy)
        current_xyz = horizontal_end
        if dz < -1e-6:
            segment("vertical descent", current_xyz, end_xyz, current_rpy)
    return planned, descriptions


def _execute_transition(jog: Jog, source: str, destination: str, target_xyz: Sequence[float],
                        target_rpy: Sequence[float], *, confirm: Callable[[str], None],
                        stop_pressed: Callable[[], bool], observe_waypoint: Callable[[], None] | None = None) -> None:
    """Precheck, confirm and send one complete transition, polling stop before every waypoint."""
    waypoints, descriptions = _precheck_transition(
        jog, source, destination, target_xyz, target_rpy, stop_pressed)
    if not waypoints:
        return
    transition = f"{source} 返回 {destination}" if destination == "ref" else f"{source} 到 {destination}"
    confirm(f"{transition}：{'；然后'.join(descriptions)}")
    for _, waypoint_xyz, waypoint_rpy in waypoints:
        if stop_pressed():
            jog.arm.stop()
            raise JogStopped("急停：已停止")
        jog.go(waypoint_xyz, waypoint_rpy, report=False)
        if observe_waypoint is not None:
            if stop_pressed():
                jog.arm.stop()
                raise JogStopped("急停：已停止")
            observe_waypoint()
            if stop_pressed():
                jog.arm.stop()
                raise JogStopped("急停：已停止")

def process(frames: dict[str, dict], poses: dict[str, list[float]], K: np.ndarray, table_z: float,
            out: Callable[[str], None] = print) -> dict[str, Any]:
    """Frames + poses -> T_flange_camera, gripper mask and quality checks. Pure; used by tests."""
    gray = {k: cv2.cvtColor(f["color"], cv2.COLOR_BGR2GRAY) for k, f in frames.items()}
    ref_depth = frames["ref"]["depth"]
    valid = ref_depth[ref_depth > 0]
    if valid.size < 0.1 * ref_depth.size:
        raise RuntimeError("参考位深度有效点太少（<10%），看不到台面？")
    far = valid[valid > 200.0]  # the fingers sit ~120 mm from the lens
    table_depth = float(np.median(far)) if far.size else float(np.median(valid))
    gripper = gripper_mask([gray["ref"], gray["yaw+"], gray["yaw-"]], ref_depth, table_depth)
    blocked = cv2.dilate(gripper, np.ones((15, 15), np.uint8))

    T0 = pose_matrix(poses["ref"])
    pairs = []
    for label in poses:
        if label == "ref":
            continue
        pa, pb = match_features(gray["ref"], gray[label], exclude=blocked)
        da, db = sample_depth(ref_depth, pa), sample_depth(frames[label]["depth"], pb)
        ok = (da > 150) & (db > 150)
        pairs.append(PairObservation(T0, pose_matrix(poses[label]), backproject(K, pa[ok], da[ok]),
                                     backproject(K, pb[ok], db[ok]), label))
        out(f"  {label}: 匹配 {len(pa)} 个特征，其中 {int(ok.sum())} 个有深度")
    if sum(len(p.x0) for p in pairs) < 100:
        raise RuntimeError("可用特征太少：台面纹理不够或深度无效；换到有孔/有纹理的台面上方再试")

    h, w = ref_depth.shape
    vs, us = np.mgrid[0:h:8, 0:w:8]
    us, vs = us.reshape(-1), vs.reshape(-1)
    d = ref_depth[vs, us]
    ok = (d > 200) & (blocked[vs, us] == 0)
    grid = backproject(K, np.column_stack([us[ok], vs[ok]]).astype(float), d[ok].astype(float))
    normal, offset, inliers = fit_plane(grid, tolerance=3.0)
    plane_points = grid[inliers][:: max(1, int(inliers.sum()) // 600)]

    result = solve_hand_eye(pairs, T0, plane_points, table_z)
    T = result["T_flange_camera"]
    tip_cam = transform(np.linalg.inv(T), np.zeros((1, 3)))[0]
    gap = jaw_gap(gripper, K, tip_cam)

    # Independent checks. 1) Each turn about the tool axis must reproduce the commanded angle when
    # the matched points are compared in base frame. 2) The dominant plane seen from every pose
    # must land at one height (a wrong camera tilt shows up as height changing with the pose).
    heights = {}
    for label, frame in frames.items():
        cam_T = pose_matrix(poses[label]) @ T
        depth = frame["depth"]
        dd = depth[vs, us]
        good = (dd > 200) & (blocked[vs, us] == 0)
        pts = transform(cam_T, backproject(K, np.column_stack([us[good], vs[good]]).astype(float), dd[good].astype(float)))
        if len(pts) > 50:
            n, _, inl = fit_plane(pts, tolerance=3.0)
            heights[label] = round(float(np.median(pts[inl][:, 2])), 2)
    spread = max(heights.values()) - min(heights.values()) if heights else math.inf
    worst = max((p["median_mm"] or math.inf) for p in result["pairs"])
    fewest = min(p["points"] for p in result["pairs"])
    checks = {"pair_median_worst_mm": worst, "pair_points_fewest": fewest, "plane_height_by_pose": heights,
              "plane_height_spread_mm": round(spread, 2), "jaw_gap": gap,
              "camera_in_flange_mm": [round(float(v), 1) for v in T[:3, 3]],
              "optical_axis_tilt_from_tool_deg": round(float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1)))), 2)}
    problems = []
    if worst > ACCEPT["median_mm"]:
        problems.append(f"配对残差中位数最大 {worst:.2f} mm > {ACCEPT['median_mm']}")
    if fewest < ACCEPT["min_points"]:
        problems.append(f"有一组只剩 {fewest} 个点")
    if spread > ACCEPT["height_spread_mm"]:
        problems.append(f"各位姿看到的台面高度相差 {spread:.2f} mm > {ACCEPT['height_spread_mm']}")
    return {"T_flange_camera": T, "K": K, "solver": result, "checks": checks, "problems": problems,
            "gripper": gripper, "table_depth_ref_mm": round(table_depth, 1)}


def _run_resume(arm: Any, camera: Any, args: argparse.Namespace, folder: Path, *,
                ask: Callable[[str], str], out: Callable[[str], None],
                stop_pressed: Callable[[], bool]) -> dict[str, Any]:
    if not getattr(args, "capture_only", False):
        raise Aborted("--resume-capture requires --capture-only")
    if not getattr(args, "no_home", False):
        raise Aborted("--resume-capture requires --no-home")
    if args.yes:
        raise Aborted("--resume-capture always requires interactive confirmation")

    guarded_state: dict[str, Any] | None = None

    def confirm(prompt: str) -> None:
        if stop_pressed():
            arm.stop()
            raise JogStopped("emergency stop: calibration stopped")
        if ask(f"{prompt} (enter y to continue; Enter cancels): ").strip().lower() != "y":
            raise Aborted(prompt)
        opening = arm.gripper_position()
        if not _gripper_is_open(opening):
            raise Aborted(f"resume requires an open gripper after confirmation; current feedback is {opening}")
        if guarded_state is not None:
            _validate_resume_state(arm.read(), guarded_state["snapshot"], guarded_state["pose"],
                                   label=guarded_state["label"])

    source, source_acquisition, source_poses, burst = _read_resume_source(
        Path(args.resume_capture), getattr(args, "arm", None), args.table_z)
    if getattr(args, "capture_burst", burst) != burst:
        raise Aborted(f"--capture-burst must match the interrupted dataset ({burst})")
    camera_info = _camera_description(camera)
    camera_match = _validate_resume_camera(source_acquisition.get("camera", {}), camera_info)
    opening = arm.gripper_position()
    if not _gripper_is_open(opening):
        raise Aborted(f"resume requires an open gripper; current feedback is {opening}")

    base_pose = [float(value) for value in source_poses["ref"]]
    interrupted_pose = [float(value) for value in source_poses["z-"]]
    base_xyz, base_rpy = base_pose[:3], base_pose[3:]
    expected_z_pose = [base_xyz[0], base_xyz[1], base_xyz[2] - 50.0, *base_rpy]
    z_target_delta, z_target_rotation = _pose_deltas(interrupted_pose, expected_z_pose)
    if z_target_delta > CAPTURE_STABILITY_MM or z_target_rotation > CAPTURE_STABILITY_DEG:
        raise Aborted("saved z- pose is outside the original reference move/bounds")
    limits = Limits(base_xyz, 80.0, base_xyz[2] - 60.0, base_xyz[2] + 20.0)
    for pose_label in (move[0] for move in MOVES):
        if pose_label in source_poses and limits.violation(source_poses[pose_label][:3]):
            raise Aborted(f"saved {pose_label} pose is outside the original reference bounds")

    interrupted_group = source_acquisition["pose_groups"]["z-"]
    last_capture = interrupted_group["captures"][-1]
    saved_snapshot = last_capture.get("robot_after") or last_capture.get("robot_before")
    guarded_state = {"snapshot": saved_snapshot, "pose": interrupted_pose, "label": "z-"}
    pose_match = _validate_resume_state(arm.read(), saved_snapshot, interrupted_pose)
    if down_tilt_deg(arm.read()["pose"][3:]) > 0.05:
        raise Aborted("saved interruption orientation is not tool-down; resume will not auto-correct it")
    if folder.exists():
        raise Aborted("recovery output folder already exists; choose a fresh --out folder")

    # Keep the original ref as the Jog origin/soft-bound center while starting at z-.
    jog = Jog(arm, limits, record_path=folder / "unused.csv", ask=ask, out=out,
              stop_pressed=stop_pressed)
    pose_match = _validate_resume_state(arm.read(), saved_snapshot, interrupted_pose)
    if not _gripper_is_open(arm.gripper_position()):
        raise Aborted("gripper open feedback fell below 250 during resume preflight")
    acquisition, poses = _prepare_recovery_copy(
        source, folder, source_acquisition, source_poses, burst=burst,
        pose_match=pose_match, camera_match=camera_match)
    prefix = f"recovery_{folder.name.rsplit('-recovery-', 1)[-1]}"
    acquisition["recovery"]["file_prefix"] = prefix
    frames: dict[str, dict[str, Any]] = {}
    acquisition_path = folder / "acquisition.json"

    def persist() -> None:
        _write_json(folder / "poses.json", poses)
        _write_json(acquisition_path, acquisition, ensure_ascii=False)

    persist()

    def capture_group(group_name: str, suffix: str) -> tuple[list[dict[str, Any]], np.ndarray]:
        return _capture_group(arm, camera, camera_info, folder, acquisition, group_name,
                              f"{prefix}_{suffix}", burst, persist, stop_pressed)

    try:
        confirm("Resume recovery: capture a fresh full z- burst at the verified saved pose?")
        pose_match = _validate_resume_state(arm.settle(), saved_snapshot, interrupted_pose)
        acquisition["recovery"]["pose_preflight_before_z_capture"] = pose_match
        persist()
        observed, representative = capture_group("z-", "zminus")
        frames["z-"] = {**observed[0], "depth": representative}
        poses["z-"] = [float(value) for value in arm.read()["pose"]]
        persist()

        current_z_group = acquisition["pose_groups"]["z-"]
        guarded_state.update({"snapshot": current_z_group["captures"][-1]["robot_after"],
                              "pose": poses["z-"], "label": "z-"})
        _validate_resume_state(arm.read(), current_z_group["captures"][-1]["robot_after"],
                               poses["z-"], label="z-")

        label, dx, dy, dz, dyaw = MOVES[-1]
        target = [base_xyz[0] + dx, base_xyz[1] + dy, base_xyz[2] + dz]
        target_rpy = [base_rpy[0], base_rpy[1], base_rpy[2] + dyaw]
        _execute_transition(jog, "z-", label, target, target_rpy,
                            confirm=confirm, stop_pressed=stop_pressed)
        state = arm.settle()
        poses[label] = [float(value) for value in state["pose"]]
        persist()
        observed, representative = capture_group(label, "z_yaw_plus")
        frames[label] = {**observed[0], "depth": representative}
        current_yaw_group = acquisition["pose_groups"][label]
        guarded_state.update({"snapshot": current_yaw_group["captures"][-1]["robot_after"],
                              "pose": poses[label], "label": label})
        _validate_resume_state(arm.read(), current_yaw_group["captures"][-1]["robot_after"],
                               poses[label], label=label)

        _execute_transition(jog, label, "ref", base_xyz, base_rpy,
                            confirm=confirm, stop_pressed=stop_pressed)
        arm.settle()
        capture_group("ref_return", "ref_return")
        ref_start = acquisition["pose_groups"]["ref"]["captures"][0]["robot_before"]
        ref_end = acquisition["pose_groups"]["ref_return"]["captures"][0]["robot_before"]
        drift = _state_delta(ref_start, ref_end)
        acquisition["return_home_drift"] = drift
        if not drift.get("stable", False):
            acquisition["capture_problems"].append(
                "ref_return: measured home pose differs from the initial ref beyond tolerance")
        acquisition["pose_groups"]["ref_return"]["color_drift_comparison"] = {
            "reference_observation": "original ref color and per-frame depth files",
            "returned_observation": "recovery ref_return color and per-frame depth files",
            "replay_note": "Compare source observations offline; no RGB averaging was applied.",
        }
        acquisition["status"] = "acquired"
        acquisition["finished_at"] = datetime.now().isoformat(timespec="seconds")
        persist()
        acquisition["status"] = "rejected" if acquisition["capture_problems"] else "complete"
        acquisition["capture_only"] = True
        persist()
        return {"acquisition": acquisition, "problems": acquisition["capture_problems"]}
    except Exception:
        acquisition["status"] = "partial"
        acquisition["interrupted_at"] = datetime.now().isoformat(timespec="seconds")
        persist()
        raise


def run(arm: Any, camera: Any, args: argparse.Namespace, folder: Path, *, ask: Callable[[str], str],
        out: Callable[[str], None], stop_pressed: Callable[[], bool]) -> dict[str, Any]:
    if getattr(args, "resume_capture", None) is not None:
        return _run_resume(arm, camera, args, folder, ask=ask, out=out, stop_pressed=stop_pressed)
    if args.yes and getattr(args, "live", False):
        raise Aborted("现场标定不能使用 --yes，必须逐步确认")

    def confirm(prompt: str) -> None:
        if stop_pressed():
            arm.stop()
            raise JogStopped("急停：已停止")
        if args.yes:
            return
        if ask(f"{prompt}（输入 y 继续，回车中止）: ").strip().lower() != "y":
            raise Aborted(prompt)

    start = arm.read()
    xyz = [float(v) for v in start["pose"][:3]]
    limits = Limits(xyz, 80.0, xyz[2] - 60.0, xyz[2] + 20.0)
    jog = Jog(arm, limits, record_path=folder / "unused.csv", ask=ask, out=out, stop_pressed=stop_pressed)
    if down_tilt_deg(jog.rpy) > 0.05:
        confirm(f"工具偏离竖直 {down_tilt_deg(jog.rpy):.2f}°，原地摆正")
        jog.snap_down()
    base_xyz, base_rpy = list(jog.xyz), list(jog.rpy)
    opening = arm.gripper_position()
    if opening < 250:
        confirm(f"夹爪要张到最大（当前 {opening}），确认手里没有东西")
        out(f"夹爪：{arm.set_gripper(255)}")
        opening = arm.gripper_position()
    if opening < 250:
        raise Aborted(f"张爪反馈 {opening} 未达到 250，未采集张爪参考")
    confirm("接下来在原地附近做 9 个小动作（转 ±15°、平移 ±30 mm、下降 50 mm），"
            "确认平移、转动、下降和返回时夹爪及相机都有净空，台面上的哑光纹理已固定")
    burst = getattr(args, "capture_burst", 1)
    if not 1 <= burst <= CAPTURE_BURST_MAX:
        raise ValueError(f"--capture-burst must be from 1 to {CAPTURE_BURST_MAX}")
    folder.mkdir(parents=True, exist_ok=True)
    frames, poses = {}, {}
    camera_info = _camera_description(camera)
    acquisition: dict[str, Any] = {
        "schema_version": 1,
        "arm": getattr(args, "arm", None),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "capture_burst": burst,
        "initial_gripper_position": opening,
        "table_z_reference_mm": float(args.table_z),
        "status": "in_progress",
        "camera": camera_info,
        "depth_burst_aggregation": {
            "method": "per-pixel temporal median of positive finite depth",
            "minimum_valid_observations": math.ceil(burst / 2),
            "depth_unit": "mm",
        },
        "pose_groups": {},
        "capture_problems": [],
    }
    acquisition_path = folder / "acquisition.json"

    def persist() -> None:
        _write_json(folder / "poses.json", poses)
        _write_json(acquisition_path, acquisition, ensure_ascii=False)

    # Empty-but-valid snapshots distinguish a cancelled first transition from an older run.
    persist()

    def capture_group(group_name: str, file_prefix: str, count: int) -> tuple[list[dict[str, Any]], np.ndarray]:
        return _capture_group(arm, camera, camera_info, folder, acquisition, group_name,
                              file_prefix, count, persist, stop_pressed)

    def save_route_observation() -> None:
        # Close-spaced images preserve correspondence on a repeating grid. The arm is at rest.
        observations = acquisition.setdefault("route_observations", [])
        frame = camera.grab(flush=2, require_fresh_pair=True) if camera_info.get("source") == "realsense" else camera.grab()
        path = folder / f"route_{len(observations):03d}.png"
        if not cv2.imwrite(str(path), frame["color"]):
            raise RuntimeError("could not save intermediate board observation")
        observations.append({"color": path.name, "from": previous_label, "to": route_destination,
                             "robot": _robot_snapshot(arm.read())})
        persist()

    observer = save_route_observation if getattr(args, "track_board_route", False) else None
    previous_label = "ref"
    for label, dx, dy, dz, dyaw in MOVES:
        route_destination = label
        target = [base_xyz[0] + dx, base_xyz[1] + dy, base_xyz[2] + dz]
        rpy = [base_rpy[0], base_rpy[1], base_rpy[2] + dyaw]
        _execute_transition(jog, previous_label, label, target, rpy,
                            confirm=confirm, stop_pressed=stop_pressed, observe_waypoint=observer)
        state = arm.settle()
        poses[label] = [float(v) for v in state["pose"]]
        persist()
        observed, representative = capture_group(label, label, burst)
        # Keep the first color image; the solver uses the temporal median depth for burst captures.
        frames[label] = {**observed[0], "depth": representative}
        out(f"{label}：位姿 {[round(v, 2) for v in poses[label]]}")
        previous_label = label
    route_destination = "ref_return"
    _execute_transition(jog, previous_label, "ref", base_xyz, base_rpy,
                        confirm=confirm, stop_pressed=stop_pressed, observe_waypoint=observer)
    if burst > 1:
        arm.settle()
        capture_group("ref_return", "ref_return", burst)
        ref_start = acquisition["pose_groups"]["ref"]["captures"][0]["robot_before"]
        ref_end = acquisition["pose_groups"]["ref_return"]["captures"][0]["robot_before"]
        drift = _state_delta(ref_start, ref_end)
        acquisition["return_home_drift"] = drift
        if not drift.get("stable", False):
            acquisition["capture_problems"].append(
                "ref_return: measured home pose differs from the initial ref beyond tolerance")
        acquisition["pose_groups"]["ref_return"]["color_drift_comparison"] = {
            "reference_observation": "ref color and per-frame depth files",
            "returned_observation": "ref_return color and per-frame depth files",
            "replay_note": "Compare source observations offline; no RGB averaging was applied.",
        }
    acquisition["status"] = "acquired"
    acquisition["finished_at"] = datetime.now().isoformat(timespec="seconds")
    persist()
    if getattr(args, "capture_only", False):
        acquisition["status"] = "rejected" if acquisition["capture_problems"] else "complete"
        acquisition["capture_only"] = True
        persist()
        return {"acquisition": acquisition, "problems": acquisition["capture_problems"]}
    out("计算相机安装位姿……")
    result = process(frames, poses, camera.K, args.table_z, out)
    result["acquisition"] = acquisition
    result.setdefault("problems", []).extend(acquisition["capture_problems"])
    acquisition["status"] = "rejected" if acquisition["capture_problems"] else "complete"
    persist()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="腕部相机免标定板手眼标定（原地小动作 + 台面特征 + 深度）")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--sim", action="store_true", help="合成相机 + 模拟臂，不连硬件")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--table-z", type=float, help="参考平面高度（默认按臂取 TABLE_Z；见结果说明）")
    parser.add_argument("--capture-burst", type=_capture_burst_value, default=1, metavar="N",
                        help="synchronized observations per pose for calibration-only capture (1..15; default 1)")
    parser.add_argument("--capture-only", action="store_true",
                        help="保存各姿态的相机与机械臂观测，供棋盘格离线求解")
    parser.add_argument("--track-board-route", action="store_true",
                        help="在现有小步到位后额外存一张图，供重复棋格的跨帧核对")
    parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)  # offline tests only
    parser.add_argument("--no-home", action="store_true", help="不先回初始位，就在当前位置标定")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=CALIBRATION_DIR)
    parser.add_argument("--resume-capture", type=Path, metavar="PATH",
                        help="resume an interrupted capture-only dataset from its saved z- pose")
    return parser


def main(argv: Sequence[str] | None = None, *, ask: Callable[[str], str] = input,
         out: Callable[[str], None] = print, arm: Any = None, camera: Any = None,
         stop_pressed: Callable[[], bool] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.live and args.yes:
        parser.error("--yes 只能与 --sim 一起使用，现场标定必须逐步确认")
    if args.table_z is None:
        args.table_z = TABLE_Z[args.arm]
    if args.resume_capture is not None:
        if not args.capture_only:
            parser.error("--resume-capture requires --capture-only")
        if not args.no_home:
            parser.error("--resume-capture requires --no-home; automatic homing is disabled")
        if args.yes:
            parser.error("--resume-capture always requires interactive confirmation")
        try:
            source, _, _, source_burst = _read_resume_source(args.resume_capture, args.arm, args.table_z)
        except Aborted as error:
            parser.error(str(error))
        provided_args = list(argv) if argv is not None else sys.argv[1:]
        burst_was_explicit = any(arg == "--capture-burst" or arg.startswith("--capture-burst=")
                                 for arg in provided_args)
        if burst_was_explicit and args.capture_burst != source_burst:
            parser.error(f"--capture-burst must match the interrupted dataset ({source_burst})")
        args.capture_burst = source_burst
        folder = _new_recovery_folder(source, args.out)
    else:
        folder = args.out / datetime.now().strftime(f"{args.arm}-%Y%m%d_%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
    if stop_pressed is None:
        if sys.platform == "win32" and not args.sim:
            from task_skills.run_cartesian_jog import _windows_keys
            stop_pressed = _windows_keys()[1]
        else:
            stop_pressed = lambda: False  # noqa: E731
    code = 1
    try:
        # A missing/unavailable wrist camera must be discovered before opening the live arm
        # session, because open_session may move the robot to home.
        if camera is None and not args.sim:
            from chem_demo.wrist_camera import LiveWristCamera
            camera = LiveWristCamera(camera_serial(f"{args.arm}_wrist"))
        if arm is None:
            if args.sim:
                from chem_demo.wrist_camera import SimWristCamera
                arm = SimArm([117.6, -344.8, 415.5], snap_down_rpy([180.0, 0.0, 90.0]))
                camera = SimWristCamera(arm, table_z=args.table_z)
            else:
                profile = json.loads(args.profile.read_text(encoding="utf-8"))
                arm = LiveArm(profile, args.arm, args.speed, log=out)
                problem = start_refusal(arm.read().get("mode"))
                if problem:
                    out(f"拒绝启动，没有发送任何指令：{problem}")
                    return 5
                if ask(f"确认 {args.arm} 臂周围无人、急停在手边，输入 {args.arm} 继续: ").strip() != args.arm:
                    out("未确认，已退出，没有发送任何指令")
                    return 1
                out(arm.open_session(home=not args.no_home))
        if camera is None:
            from chem_demo.wrist_camera import SimWristCamera
            camera = SimWristCamera(arm, table_z=args.table_z)
        result = run(arm, camera, args, folder, ask=ask, out=out, stop_pressed=stop_pressed)
        if args.capture_only:
            if result["problems"]:
                out("采集质量检查未通过：" + "；".join(result["problems"]))
                return 6
            out(f"采集完成，待离线标定：{folder}")
            return 0
        checks = result["checks"]
        out(json.dumps({"checks": checks, "solver_pairs": result["solver"]["pairs"],
                        "table_z_residual": result["solver"]["table_z_residual_mm"]}, ensure_ascii=False, indent=1))
        cv2.imwrite(str(folder / "gripper_mask.png"), result["gripper"] * 255)
        record = {"arm": args.arm, "time": datetime.now().isoformat(timespec="seconds"), "frames": str(folder),
                  "acquisition_file": str(folder / "acquisition.json"),
                  "camera": result["acquisition"]["camera"], "capture_burst": args.capture_burst,
                  "T_flange_camera": result["T_flange_camera"], "K": result["K"], "checks": checks,
                  "solver": result["solver"] | {"T_flange_camera": None}, "problems": result["problems"],
                  "table_z_reference": args.table_z,
                  "notes": ["相机沿工具轴的高度由参考平面高度决定；标定处看到的若是金属台面而非白色台面，"
                            "该高度不准，run_bottle_pick 每次用白色台面深度重新测定"]}
        if result["problems"]:
            out("标定未通过，不保存：" + "；".join(result["problems"]))
            (folder / "result_rejected.json").write_text(
                json.dumps({**record, "T_flange_camera": result["T_flange_camera"].tolist(), "K": result["K"].tolist()},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            code = 6
        else:
            # A --sim result stays next to its frames and never replaces the live calibration.
            path = save_calibration(args.arm, record, result["gripper"], folder if args.sim else args.out)
            out(f"标定通过，已保存 {path}")
            code = 0
    except Aborted as reason:
        out(f"已中止（{reason}）")
    except JogRefused as refusal:
        out(f"拒绝，未发送这一步的运动：{refusal}")
        code = 4
    except JogStopped as stopped:
        out(str(stopped))
        code = 3
    except RuntimeError as error:
        out(f"运行错误，已停止：{error}")
        code = 4
    except (ImportError, OSError) as error:
        out(f"相机或硬件初始化失败；机械臂未开始标定：{error}")
        code = 4
    finally:
        if camera is not None:
            camera.close()
        if arm is not None:
            arm.close()
    out(f"记录目录：{folder}")
    return code


if __name__ == "__main__":
    sys.exit(main())
