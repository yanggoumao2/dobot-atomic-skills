"""Read-only wrist observation for the first bottle-pick stage.

This module accepts only the checked open-wrist camera model's recorded poses. Live robot and
gripper state are read through passive interfaces; it never creates a motion driver.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from chem_demo.wrist_geometry import backproject, pose_matrix, rotation_angle_deg, sample_depth
from chem_demo.wrist_optical import OpenWristCameraModel
from chem_demo.wrist_runtime import ROOT, camera_serial, save_frame
from chem_demo.wrist_landmarks import measure_open_pad_tips


DEFAULT_OPEN_WRIST_MODEL = (
    ROOT / "calibration-data" / "chem-demo" / "wrist-board" /
    "left-20261121_152610" / "checker_plane_validation.json"
)
POSITION_STABILITY_MM = 0.5
ANGLE_STABILITY_DEG = 0.5
JOINT_STABILITY_DEG = 0.5
ZERO_SPEED_TOLERANCE = 0.5
GRIPPER_OPEN_MINIMUM = 250


def _angle_delta(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    pose_a, pose_b = before.get("pose"), after.get("pose")
    joints_a, joints_b = before.get("joints"), after.get("joints")
    if not pose_a or not pose_b or not joints_a or not joints_b:
        return {"available": False, "stable": False, "reason": "missing six-axis pose or joints"}
    if len(pose_a) != 6 or len(pose_b) != 6 or len(joints_a) != 6 or len(joints_b) != 6:
        return {"available": False, "stable": False, "reason": "pose or joints are not six-axis"}
    values = np.asarray([*pose_a, *pose_b, *joints_a, *joints_b], dtype=float)
    if not np.isfinite(values).all():
        return {"available": False, "stable": False, "reason": "pose or joints contain non-finite values"}
    position = math.dist(pose_a[:3], pose_b[:3])
    orientation = rotation_angle_deg(pose_matrix(pose_a)[0:3, 0:3].T @ pose_matrix(pose_b)[0:3, 0:3])
    joint = max(_angle_delta(joints_a[i], joints_b[i]) for i in range(6))
    stable = (position <= POSITION_STABILITY_MM and orientation <= ANGLE_STABILITY_DEG and
              joint <= JOINT_STABILITY_DEG)
    return {"available": True, "stable": stable,
            "translation_mm": round(position, 4), "rotation_deg": round(orientation, 4),
            "joint_max_deg": round(joint, 4),
            "limits": {"translation_mm": POSITION_STABILITY_MM,
                       "rotation_deg": ANGLE_STABILITY_DEG, "joint_deg": JOINT_STABILITY_DEG}}


def _state_gate(before: dict[str, Any], after: dict[str, Any], *, expected_user: int,
                expected_tool: int, require_full_telemetry: bool) -> tuple[dict[str, Any], list[str]]:
    problems: list[str] = []
    delta = _state_delta(before, after)
    if not delta.get("stable"):
        problems.append("robot pose changed during RGB-D capture" if delta.get("available") else
                        "robot pose stability could not be verified")

    for label, state in (("before", before), ("after", after)):
        if state.get("mode") not in (4, 5):
            problems.append(f"{label} RobotMode={state.get('mode')} is not idle/disabled")
        enabled = state.get("enabled")
        if enabled is not None and int(enabled) != (1 if state.get("mode") == 5 else 0):
            problems.append(f"{label} enable status conflicts with RobotMode")
        if require_full_telemetry and enabled is None:
            problems.append(f"{label} feedback lacks enable status")
        running = state.get("running")
        if running is not None and int(running) != 0:
            problems.append(f"{label} controller reports a running command")
        if require_full_telemetry and running is None:
            problems.append(f"{label} feedback lacks running status")
        for key, expected, label_text in (("user", expected_user, "User"),
                                          ("tool", expected_tool, "Tool")):
            actual = state.get(key)
            if actual is not None and int(actual) != expected:
                problems.append(f"{label} selected {label_text}{actual}; expected {label_text}{expected}")
            if require_full_telemetry and actual is None:
                problems.append(f"{label} feedback lacks selected {label_text}")
        for key, tolerance, axes, name in (
                ("tcp_speed", ZERO_SPEED_TOLERANCE, slice(0, 3), "TCP linear speed"),
                ("tcp_speed", ZERO_SPEED_TOLERANCE, slice(3, 6), "TCP angular speed"),
                ("joint_speed", ZERO_SPEED_TOLERANCE, slice(0, 6), "joint speed")):
            speed = state.get(key)
            if speed is None:
                if require_full_telemetry:
                    problems.append(f"{label} feedback lacks {name}")
            else:
                expected_length = 6
                try:
                    speed_values = np.asarray(speed, dtype=float)
                except (TypeError, ValueError):
                    speed_values = np.asarray([], dtype=float)
                if speed_values.shape != (expected_length,) or not np.isfinite(speed_values).all():
                    problems.append(f"{label} feedback has invalid {name}")
                elif max((abs(float(value)) for value in speed_values[axes]), default=0.0) > tolerance:
                    problems.append(f"{label} {name} exceeds {tolerance:g}")

    if before.get("mode") != after.get("mode"):
        problems.append("RobotMode changed during capture")
    if before.get("running") != after.get("running"):
        problems.append("running status changed during capture")
    return {"stable": not problems, "before": before, "after": after, "delta": delta,
            "expected_user": expected_user, "expected_tool": expected_tool}, problems


def read_passive_robot_state(ip: str) -> dict[str, Any]:
    """Read a fresh 30004 state packet on a new socket; no dashboard or motion commands."""
    from dobot_control.robots.dobot_api import DobotApiStatus

    stream = DobotApiStatus(ip, 30004)
    try:
        values = stream.get_feedback()
        return {
            "source": "30004 feedback",
            "mode": int(values["robot_mode"]),
            "enabled": int(values["enable_status"]),
            "running": int(values["running_status"]),
            "tool": int(values["tool"]),
            "user": int(values["user"]),
            "pose": [float(value) for value in values["tool_vector_actual"]],
            "joints": [float(value) for value in values["q_actual"]],
            "tcp_speed": [float(value) for value in values["TCP_speed_actual"]],
            "joint_speed": [float(value) for value in values["qd_actual"]],
        }
    finally:
        stream.close()


def read_gripper_position(port_name: str, servo_id: int, servo_range: list[int]) -> dict[str, Any]:
    """Read the gripper without using DobotGripper, whose constructor writes torque limit."""
    from third_party.feetech.scservo_sdk import PortHandler, sms_sts

    if len(servo_range) != 2 or int(servo_range[0]) == int(servo_range[1]):
        raise ValueError("profile gripper servo_pos must contain two distinct raw positions")
    port = PortHandler(port_name)
    try:
        if not port.setBaudRate(1_000_000):
            raise RuntimeError(f"could not set gripper baud rate on {port_name}")
        servo = sms_sts(port)
        raw, result, error = servo.ReadPos(int(servo_id))
        if result != 0 or error != 0:
            raise RuntimeError(f"gripper ReadPos failed (comm_result={result}, servo_error={error})")
        low, high = (int(value) for value in servo_range)
        position = int((raw - low) / (high - low) * 255)
        return {"position": position, "raw_position": int(raw), "source": "read-only Feetech ReadPos"}
    finally:
        if port.is_open:
            port.closePort()


def _read_replay(folder: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
    camera = metadata["camera"]
    before, after = dict(metadata["before"]), dict(metadata["after"])
    frame = {"color": cv2.imread(str(folder / "ref.png")),
             "depth": np.load(folder / "ref_depth_temporal_median_mm.npy", allow_pickle=False),
             "capture_metadata": {"source": "saved capture replay", "capture": str(folder),
                                  "frames": metadata.get("frames", [])}}
    for state in (before, after):
        state.setdefault("pose", state.get("tcp_pose_mm_deg"))
        state.setdefault("joints", state.get("joints_deg"))
        state.setdefault("running", state.get("running_status"))
    gripper = {"before": metadata.get("gripper_position_before"),
               "after": metadata.get("gripper_position_after"), "source": "saved capture metadata"}
    return camera, before, after, {"frame": frame, "gripper": gripper}


def _pose_match(model: OpenWristCameraModel, pose: list[float]) -> dict[str, Any]:
    candidates = []
    actual = pose_matrix(pose)[:3, :3]
    for index, calibrated in enumerate(model.poses):
        calibrated_pose = [float(value) for value in calibrated]
        distance = math.dist(pose[:3], calibrated_pose[:3])
        angle = rotation_angle_deg(pose_matrix(calibrated_pose)[:3, :3].T @ actual)
        candidates.append((max(distance / 0.5, angle / 0.5), index, distance, angle))
    _, index, distance, angle = min(candidates)
    return {"nearest_recorded_pose_index": index,
            "translation_error_mm": round(distance, 4), "rotation_error_deg": round(angle, 4),
            "accepted_by_model": False}


def _jaw_report(color: np.ndarray, depth: np.ndarray, K: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    landmarks = measure_open_pad_tips(color)
    report: dict[str, Any] = {"ok": bool(landmarks.ok), "state": landmarks.state,
                              "left_tip_px": landmarks.left, "right_tip_px": landmarks.right,
                              "gap_px": landmarks.gap_px, "confidence": round(float(landmarks.confidence), 4),
                              "reason": landmarks.reason}
    if landmarks.ok and landmarks.left is not None and landmarks.right is not None:
        pixels = np.asarray([landmarks.left, landmarks.right], dtype=float)
        depths = sample_depth(depth, pixels, window=2)
        if np.all(depths > 0):
            points = backproject(K, pixels, depths)
            report["gap_mm_camera_frame"] = round(float(np.linalg.norm(points[1] - points[0])), 2)
            report["tip_depth_mm"] = [round(float(value), 2) for value in depths]
    overlay = color.copy()
    for name, point, color_bgr in (("L", landmarks.left, (0, 255, 0)),
                                   ("R", landmarks.right, (0, 200, 255))):
        if point is not None:
            pixel = tuple(int(round(value)) for value in point)
            cv2.drawMarker(overlay, pixel, color_bgr, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(overlay, name, (pixel[0] + 6, pixel[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color_bgr, 2, cv2.LINE_AA)
    return report, overlay


def _camera_info(camera: Any) -> dict[str, Any]:
    describe = getattr(camera, "describe", None)
    if callable(describe):
        return describe()
    K = getattr(camera, "K", None)
    return {"source": "injected camera", "serial": getattr(camera, "serial", None),
            "intrinsics": {"matrix_px": np.asarray(K, float).tolist()} if K is not None else None,
            "depth_output_unit": "mm", "depth_alignment": "aligned to color"}


def run_observation(args: Any, out_dir: Path, *, out: Callable[[str], None] = print,
                    camera: Any = None, profile: dict[str, Any] | None = None,
                    state_reader: Callable[[], dict[str, Any]] | None = None,
                    gripper_reader: Callable[[], dict[str, Any]] | None = None) -> tuple[dict[str, Any], int]:
    """Capture or replay one pose-gated observation and persist the evidence report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.open_wrist_model or DEFAULT_OPEN_WRIST_MODEL
    report: dict[str, Any] = {
        "purpose": "read-only bottle observation; not a motion or hover authorization",
        "source": "saved replay" if args.replay is not None else "live passive",
        "arm": args.arm,
        "model": str(model_path),
        "status": "rejected",
        "limitations": ["OpenWristCameraModel only supports its recorded calibration postures; "
                        "this report does not authorize arbitrary hovering or motion."],
    }
    opened_camera = False
    model_accepted = False
    try:
        if args.arm != "left":
            raise ValueError("the verified open-wrist observation model currently supports only --arm left")

        model_record = json.loads(model_path.read_text(encoding="utf-8"))
        model = OpenWristCameraModel(model_path, serial=model_record["serial"],
                                     K=np.asarray(model_record["K"], dtype=float))
        if model.arm != "left":
            raise ValueError("loaded open-wrist model is not for the left arm")

        if args.replay is not None:
            camera_info, before, after, replay_data = _read_replay(args.replay)
            frame = replay_data["frame"]
            gripper = replay_data["gripper"]
            expected_user, expected_tool = 0, 0
            state_source = "saved capture metadata"
            require_full_telemetry = False
            K = np.asarray(camera_info["intrinsics"]["matrix_px"], dtype=float)
            serial = camera_info["serial"]
        else:
            if profile is None:
                profile = json.loads(args.profile.read_text(encoding="utf-8"))
            frame_config = profile.get("frame", {})
            expected_user, expected_tool = int(frame_config.get("user", -1)), int(frame_config.get("tool", -1))
            if expected_user != 0 or expected_tool != 0:
                raise ValueError("this verified model was captured with Tool0/User0; profile frame must be 0/0")
            binding = profile["arms"]["left"]
            if state_reader is None:
                state_reader = lambda: read_passive_robot_state(binding["ip"])
            if gripper_reader is None:
                gripper_reader = lambda: read_gripper_position(
                    binding["gripper_port"], binding["gripper_id"], binding["servo_pos"])
            if camera is None:
                from chem_demo.wrist_camera import LiveWristCamera
                camera = LiveWristCamera(camera_serial("left_wrist"))
                opened_camera = True
            camera_info = _camera_info(camera)
            K = np.asarray(camera_info["intrinsics"]["matrix_px"], dtype=float)
            serial = camera_info.get("serial") or camera_serial("left_wrist")
            before = state_reader()
            capture_problems: list[str] = []
            try:
                gripper_before = gripper_reader()
            except Exception as error:  # noqa: BLE001 - preserve the frame as rejected evidence
                gripper_before = {}
                capture_problems.append(f"gripper ReadPos before capture failed: {error}")
            frame = camera.grab(flush=4, require_fresh_pair=True)
            try:
                gripper_after = gripper_reader()
            except Exception as error:  # noqa: BLE001 - preserve the frame as rejected evidence
                gripper_after = {}
                capture_problems.append(f"gripper ReadPos after capture failed: {error}")
            try:
                after = state_reader()
            except Exception as error:  # noqa: BLE001 - preserve the frame as rejected evidence
                after = {}
                capture_problems.append(f"30004 feedback after capture failed: {error}")
            gripper = {"before": gripper_before.get("position"),
                       "after": gripper_after.get("position"),
                       "raw_before": gripper_before.get("raw_position"),
                       "raw_after": gripper_after.get("raw_position"),
                       "source": gripper_before.get("source")}
            state_source = "30004 feedback"
            require_full_telemetry = True
            report["live_read_errors"] = capture_problems

        if not isinstance(frame.get("color"), np.ndarray) or not isinstance(frame.get("depth"), np.ndarray):
            raise ValueError("capture is missing color or depth")
        if frame["color"].ndim != 3 or frame["color"].shape[2] != 3 or frame["depth"].shape != frame["color"].shape[:2]:
            raise ValueError("color and aligned depth dimensions do not match")
        if not np.isfinite(frame["depth"]).all() or np.any(frame["depth"] < 0):
            raise ValueError("depth contains invalid values")
        if not camera_info.get("intrinsics", {}).get("distortion_removed_from_output", True):
            raise ValueError("OpenWristCameraModel requires undistorted color frames")
        if serial != model_record["serial"] or not np.allclose(K, model.K, rtol=1e-6, atol=1e-5):
            raise ValueError("camera serial or intrinsics differ from the verified open-wrist model")
        robot_report, state_problems = _state_gate(before, after, expected_user=expected_user,
                                                   expected_tool=expected_tool,
                                                   require_full_telemetry=require_full_telemetry)
        state_problems.extend(report.get("live_read_errors", []))
        report["robot"] = {"source": state_source, **robot_report}
        report["robot"]["problems"] = state_problems
        report["camera"] = camera_info
        report["capture"] = frame.get("capture_metadata", {})
        report["gripper"] = {**gripper, "open_minimum": GRIPPER_OPEN_MINIMUM}
        pose = after.get("pose") or after.get("tcp_pose_mm_deg")
        if pose is None:
            pose = before.get("pose") or before.get("tcp_pose_mm_deg")
        pose = [float(value) for value in pose]
        if len(pose) != 6 or not np.isfinite(pose).all():
            raise ValueError("captured TCP pose must contain six finite values")
        report["tcp_pose_mm_deg"] = pose
        report["model_pose_match"] = _pose_match(model, pose)

        jaw, overlay = _jaw_report(frame["color"], frame["depth"], K)
        report["jaw"] = jaw
        image_open = jaw["ok"] and jaw["state"] == "open"
        positions = [value for value in (gripper.get("before"), gripper.get("after")) if value is not None]
        telemetry_open = not positions or all(float(value) >= GRIPPER_OPEN_MINIMUM for value in positions)
        telemetry_stable = len(positions) < 2 or abs(float(positions[0]) - float(positions[1])) <= 5
        if not image_open:
            report["model_error"] = "image evidence does not show two confidently open jaws"
        elif not telemetry_open or not telemetry_stable:
            report["model_error"] = "gripper position feedback is not fully open and stable"
        elif state_problems:
            report["model_error"] = "; ".join(state_problems)
        else:
            try:
                measured = model.reference.measure(frame["color"], frame["depth"], arm="left")
                report["jaw_registration"] = {
                    "ok": bool(measured.ok), "matches": int(measured.matches),
                    "inliers": int(measured.inliers), "left_inliers": int(measured.left_inliers),
                    "right_inliers": int(measured.right_inliers),
                    "median_residual_px": measured.median_residual_px,
                    "p90_residual_px": measured.p90_residual_px, "reason": measured.reason,
                }
                T_base_camera = model.at_recorded_pose(pose, frame["color"], frame["depth"], gripper_open=True)
                report["model_pose_match"]["accepted_by_model"] = True
                report["T_base_camera"] = T_base_camera.tolist()
                model_accepted = True
                report["operator_message"] = (
                    "影像通过了已验证的腕部相机模型，且姿态匹配记录中的标定姿态。"
                    "这不代表可以在任意位置悬停或驱动机械臂。")
            except (ValueError, RuntimeError, cv2.error) as error:
                report["model_error"] = str(error)

        paths = save_frame(frame, out_dir, "wrist_observation")
        depth_path = out_dir / "wrist_observation_depth_mm.npy"
        np.save(depth_path, np.asarray(frame["depth"], dtype=np.float32), allow_pickle=False)
        overlay_path = out_dir / "wrist_observation_jaws.jpg"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError("could not save jaw evidence overlay")
        required_files = [Path(paths["color"]), Path(paths["depth"]), depth_path, overlay_path]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required_files):
            raise RuntimeError("one or more wrist evidence files were not saved")
        report["evidence"] = {"color": paths["color"], "depth_png": paths["depth"],
                              "depth_mm_npy": str(depth_path), "jaw_overlay": str(overlay_path)}
        if model_accepted:
            report["status"] = "recorded pose accepted"
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError, cv2.error) as error:
        report["status"] = "rejected"
        report["error"] = f"{type(error).__name__}: {error}"
        out(f"只读观测未通过：{error}")
    finally:
        if opened_camera and camera is not None:
            camera.close()

    report_path = out_dir / "observation.json"
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False, default=str),
                               encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        report["status"] = "rejected"
        out(f"无法保存观测记录：{error}")
        return report, 1
    if report["status"] == "recorded pose accepted":
        out(report["operator_message"])
        pose = report["tcp_pose_mm_deg"]
        match = report["model_pose_match"]
        before = report["robot"]["before"]
        out(f"TCP=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) mm；最近标定姿态误差 "
            f"{match['translation_error_mm']:.3f} mm / {match['rotation_error_deg']:.3f}°；"
            f"RobotMode={before.get('mode')}，Tool={before.get('tool')}，User={before.get('user')}")
        out(f"夹爪图像：{report['jaw']['state']}，开口 {report['jaw'].get('gap_px')} px，"
            f"置信度 {report['jaw']['confidence']:.2f}；证据文件：{report.get('evidence')}")
        if report["gripper"].get("before") is not None:
            out(f"夹爪串口位置：采集前 {report['gripper']['before']}，采集后 {report['gripper']['after']} / 255")
        else:
            out("夹爪串口位置未记录；张开状态仅由图像判断")
        return report, 0
    if "model_error" in report:
        out(f"相机模型未通过：{report['model_error']}")
    out(f"观测证据：{report_path}")
    return report, 1


__all__ = ["DEFAULT_OPEN_WRIST_MODEL", "read_gripper_position", "read_passive_robot_state", "run_observation"]
