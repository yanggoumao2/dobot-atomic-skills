"""One-time, operator-guided Z and gripper commissioning for the neck pick.

The caller must already be at the accepted hover pose. This module never
chooses XY or yaw: it changes Z in checked 1/5/10 mm increments, then issues
the configured low-speed close command while the empty bottle remains
supported on the bench. A profile is written only after the operator confirms
neck contact and the same close command produces distinct loaded and unloaded
feedback readings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from task_skills.run_cartesian_jog import Jog, JogRefused, JogStopped

PROFILE_SCHEMA_VERSION = "1.0.0"
EXPECTED_ARM = "left"
EXPECTED_RPY = (180.0, 0.0, -90.0)
EXPECTED_HOVER_Z_MM = 485.0
Z_STEPS_MM = {"1": 1.0, "5": 5.0, "0": 10.0}
TEACH_SPEED = 5
OPEN_POSITION = 255
NECK_GRIP_TARGET = 0
MAX_GRIP_TARGET = 254
DEFAULT_Z_FLOOR_MM = 350.0
STABILITY_SAMPLES = 5
STABILITY_INTERVAL_S = 0.05
STABILITY_SPAN_MM = 0.5
STABILITY_SPAN_DEG = 0.5
POSE_TOL_MM = 0.5
POSE_TOL_DEG = 0.5
MIN_CONTACT_CLEARANCE_MM = 15.0


class CommissioningError(RuntimeError):
    """Commissioning could not complete without violating its safety bounds."""


class CommissioningAborted(CommissioningError):
    """The operator selected abort or declined a required confirmation."""


def _angle_delta(left: float, right: float) -> float:
    return (float(left) - float(right) + 180.0) % 360.0 - 180.0


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _feedback_final(feedback: Any, label: str, expected_target: int | None = None) -> int:
    if not isinstance(feedback, dict) or not _finite_number(feedback.get("final")):
        raise CommissioningError(f"{label} returned no stable final servo-position reading; no profile was saved")
    value = int(feedback["final"])
    if not 0 <= value <= 255:
        raise CommissioningError(f"{label} returned out-of-range servo-position reading {value}")
    samples = feedback.get("samples")
    if not isinstance(samples, (list, tuple)) or len(samples) < 5 or not all(_finite_number(v) for v in samples):
        raise CommissioningError(f"{label} lacks five stable feedback samples; no profile was saved")
    observed = [int(sample) for sample in samples[-5:]]
    if max(observed) - min(observed) > 1 or abs(value - observed[-1]) > 1:
        raise CommissioningError(f"{label} feedback is not stable; no profile was saved")
    if expected_target is not None:
        try:
            target = int(feedback["target"])
            start = int(feedback["start"])
        except (KeyError, TypeError, ValueError) as error:
            raise CommissioningError(f"{label} lacks start/target feedback; no profile was saved") from error
        if target != expected_target:
            raise CommissioningError(f"{label} acknowledged target {target}, expected {expected_target}")
        if abs(value - target) > 2 and abs(value - start) < 3:
            raise CommissioningError(f"{label} feedback neither reached target nor showed travel/blockage")
    return value


def _checkpoint(arm: Any, stop_pressed: Callable[[], bool]) -> None:
    if stop_pressed():
        stop = getattr(arm, "stop", None)
        if callable(stop):
            stop()
        raise JogStopped("Keyboard stop detected; arm stop requested")


def _require_yes(ask: Callable[[str], str], arm: Any, stop_pressed: Callable[[], bool], prompt: str) -> None:
    _checkpoint(arm, stop_pressed)
    answer = ask(prompt).strip().lower()
    _checkpoint(arm, stop_pressed)
    if answer != "y":
        raise CommissioningAborted("Required operator confirmation was declined")


def _pose_samples(
    arm: Any,
    *,
    ask: Callable[[str], str],
    out: Callable[[str], None],
    stop_pressed: Callable[[], bool],
    sleep: Callable[[float], None],
    label: str,
) -> tuple[list[list[float]], dict[str, Any]]:
    samples: list[list[float]] = []
    modes: list[int] = []
    for index in range(STABILITY_SAMPLES):
        _checkpoint(arm, stop_pressed)
        state = arm.read()
        pose = state.get("pose") if isinstance(state, dict) else None
        mode = state.get("mode") if isinstance(state, dict) else None
        if not isinstance(pose, (list, tuple)) or len(pose) < 6 or not all(_finite_number(v) for v in pose[:6]):
            raise CommissioningError(f"{label} stability sample {index + 1} has an invalid pose")
        if mode != 5:
            raise CommissioningError(f"{label} stability sample found RobotMode={mode}; commissioning stopped")
        samples.append([float(value) for value in pose[:6]])
        modes.append(int(mode))
        if index + 1 < STABILITY_SAMPLES:
            sleep(STABILITY_INTERVAL_S)

    xyz_spans = [max(sample[axis] for sample in samples) - min(sample[axis] for sample in samples)
                 for axis in range(3)]
    # Use circular spread so equivalent rotations around +/-180 degrees remain stable.
    reference = samples[0][3:]
    wrapped_rpy_spans = [max(abs(_angle_delta(sample[axis], reference[axis - 3])) for sample in samples)
                         for axis in range(3, 6)]
    max_xyz_span = max(xyz_spans)
    max_rpy_span = max(wrapped_rpy_spans)
    report = {
        "sample_count": len(samples),
        "robot_modes": modes,
        "xyz_span_mm": [round(value, 4) for value in xyz_spans],
        "max_xyz_span_mm": round(max_xyz_span, 4),
        "rpy_span_deg": [round(value, 4) for value in wrapped_rpy_spans],
        "max_rpy_span_deg": round(max_rpy_span, 4),
        "stable": max_xyz_span <= STABILITY_SPAN_MM and max_rpy_span <= STABILITY_SPAN_DEG,
    }
    if not report["stable"]:
        out(f"{label} pose is moving (XYZ span {max_xyz_span:.3f} mm, RPY span {max_rpy_span:.3f} deg).")
        raise CommissioningError(f"{label} pose was not stable; no profile was saved")
    return samples, report


def _check_pose(samples: Sequence[Sequence[float]], expected: Sequence[float], label: str) -> None:
    measured = [statistics.median(sample[axis] for sample in samples) for axis in range(6)]
    xyz_error = max(abs(measured[axis] - float(expected[axis])) for axis in range(3))
    rpy_error = max(abs(_angle_delta(measured[axis], float(expected[axis]))) for axis in range(3, 6))
    if xyz_error > POSE_TOL_MM or rpy_error > POSE_TOL_DEG:
        raise CommissioningError(
            f"{label} pose differs from the accepted pose by {xyz_error:.3f} mm / "
            f"{rpy_error:.3f} deg; no profile was saved"
        )


def _checked_z_move(
    arm: Any,
    jog: Jog,
    target_z: float,
    *,
    hover_z: float,
    floor_z: float,
    fixed_xy: Sequence[float],
    fixed_rpy: Sequence[float],
    ask: Callable[[str], str],
    out: Callable[[str], None],
    stop_pressed: Callable[[], bool],
    label: str,
) -> None:
    target_z = float(target_z)
    if not math.isfinite(target_z) or target_z < floor_z - 1e-6 or target_z > hover_z + 1e-6:
        raise CommissioningError(f"Requested Z={target_z:.2f} mm is outside the supervised range [{floor_z:.2f}, {hover_z:.2f}] mm")
    target = [float(fixed_xy[0]), float(fixed_xy[1]), round(target_z, 4)]
    _checkpoint(arm, stop_pressed)
    try:
        finals = jog.check_route([target])
    except JogRefused:
        raise
    _checkpoint(arm, stop_pressed)
    jog.run_route([target], finals)
    state = arm.read()
    pose = state.get("pose") if isinstance(state, dict) else None
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        raise CommissioningError(f"{label} returned an invalid measured pose")
    xyz_error = max(abs(float(pose[index]) - target[index]) for index in range(3))
    rpy_error = max(abs(_angle_delta(float(pose[index]), float(fixed_rpy[index - 3])))
                    for index in range(3, 6))
    if xyz_error > POSE_TOL_MM or rpy_error > POSE_TOL_DEG:
        arm.stop()
        raise CommissioningError(
            f"{label} did not preserve XY/orientation: {xyz_error:.3f} mm / {rpy_error:.3f} deg; arm stopped"
        )
    out(f"{label}: measured Z={float(pose[2]):.2f} mm; XY and yaw held.")


def _grip_to(
    arm: Any,
    target: int,
    *,
    ask: Callable[[str], str],
    stop_pressed: Callable[[], bool],
    label: str,
) -> dict[str, Any]:
    method = getattr(arm, "grip_to", None)
    if not callable(method):
        raise CommissioningError("Arm adapter must provide cancellable grip_to(position, speed)")
    _checkpoint(arm, stop_pressed)
    feedback = method(int(target), TEACH_SPEED)
    _checkpoint(arm, stop_pressed)
    return feedback


def _stable_neck_grasp(data: Any, profile_id: str) -> bool:
    if not isinstance(data, dict) or data.get("profile_id") != profile_id:
        return False
    neck = data.get("neck_grasp")
    if not isinstance(neck, dict) or neck.get("arm") != EXPECTED_ARM or neck.get("frame") != "base" or neck.get("units") != "mm":
        return False
    if not all(_finite_number(neck.get(key)) for key in ("tcp_z", "hover_z")):
        return False
    grip = neck.get("gripper")
    commissioning = neck.get("commissioning")
    if not isinstance(grip, dict) or not isinstance(commissioning, dict):
        return False
    if grip.get("open") != OPEN_POSITION or commissioning.get("confirmed") is not True or \
            commissioning.get("actual_closed_around_neck") is not True:
        return False
    try:
        target = int(grip["target"])
        free = int(grip["free_closed_final"])
        loaded = int(grip["loaded_closed_final"])
        speed = int(grip["speed"])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        verified_z = float(commissioning["verified_tcp_z_mm"])
        floor_z = float(commissioning["z_soft_floor_mm"])
        z_range = commissioning["z_range_mm"]
        range_min, range_max = float(z_range["min"]), float(z_range["max"])
    except (KeyError, TypeError, ValueError):
        return False
    return (target == NECK_GRIP_TARGET and 1 <= speed <= TEACH_SPEED and
            0 <= free <= 255 and 0 <= loaded <= 255 and loaded - free > 4 and
            abs(verified_z - float(neck["tcp_z"])) <= 1.0 and
            floor_z >= DEFAULT_Z_FLOOR_MM and range_min >= DEFAULT_Z_FLOOR_MM and
            abs(range_min - floor_z) <= 1.0 and abs(range_max - float(neck["hover_z"])) <= 1.0 and
            range_min <= float(neck["tcp_z"]) < range_max - MIN_CONTACT_CLEARANCE_MM and
            isinstance(commissioning.get("profile_identity"), dict) and
            commissioning["profile_identity"].get("profile_id") == profile_id and
            isinstance(commissioning.get("evidence"), str) and bool(commissioning["evidence"].strip()) and
            all(isinstance(commissioning.get(key), dict) and commissioning[key].get("stable") is True
                for key in ("pre_stability", "post_stability")) and
            isinstance(commissioning.get("same_bottle_designation"), str) and
            bool(commissioning["same_bottle_designation"].strip()) and
            all(_finite_number(value) for value in neck.get("rpy", [])) and
            len(neck.get("rpy", [])) == 3 and
            max(abs(_angle_delta(float(neck["rpy"][index]), EXPECTED_RPY[index]))
                for index in range(3)) <= POSE_TOL_DEG)


def is_neck_grasp_commissioned(path: str | Path, profile_id: str) -> bool:
    """Return whether ``path`` contains a complete commissioning for this profile."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    return _stable_neck_grasp(data, profile_id)


def load_neck_grasp(path: str | Path, profile_id: str) -> dict[str, Any] | None:
    """Load a verified neck grasp only when it belongs to the caller's profile."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not _stable_neck_grasp(data, profile_id):
        return None
    return data["neck_grasp"]


def _atomic_save(path: Path, payload: dict[str, Any], profile_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CommissioningError(f"Cannot safely preserve existing commissioning file {path}: {error}") from error
        existing_id = existing.get("profile_id") if isinstance(existing, dict) else None
        if isinstance(existing, dict) and isinstance(existing.get("neck_grasp"), dict):
            if _stable_neck_grasp(existing, str(existing_id or "")):
                raise CommissioningError(
                    f"A valid commissioning already exists for profile {existing_id!r}; refusing to overwrite {path}"
                )
        merged = dict(existing) if isinstance(existing, dict) else {}
    else:
        merged = {}
    if merged.get("profile_id") not in (None, profile_id):
        # An incomplete/stale entry may be safely replaced, but unrelated fields remain intact.
        merged.pop("neck_grasp", None)
    merged.update(payload)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            json.dump(merged, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def commission(
    arm: Any,
    jog: Jog,
    *,
    profile: dict[str, Any],
    profile_path: str | Path,
    same_bottle_designation: str,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    stop_pressed: Callable[[], bool] = lambda: False,
    z_floor_mm: float = DEFAULT_Z_FLOOR_MM,
    accepted_contact_z: float | None = None,
    defer_save: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Commission a neck grasp and return with the bottle supported and jaws closed at contact.

    ``same_bottle_designation`` is an operator-provided short label for the same
    empty bottle used during this single supervised session. ``profile_path`` is
    the dedicated commissioned profile path, not the hardware profile path.
    ``accepted_contact_z`` replays a prior operator-approved depth from the same
    gate. The caller remains responsible for checking that gate before passing it.
    ``defer_save=True`` returns a validated candidate without writing it; the
    launcher should persist it only after the candidate passes the verified-lift check.
    """
    profile_id = profile.get("profile_id") if isinstance(profile, dict) else None
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise CommissioningError("Hardware profile has no stable profile_id")
    bottle_label = str(same_bottle_designation).strip()
    if not bottle_label:
        raise CommissioningError("A same-bottle designation is required")
    if type(defer_save) is not bool:
        raise CommissioningError("defer_save must be a bool")
    if accepted_contact_z is not None and not defer_save:
        raise CommissioningError(
            "Replayed contact-Z candidates must defer persistence until the verified-lift check passes"
        )
    destination = Path(profile_path)
    if is_neck_grasp_commissioned(destination, profile_id):
        raise CommissioningError(f"Commissioning is already valid for profile {profile_id}; no motion was sent")
    if not _finite_number(z_floor_mm):
        raise CommissioningError("The supervised Z floor must be a finite number")

    arm_name = getattr(arm, "arm", EXPECTED_ARM)
    if arm_name != EXPECTED_ARM:
        raise CommissioningError(f"Neck commissioning currently requires the left arm, got {arm_name!r}")
    if not hasattr(jog, "xyz") or not hasattr(jog, "rpy"):
        raise CommissioningError("Commissioning must start from the accepted hover in an initialized Jog")
    start_target = [float(value) for value in jog.xyz]
    start_rpy = [float(value) for value in jog.rpy]
    if len(start_target) != 3 or len(start_rpy) != 3 or not all(_finite_number(value) for value in start_target + start_rpy):
        raise CommissioningError("Accepted hover pose is invalid")
    if max(abs(_angle_delta(start_rpy[index], EXPECTED_RPY[index])) for index in range(3)) > POSE_TOL_DEG:
        raise CommissioningError(f"Accepted hover orientation must remain {list(EXPECTED_RPY)} degrees")
    hover_z = start_target[2]
    if abs(hover_z - EXPECTED_HOVER_Z_MM) > POSE_TOL_MM:
        raise CommissioningError(
            f"Commissioning requires the verified open-air hover near Z={EXPECTED_HOVER_Z_MM:.1f} mm; "
            "low-start resume is refused before any gripper or arm movement"
        )
    floor_z = max(DEFAULT_Z_FLOOR_MM, float(z_floor_mm),
                  float(getattr(getattr(jog, "limits", None), "z_min", z_floor_mm)))
    if not floor_z <= start_target[2] <= hover_z + 0.5:
        raise CommissioningError("Start Z outside the supervised range")
    if floor_z >= hover_z:
        raise CommissioningError(f"Supervised Z floor {floor_z:.2f} is not below accepted hover {hover_z:.2f}")
    if hover_z - floor_z > 200.0:
        raise CommissioningError("Supervised Z range exceeds the 200 mm commissioning envelope")
    if accepted_contact_z is not None:
        if not _finite_number(accepted_contact_z):
            raise CommissioningError("Prior operator-approved contact Z must be finite")
        accepted_contact_z = float(accepted_contact_z)
        if not floor_z <= accepted_contact_z < hover_z - MIN_CONTACT_CLEARANCE_MM:
            raise CommissioningError("Prior contact Z is outside the guarded commissioning range")

    grip_position_method = getattr(arm, "gripper_position", None)
    set_speed_method = getattr(arm, "set_linear_speed", None)
    if not callable(grip_position_method):
        raise CommissioningError("Arm adapter must expose gripper_position()")
    if not callable(set_speed_method):
        raise CommissioningError("Arm adapter must expose set_linear_speed(percent) for slow supervised motion")
    original_speed = getattr(arm, "speed", None)
    if not _finite_number(original_speed):
        original_speed = 8

    if accepted_contact_z is None:
        _require_yes(
            ask, arm, stop_pressed,
            "確認使用同一支空瓶，瓶身直立且底部穩放在工作台；夾爪已張開，手已離開作業區；" 
            "急停鍵在手邊。輸入 y 開始：",
        )
    start_samples, pre_stability = _pose_samples(arm, ask=ask, out=out, stop_pressed=stop_pressed,
                                                  sleep=sleep, label="Pre-commission")
    _check_pose(start_samples, [*start_target, *start_rpy], "Accepted hover")
    fixed_xy = start_target[:2]
    contact_z: float | None = None
    contact_stability: dict[str, Any] | None = None
    speed_changed = False
    try:
        set_speed_method(5)
        speed_changed = True
        current_grip = int(grip_position_method())
        if not 250 <= current_grip <= 255:
            raise CommissioningError(
                f"Gripper is at {current_grip}; it must be open at 250-255 at the accepted hover. "
                "No bottle motion was attempted."
            )
        if current_grip < OPEN_POSITION:
            open_feedback = _grip_to(arm, OPEN_POSITION, ask=ask, stop_pressed=stop_pressed,
                                     label="Open at accepted hover")
            if _feedback_final(open_feedback, "Open at accepted hover") < 250:
                raise CommissioningError("Gripper did not reach the open range; stop and inspect manually")

        # Measure the free reference at the verified open-air hover before any Z motion.
        out("先在已核實的空中懸停位執行目標 0，取得穩定空夾爪讀值，然後重新張開。")
        free_feedback = _grip_to(arm, NECK_GRIP_TARGET, ask=ask, stop_pressed=stop_pressed,
                                 label="Free open-air baseline")
        try:
            free_final = _feedback_final(free_feedback, "Free open-air baseline", NECK_GRIP_TARGET)
        except CommissioningError:
            # At the clear hover there is no supported load. Restore the jaws
            # open before refusing an unstable free reference.
            try:
                opening = _grip_to(arm, OPEN_POSITION, ask=ask, stop_pressed=stop_pressed,
                                   label="Reopen after invalid free baseline")
                _feedback_final(opening, "Reopen after invalid free baseline", OPEN_POSITION)
            except Exception:
                arm.stop()
            raise
        opening = _grip_to(arm, OPEN_POSITION, ask=ask, stop_pressed=stop_pressed,
                           label="Reopen after free baseline")
        if _feedback_final(opening, "Reopen after free baseline", OPEN_POSITION) < 250:
            raise CommissioningError("Gripper did not reopen at hover; no Z movement was attempted")
        free_evidence = {"target": NECK_GRIP_TARGET, "final": free_final,
                         "samples": [int(value) for value in free_feedback["samples"][-5:]],
                         "measured_at": "verified_open_air_hover_before_Z"}

        manual_depth = accepted_contact_z is None
        initial_accepted_z = accepted_contact_z
        depth_source = ("operator_guided_incremental" if manual_depth
                        else "prior_operator_approved_contact_z_replayed")
        retune_count = 0
        step_z = 5.0
        contact_z = None
        contact_stability = None
        post_stability = None
        loaded_feedback = None
        loaded_final = None
        while True:
            if manual_depth:
                while True:
                    _checkpoint(arm, stop_pressed)
                    command = ask(f"Z={jog.xyz[2]:.2f} mm，步長={step_z:g} mm [1/5/0,d,u,c,q]：").strip().lower()
                    _checkpoint(arm, stop_pressed)
                    if command == "q":
                        raise CommissioningAborted("Operator aborted during Z commissioning")
                    if command in Z_STEPS_MM:
                        step_z = Z_STEPS_MM[command]
                        continue
                    if command in ("d", "u"):
                        delta = step_z if command == "u" else -step_z
                        target_z = round(float(jog.xyz[2]) + delta, 4)
                        if not floor_z <= target_z <= hover_z:
                            out(f"可用 Z 範圍 {floor_z:.1f}–{hover_z:.1f} mm，請調整步長或方向。")
                            continue
                        _checked_z_move(arm, jog, target_z, hover_z=hover_z, floor_z=floor_z,
                                        fixed_xy=fixed_xy, fixed_rpy=start_rpy, ask=ask, out=out,
                                        stop_pressed=stop_pressed, label="Commissioning Z jog")
                        continue
                    if command == "c":
                        break
                    out("請選 1/5/0 設定步長，再用 d/u 移動；c 只確認 Z，q 取消。")
            else:
                target_z = float(accepted_contact_z)
                out(f"沿通過同一 gate 核驗的垂直路徑重用先前接受的 Z={target_z:.2f} mm。")
                while float(jog.xyz[2]) > target_z + 1e-4:
                    _checkpoint(arm, stop_pressed)
                    next_z = max(target_z, float(jog.xyz[2]) - 10.0)
                    _checked_z_move(arm, jog, next_z, hover_z=hover_z, floor_z=floor_z,
                                    fixed_xy=fixed_xy, fixed_rpy=start_rpy, ask=ask, out=out,
                                    stop_pressed=stop_pressed, label="Accepted contact Z replay")

            contact_samples, contact_stability = _pose_samples(
                arm, ask=ask, out=out, stop_pressed=stop_pressed, sleep=sleep,
                label="Neck contact candidate")
            _check_pose(contact_samples, [fixed_xy[0], fixed_xy[1], float(jog.xyz[2]), *start_rpy],
                        "Neck contact candidate")
            contact_z = statistics.median(sample[2] for sample in contact_samples)
            if manual_depth:
                close_prompt = (
                    "目視確認兩片張開的夾墊已對準同一空瓶的瓶頸外側，瓶子直立且底部穩放。"
                    "輸入 y 後以低速 5 執行夾爪目標 0；q 取消。"
                )
            else:
                close_prompt = (
                    f"此 Z={contact_z:.2f} mm 重用同一 gate 先前接受的接觸高度，不代表本次重新目視教導。"
                    "確認仍是同一支空瓶且瓶底穩放後，輸入 y 以低速 5 執行目標 0；q 取消。"
                )
            if manual_depth:
                _require_yes(ask, arm, stop_pressed, close_prompt)
            loaded_feedback = _grip_to(
                arm, NECK_GRIP_TARGET, ask=ask, stop_pressed=stop_pressed,
                label="Low-speed full-close on supported bottle")
            loaded_final = _feedback_final(loaded_feedback, "Loaded neck contact", NECK_GRIP_TARGET)
            pose_samples, post_stability = _pose_samples(
                arm, ask=ask, out=out, stop_pressed=stop_pressed, sleep=sleep,
                label="Loaded contact pose")
            _check_pose(pose_samples, [fixed_xy[0], fixed_xy[1], contact_z, *start_rpy],
                        "Loaded contact pose")

            if not manual_depth:
                break
            while True:
                _checkpoint(arm, stop_pressed)
                response = ask(
                    f"目標 0 的穩定瓶頸讀值={loaded_final} counts。瓶子仍由工作台支撐。"
                    "按 o 在支撐狀態下張開並以 1 mm 微調；按 y 接受並保持閉合在接觸位；按 q 取消並保持原狀："
                ).strip().lower()
                _checkpoint(arm, stop_pressed)
                if response in ("o", "y", "q"):
                    break
                out("請按 o 張開微調、y 接受並保持閉合，或 q 取消。")
            if response == "y":
                break
            if response == "q":
                raise CommissioningAborted("Operator cancelled after loaded close; gripper remains closed at contact")
            opening = _grip_to(arm, OPEN_POSITION, ask=ask, stop_pressed=stop_pressed,
                               label="Retune open while bottle is supported")
            if _feedback_final(opening, "Retune open", OPEN_POSITION) < 250:
                raise CommissioningError("Gripper did not reopen at the supported bottle; Z was not moved")
            manual_depth = True
            depth_source = ("operator_guided_retune_after_prior_contact_z"
                            if depth_source == "prior_operator_approved_contact_z_replayed" else "operator_guided_incremental")
            retune_count += 1
            step_z = 1.0
            out("夾爪已在支撐狀態下張開。請用 1 mm 步長調整 Z；接受後會保持閉合於接觸位。")

        assert contact_z is not None and contact_stability is not None
        assert loaded_feedback is not None and loaded_final is not None and post_stability is not None
        if loaded_final - free_final <= 4:
            out("瓶頸接觸與空中空夾爪讀值相差不超過 4 counts；沒有候選。夾爪保持閉合於接觸位，請勿抬瓶。")
            raise CommissioningError(
                f"Same-command feedback is not distinguishable: free={free_final}, loaded={loaded_final}; "
                "retune at the supported station or abort"
            )
        measured_contact_z = float(contact_z)
        if not floor_z <= measured_contact_z < hover_z - MIN_CONTACT_CLEARANCE_MM:
            raise CommissioningError(
                f"Measured TCP Z must be within the supervised range and at least {MIN_CONTACT_CLEARANCE_MM:.0f} mm "
                "below hover for the cycle; no profile was saved"
            )
        profile_digest = hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        commissioning = {
            "confirmed": True,
            "actual_closed_around_neck": True,
            "evidence": (
                "The caller supplied a prior operator-approved contact Z after same-gate checks. This invocation "
                "replayed that depth and recorded a stable loaded-versus-free encoder difference while the same empty "
                "bottle remained supported. No new operator confirmation or visual contact claim was collected; "
                "the candidate must pass the verified-lift check before persistence."
                if depth_source == "prior_operator_approved_contact_z_replayed" else
                "Operator guided Z, visually confirmed the open pads at the outside of the same empty supported bottle "
                "neck, and accepted the stable loaded close; the free baseline was measured at the verified hover first."
            ),
            "profile_identity": {"profile_id": profile_id, "sha256": profile_digest},
            "same_bottle_designation": bottle_label,
            "verified_tcp_z_mm": round(measured_contact_z, 3),
            "contact_depth_source": depth_source,
            "accepted_prior_contact_z_mm": initial_accepted_z,
            "loaded_closure_confirmation": (
                "same_gate_authorized_stable_feedback_candidate" if not manual_depth else
                "operator_accepted_after_stable_feedback"
            ),
            "contact_visual_confirmation": manual_depth,
            "fresh_operator_confirmation": manual_depth,
            "retune_count": retune_count,
            "z_range_mm": {"min": round(floor_z, 3), "max": round(hover_z, 3)},
            "commanded_width": {"servo_position_counts": NECK_GRIP_TARGET, "units": "gripper position counts; 0 closed, 255 open"},
            "observed_width": {"loaded_servo_position_counts": loaded_final,
                               "free_servo_position_counts": free_final,
                               "units": "measured servo position counts"},
            "observed_width_counts": loaded_final,
            "z_soft_floor_mm": round(floor_z, 3),
            "pre_stability": pre_stability,
            "contact_stability": contact_stability,
            "post_stability": post_stability,
            "free_baseline": free_evidence,
            "loaded_feedback": {"target": NECK_GRIP_TARGET, "final": loaded_final,
                                "samples": [int(value) for value in loaded_feedback["samples"][-5:]]},
            "jaw_adjustments": [{"commanded_position": NECK_GRIP_TARGET, "speed": TEACH_SPEED,
                                  "loaded_observed_final": loaded_final,
                                  "free_observed_final": free_final}],
            "unattended_verification": {
                "mode": "same_command_at_verified_open_air_hover_before_Z",
                "hover_z_mm": round(hover_z, 3),
                "commanded_position_counts": NECK_GRIP_TARGET,
                "free_closed_final": free_final,
                "loaded_closed_final": loaded_final,
                "distinct_feedback": True,
            },
            "commissioned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        neck_grasp = {
            "arm": EXPECTED_ARM,
            "frame": "base",
            "units": "mm",
            "tcp_z": round(measured_contact_z, 3),
            "hover_z": round(hover_z, 3),
            "rpy": list(EXPECTED_RPY),
            "gripper": {
                "open": OPEN_POSITION,
                "target": NECK_GRIP_TARGET,
                "speed": TEACH_SPEED,
                "free_closed_final": free_final,
                "loaded_closed_final": loaded_final,
            },
            "commissioning": commissioning,
        }
        payload = {"schema_version": PROFILE_SCHEMA_VERSION,
                   "profile_id": profile_id,
                   "neck_grasp": neck_grasp}
        if not _stable_neck_grasp(payload, profile_id):
            raise CommissioningError("Generated neck profile did not pass its own validity checks")
        if not defer_save:
            _atomic_save(destination, payload, profile_id)
            out(f"已儲存教導資料：{destination}。夾爪保持閉合於接觸位。")
        else:
            out("候選資料尚未持久化；夾爪保持閉合於接觸位，先完成核驗提起後再儲存。")
        return payload
    finally:
        if speed_changed:
            try:
                set_speed_method(int(original_speed))
            except Exception as error:  # leave the safe arm pose alone; caller reports adapter errors
                out(f"Could not restore the previous arm speed setting ({type(error).__name__}: {error}).")

