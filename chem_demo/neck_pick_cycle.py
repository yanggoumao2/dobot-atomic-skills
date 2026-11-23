"""Single-start, no-home cycle for grasping the commissioned bottle neck.

All perception and live-device setup are injected by the launcher. This module
only owns the checked Cartesian sequence and its no-release-on-uncertainty rule.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from chem_demo.stations import near_photo, photo_pose
from task_skills.run_cartesian_jog import (
    Jog,
    JogStopped,
    Limits,
    _rotation_angle_deg,
)


ARM = "left"
HOME_XY = (117.6, -344.8)
HOME_Z = 415.5
HOVER_Z = 485.0
TARGET_X = (220.0, 305.0)
TARGET_Y = (-425.0, -350.0)
MIN_TCP_Z = 350.0
MAX_START_XY_ERROR_MM = 5.0
MAX_START_Z_ERROR_MM = 5.0
MAX_START_ORIENTATION_ERROR_DEG = 3.0
START_RPY = (180.0, 0.0, -90.0)
PHOTO_XYZ, PHOTO_RPY = photo_pose(ARM)
YAW_ALIGN_STEP_DEG = 10.0
START_ROUTE_RADIUS_MM = 400.0
VERIFY_LIFT_MM = 15.0
GRIPPER_CODE_TOLERANCE = 2


MAX_HOVER_ADJUSTS = 3
HOVER_ADJUST_DONE_MM = 1.5    # rim within this of the jaw-centred target: descend
HOVER_ADJUST_STEP_MM = 8.0    # largest single correction at the hover height
HOVER_ADJUST_TOTAL_MM = 15.0  # never wander further than this from the top-camera target


class NeckPickRefused(RuntimeError):
    """The single-start cycle could not prove its preconditions."""


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise NeckPickRefused(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise NeckPickRefused(f"{label} must be a finite number")
    return number


def _vector(value: Any, length: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise NeckPickRefused(f"{label} must contain {length} numbers")
    try:
        result = [_finite_number(item, label) for item in value]
    except TypeError as error:
        raise NeckPickRefused(f"{label} must contain {length} numbers") from error
    if len(result) != length:
        raise NeckPickRefused(f"{label} must contain {length} numbers")
    return result


def validate_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated commissioned parameters without touching hardware."""
    if not isinstance(profile, Mapping):
        raise NeckPickRefused("profile must be an object")
    neck = profile.get("neck_grasp")
    if not isinstance(neck, Mapping):
        raise NeckPickRefused("profile has no commissioned neck_grasp section")
    if neck.get("arm") != ARM or neck.get("frame") != "base" or neck.get("units") != "mm":
        raise NeckPickRefused("neck_grasp must be commissioned for the left arm base frame in mm")
    hover_z = _finite_number(neck.get("hover_z"), "neck_grasp.hover_z")
    if abs(hover_z - HOVER_Z) > 0.5:
        raise NeckPickRefused(f"neck_grasp.hover_z must be {HOVER_Z:.1f} mm")
    rpy = _vector(neck.get("rpy"), 3, "neck_grasp.rpy")
    if _rotation_angle_deg(rpy, START_RPY) > MAX_START_ORIENTATION_ERROR_DEG:
        raise NeckPickRefused("neck_grasp.rpy must match the commissioned left wrist orientation near [180, 0, -90]")

    tcp_z = _finite_number(neck.get("tcp_z"), "neck_grasp.tcp_z")
    if tcp_z < MIN_TCP_Z:
        raise NeckPickRefused(f"neck_grasp.tcp_z must be at least {MIN_TCP_Z:.1f} mm")
    if tcp_z >= hover_z - 15.0:
        raise NeckPickRefused("neck_grasp.tcp_z must be at least 15 mm below the hover height")

    gripper = neck.get("gripper")
    commissioning = neck.get("commissioning")
    if not isinstance(gripper, Mapping) or not isinstance(commissioning, Mapping):
        raise NeckPickRefused("neck_grasp needs taught gripper and commissioning evidence")
    if commissioning.get("confirmed") is not True or commissioning.get("actual_closed_around_neck") is not True:
        raise NeckPickRefused("neck grasp depth is not confirmed as actually closed around this bottle neck")
    if not commissioning.get("evidence"):
        raise NeckPickRefused("neck grasp commissioning evidence is missing")
    commissioned_z = _finite_number(commissioning.get("verified_tcp_z_mm"),
                                    "neck_grasp.commissioning.verified_tcp_z_mm")
    if abs(commissioned_z - tcp_z) > 1.0:
        raise NeckPickRefused("neck_grasp.tcp_z does not match the measured commissioning pose")
    soft_floor_z = _finite_number(commissioning.get("z_soft_floor_mm"),
                                  "neck_grasp.commissioning.z_soft_floor_mm")
    z_range = commissioning.get("z_range_mm")
    if not isinstance(z_range, Mapping):
        raise NeckPickRefused("neck grasp commissioning z_range_mm is missing")
    range_floor = _finite_number(z_range.get("min"), "neck_grasp.commissioning.z_range_mm.min")
    range_hover = _finite_number(z_range.get("max"), "neck_grasp.commissioning.z_range_mm.max")
    if soft_floor_z < MIN_TCP_Z or range_floor < MIN_TCP_Z or abs(range_hover - hover_z) > 0.5:
        raise NeckPickRefused("neck grasp commissioned Z range does not match the guarded workspace")
    if tcp_z < max(MIN_TCP_Z, soft_floor_z, range_floor) or tcp_z >= range_hover - 15.0:
        raise NeckPickRefused("neck_grasp.tcp_z is outside its commissioned safe Z range")

    open_position = gripper.get("open")
    target = gripper.get("target")
    speed = gripper.get("speed")
    free_code = gripper.get("free_closed_final")
    loaded_code = gripper.get("loaded_closed_final")
    for name, value in (("open", open_position), ("target", target),
                        ("free_closed_final", free_code), ("loaded_closed_final", loaded_code)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise NeckPickRefused(f"neck_grasp.gripper.{name} must be an integer from 0 to 255")
    if open_position != 255 or target != 0:
        raise NeckPickRefused("neck_grasp must use the taught 255-open position and full-close target 0")
    if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= 10:
        raise NeckPickRefused("neck_grasp.gripper.speed must be a taught low speed from 1 to 10")
    if loaded_code <= free_code or loaded_code - free_code <= 2 * GRIPPER_CODE_TOLERANCE:
        raise NeckPickRefused("loaded feedback must exceed empty feedback by more than four counts")

    return {
        "arm": ARM,
        "tcp_z": tcp_z,
        "hover_z": hover_z,
        "rpy": rpy,
        "open": open_position,
        "close_target": target,
        "close_speed": speed,
        "free_closed_final": free_code,
        "loaded_closed_final": loaded_code,
        "commissioning_evidence": str(commissioning["evidence"]),
        "z_soft_floor_mm": soft_floor_z,
        "z_range_mm": {"min": range_floor, "max": range_hover},
    }


def validate_start(state: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> str:
    """Accept the platform photo pose, the calibrated home or its high hover in the left clear area.

    The photo pose is the platform start. It may carry the commissioned wrist yaw or the
    +90 degree yaw recorded in poses.json; in the second case the cycle rotates the wrist
    in place first (see ``needs_yaw_alignment``).
    """
    if not isinstance(state, Mapping):
        raise NeckPickRefused("arm state is missing")
    pose = _vector(state.get("pose"), 6, "arm pose")
    if state.get("mode") not in (4, 5):
        raise NeckPickRefused(f"arm must be disabled or enabled-idle at start; RobotMode={state.get('mode')}")
    rpy = config["rpy"] if config is not None else list(START_RPY)
    if near_photo(ARM, pose, xy_tol=MAX_START_XY_ERROR_MM, z_tol=MAX_START_Z_ERROR_MM):
        if (_rotation_angle_deg(pose[3:], rpy) > MAX_START_ORIENTATION_ERROR_DEG and
                _rotation_angle_deg(pose[3:], PHOTO_RPY) > MAX_START_ORIENTATION_ERROR_DEG):
            raise NeckPickRefused("photo-pose wrist orientation is neither the commissioned yaw nor the recorded photo yaw")
        return "photo"
    if _rotation_angle_deg(pose[3:], rpy) > MAX_START_ORIENTATION_ERROR_DEG:
        raise NeckPickRefused("start wrist orientation is not the commissioned fixed neck-grasp orientation")
    xy_error = math.dist(pose[:2], HOME_XY)
    if abs(pose[2] - HOME_Z) <= MAX_START_Z_ERROR_MM:
        if xy_error > MAX_START_XY_ERROR_MM:
            raise NeckPickRefused("home start TCP is outside the left clear home area")
        return "home"
    if abs(pose[2] - HOVER_Z) <= MAX_START_Z_ERROR_MM:
        if xy_error > MAX_START_XY_ERROR_MM and not (
                TARGET_X[0] <= pose[0] <= TARGET_X[1] and TARGET_Y[0] <= pose[1] <= TARGET_Y[1]):
            raise NeckPickRefused("high-hover TCP is outside the left clear start and target areas")
        return "high_hover"
    raise NeckPickRefused(f"start Z must be near home {HOME_Z:.1f} or high hover {HOVER_Z:.1f} mm")


def preflight(profile: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the commissioned profile and accepted start pose before enabling/moving."""
    config = validate_profile(profile)
    config["start_kind"] = validate_start(state, config)
    return config


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return repr(value)


class NeckPickCycle:
    """Run one automatic left-arm bottle-neck grasp and leave the gripper closed."""

    def __init__(self, arm: Any, profile: Mapping[str, Any], *,
                 localize: Callable[[str, Sequence[float]], Mapping[str, Any]],
                 check_wrist: Callable[[Sequence[float], Mapping[str, Any]], Mapping[str, Any]],
                 read_wrist: Callable[[], Any],
                 verify_held: Callable[[Any, Any], Mapping[str, Any]],
                 stop_pressed: Callable[[], bool] = lambda: False,
                 out: Callable[[str], None] = print,
                 log_path: Path | None = None, lift_mm: int = VERIFY_LIFT_MM):
        if lift_mm not in (VERIFY_LIFT_MM, 30):
            raise NeckPickRefused("Select a 15 or 30 mm vertical lift")
        self.lift_mm = lift_mm
        self.arm = arm
        self.profile = profile
        self.localize = localize
        self.check_wrist = check_wrist
        self.read_wrist = read_wrist
        self.verify_held = verify_held
        self.stop_pressed = stop_pressed
        self.out = out
        self.log_path = Path(log_path) if log_path is not None else None
        self.config: dict[str, Any] | None = None
        self.jog: Jog | None = None
        self.moved = False
        self.log: dict[str, Any] = {
            "schema_version": "neck-pick-cycle/1",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "arm": ARM,
            "target_bounds_mm": {"x": list(TARGET_X), "y": list(TARGET_Y)},
            "steps": [],
        }

    def _step(self, name: str, **fields: Any) -> None:
        entry = {"step": name, "time": datetime.now().isoformat(timespec="milliseconds")}
        entry.update(_jsonable(fields))
        self.log["steps"].append(entry)
        self.out(f"neck-pick: {name}")

    def _save_log(self) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.log_path.with_suffix(self.log_path.suffix + ".tmp")
        temporary.write_text(json.dumps(_jsonable(self.log), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.log_path)

    def _checkpoint(self) -> None:
        if self.stop_pressed():
            self.arm.stop()
            raise JogStopped("stop requested; neck-pick cycle halted")

    def _validate_adapter(self) -> None:
        missing = [name for name in ("read", "plan", "forward", "move", "wait", "stop",
                                     "gripper_position", "grip_to") if not callable(getattr(self.arm, name, None))]
        if missing:
            raise NeckPickRefused("arm adapter is missing required checked-motion methods: " + ", ".join(missing))
        if getattr(self.arm, "arm", ARM) != ARM:
            raise NeckPickRefused("neck-pick cycle is restricted to the left arm")

    def _target_from_localization(self, result: Mapping[str, Any]) -> list[float]:
        if not isinstance(result, Mapping):
            raise NeckPickRefused("localizer did not return a result object")
        if "ok" in result and result["ok"] is not True:
            raise NeckPickRefused(f"bottle localization failed: {result.get('reason', result.get('error', 'not ok'))}")
        if result.get("empty") is False or result.get("known_empty") is False:
            raise NeckPickRefused("localized bottle is not confirmed empty")
        xy = _vector(result.get("xy"), 2, "localized bottle xy")
        if not TARGET_X[0] <= xy[0] <= TARGET_X[1] or not TARGET_Y[0] <= xy[1] <= TARGET_Y[1]:
            raise NeckPickRefused(f"localized bottle xy {xy} is outside the commissioned left platform patch")
        return xy

    def _make_jog(self, state: Mapping[str, Any]) -> Jog:
        assert self.config is not None
        jog = Jog(
            self.arm,
            Limits(state["pose"][:3], START_ROUTE_RADIUS_MM, MIN_TCP_Z,
                   max(HOVER_Z, state["pose"][2]) + 15.0),
            record_path=Path("unused-neck-pick.csv"),
            ask=lambda _prompt: "",
            out=self.out,
            stop_pressed=self.stop_pressed,
        )
        jog.rpy = list(self.config["rpy"])
        state = self.arm.read()
        validate_start(state, self.config)
        if state.get("mode") != 5:
            raise NeckPickRefused("left arm left enabled-idle mode before motion")
        return jog

    def _align_yaw(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Rotate the wrist in place to the commissioned yaw in small, IK-checked steps.

        Needed when the arm waits at the photo pose with the +90 degree yaw recorded in
        poses.json. Both rotation directions are checked first; the first that keeps every
        step under the joint-jump limit is executed. Nothing moves if neither passes.
        """
        assert self.jog is not None and self.config is not None
        pose = _vector(state.get("pose"), 6, "arm pose")
        target = list(self.config["rpy"])
        xyz = pose[:3]
        delta = math.remainder(target[2] - pose[5], 360.0)
        failures: list[str] = []
        for total in (delta, delta - math.copysign(360.0, delta)):
            sign = math.copysign(1.0, total)
            count = max(1, math.ceil(abs(total) / YAW_ALIGN_STEP_DEG - 1e-9))
            plan: list[tuple[list[float], list[float]]] = []
            seed = list(self.jog.joints)
            try:
                for k in range(1, count + 1):
                    yaw = math.remainder(pose[5] + total * k / count, 360.0)
                    rpy = [pose[3], pose[4], yaw] if k < count else target
                    seed = self.jog.check(xyz, rpy, seed)
                    plan.append((rpy, list(seed)))
            except Exception as refusal:  # noqa: BLE001 - try the other direction, then refuse
                failures.append(f"direction {sign:+.0f}: {refusal}")
                continue
            self._checkpoint()
            self._step("align_yaw", direction=sign, steps=count, from_yaw=pose[5], to_yaw=target[2])
            self.moved = True
            for rpy, joints in plan:
                self.arm.move(list(xyz), list(rpy), list(joints))
                self.arm.wait(list(xyz), list(rpy), list(joints), self.stop_pressed)
                # A pure rotation has no position error, so wait() can return before the wrist has turned.
                deadline = time.monotonic() + 20.0
                while _rotation_angle_deg(_vector(self.arm.read()["pose"], 6, "yaw step pose")[3:], rpy) > 1.0:
                    self._checkpoint()
                    if time.monotonic() > deadline:
                        raise NeckPickRefused("wrist yaw step did not complete within 20 s")
                    time.sleep(0.1)
                self._checkpoint()
            self.jog.rpy, self.jog.joints, self.jog.xyz = list(target), list(plan[-1][1]), list(xyz)
            settled = self.arm.read()
            measured = _vector(settled.get("pose"), 6, "yaw-aligned pose")
            if (settled.get("mode") != 5 or max(abs(measured[i] - xyz[i]) for i in range(3)) > 0.75 or
                    _rotation_angle_deg(measured[3:], target) > 1.0):
                raise NeckPickRefused("wrist yaw alignment did not settle at the commissioned orientation")
            self._step("align_yaw_settled", measured=measured)
            return settled
        raise NeckPickRefused("wrist yaw alignment is not reachable in small checked steps: " + "; ".join(failures))

    def _move_route(self, name: str, legs: Sequence[Sequence[float]]) -> None:
        assert self.jog is not None
        route = [[_finite_number(v, f"{name} target") for v in leg] for leg in legs]
        if any(len(leg) != 3 for leg in route):
            raise NeckPickRefused(f"{name} route contains an invalid waypoint")
        filtered: list[list[float]] = []
        previous = list(self.jog.xyz)
        for leg in route:
            if math.dist(previous, leg) > 1e-6:
                filtered.append(leg)
            previous = leg
        route = filtered
        if not route:
            self._step(name, moved=False)
            return
        finals = self.jog.check_route(route)
        self._checkpoint()
        self._step(name, planned=route, checked_legs=len(route))
        self.moved = True
        self.jog.run_route(route, finals)
        state = self.arm.read()
        pose = _vector(state.get("pose"), 6, f"{name} measured pose")
        target = route[-1]
        position_error = max(abs(pose[index] - target[index]) for index in range(3))
        orientation_error = _rotation_angle_deg(pose[3:], self.config["rpy"])
        if state.get("mode") != 5 or position_error > 0.75 or orientation_error > 1.0:
            raise NeckPickRefused(f"{name} did not settle at the checked target ({position_error:.2f} mm, "
                                  f"{orientation_error:.2f} deg, mode {state.get('mode')})")
        self._step(name + "_settled", measured=pose, position_error_mm=position_error,
                   orientation_error_deg=orientation_error)

    def _verify_lift(self, contact_pose: Sequence[float]) -> dict[str, Any]:
        """Capture at closed contact, lift vertically, then verify against the bench."""
        assert self.config is not None
        contact = _vector(contact_pose, 6, "closed contact pose")
        self.log["verification_stage"] = "before_lift_capture"
        state = self.arm.read()
        current = _vector(state.get("pose"), 6, "current closed-contact pose")
        if state.get("mode") != 5 or any(abs(current[index] - contact[index]) > 1.0 for index in range(3)):
            raise NeckPickRefused("arm moved or left enabled-idle mode before the closed-grasp lift")
        if _rotation_angle_deg(current[3:], self.config["rpy"]) > 1.0:
            raise NeckPickRefused("wrist attitude changed before the closed-grasp lift")
        final = self.arm.gripper_position()
        if (abs(final - self.config["loaded_closed_final"]) > GRIPPER_CODE_TOLERANCE or
                abs(final - self.config["free_closed_final"]) <= GRIPPER_CODE_TOLERANCE):
            raise NeckPickRefused("current gripper feedback no longer matches the taught loaded-close code")
        if self.jog is None:
            raise NeckPickRefused("checked Cartesian motion was not initialized")
        self.jog.xyz = current[:3]
        self.jog.joints = _vector(state.get("joints"), 6, "current arm joints")
        # The wrist camera only gives evidence here; losing it must not stop the arm with the bottle
        # clamped on the table (2026-11-23 16:19: the left D405 dropped off USB right after the
        # close). Without both frames the verdict is "uncertain" and the caller falls back to the
        # jaw reading after the lift, as for any ambiguous image check.
        before = self._read_wrist_or_none("before_lift")
        self._step("wrist_before_lift_captured", close_feedback=final, camera_ok=before is not None)
        lift = [current[0], current[1], current[2] + self.lift_mm]
        if lift[2] >= self.config["hover_z"]:
            raise NeckPickRefused("verification lift would reach the hover height")
        self.log["verification_stage"] = "lifting"
        self._move_route("short_verification_lift", [lift])
        self.log["verification_stage"] = "after_lift"
        after = self._read_wrist_or_none("after_lift") if before is not None else None
        self._step("wrist_after_lift_captured", lift_mm=self.lift_mm, camera_ok=after is not None)
        self._checkpoint()
        self.log["verification_stage"] = "held_verifier"
        if before is None or after is None:
            verification = {"held": None, "reason": "left wrist camera unavailable: "
                            + "; ".join(self.log.get("wrist_camera_errors", []))}
        else:
            verification = self.verify_held(before, after)
        held = verification.get("held") if isinstance(verification, Mapping) else None
        if held not in (True, False, None):
            held = None
        result = {
            "success": held is True,
            "held": held,
            "status": "held_and_verified" if held is True else ("not_held" if held is False else "hold_uncertain"),
            "gripper_final": final,
            "verification": _jsonable(verification),
            "gripper_released": False,
        }
        self.log["held_verification"] = _jsonable(verification)
        self.log["result"] = result
        self.log["verification_stage"] = "complete"
        self._step("held_verification_complete", result=result)
        return self.log

    def _read_wrist_or_none(self, stage: str) -> Any:
        """One wrist capture; a camera/USB error (RuntimeError, OSError) is logged and gives None."""
        try:
            return self.read_wrist()
        except (RuntimeError, OSError) as error:
            message = f"{stage}: {type(error).__name__}: {error}"
            self.log.setdefault("wrist_camera_errors", []).append(message)
            self.out(f"左腕相机取帧失败（{message}），抬起后改用夹爪读数判断")
            return None

    def verify_first_closed(self, contact_pose: Sequence[float]) -> dict[str, Any]:
        """Verify an already-closed commissioned grasp with no localization or jaw command.

        ``contact_pose`` is the six-value TCP pose recorded by the caller at closed
        contact. The current state must remain at that pose, at the commissioned
        orientation and Z, with the encoder at the taught loaded code.
        """
        try:
            if self.log["steps"] or self.moved:
                raise NeckPickRefused("verify_first_closed requires a fresh cycle instance")
            self.config = validate_profile(self.profile)
            self._validate_adapter()
            contact = _vector(contact_pose, 6, "closed contact pose")
            state = self.arm.read()
            current = _vector(state.get("pose"), 6, "current arm pose")
            if state.get("mode") != 5:
                raise NeckPickRefused("closed-contact verification requires enabled-idle mode")
            if _rotation_angle_deg(contact[3:], self.config["rpy"]) > 1.0 or \
                    _rotation_angle_deg(current[3:], self.config["rpy"]) > 1.0:
                raise NeckPickRefused("closed-contact attitude differs from the commissioned wrist orientation")
            if _rotation_angle_deg(contact[3:], current[3:]) > 1.0:
                raise NeckPickRefused("wrist attitude changed since the caller recorded closed contact")
            if abs(contact[2] - self.config["tcp_z"]) > 1.0:
                raise NeckPickRefused("caller contact Z differs from the commissioned grasp TCP Z")
            deltas = [current[index] - contact[index] for index in range(3)]
            if any(abs(delta) > 1.0 for delta in deltas):
                raise NeckPickRefused(f"arm moved since closed contact; signed XYZ deltas are {deltas}")
            for pose in (contact, current):
                if not (TARGET_X[0] <= pose[0] <= TARGET_X[1] and TARGET_Y[0] <= pose[1] <= TARGET_Y[1]):
                    raise NeckPickRefused("closed-contact XY is outside the commissioned left platform patch")
            if abs(current[2] - self.config["tcp_z"]) > 1.0:
                raise NeckPickRefused("current TCP Z differs from the commissioned grasp pose")
            self._checkpoint()
            feedback = self.arm.gripper_position()
            if abs(feedback - self.config["loaded_closed_final"]) > GRIPPER_CODE_TOLERANCE:
                raise NeckPickRefused("current gripper feedback does not match the taught loaded-close code")
            if abs(feedback - self.config["free_closed_final"]) <= GRIPPER_CODE_TOLERANCE:
                raise NeckPickRefused("current gripper feedback matches the taught empty-close code")

            self.log["operation"] = "verify_first_closed"
            self.log["profile"] = {key: value for key, value in self.config.items()
                                   if key not in {"commissioning_evidence"}}
            self.log["closed_contact"] = {"caller_pose": contact, "current_pose": current,
                                          "signed_xyz_delta_mm": deltas,
                                          "loaded_gripper_position": feedback,
                                          "measured_rpy": current[3:]}
            self._step("closed_contact_validated", pose=current, signed_xyz_delta_mm=deltas,
                       gripper_position=feedback)
            self.jog = Jog(
                self.arm,
                Limits(current[:3], 1.0, self.config["z_soft_floor_mm"], self.config["hover_z"] + 15.0),
                record_path=Path("unused-neck-verify.csv"),
                ask=lambda _prompt: "",
                out=self.out,
                stop_pressed=self.stop_pressed,
            )
            self.jog.rpy = list(current[3:])
            return self._verify_lift(current)
        except Exception as error:
            self.log["failure"] = {"type": type(error).__name__, "message": str(error)}
            self.log["result"] = {"success": False, "held": None,
                                  "status": ("verification_failed_after_lift" if self.log.get("verification_stage") in
                                             {"after_lift", "held_verifier"} else
                                             "verification_lift_interrupted" if self.log.get("verification_stage") == "lifting" else
                                             "stopped_with_gripper_state_preserved"),
                                  "gripper_released": False, "reason": str(error)}
            if self.moved:
                try:
                    self.arm.stop()
                except Exception as stop_error:
                    self.log["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
            raise
        finally:
            exception_active = sys.exc_info()[0] is not None
            try:
                self._save_log()
            except Exception as log_error:
                self.log["log_write_error"] = f"{type(log_error).__name__}: {log_error}"
                if not exception_active:
                    raise

    def run(self) -> dict[str, Any]:
        """Execute once. Raises on unsafe preconditions; outcome keeps the gripper closed."""
        try:
            self.config = validate_profile(self.profile)
            if self.config["tcp_z"] + self.lift_mm >= self.config["hover_z"]:
                raise NeckPickRefused("requested lift would reach the hover height")
            self._validate_adapter()
            initial = self.arm.read()
            start_kind = validate_start(initial, self.config)
            if initial.get("mode") != 5:
                raise NeckPickRefused("live launcher must enable the arm without homing before running the cycle")
            if self.arm.gripper_position() < self.config["open"] - 5:
                raise NeckPickRefused("gripper must already be fully open and empty at the accepted start")
            self.log["profile"] = {key: value for key, value in self.config.items()
                                   if key not in {"commissioning_evidence"}}
            self.log["start"] = {"kind": start_kind, "state": _jsonable(initial)}
            self._step("preflight_passed", start_kind=start_kind, gripper_position=self.arm.gripper_position())
            self._checkpoint()

            localized = self.localize("start", initial["pose"])
            target_xy = self._target_from_localization(localized)
            self.log["localization"] = _jsonable(localized)
            self._step("localized", xy=target_xy, source=localized.get("source"))

            self.jog = self._make_jog(initial)
            if _rotation_angle_deg(initial["pose"][3:], self.config["rpy"]) > MAX_START_ORIENTATION_ERROR_DEG:
                initial = self._align_yaw(initial)
            start_xyz = list(self.jog.xyz)
            hover_legs = []
            # From the photo pose the straight leg to the hover point already descends gently; crossing at
            # photo height first is unreachable over the bottle (wrist too high for the left arm's reach).
            if start_kind != "photo" and abs(start_xyz[2] - self.config["hover_z"]) > 0.5:
                hover_legs.append([start_xyz[0], start_xyz[1], self.config["hover_z"]])
            hover_legs.append([target_xy[0], target_xy[1], self.config["hover_z"]])
            self._move_route("translate_at_hover", hover_legs)

            hover_state = self.arm.read()
            hover_pose = _vector(hover_state.get("pose"), 6, "hover pose")
            wrist_target = {"xy": target_xy, "z": self.config["hover_z"]}
            self._checkpoint()
            wrist = self.check_wrist(hover_pose, wrist_target)
            # Close-range fine alignment: move by the measured rim-vs-jaw offset and look again, instead
            # of refusing because the view is not identical to the reference (Kunyu, 2026-11-22).
            previous = None
            for adjust in range(MAX_HOVER_ADJUSTS):
                correction = wrist.get("correction_xy_mm") if isinstance(wrist, Mapping) else None
                if correction is None:
                    break
                size = math.hypot(*correction)
                if size <= HOVER_ADJUST_DONE_MM:
                    break
                if previous is not None and size > 0.8 * previous:
                    raise NeckPickRefused(f"wrist fine alignment is not converging ({previous:.1f} -> {size:.1f} mm)")
                step = min(1.0, HOVER_ADJUST_STEP_MM / size)
                new_xy = [hover_pose[0] + step * correction[0], hover_pose[1] + step * correction[1]]
                if math.dist(new_xy, target_xy) > HOVER_ADJUST_TOTAL_MM:
                    raise NeckPickRefused(f"wrist fine alignment would move {math.dist(new_xy, target_xy):.0f} mm "
                                          "from the top-camera target")
                self.log.setdefault("hover_adjusts", []).append(_jsonable({"evidence": wrist, "move_xy": new_xy}))
                self._move_route(f"hover_adjust_{adjust + 1}", [[new_xy[0], new_xy[1], self.config["hover_z"]]])
                hover_pose = _vector(self.arm.read().get("pose"), 6, "hover pose")
                previous = size
                self._checkpoint()
                wrist = self.check_wrist(hover_pose, wrist_target)
            if not isinstance(wrist, Mapping) or wrist.get("ok") is not True:
                reason = wrist.get("reason", wrist.get("error", "wrist check did not pass")) if isinstance(wrist, Mapping) else "wrist check did not return an object"
                raise NeckPickRefused(f"wrist alignment gate failed: {reason}")
            self.log["wrist_check"] = _jsonable(wrist)
            self._step("wrist_check_passed", evidence=wrist)

            assert self.jog is not None
            vertical_xy = [hover_pose[0], hover_pose[1]]
            descend = [vertical_xy[0], vertical_xy[1], self.config["tcp_z"]]
            self._move_route("descend_to_commissioned_neck_tcp", [descend])

            self._checkpoint()
            grip_feedback = self.arm.grip_to(self.config["close_target"], self.config["close_speed"])
            if not isinstance(grip_feedback, Mapping):
                raise NeckPickRefused("gripper adapter returned no closure feedback")
            try:
                start = int(grip_feedback["start"])
                commanded = int(grip_feedback["target"])
                final = int(grip_feedback["final"])
            except (KeyError, TypeError, ValueError) as error:
                raise NeckPickRefused("gripper closure feedback lacks valid start, target, or final position") from error
            try:
                samples = [int(value) for value in grip_feedback["samples"]]
            except (KeyError, TypeError, ValueError) as error:
                raise NeckPickRefused("gripper closure feedback lacks stable position samples") from error
            if (not 0 <= start <= 255 or start < self.config["open"] - 5 or commanded != 0 or
                    not 0 <= final <= 255 or len(samples) < 5 or
                    any(not 0 <= value <= 255 for value in samples)):
                raise NeckPickRefused("gripper closure feedback is outside the full-close contract")
            if any(abs(value - final) > 3 for value in samples[-5:]):
                raise NeckPickRefused("gripper feedback did not stabilize at the final position")
            if abs(final - start) < 10 and abs(final - commanded) > 3:
                raise NeckPickRefused("gripper closure feedback shows neither meaningful travel nor target arrival")
            self.log["gripper_close"] = _jsonable(grip_feedback)
            self._step("gripper_closed", feedback=grip_feedback)

            if abs(final - self.config["free_closed_final"]) <= GRIPPER_CODE_TOLERANCE:
                result = {"success": False, "held": False, "status": "empty_close", "gripper_released": False,
                          "reason": "feedback matches the taught empty-gripper code; left closed at grasp TCP"}
                self.log["result"] = result
                self._step("empty_close_detected", final=final)
                return self.log
            if abs(final - self.config["loaded_closed_final"]) > GRIPPER_CODE_TOLERANCE:
                result = {"success": False, "held": None, "status": "unknown_close", "gripper_released": False,
                          "reason": "feedback matches neither taught empty nor loaded code; left closed at grasp TCP"}
                self.log["result"] = result
                self._step("unknown_close_detected", final=final)
                return self.log

            closed_state = self.arm.read()
            contact = _vector(closed_state.get("pose"), 6, "closed contact pose")
            if (closed_state.get("mode") != 5 or abs(contact[2] - self.config["tcp_z"]) > 0.75 or
                    _rotation_angle_deg(contact[3:], self.config["rpy"]) > 1.0):
                raise NeckPickRefused("arm state changed after closing at the commissioned neck pose")
            return self._verify_lift(contact)
        except Exception as error:
            self.log["failure"] = {"type": type(error).__name__, "message": str(error)}
            self.log["result"] = {"success": False, "held": None,
                                  "status": ("verification_failed_after_lift" if self.log.get("verification_stage") in
                                             {"after_lift", "held_verifier"} else
                                             "verification_lift_interrupted" if self.log.get("verification_stage") == "lifting" else
                                             "stopped_with_gripper_state_preserved"),
                                  "gripper_released": False, "reason": str(error)}
            if self.moved:
                try:
                    self.arm.stop()
                except Exception as stop_error:  # preserve the original motion failure
                    self.log["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
            raise
        finally:
            exception_active = sys.exc_info()[0] is not None
            try:
                self._save_log()
            except Exception as log_error:
                self.log["log_write_error"] = f"{type(log_error).__name__}: {log_error}"
                if not exception_active:
                    raise

