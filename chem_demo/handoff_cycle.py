"""Left-to-right bottle handoff: left lifts, right takes the body from the side.

Nothing here knows a coordinate. The handoff point and the right-hand pose come from
``plan_fn`` (a reachability search over the live left pose), the final alignment
from ``observe_fn`` (right wrist camera, closed loop), and the transfer is checked
by ``transfer_fn`` before the left hand lets go. Hardware, planner and perception
are injected so the order of operations and every refusal can be tested offline.

Release rules (the part that must not be bent):
  * the right gripper closes first; the left opens only after the right close
    feedback is inside the expected "holding the body" window;
  * after the left opens, the right retreats a few mm and ``transfer_fn`` must say
    the bottle moved with it; anything else re-closes the left gripper on the neck;
  * a stop, a refusal or an exception never opens a gripper.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from chem_demo.handoff_plan import _rot
from chem_demo.neck_pick_cycle import _jsonable, _vector
from task_skills.run_cartesian_jog import Jog, JogStopped, Limits, _rotation_angle_deg

ALIGN_TOL_MM = 3.0
MAX_SERVO_ITERATIONS = 6
MAX_CORRECTION_MM = 12.0
APPROACH_STEP_MM = 12.0
# Smooth approach (Kunyu 2026-11-23: "一段一段动的，改成平滑移动"): from the aligned standoff one
# continuous MovL to FINAL_LOOK_MM short of the axis, one look there (depth is still reliable at
# ~4 cm; at 1-2 cm the glass returns background depth), then one continuous MovL onto the body.
FINAL_LOOK_MM = 35.0
MAX_APPROACH_STEPS = 30
RANGE_SANITY_MM = 60.0       # measured pad-to-bottle range vs the planned standoff
# First look: the held bottle's grasp point vs the pads, vertically. The plan already puts the pads at
# the predicted grasp height, so only the flange->tips estimate (+-20 mm) and hang errors remain; a
# larger value means the mask is another vessel (the beaker on the balance below, 2026-11-22 21:35).
VERTICAL_SANITY_MM = 50.0
OVERSHOOT_REFUSE_MM = -6.0   # bottle already past the pad midpoint
# A single bad frame (2026-11-23 16:07: the mask merged another object) must not end the handoff:
# an approach reading that is off-axis, not ok, or whose range did not shrink by about the last step
# is re-taken at the same pose (no motion) this many times before the guards act on it.
OBSERVATION_RETRIES = 2
RANGE_JUMP_MM = 25.0
# Near the bottle the allowed range jump shrinks (17:13: at ~3 mm to go the depth through the glass
# read 18 mm, the hand stepped 12 mm more and ended 9 mm past the axis). Below DEAD_RECKON_BELOW_MM a
# reading that still disagrees after the retakes is replaced by the last trusted range minus the
# distance actually commanded (arm repeatability << 1 mm), with no sideways correction.
RANGE_JUMP_MIN_MM = 8.0
RANGE_JUMP_SHARE = 0.3
DEAD_RECKON_BELOW_MM = 40.0
# During the final approach the wrist reading is noisy (~+-5 mm per frame at 10-15 cm, 2026-11-23
# 16:48). Correcting the full reading every 12 mm step made it oscillate (-5, +5, -6.5 mm) until the
# old 6 mm drift limit fired. Now: half-gain correction with a deadband, and the drift limits are what
# the open jaws can still take (92 mm opening on a 68 mm body: ~12 mm each side; the pads are tall).
APPROACH_GAIN = 0.5
APPROACH_LATERAL_LIMIT_MM = 10.0
APPROACH_VERTICAL_LIMIT_MM = 15.0
RETREAT_CHECK_MM = 15.0
ARRIVAL_POSITION_TOL_MM = 1.0
ARRIVAL_ORIENTATION_TOL_DEG = 1.0
GRIPPER_CODE_TOLERANCE = 2


def segment_ends(waypoints: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Last sample of every run of equal ``step_kind`` (the start pose excluded).

    The planner samples each straight segment every 10 mm / 10 deg (position linear, orientation
    slerp over the same fraction, which is what MovL runs), so only segment ends are sent.
    """
    moving = [w for w in waypoints if w.get("step_kind") != "start"]
    return [w for i, w in enumerate(moving) if i == len(moving) - 1 or moving[i + 1].get("step_kind") != w.get("step_kind")]


def move_timeout_s(start: Sequence[float], end: Sequence[float]) -> float:
    """Arrival timeout for one MovL: 30 s plus 0.5 s per mm and 0.3 s per degree (slow test speeds)."""
    distance = math.dist(list(start[:3]), list(end[:3]))
    return 30.0 + 0.5 * distance + 0.3 * _rotation_angle_deg(list(start[3:]), list(end[3:]))


class HandoffRefused(RuntimeError):
    """A precondition, a measurement or a safety gate failed; grippers stay as they are."""


class HandoffCycle:
    def __init__(self, left: Any, right: Any, *, left_loaded_code: int,
                 plan_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
                 refine_fn: Callable[[Mapping[str, Any], Sequence[float]], Mapping[str, Any]],
                 observe_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                 transfer_fn: Callable[[Any, Any], Mapping[str, Any]],
                 capture_fn: Callable[[], Any] = lambda: None,
                 verify_hold_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
                 stop_after_hang: bool = False,
                 right_hold_window: tuple[int, int] = (30, 240),
                 left_open: int = 255, right_open: int = 255,
                 close_speed: int = 5, release_speed: int = 5,
                 stop_pressed: Callable[[], bool] = lambda: False,
                 out: Callable[[str], None] = print, log_path: Path | None = None):
        self.left, self.right = left, right
        self.left_loaded_code = int(left_loaded_code)
        self.plan_fn, self.refine_fn = plan_fn, refine_fn
        self.observe_fn, self.transfer_fn, self.capture_fn = observe_fn, transfer_fn, capture_fn
        self.verify_hold_fn = verify_hold_fn
        self.stop_after_hang = stop_after_hang
        self.right_hold_window = right_hold_window
        self.left_open, self.right_open = left_open, right_open
        self.close_speed, self.release_speed = close_speed, release_speed
        self.stop_pressed, self.out = stop_pressed, out
        self.log_path = Path(log_path) if log_path is not None else None
        self.right_jaws_closed = False
        self.left_jaws_open = False
        self.log: dict[str, Any] = {"schema_version": "handoff-cycle/1",
                                    "started_at": datetime.now().isoformat(timespec="seconds"),
                                    "steps": []}

    # ------------------------------------------------------------------ helpers
    def _step(self, name: str, **fields: Any) -> None:
        entry = {"step": name, "time": datetime.now().isoformat(timespec="milliseconds")}
        entry.update(_jsonable(fields))
        self.log["steps"].append(entry)
        self.out(f"handoff: {name}")

    def _save(self) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.log_path.with_suffix(self.log_path.suffix + ".tmp")
        temporary.write_text(json.dumps(_jsonable(self.log), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.log_path)

    def _checkpoint(self) -> None:
        if self.stop_pressed():
            self.left.stop()
            self.right.stop()
            raise JogStopped("stop requested; handoff halted with both grippers unchanged")

    def _left_holds(self) -> int:
        code = self.left.gripper_position()
        if abs(code - self.left_loaded_code) > GRIPPER_CODE_TOLERANCE:
            raise HandoffRefused(f"left gripper reads {code}, not the loaded-neck code {self.left_loaded_code}")
        return code

    def _enabled_idle(self, arm: Any, label: str) -> dict[str, Any]:
        state = arm.read()
        if state.get("mode") != 5:
            raise HandoffRefused(f"{label} arm is not enabled-idle (RobotMode={state.get('mode')})")
        return state

    def _right_goto(self, name: str, waypoint: Mapping[str, Any]) -> dict[str, Any]:
        xyz = _vector(waypoint["xyz"], 3, f"{name} xyz")
        rpy = _vector(waypoint["rpy"], 3, f"{name} rpy")
        joints = _vector(waypoint["joints"], 6, f"{name} joints")
        self._checkpoint()
        here = _vector(self.right.read()["pose"], 6, f"{name} start pose")
        timeout = move_timeout_s(here, [*xyz, *rpy])
        if waypoint.get("step_kind") == "joint_approach":
            timeout += 60.0  # joint-space move: Cartesian distance understates it
            self.right.move_joint(xyz, rpy, joints)
        else:
            self.right.move(xyz, rpy, joints)
        self.right.wait(xyz, rpy, joints, self.stop_pressed, timeout_s=timeout)
        state = self._enabled_idle(self.right, "right")
        pose = _vector(state["pose"], 6, f"{name} measured pose")
        position_error = max(abs(pose[i] - xyz[i]) for i in range(3))
        orientation_error = _rotation_angle_deg(pose[3:], rpy)
        if position_error > ARRIVAL_POSITION_TOL_MM or orientation_error > ARRIVAL_ORIENTATION_TOL_DEG:
            raise HandoffRefused(f"right arm did not settle at {name} ({position_error:.2f} mm, "
                                 f"{orientation_error:.2f} deg)")
        self._step(name, target=xyz, rpy=rpy, measured=pose)
        return state

    # --------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        try:
            return self._run()
        except BaseException as error:
            self.log["failure"] = {"type": type(error).__name__, "message": str(error)}
            self.log["result"] = {"success": False, "status": "stopped_grippers_unchanged",
                                  "right_jaws_closed": self.right_jaws_closed,
                                  "left_jaws_open": self.left_jaws_open, "reason": str(error)}
            for arm in (self.left, self.right):
                try:
                    arm.stop()
                except Exception as stop_error:  # noqa: BLE001 - keep the original failure
                    self.log.setdefault("stop_errors", []).append(f"{type(stop_error).__name__}: {stop_error}")
            raise
        finally:
            try:
                self._save()
            except Exception as log_error:  # noqa: BLE001
                self.log["log_write_error"] = str(log_error)
                if sys.exc_info()[0] is None:
                    raise

    def _run(self) -> dict[str, Any]:
        left_state = self._enabled_idle(self.left, "left")
        right_state = self._enabled_idle(self.right, "right")
        left_code = self._left_holds()
        if self.right.gripper_position() < self.right_open - 5:
            raise HandoffRefused("right gripper must start fully open")
        self._step("preflight_passed", left_gripper=left_code, left_pose=left_state["pose"],
                   right_pose=right_state["pose"])

        plan = dict(self.plan_fn(left_state, right_state))
        self.log["plan"] = _jsonable(plan)
        self._step("plan_ready", handoff_left_tcp=plan.get("handoff_left_tcp"),
                   candidate=plan.get("candidate"), rejected=plan.get("rejected_summary"))
        left_legs = [_vector(leg, 3, "left leg") for leg in plan["left_legs"]]
        waypoints = list(plan["right_waypoints"])
        standoff_name = (plan.get("standoff_pose") or {}).get("name", "standoff")
        standoff_index = next((i for i, w in enumerate(waypoints) if w.get("name") == standoff_name), None)
        if standoff_index is None:
            raise HandoffRefused("plan has no standoff waypoint")

        # 1. Left lifts the bottle to the planned handoff point (vertical first, then across).
        self._checkpoint()
        left_pose = _vector(left_state["pose"], 6, "left pose")
        z_top = max([left_pose[2]] + [leg[2] for leg in left_legs]) + 15.0
        jog = Jog(self.left, Limits(left_pose[:3], 450.0, 350.0, z_top), record_path=Path("unused-handoff-left.csv"),
                  ask=lambda _p: "", out=self.out, stop_pressed=self.stop_pressed)
        jog.rpy = list(left_pose[3:])
        finals = jog.check_route(left_legs)
        self._step("left_route_checked", legs=left_legs)
        jog.run_route(left_legs, finals)
        self._left_holds()
        left_at = self.left.read()
        self._step("left_at_handoff", pose=left_at["pose"], gripper=self.left.gripper_position())

        # 1b. Top camera measures where the bottle really hangs before the right arm moves toward it.
        if self.verify_hold_fn is not None:
            self._checkpoint()
            hang = dict(self.verify_hold_fn(left_at))
            self.log["hang_check"] = _jsonable(hang)
            if hang.get("ok") is not True:
                raise HandoffRefused(f"top camera does not confirm the bottle hang: {hang.get('reason')}")
            self._step("hang_confirmed", xy_offset_mm=hang.get("xy_offset_mm"),
                       bottom_offset_mm=hang.get("bottom_offset_mm"))
        if self.stop_after_hang:
            # Photo-only rehearsal: the left holds the bottle at the station, the right never moved.
            self.log["result"] = {"success": True, "status": "stopped_after_hang_check",
                                  "right_jaws_closed": False, "left_jaws_open": False}
            return self.log

        # 2. Right arm goes to the standoff pose in ONE continuous MovL. The planner checked IK, joint
        # jumps and clearance every 10 mm / 10 deg along exactly the line the controller runs
        # (position linear, orientation slerp over the same fraction), so the samples need not be
        # sent one by one; stopping at each of them made the motion jerky.
        for end in segment_ends(waypoints[:standoff_index + 1]):
            self._right_goto(str(end.get("name", "right_step")), end)

        # 3. Closed-loop alignment from the right wrist camera, then advance in short steps.
        current = dict(waypoints[standoff_index])
        standoff_mm = (plan.get("candidate") or {}).get("standoff_mm")
        for iteration in range(MAX_SERVO_ITERATIONS):
            self._checkpoint()
            observation = dict(self.observe_fn(self.right.read()))
            self.log.setdefault("observations", []).append(_jsonable(observation))
            if observation.get("ok") is not True:
                raise HandoffRefused(f"right wrist alignment unavailable: {observation.get('reason', 'not ok')}")
            vertical, lateral, rng = _vector(observation["error_tool_mm"], 3, "alignment error")
            if iteration == 0 and standoff_mm is not None and abs(rng - standoff_mm) > RANGE_SANITY_MM:
                raise HandoffRefused(
                    f"measured pad-to-bottle range {rng:.0f} mm differs from the planned standoff {standoff_mm:.0f} mm "
                    "by more than the sanity limit (right tool-to-pad offset or the plan is wrong); nothing closed")
            if iteration == 0 and abs(vertical) > VERTICAL_SANITY_MM:
                raise HandoffRefused(
                    f"measured grasp height differs from the plan by {vertical:.0f} mm (limit {VERTICAL_SANITY_MM:.0f}); "
                    "the wrist camera may be looking at another vessel; nothing closed")
            if max(abs(lateral), abs(vertical)) <= ALIGN_TOL_MM:
                break
            # Tool0 axes: X = vertical (along the bottle axis), Y = lateral (jaw axis), Z = range.
            delta = [max(-MAX_CORRECTION_MM, min(MAX_CORRECTION_MM, vertical)),
                     max(-MAX_CORRECTION_MM, min(MAX_CORRECTION_MM, lateral)), 0.0]
            current = dict(self.refine_fn(current, delta))
            self._right_goto(f"align_{iteration + 1}", current)
        else:
            raise HandoffRefused("right wrist alignment did not converge")
        self._step("aligned", error_tool_mm=[vertical, lateral, rng])

        safe_to_close = False
        expected_rng = rng
        for advance in range(MAX_APPROACH_STEPS):
            for attempt in range(OBSERVATION_RETRIES + 1):
                self._checkpoint()
                observation = dict(self.observe_fn(self.right.read()))
                suspect = None
                if observation.get("ok") is not True:
                    suspect = f"not ok: {observation.get('reason', 'not ok')}"
                else:
                    vertical, lateral, rng = _vector(observation["error_tool_mm"], 3, "alignment error")
                    if abs(lateral) > APPROACH_LATERAL_LIMIT_MM or abs(vertical) > APPROACH_VERTICAL_LIMIT_MM:
                        suspect = f"off axis ({vertical:.0f}, {lateral:.0f}) mm"
                    elif abs(rng - expected_rng) > max(RANGE_JUMP_MIN_MM,
                                                       min(RANGE_JUMP_MM, RANGE_JUMP_SHARE * abs(expected_rng))):
                        suspect = f"range {rng:.0f} mm, expected ~{expected_rng:.0f} mm"
                if suspect is None or attempt == OBSERVATION_RETRIES:
                    break
                self.log.setdefault("observation_retries", []).append(
                    {"advance": advance, "attempt": attempt + 1, "why": suspect})
                self.out(f"右腕这一帧可疑（{suspect}），原地重拍")
            dead_reckoned = False
            if (suspect is not None and suspect.startswith("range") and advance > 0
                    and expected_rng <= DEAD_RECKON_BELOW_MM):
                self.log.setdefault("dead_reckoning", []).append(
                    {"advance": advance, "measured_range_mm": rng, "used_range_mm": expected_rng})
                self.out(f"近距离测距与实际走过的距离不符（{suspect}），改用上次可信读数推算 {expected_rng:.1f} mm")
                vertical, lateral, rng = 0.0, 0.0, expected_rng
                dead_reckoned = True
            if observation.get("ok") is not True:
                raise HandoffRefused(f"right wrist range unavailable: {observation.get('reason', 'not ok')}")
            if not dead_reckoned:
                vertical, lateral, rng = _vector(observation["error_tool_mm"], 3, "alignment error")
            if abs(lateral) > APPROACH_LATERAL_LIMIT_MM or abs(vertical) > APPROACH_VERTICAL_LIMIT_MM:
                raise HandoffRefused("right gripper drifted off the bottle axis during the approach")
            if rng < OVERSHOOT_REFUSE_MM:
                raise HandoffRefused(f"right pads are {-rng:.1f} mm past the bottle axis; refusing to close")
            if abs(rng) <= ALIGN_TOL_MM:
                safe_to_close = observation.get("safe_to_close", True) is True
                break
            step = rng - FINAL_LOOK_MM if rng > FINAL_LOOK_MM + APPROACH_STEP_MM else rng
            expected_rng = rng - step
            fix = [0.0 if abs(v) <= ALIGN_TOL_MM else APPROACH_GAIN * v for v in (vertical, lateral)]
            current = dict(self.refine_fn(current, [fix[0], fix[1], step]))
            self._right_goto(f"approach_{advance + 1}", current)
        else:
            raise HandoffRefused("right gripper did not reach the bottle body in the allowed steps")
        if not safe_to_close:
            raise HandoffRefused(f"close gates failed at the final observation: {observation.get('reason')}")
        contact = dict(current)

        # 4. Right closes on the body; only a plausible hold lets the left let go.
        self._checkpoint()
        self.right_jaws_closed = True  # set before the command: a timed-out close may still be partly closed
        feedback = self.right.grip_to(0, self.close_speed)
        final = int(feedback["final"])
        low, high = self.right_hold_window
        self._step("right_closed", feedback=feedback)
        if not low <= final <= high or abs(final - int(feedback["start"])) < 10:
            raise HandoffRefused(f"right close feedback {final} is outside the body-hold window {self.right_hold_window}")

        before = self.capture_fn()
        self._checkpoint()
        self.left.grip_to(self.left_open, self.release_speed)
        left_code = self.left.gripper_position()
        if left_code < self.left_open - 5:
            raise HandoffRefused(f"left gripper only reads {left_code} after the open command; not retreating")
        self.left_jaws_open = True
        self._step("left_released", left_gripper=left_code)

        # 5. Move the bottle straight down a little with the right hand (neck slides inside the open left
        #    jaws) and prove it came along: a pure image translation is easy to tell from a static bottle.
        # World-vertical down, expressed in the tool frame (the tool may be pitched down to keep the
        # right forearm high, so tool X is no longer vertical).
        down_tool = (_rot(contact["rpy"]).T @ [0.0, 0.0, -RETREAT_CHECK_MM]).tolist()
        shifted = dict(self.refine_fn(contact, down_tool))
        self._right_goto("right_down_check", shifted)
        after = self.capture_fn()
        verdict = dict(self.transfer_fn(before, after))
        self.log["transfer_verdict"] = _jsonable(verdict)
        held = verdict.get("held")
        if held is None:
            # Vision undecided: the right jaw is the physical evidence. The bottle hangs from it alone
            # (left open) and was just moved 15 mm; a slipping body would let the servo, which keeps
            # pushing toward 0, run on past the close reading.
            now = int(self.right.gripper_position())
            self._step("right_jaw_after_down_check", position=now, at_close=final)
            if abs(now - final) <= GRIPPER_CODE_TOLERANCE + 1 and low <= now <= high:
                verdict = {**verdict, "held": True, "source": "right jaw reading unchanged after the down check",
                           "right_jaw": now}
                self.log["transfer_verdict"] = _jsonable(verdict)
                held = True
        if held is True:
            self._step("transfer_confirmed", verdict=verdict)
            self.log["result"] = {"success": True, "status": "handed_over_to_right",
                                  "right_jaws_closed": True, "left_jaws_open": True}
            return self.log
        if held is None:
            # Unknown: do not close the left on a bottle the right may be clamping. Stop and hand over to the operator.
            self._step("transfer_uncertain", verdict=verdict)
            self.log["result"] = {"success": False, "status": "transfer_uncertain_both_unchanged",
                                  "right_jaws_closed": True, "left_jaws_open": True,
                                  "reason": "vision could not tell whether the bottle moved with the right hand"}
            return self.log
        return self._recover(contact, verdict)

    def _recover(self, contact: Mapping[str, Any], verdict: Mapping[str, Any]) -> dict[str, Any]:
        """The right hand did not demonstrably hold the bottle: go back and re-close the left."""
        self._step("transfer_not_confirmed", verdict=verdict)
        self._right_goto("right_back_to_contact", contact)
        self.left.grip_to(0, self.close_speed)
        self.left_jaws_open = False
        self._left_holds()
        self._step("left_regrasped")
        # The left hand holds the neck again, so the right hand can safely let go.
        self.right.grip_to(self.right_open, self.release_speed)
        self.right_jaws_closed = False
        self._step("right_released_after_recovery")
        self.log["result"] = {"success": False, "status": "handoff_aborted_left_regrasped",
                              "right_jaws_closed": False, "left_jaws_open": False,
                              "reason": f"transfer verdict {verdict.get('held')!r}"}
        return self.log
