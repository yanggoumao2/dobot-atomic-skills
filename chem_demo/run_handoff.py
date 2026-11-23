"""Left picks the blue-rimmed bottle by the neck, lifts it, and the right hand takes it by the body.

Start pose of both arms is the platform photo pose. Nothing is taught: the bottle comes from
the top camera, the handoff point and the right-hand pose from a reachability search over the
live left pose, the final alignment from the right wrist camera.

  --preflight   read-only: localize the bottle, predict the lifted left pose and check the handoff
                with controller IK queries. No arm is enabled, nothing moves.
  --live        pick, check the plan again from the real lifted pose, hand over, retreat left.
  --reset       put a bottle still held by the left jaw back where it was lifted, open the jaw,
                and bring both arms to the photo pose.

The handoff happens at one stored station (poses.json "handoff_station"), found once by the full
reachability search (``--preflight --search``). Each run only re-checks that station from the live
poses, which takes seconds instead of a minute.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from chem_demo.handoff_cycle import RETREAT_CHECK_MM, HandoffCycle, HandoffRefused, move_timeout_s
from chem_demo.handoff_plan import HandoffInfeasible
from chem_demo.bottle_observation import read_passive_robot_state
from chem_demo.handoff_plan import _rot
from chem_demo.handoff_runtime import (MAX_TOTAL_REFINE_MM, STANDOFF_MM, handoff_config, make_plan_fn,
                                         make_refine_fn)
from chem_demo.neck_lift import verify_neck_lift
from chem_demo.neck_pick_cycle import NeckPickCycle, preflight
from chem_demo.run_neck_pick import (DATA, Cameras, NeckArm, _check_idle, _checked_target, _load_profile,
                                     _stop_key, make_read_wrist)
from chem_demo.neck_pick_cycle import GRIPPER_CODE_TOLERANCE
from chem_demo.stations import handoff_station, near_photo, photo_pose, save_handoff_station
from task_skills.run_cartesian_jog import Jog, Limits
from task_skills.chemistry.driver import LiveDriver

HANDOFF_DATA = DATA.parent / "handoff"
RIGHT_SPEED_PERCENT = 12
HOVER_Z_LIMIT = 500.0   # highest left TCP from which a held bottle is put straight back down
HOVER_Z_RESET = 485.0   # commissioned neck hover height
LEFT_SPEED_PERCENT = 12


class SharedArm(NeckArm):
    """NeckArm that borrows one LiveDriver, so both arms share the machine-wide device lease.

    ``close`` only disables this arm; the runtime closes the driver once, after both arms.
    """

    def __init__(self, profile, arm, speed, driver, log=print):
        self.profile, self.arm, self.speed, self.log = profile, arm, speed, log
        self.session_open = False
        self.driver = driver
        self.rig = driver._rig(arm, profile, with_motion=False)

    def close(self):
        if self.session_open:
            try:
                self.log(self.disable())
            except Exception as error:  # noqa: BLE001 - the other arm and the lease must still be released
                self.log(f"{self.arm} 下使能失败，请在示教器上手动下使能：{type(error).__name__}: {error}")


# JointMovJ speed ratio. The controller multiplies it by the global SpeedFactor (12 %): at SpeedJ 12 %
# the right arm's long joint move to the standoff ran at ~1.4 % and took 98 s (2026-11-23 17:12)
# while the left's short straight moves took 14 s. The joint path is checked sample by sample by the
# planner (open fingers >= ~10 cm from the bottle), so it may run faster; linear moves stay at 12 %.
JOINT_SPEED_PERCENT = 50


def _joint_speed(arm):
    """open_session(home=False) sets only the linear speed; JointMovJ uses SpeedJ/AccJ, which were
    left at the controller's previous (slow) values: the right joint move took ~110 s on 2026-11-22."""
    arm.driver._configure_speed(arm.rig, arm.speed, linear=False)
    arm.driver._configure_speed(arm.rig, arm.speed, linear=True)
    # Raise only the joint-move ratios; SpeedFactor (global) and SpeedL/AccL keep arm.speed.
    from task_skills.chemistry.driver import _reply_status
    for name in ("SpeedJ", "AccJ"):
        if _reply_status(getattr(arm.rig.dashboard, name)(JOINT_SPEED_PERCENT)) != 0:
            raise RuntimeError(f"{name} on {arm.arm} was not acknowledged with status 0")


def _idle_states(profile):
    """Both arms idle and disabled at the platform start. ``_check_idle`` validates and returns the left."""
    left = _check_idle(profile)
    right = read_passive_robot_state(profile["arms"]["right"]["ip"])
    return {"left": left, "right": right}


def _verify_both_disabled(profile):
    for name, binding in profile["arms"].items():
        state = read_passive_robot_state(binding["ip"])
        if state["mode"] != 4 or state["enabled"]:
            raise RuntimeError(f"未确认{name}臂已下使能，请查看现场控制器")


def _t_right_left(sight) -> np.ndarray:
    return sight.right_camera.T_base_camera @ np.linalg.inv(sight.left_camera.T_base_camera)


def _predicted_left_state(left, xy, profile, lift_mm):
    """Left state after the commissioned grasp and lift, from raw (read-only) controller IK."""
    neck = profile["neck_grasp"]
    rpy = list(neck["rpy"])
    pose = [xy[0], xy[1], float(neck["tcp_z"]) + lift_mm, *rpy]
    # The start pose may carry the other wrist yaw; seed from the commissioned yaw at the start position.
    here = left.read()
    seed = left.plan(here["pose"][:3], rpy, here["joints"])
    joints = left.plan(pose[:3], rpy, seed)
    return {"pose": pose, "joints": joints, "mode": 5}


def _refiner(right):
    """Right-pose refinement whose displacement cap is measured from the first (standoff) pose."""
    holder = {}

    def refine(current, delta):
        if "fn" not in holder:
            holder["fn"] = make_refine_fn(right, current, handoff_config([0.0, 0.0, 0.0]))
        return holder["fn"](current, delta)

    return refine


def _gripper_confirms_hold(position, gripper):
    """The jaw stopped at the taught loaded code and far from the empty-close code.

    The servo keeps pushing toward its close target, so a bottle that slipped out during the lift
    would let the jaw run on to the empty code; still reading the loaded code after the lift is
    physical evidence that the neck is between the jaws.
    """
    return (abs(position - gripper["loaded_closed_final"]) <= GRIPPER_CODE_TOLERANCE and
            abs(position - gripper["free_closed_final"]) > 4 * GRIPPER_CODE_TOLERANCE)


def _exit_left(left, out):
    """Back away from the held bottle: straight up, then across to the photo position."""
    start = left.read()
    pose = start["pose"]
    target, _ = photo_pose("left")
    travel_z = max(pose[2], target[2])
    jog = Jog(left, Limits(pose[:3], 700.0, 350.0, travel_z + 15.0), record_path=Path("unused-handoff-exit.csv"),
              ask=lambda _p: "", out=out, stop_pressed=_stop_key)
    jog.rpy = list(pose[3:])
    legs = jog.travel_legs(target, travel_z)
    finals = jog.check_route(legs)
    jog.run_route(legs, finals)
    return legs


def _latest_cycle_log():
    """The newest handoff_cycle.json (plan and executed steps of the last live handoff), or None."""
    logs = sorted(HANDOFF_DATA.glob("run-*/handoff_cycle.json"), key=lambda path: path.stat().st_mtime)
    return json.loads(logs[-1].read_text(encoding="utf-8")) if logs else None


def _right_retreat_path(cycle_log, right_state):
    """Planned right waypoints from the last one reached back to the photo pose, newest first.

    Only the poses the arm already passed through on the way in are used, each with the joints the
    plan checked, so the way out sweeps exactly the space the way in did.
    """
    if cycle_log is None or "plan" not in cycle_log:
        raise RuntimeError("右臂不在拍摄位，且找不到上次交接的路径记录，请用示教器手动移回")
    waypoints = cycle_log["plan"]["right_waypoints"]
    reached = {step["step"] for step in cycle_log["steps"]}
    last = max((i for i, w in enumerate(waypoints) if w["name"] in reached), default=None)
    if last is None:
        # Stopped inside the first continuous segment (only segment ends are logged): take the
        # planned sample just behind the arm, i.e. one it has already passed.
        pose = np.array(right_state["pose"][:3], float)
        distances = [float(np.linalg.norm(pose - np.array(w["xyz"], float))) for w in waypoints]
        nearest = int(np.argmin(distances))
        if distances[nearest] > 15.0:
            raise RuntimeError("右臂不在上次规划的路径上（偏离超过 15 mm），请用示教器手动移回")
        last = max(nearest - 1, 0)
    # The arm may have been stopped on its way to the next waypoint: accept either as the start.
    pose = np.array(right_state["pose"][:3], float)
    near = [i for i in (last, last + 1) if i < len(waypoints)
            and np.linalg.norm(pose - np.array(waypoints[i]["xyz"], float)) <= 15.0]
    if not near:
        raise RuntimeError("右臂当前位置离上次记录的路径超过 15 mm，请用示教器手动移回")
    joints = np.array(right_state["joints"], float)
    if np.abs(joints - np.array(waypoints[last]["joints"], float)).max() > 15.0:
        raise RuntimeError("右臂关节构型与记录的路径不一致，请用示教器手动移回")
    # Never step forward toward the bottle: start from the last waypoint actually reached.
    return list(reversed(waypoints[:last + 1]))


CORRIDOR_ORIENTATION_DEG = 2.0
RETREAT_DOWN_MM = RETREAT_CHECK_MM  # the right's down check after the left let go
CORRIDOR_SLACK_MM = 5.0


def _right_corridor_back(cycle_log, right_state):
    """Way out for a right arm stopped during the wrist-camera alignment/approach after the standoff.

    That phase only shifts the hand sideways (<= MAX_TOTAL_REFINE_MM in tool X/Y) and forward along
    the tool axis from the planned standoff, keeping its orientation, so the stop pose is off the
    recorded path (2026-11-23 16:07: 32 mm). If the arm is inside that corridor, return
    ``(targets, standoff_state)``: first straight back along the tool axis to the standoff depth
    (away from the bottle), then the sideways shift back onto the standoff; ``standoff_state`` is the
    recorded standoff waypoint, from which the normal recorded retreat continues. Otherwise None.
    """
    if cycle_log is None or "plan" not in cycle_log:
        return None
    plan = cycle_log["plan"]
    name = (plan.get("standoff_pose") or {}).get("name")
    standoff = next((w for w in plan.get("right_waypoints", []) if w.get("name") == name), None)
    reached = {step.get("step") for step in cycle_log.get("steps", [])}
    if standoff is None or name not in reached:
        return None
    pose = np.asarray(right_state["pose"], float)
    R_here, R_stand = _rot([float(v) for v in pose[3:]]), _rot([float(v) for v in standoff["rpy"]])
    cos = (np.trace(R_stand.T @ R_here) - 1.0) / 2.0
    if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > CORRIDOR_ORIENTATION_DEG:
        return None
    stand_xyz = np.asarray(standoff["xyz"], float)
    shift = R_stand.T @ (pose[:3] - stand_xyz)  # tool X/Y sideways, Z forward toward the bottle
    if (max(abs(shift[0]), abs(shift[1])) > MAX_TOTAL_REFINE_MM + CORRIDOR_SLACK_MM
            or not -CORRIDOR_SLACK_MM <= shift[2] <= STANDOFF_MM + CORRIDOR_SLACK_MM):
        return None
    rpy = [float(v) for v in standoff["rpy"]]
    back = pose[:3] - R_stand[:, 2] * shift[2]
    targets = [{"name": "corridor_back", "xyz": [float(v) for v in back], "rpy": rpy},
               {"name": "corridor_to_standoff", "xyz": [float(v) for v in stand_xyz], "rpy": rpy}]
    return targets, {"pose": [*stand_xyz.tolist(), *rpy], "joints": list(standoff["joints"])}


def _handback_contact(cycle_log, right_state, left_state, right_jaw, left_jaw, window):
    """After a handoff the right holds the bottle and the left stands open at the station.

    Returns the right pose at which it closed (``{"xyz", "rpy"}``) when the live state matches the
    last log: right jaw in the body-hold window, left open at the logged station, right at that
    contact pose or straight below it (the 15 mm down check), same orientation. Otherwise None.
    """
    if cycle_log is None or "plan" not in cycle_log:
        return None
    steps = cycle_log.get("steps", [])
    names = [step.get("step") for step in steps]
    if "right_closed" not in names or "left_released" not in names:
        return None
    if not (window[0] <= right_jaw <= window[1]) or left_jaw < 250:
        return None
    closed_at = names.index("right_closed")
    contact = next((step for step in reversed(steps[:closed_at]) if "measured" in step), None)
    legs = next((step["legs"] for step in steps if step.get("step") == "left_route_checked"), None)
    if contact is None or not legs or math.dist(left_state["pose"][:3], legs[-1]) > 5.0:
        return None
    xyz = [float(v) for v in contact["measured"][:3]]
    rpy = [float(v) for v in contact.get("rpy") or contact["measured"][3:]]
    pose = np.asarray(right_state["pose"], float)
    R_here, R_contact = _rot([float(v) for v in pose[3:]]), _rot(rpy)
    cos = (np.trace(R_contact.T @ R_here) - 1.0) / 2.0
    if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > CORRIDOR_ORIENTATION_DEG:
        return None
    d = pose[:3] - np.asarray(xyz)
    if math.hypot(d[0], d[1]) > 3.0 or not -RETREAT_DOWN_MM - 3.0 <= d[2] <= 3.0:
        return None
    return {"xyz": xyz, "rpy": rpy}


def _left_return_target(cycle_log, left_state, right_state, right_jaw, left_jaw, window):
    """After a completed handoff: the right holds the bottle, the left (open) went back to the photo pose.

    ``_exit_left`` left the station straight across at the station height and then down onto the photo
    pose, so the neck slid out of the open jaws along that line. Returns the legs that retrace it
    (up to the station height over the photo pose, then straight across to the station) when the live
    state matches the last log: transfer confirmed, left open at the photo pose, right jaw in the hold
    window at the logged down-check pose. Otherwise None.
    """
    if cycle_log is None:
        return None
    steps = cycle_log.get("steps", [])
    names = [step.get("step") for step in steps]
    if "transfer_confirmed" not in names or left_jaw < 250 or not (window[0] <= right_jaw <= window[1]):
        return None
    if not near_photo("left", left_state["pose"]):
        return None
    legs = next((step["legs"] for step in steps if step.get("step") == "left_route_checked"), None)
    down = next((step for step in steps if step.get("step") == "right_down_check"), None)
    if not legs or down is None:
        return None
    if math.dist(right_state["pose"][:3], down["measured"][:3]) > 3.0:
        return None
    station = [float(v) for v in legs[-1]]
    here = [float(v) for v in left_state["pose"][:3]]
    route = [[here[0], here[1], station[2]], station] if station[2] > here[2] + 1e-6 else [station]
    return route


def _hand_back(right, left, contact, gripper, outcome):
    """Right lifts the bottle back to where it took it, the left re-closes on the neck, the right lets go."""
    here = right.read()
    joints = right.plan(contact["xyz"], contact["rpy"], here["joints"])
    right.move(contact["xyz"], contact["rpy"], joints)
    right.wait(contact["xyz"], contact["rpy"], joints, _stop_key,
               timeout_s=move_timeout_s(here["pose"], [*contact["xyz"], *contact["rpy"]]))
    if _stop_key():
        raise RuntimeError("操作员停止")
    closed = left.grip_to(int(gripper["target"]), int(gripper["speed"]))  # neck_grasp.gripper keys
    outcome["left_regrasp"] = closed
    if not _gripper_confirms_hold(int(closed["final"]), gripper):
        raise RuntimeError(f"左爪重新夹瓶颈的读数 {closed['final']} 不是带瓶读数，右爪保持夹住，停止")
    opened = right.grip_to(255, int(gripper["speed"]))
    outcome["right_open"] = opened
    if int(opened["final"]) < 250:
        raise RuntimeError(f"右爪未张开到位（读数 {opened['final']}），停止")
    position = left.gripper_position()
    if not _gripper_confirms_hold(position, gripper):
        raise RuntimeError(f"右爪松开后左爪读数 {position} 不再是带瓶读数，停止")
    return position


def _back_to_standoff(right, targets, standoff_joints):
    """Straight MovL moves of ``_right_corridor_back``; the arm must end in the standoff configuration."""
    for target in targets:
        if _stop_key():
            raise RuntimeError("操作员停止")
        here = right.read()
        if math.dist(here["pose"][:3], target["xyz"]) < 0.5:
            continue
        joints = right.plan(target["xyz"], target["rpy"], here["joints"])
        right.move(target["xyz"], target["rpy"], joints)
        right.wait(target["xyz"], target["rpy"], joints, _stop_key,
                   timeout_s=move_timeout_s(here["pose"], [*target["xyz"], *target["rpy"]]))
    joints = np.asarray(right.read()["joints"], float)
    if np.abs(joints - np.asarray(standoff_joints, float)).max() > 15.0:
        raise RuntimeError("右臂回到停靠点后关节构型与记录不一致，停止；请用示教器处理")


def _retreat_right(right, path):
    """Back along the checked approach, one continuous MovL per planned segment, to the photo pose."""
    # ``path`` runs newest -> start. A segment boundary going back is the first sample of a kind
    # in forward order, i.e. where the step_kind changes while walking backwards.
    targets = [w for i, w in enumerate(path[1:], 1) if w.get("step_kind") != path[i - 1].get("step_kind")]
    if not targets or targets[-1] is not path[-1]:
        targets.append(path[-1])
    for target in targets:
        if _stop_key():
            raise RuntimeError("操作员停止")
        here = right.read()["pose"]
        timeout = move_timeout_s(here, [*target["xyz"], *target["rpy"]])
        # Leaving a joint-space segment backwards is again a joint move (the way in was JointMovJ).
        index = path.index(target)
        kind_ahead = path[index - 1].get("step_kind") if index > 0 else target.get("step_kind")
        if kind_ahead == "joint_approach":
            right.move_joint(target["xyz"], target["rpy"], target["joints"])
            timeout += 60.0
        else:
            right.move(target["xyz"], target["rpy"], target["joints"])
        right.wait(target["xyz"], target["rpy"], target["joints"], _stop_key, timeout_s=timeout)


def _reset(args, directory, outcome):
    """Put a held bottle back down where it was lifted, open the left jaw, both arms to the photo pose."""
    profile = _load_profile()
    neck = profile["neck_grasp"]
    gripper = neck["gripper"]
    states = {}
    for name, binding in profile["arms"].items():
        state = read_passive_robot_state(binding["ip"])
        if state["mode"] != 4 or state["enabled"] or state["tool"] != 0 or state["user"] != 0:
            raise RuntimeError(f"{name} 臂需在下使能空闲（mode 4，User0/Tool0），当前 mode {state['mode']}")
        states[name] = state
    with ExitStack() as stack:
        driver = LiveDriver(profile, allow_live=True, confirmation=lambda: True, log=print)
        stack.callback(driver.close)
        left = SharedArm(profile, "left", LEFT_SPEED_PERCENT, driver)
        right = SharedArm(profile, "right", RIGHT_SPEED_PERCENT, driver)
        right_jaw, left_jaw = right.gripper_position(), left.gripper_position()
        outcome.update(left_jaw=left_jaw, right_jaw=right_jaw)
        cycle_log = _latest_cycle_log()
        window = tuple(getattr(args, "right_hold_window", (30, 240)))
        return_route = None
        if right_jaw < 250:
            return_route = _left_return_target(cycle_log, states["left"], states["right"], right_jaw, left_jaw, window)
        handback = None
        right_plan_state = states["right"]
        if right_jaw < 250:
            handback = _handback_contact(cycle_log, states["right"], states["left"], right_jaw, left_jaw, window)
            if handback is None and return_route is not None:
                # The left will first retrace its exit onto the station; plan the hand-back from there.
                station_state = {"pose": [*return_route[-1], *states["left"]["pose"][3:]], "joints": states["left"]["joints"]}
                handback = _handback_contact(cycle_log, states["right"], station_state, right_jaw, left_jaw, window)
            if handback is None:
                raise RuntimeError(f"右爪未张开（读数 {right_jaw}），且不是上次交接后的状态，请人工处理")
            # After the hand-back the right stands at its contact pose, empty.
            right_plan_state = {"pose": [*handback["xyz"], *handback["rpy"]], "joints": states["right"]["joints"]}
        right_path, corridor = None, None
        if not near_photo("right", right_plan_state["pose"]):
            try:
                right_path = _right_retreat_path(cycle_log, right_plan_state)
            except RuntimeError:
                found = _right_corridor_back(cycle_log, right_plan_state)
                if found is None:
                    raise
                corridor, standoff_state = found
                right_path = _right_retreat_path(cycle_log, standoff_state)
        pose = states["left"]["pose"] if return_route is None else [*return_route[-1], *states["left"]["pose"][3:]]
        held = left_jaw < 250 or handback is not None
        left_back = []
        if held and cycle_log is not None:
            legs = next((s["legs"] for s in cycle_log["steps"] if s["step"] == "left_route_checked"), None)
            if legs and math.dist(pose[:3], legs[-1]) <= 5.0:
                # Still at the handoff station: retrace the route that brought it there.
                left_back = [list(leg) for leg in reversed(legs[:-1])]
                pose = [*left_back[-1], *pose[3:]]
        if held:
            x, y, z = pose[:3]
            if not (220 <= x <= 305 and -425 <= y <= -350):
                raise RuntimeError(f"左爪夹着瓶子但不在原放瓶区上方（{x:.0f}, {y:.0f}），请人工放回")
            if not float(neck["tcp_z"]) - 2 <= z <= HOVER_Z_LIMIT:
                raise RuntimeError(f"左臂高度 {z:.0f} mm 不在可直接放回的范围")
            if handback is None and not _gripper_confirms_hold(left_jaw, gripper):
                raise RuntimeError(f"左爪读数 {left_jaw} 既不是张开也不是带瓶读数，请人工检查")
        attempted = False
        try:
            if _stop_key():
                raise RuntimeError("已取消")
            attempted = True
            if handback is not None:
                right.open_session(home=False)
                left.open_session(home=False)
                if return_route is not None:
                    lhere = left.read()
                    top = max(float(lhere["pose"][2]), return_route[-1][2]) + 15.0
                    ljog = Jog(left, Limits(lhere["pose"][:3], 700.0, 350.0, top),
                               record_path=Path("unused-handoff-reset.csv"), ask=lambda _p: "", out=print,
                               stop_pressed=_stop_key)
                    ljog.rpy = list(lhere["pose"][3:])
                    ljog.run_route(return_route, ljog.check_route(return_route))
                    outcome["left_returned_to_station"] = return_route
                outcome["handed_back_left_jaw"] = _hand_back(right, left, handback, gripper, outcome)
            if right_path is not None:
                if not right.session_open:
                    right.open_session(home=False)
                _joint_speed(right)
                if corridor is not None:
                    _back_to_standoff(right, corridor, right_path[0]["joints"])
                    outcome["right_corridor_back"] = [t["xyz"] for t in corridor]
                _retreat_right(right, right_path)
                outcome["right_retreated_waypoints"] = len(right_path)
            if not left.session_open:
                left.open_session(home=False)
            here = left.read()
            # Ceiling: the higher of where the left stands now (the 610 mm handoff station) and the photo pose.
            ceiling = max(float(here["pose"][2]), photo_pose("left")[0][2]) + 15.0
            jog = Jog(left, Limits(here["pose"][:3], 700.0, float(neck["tcp_z"]) - 5, ceiling),
                      record_path=Path("unused-handoff-reset.csv"), ask=lambda _p: "", out=print,
                      stop_pressed=_stop_key)
            jog.rpy = list(here["pose"][3:])
            if left_back:
                jog.run_route(left_back, jog.check_route(left_back))
                here = left.read()
            if held:
                down = [[here["pose"][0], here["pose"][1], float(neck["tcp_z"])]]
                jog.run_route(down, jog.check_route(down))
                opened = left.grip_to(int(gripper["open"]), int(gripper["speed"]))
                outcome["left_open"] = opened
                if opened["final"] < 250:
                    raise RuntimeError(f"左爪未张开到位（读数 {opened['final']}），停止")
                up = [[here["pose"][0], here["pose"][1], HOVER_Z_RESET]]
                jog.run_route(up, jog.check_route(up))
            target, _ = photo_pose("left")
            jog.limits = Limits(jog.xyz, 700.0, float(neck["tcp_z"]) - 5, ceiling)
            legs = [list(target)] if jog.xyz[2] >= HOVER_Z_RESET - 1 else [[*jog.xyz[:2], HOVER_Z_RESET], list(target)]
            jog.run_route(legs, jog.check_route(legs))
            if right_path is None and not near_photo("right", states["right"]["pose"]):
                right.open_session(home=False)
                rhere = right.read()
                rjog = Jog(right, Limits(rhere["pose"][:3], 700.0, 300.0, 560.0),
                           record_path=Path("unused-handoff-reset.csv"), ask=lambda _p: "", out=print,
                           stop_pressed=_stop_key)
                rjog.rpy = list(rhere["pose"][3:])
                rtarget, _ = photo_pose("right")
                rlegs = rjog.travel_legs(rtarget, max(rhere["pose"][2], rtarget[2]))
                rjog.run_route(rlegs, rjog.check_route(rlegs))
            outcome["bottle_put_back"] = held
            return True
        except BaseException:
            if attempted:
                for arm in (left, right):
                    if arm.session_open:
                        try:
                            arm.stop()
                        except Exception:  # noqa: BLE001
                            pass
            raise
        finally:
            for arm in (left, right):
                arm.close()
            if attempted:
                _verify_both_disabled(profile)


def _execute(args, directory, outcome):
    profile = _load_profile()
    if "neck_grasp" not in profile:
        raise ValueError("缺少瓶颈首设记录，请先完成 run_neck_pick 的首次设置")
    states = _idle_states(profile)
    preflight(profile, states["left"])
    with ExitStack() as stack:
        sight = Cameras(stack, directory)
        driver = LiveDriver(profile, allow_live=True, confirmation=lambda: True, log=print)
        stack.callback(driver.close)
        left = SharedArm(profile, "left", LEFT_SPEED_PERCENT, driver)
        right = SharedArm(profile, "right", RIGHT_SPEED_PERCENT, driver)
        T = _t_right_left(sight)

        target = _checked_target(sight.localize("handoff-preflight", states["left"]["pose"]))
        predicted = _predicted_left_state(left, target, profile, args.lift_mm)
        right_state = {"pose": states["right"]["pose"], "joints": states["right"]["joints"], "mode": 5}
        station = None if args.search else handoff_station()
        if station is None and args.live:
            raise RuntimeError("还没有交接站：先运行 --preflight --search 找到并保存交接站")
        plan_fn = make_plan_fn(left, right, T, wide=args.wide_search, station=station,
                               pitch_deg=args.grasp_pitch if args.search else None)
        try:
            plan = plan_fn(predicted, right_state)
        except HandoffInfeasible as infeasible:
            outcome["plan_error"] = str(infeasible)
            outcome["rejected_summary"] = dict(infeasible.histogram)
            raise
        if station is None:
            save_handoff_station(plan["left_handoff_xyz"], plan["candidate"]["approach_offset_deg"], directory.name,
                                 plan["candidate"].get("pitch_down_deg", 0.0))
            print("已保存交接站到 chem_demo/poses.json：", plan["left_handoff_xyz"], flush=True)
        (directory / "preflight_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        outcome["preflight_plan"] = {"handoff_left_tcp": plan["left_handoff_xyz"], "candidate": plan["candidate"],
                                     "bottle_xy": target}
        print("交接规划（只读查询）可行：", json.dumps(outcome["preflight_plan"], ensure_ascii=False), flush=True)
        if args.preflight:
            return True

        from chem_demo.handoff_observer import RightWristObserver
        observer = RightWristObserver(stack, sight.roles["right_wrist"], directory)
        observer.expected_range_mm = STANDOFF_MM   # soft prior for picking the bottle mask
        attempted = False
        try:
            if _stop_key():
                raise RuntimeError("已取消")
            attempted = True
            left.open_session(home=False)
            right.open_session(home=False)
            for arm in (left, right):
                _joint_speed(arm)

            read_wrist = make_read_wrist(left, sight, directory)
            pick = NeckPickCycle(left, profile, localize=sight.localize, check_wrist=sight.check_wrist,
                                 read_wrist=read_wrist, verify_held=verify_neck_lift, stop_pressed=_stop_key,
                                 log_path=directory / "pick_cycle.json", lift_mm=args.lift_mm)
            picked = pick.run()["result"]
            outcome.update(pick_status=picked.get("status"), pick_held=picked.get("held"))
            gripper = profile["neck_grasp"]["gripper"]
            if picked.get("status") == "hold_uncertain":
                # The lift camera check could not decide; fall back to the jaw reading after the lift.
                position = left.gripper_position()
                outcome["held_by_gripper"] = position
                if _gripper_confirms_hold(position, gripper):
                    print(f"抬起后视觉无法判断，夹爪读数 {position} 与带瓶读数一致，按已夹住继续", flush=True)
                    picked = {**picked, "success": True, "held": True, "status": "held_by_gripper_feedback"}
                    outcome["pick_status"] = picked["status"]
            sight.pause_streams()  # only the right wrist streams during the handoff
            if picked.get("success") is not True:
                raise HandoffRefused(f"瓶子未确认被左爪夹住（{picked.get('status')}），不进行交接")

            def verify_hold(left_state):
                """Top frame at the station -> GPU worker fits the hanging bottle (held_bottle_top)."""
                image = sight.top_frame()
                stem = directory / "hang_top"
                path = stem.with_suffix(".png")
                cv2.imwrite(str(path), image)
                reply = observer.client.request({"cmd": "held_bottle", "image": str(path),
                                                 "pose": list(left_state["pose"]),
                                                 "overlay": str(directory / "hang_top_fit.png")})
                sight.pause_streams()  # the right wrist streams next
                if not reply.get("ok"):
                    return {"ok": False, "reason": reply.get("error", "vision worker failed")}
                measurement = reply["measurement"]
                print("顶视实测瓶子悬挂：", json.dumps({k: measurement.get(k) for k in
                      ("ok", "reason", "xy_offset_mm", "bottom_offset_mm", "along_ray_mm", "across_ray_mm", "rms_px")}, ensure_ascii=False), flush=True)
                return measurement

            loaded = gripper["loaded_closed_final"]
            cycle = HandoffCycle(
                left, right, left_loaded_code=loaded,
                plan_fn=plan_fn,
                refine_fn=_refiner(right),
                observe_fn=observer.observe, transfer_fn=observer.transfer, capture_fn=observer.capture,
                verify_hold_fn=verify_hold,
                stop_after_hang=args.until_hang,
                right_hold_window=tuple(args.right_hold_window), stop_pressed=_stop_key,
                log_path=directory / "handoff_cycle.json")
            result = cycle.run()["result"]
            outcome.update(handoff_status=result.get("status"))
            if result.get("status") == "stopped_after_hang_check":
                print("已按 --until-hang 在顶视实测后停止：左爪仍夹瓶停在交接站，右臂未动。"
                      f"叠加图 {directory / 'hang_top_fit.png'}；放回用 --reset --path-clear", flush=True)
                return True
            if result.get("success") is True:
                _exit_left(left, print)
            return result.get("success") is True
        except BaseException:
            if attempted:
                for arm in (left, right):
                    if arm.session_open:
                        try:
                            arm.stop()
                        except Exception:  # noqa: BLE001
                            pass
            raise
        finally:
            for arm in (left, right):
                arm.close()
            if attempted:
                _verify_both_disabled(profile)


def main(argv=None):
    parser = argparse.ArgumentParser(description="左臂夹瓶颈抬起，右臂水平夹瓶身交接（全程视觉定位，不用示教坐标）")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="现场执行：抓取、交接、左臂撤回拍摄位")
    mode.add_argument("--preflight", action="store_true", help="只读：定位瓶子并用控制器逆解查询交接是否可行，不动机械臂")
    mode.add_argument("--reset", action="store_true", help="把左爪还夹着的瓶子放回抬起处、张开，两臂回拍摄位")
    parser.add_argument("--search", action="store_true",
                        help="与 --preflight 一起用：重新全量搜索交接站并保存（约 1 分钟）")
    parser.add_argument("--path-clear", action="store_true",
                        help="确认左臂抬瓶/横移路径和右臂接近路径上没有天平、烧杯、试管架等器材（--live 必须）")
    parser.add_argument("--until-hang", action="store_true",
                        help="与 --live 一起用：抓瓶、左臂到交接站、顶视实测瓶子悬挂后停止，右臂不动（只拍照验证）")
    parser.add_argument("--lift-mm", type=int, choices=(15, 30), default=30)
    parser.add_argument("--grasp-pitch", type=float, choices=(0.0, 20.0, 35.0), default=None,
                        help="与 --search 一起用：右爪下倾角（0=水平夹）。不给则在 35/20 中自动选")
    parser.add_argument("--wide-search", action="store_true", help="更大的交接点/接近方位搜索（更慢）")
    parser.add_argument("--right-hold-window", type=int, nargs=2, default=(30, 240), metavar=("MIN", "MAX"),
                        help="右爪夹住瓶身后允许的夹爪读数范围（首次现场实测后收紧）")
    args = parser.parse_args(argv)
    if args.until_hang and not args.live:
        parser.error("--until-hang 只能和 --live 一起用")
    if args.search and not args.preflight:
        parser.error("--search 只能和 --preflight 一起用")
    if (args.live or args.reset) and not args.path_clear:
        parser.error("--live 需要加 --path-clear：规划只检查可达性，不知道台面上有哪些器材，须由现场确认扫过的区域为空")
    directory = HANDOFF_DATA / (datetime.now().strftime("run-%Y%m%d_%H%M%S") + "-" + uuid4().hex[:10])
    directory.mkdir(parents=True)
    started = time.perf_counter()
    result = {"mode": "live" if args.live else "reset" if args.reset else "preflight"}
    print(f"左右臂交接 | 记录：{directory}", flush=True)
    if args.live:
        print("运行时按住空格停止。空瓶、工作区净空、两只手边都有急停。", flush=True)
    try:
        passed = (_reset if args.reset else _execute)(args, directory, result)
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = str(exc) or type(exc).__name__
        print(f"已停止：{result['error']}", flush=True)
        passed = False
    result.update(success=passed, elapsed_s=round(time.perf_counter() - started, 3))
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
