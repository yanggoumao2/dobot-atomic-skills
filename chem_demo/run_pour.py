"""Right arm pours the bottle it holds (after run_handoff) into the beaker on the left balance.

Start state: the right jaws hold the bottle body at the hold pose the handoff leaves (tool axis
horizontal, bottle upright), the left arm at the photo pose, both arms disabled (mode 4).

  --preflight            read-only: top frame -> GPU worker locates the vessels again (the balances
                         move between sessions), plan with controller IK queries, save plan + picture.
  --live --until side    lift, carry the bottle upright beside the beaker, take top + front frames, back.
  --live --until pretilt ... and tip to the pre-tilt with the lip over the beaker (no liquid flows yet).
  --live                 full pour: tip about the lip to theta_max, dwell, the same way back to the hold.

The right arm stays in the IK branch it is in (every sample is sent as a JointMovJ to the planned
joints); the pour needs the wrist-up branch (J4 above the flange), which the handoff has to choose
(HandoffConfig.right_branch_seed), so a hold in the wrist-down branch is refused here.
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
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from chem_demo.gripper_geometry import FINGERTIP_FROM_FLANGE_MM, table_z_real
from chem_demo.handoff_plan import HandoffConfig, _rot, _tool_axis, arm_capsules_in
from chem_demo.pour_plan import PourConfig, PourInfeasible, PourPlan, Vessel, plan_pour, reverse_steps, side_view_svg

POUR_DATA = Path(__file__).resolve().parents[1] / "calibration-data" / "chem-demo" / "pour"
RIGHT_SPEED_PERCENT = 16
CP_RATIO = 80                  # blend between queued joint samples (50 left a visible halt at each sample)
DEFAULT_DWELL_S = 4.0
ARRIVAL_POSITION_TOL_MM = 1.5
ARRIVAL_ORIENTATION_TOL_DEG = 1.0
# Not seen as 3-D objects by the top-camera locate (ESTIMATES):
ROD_RADIUS_MM, ROD_ABOVE_RIM_MM = 8.0, 120.0     # glass rod standing in the centre beaker
RACK_RADIUS_MM, RACK_HEIGHT_MM = 90.0, 110.0     # tube rack footprint / height above the table
PAN_RADIUS_MM = 110.0                            # balance footprint around a pan beaker
MEASURED_GRASP_OFFSET_MM = (16.4, 25.3, 19.5)    # bottle in the right jaws vs the model (see add_pour_arguments)


# ----------------------------------------------------------------------------- pure helpers (tested)

def scene_from_locate(vessels: Mapping[str, Mapping[str, Any]], target: str = "left_beaker", *,
                      rack_xy: Sequence[float] | None = None,
                      fingertip_mm: float = FINGERTIP_FROM_FLANGE_MM) -> tuple[Vessel, list[Vessel]]:
    """Target beaker and obstacles in controller Z from a chem_demo.locate report (cw heights).

    locate gives axis XY in the right base frame and heights in the fingertip convention (cw);
    controller Z = cw - flange->tips. The pour planner adds the balance under the target itself.
    """
    if target not in vessels:
        raise RuntimeError(f"顶视没有找到目标烧杯 {target}（找到：{', '.join(vessels) or '无'}）")

    def vessel(name: str, v: Mapping[str, Any]) -> Vessel:
        return Vessel(name, float(v["x"]), float(v["y"]), float(v["radius"]),
                      float(v["base_z"]) - fingertip_mm, float(v["rim_z"]) - fingertip_mm)

    beaker = vessel(target, vessels[target])
    table = table_z_real("right", fingertip_mm)
    obstacles: list[Vessel] = []
    for name, v in vessels.items():
        if name == target or v.get("role") == "bottle":   # the bottle is the one in the hand
            continue
        o = vessel(name, v)
        obstacles.append(o)
        if v.get("support", "").endswith("_pan"):
            obstacles.append(Vessel(f"{name}_balance", o.x, o.y, PAN_RADIUS_MM, table, o.bottom_z))
        if name == "centre_beaker":
            obstacles.append(Vessel("centre_rod", o.x, o.y, ROD_RADIUS_MM, o.top_z, o.top_z + ROD_ABOVE_RIM_MM))
    if rack_xy is not None:
        obstacles.append(Vessel("rack", float(rack_xy[0]), float(rack_xy[1]), RACK_RADIUS_MM, table,
                                table + RACK_HEIGHT_MM))
    return beaker, obstacles


def lip_overlay(image, lip_xyz: Sequence[float], beaker: Vessel,
                fingertip_mm: float = FINGERTIP_FROM_FLANGE_MM):
    """Top frame with the planned lip (red dot) and the target rim circle (green) projected on it.

    Controller Z -> fingertip convention (cw) of the top-camera calibration: cw = real + flange->tips.
    """
    import cv2

    from chem_demo.camera_drift import load_corrected_camera
    from vision_pick.cup_locate import DEFAULT_SESSION

    camera = load_corrected_camera(DEFAULT_SESSION, "right")
    out = image.copy()
    a = np.linspace(0.0, 2.0 * math.pi, 72)
    rim = np.column_stack([beaker.x + beaker.radius * np.cos(a), beaker.y + beaker.radius * np.sin(a),
                           np.full_like(a, beaker.top_z + fingertip_mm)])
    axis = np.array([[beaker.x, beaker.y, beaker.top_z + fingertip_mm]])
    lip = np.array([[lip_xyz[0], lip_xyz[1], lip_xyz[2] + fingertip_mm]])
    cv2.polylines(out, [camera.project(rim).astype(np.int32)], True, (0, 220, 0), 2)
    for pts, colour in ((axis, (0, 220, 0)), (lip, (0, 0, 255))):
        u, v = camera.project(pts)[0]
        cv2.circle(out, (int(round(u)), int(round(v))), 5, colour, -1)
    cv2.putText(out, "red: planned lip, green: rim + axis", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2)
    return out


def wrist_up(joints: Sequence[float], flange_z: float) -> bool:
    """True if J4 sits above the flange (the branch the pour planner can use)."""
    from chem_demo.nova2_model import link_points
    return float(link_points(joints)["wrist1"][2]) > flange_z + 50.0


def check_hold(state: Mapping[str, Any], jaw: int, window: Sequence[int]) -> list[str]:
    """Reasons the right arm is not in a pourable hold (empty = fine)."""
    problems = []
    pose, joints = list(state["pose"]), list(state["joints"])
    axis = _tool_axis(pose[3:])
    if abs(float(axis[2])) > 0.1:
        problems.append(f"右爪工具轴不水平（z 分量 {axis[2]:.2f}），不是交接后的持瓶姿态")
    if not window[0] <= jaw <= window[1]:
        problems.append(f"右爪读数 {jaw} 不在持瓶范围 {window[0]}–{window[1]}，可能没夹着瓶子")
    if not wrist_up(joints, pose[2]):
        problems.append("右臂在腕部朝下的分支（J4 在法兰下方）：倒液时前臂会扫过天平和中间烧杯。"
                        "需要交接时选腕部朝上的分支（HandoffConfig.right_branch_seed）")
    return problems


def phases(plan: PourPlan, until: str | None) -> list[tuple[str, list[dict[str, Any]]]]:
    """Forward samples grouped by phase, cut after ``until`` (side = end of transit)."""
    stop_after = {"side": "transit", "pretilt": "pretilt", None: "pour"}[until]
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for step in plan.steps[1:]:
        if groups and groups[-1][0] == stop_after and step["phase"] != stop_after:
            break
        if not groups or groups[-1][0] != step["phase"]:
            groups.append((step["phase"], []))
        groups[-1][1].append(step)
    return groups


def segments(forward: Sequence[tuple[str, list[dict[str, Any]]]], continuous: bool
             ) -> list[tuple[str, list[dict[str, Any]]]]:
    """Groups the controller runs without stopping. The full pour queues lift -> pour as ONE blended
    motion (no arrival wait, jaw read or photo between phases, which made it stop-and-go); the
    --until rehearsals keep one group per phase so each phase can be photographed."""
    if not continuous or not forward:
        return [(name, list(steps)) for name, steps in forward]
    return [("+".join(name for name, _ in forward), [s for _, steps in forward for s in steps])]


def way_back(forward: Sequence[tuple[str, list[dict[str, Any]]]], hold: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every executed sample in reverse, ending at the hold sample."""
    done = [s for _, steps in forward for s in steps]
    return list(reversed(done[:-1])) + [dict(hold)]


# ----------------------------------------------------------------------------- live

def _run(args, directory: Path, outcome: dict) -> bool:
    import cv2

    from chem_demo.bottle_observation import read_passive_robot_state
    from chem_demo.cameras import grab_c920
    from chem_demo.handoff_runtime import CachedIK
    from chem_demo.run_handoff import SharedArm, _joint_speed, _t_right_left
    from chem_demo.run_neck_pick import Cameras, _load_profile, _stop_key
    from chem_demo.stations import near_photo
    from chem_demo.wrist_runtime import VisionClient
    from task_skills.chemistry.driver import LiveDriver, _reply_status
    from task_skills.run_cartesian_jog import _rotation_angle_deg

    profile = _load_profile()
    left_state = read_passive_robot_state(profile["arms"]["left"]["ip"])
    right_state = read_passive_robot_state(profile["arms"]["right"]["ip"])
    for name, st in (("left", left_state), ("right", right_state)):
        if st["mode"] != 4 or st["enabled"]:
            raise RuntimeError(f"{name} 臂应为已下使能空闲（mode 4），当前 mode {st['mode']}")
    if not near_photo("left", left_state["pose"]):
        raise RuntimeError("左臂不在拍摄位：倒液规划假定左臂停在拍摄位")
    outcome["start"] = {"left": left_state["pose"], "right": right_state["pose"], "right_joints": right_state["joints"]}

    def front(name: str) -> None:
        if args.no_front:
            return
        try:
            cv2.imwrite(str(directory / f"front_{name}.png"), grab_c920())
        except Exception as error:  # noqa: BLE001 - the observer camera never blocks the run
            print(f"正面相机取帧失败（{name}）：{error}", flush=True)

    with ExitStack() as stack:
        sight = Cameras(stack, directory)
        driver = LiveDriver(profile, allow_live=True, confirmation=lambda: True, log=print)
        stack.callback(driver.close)
        right = SharedArm(profile, "right", RIGHT_SPEED_PERCENT, driver)
        jaw = int(right.gripper_position())
        problems = check_hold(right_state, jaw, args.right_hold_window)
        outcome["right_jaw_start"] = jaw
        if problems:
            raise RuntimeError("；".join(problems))

        # --- scene: the balances move between sessions, so the vessels are located now
        top_path = directory / "top_start.png"
        cv2.imwrite(str(top_path), sight.top_frame())
        if args.scene_image:
            # The held bottle hides the target beaker from the top camera at the hold pose, so the
            # vessels are located in a frame taken just before the handoff (arms at the photo pose).
            scene = cv2.imread(str(args.scene_image))
            if scene is None:
                raise RuntimeError(f"读不到场景图 {args.scene_image}")
            top_path = directory / "top_scene.png"
            cv2.imwrite(str(top_path), scene)
            outcome["scene_image"] = str(args.scene_image)
        client = VisionClient()
        stack.callback(client.close)
        beaker, obstacles = locate_scene(client, top_path, directory, args.target, outcome)
        right_state = {**right_state}
        pour_cfg = pour_config(args)
        plan = plan_for_hold(right, sight, left_state, right_state, beaker, obstacles, pour_cfg, directory, outcome)
        if args.preflight:
            return True
        try:
            if _stop_key():
                raise RuntimeError("已取消")
            front("before")
            right.open_session(home=False)
            _joint_speed(right)
            return execute_pour(right, sight, plan, beaker, until=args.until, dwell_s=args.dwell_s, jaw=jaw,
                                jaw_change_limit=args.jaw_change_limit, directory=directory, outcome=outcome,
                                front=front)
        except BaseException:
            if right.session_open:
                try:
                    right.stop()
                except Exception:  # noqa: BLE001
                    pass
            raise
        finally:
            right.close()


# ----------------------------------------------------------------------------- reusable steps
# (run_handoff --then-pour calls these with arms that stay enabled after the handoff)

def pour_config(args) -> PourConfig:
    kw = dict(theta_max_deg=args.theta_max, grasp_along_tool_mm=args.grasp_shift_mm, grasp_up_mm=args.grasp_up_mm,
              grasp_along_pour_mm=args.grasp_pour_mm, lip_back_mm=args.lip_back_mm)
    if getattr(args, "fast_plan", False):
        # The combination that passed every real run so far first: ~40 IK queries instead of ~380.
        kw.update(lip_heights_mm=(40.0, 35.0, 45.0, 50.0, 30.0, 60.0), pretilt_candidates_deg=(60.0, 50.0, 40.0))
    return PourConfig(**kw)


def locate_scene(client, top_path: Path, directory: Path, target: str, outcome: dict) -> tuple[Vessel, list[Vessel]]:
    """Top frame -> GPU worker -> target beaker and obstacles (controller Z). Vision, no stored coordinates."""
    reply = client.locate_top(str(top_path), str(directory / "locate"))
    if not reply.get("ok"):
        raise RuntimeError(f"顶视定位失败：{reply.get('error')}")
    report = json.loads((directory / "locate" / "locate.json").read_text(encoding="utf-8"))
    rack = report.get("rack") or {}
    beaker, obstacles = scene_from_locate(reply["vessels"], target, rack_xy=rack.get("approx_xy"))
    outcome["scene"] = {"beaker": beaker.to_dict(), "obstacles": [o.to_dict() for o in obstacles],
                        "notes": reply.get("notes")}
    print("目标烧杯（控制器坐标）：", json.dumps(beaker.to_dict(), ensure_ascii=False), flush=True)
    return beaker, obstacles


def plan_for_hold(right, sight, left_state, right_state, beaker, obstacles, pour_cfg, directory, outcome) -> PourPlan:
    """Plan from the right arm's actual hold with controller IK queries (read-only)."""
    from chem_demo.handoff_runtime import CachedIK
    from chem_demo.run_handoff import _t_right_left

    T = _t_right_left(sight)
    cfg = HandoffConfig(body_model=True)
    other = arm_capsules_in(np.asarray(T), left_state["joints"], left_state["pose"][3:], cfg, "left")
    ik = CachedIK(right.plan, "right")
    started = time.monotonic()
    try:
        plan = plan_pour(right_state["pose"][:3], right_state["pose"][3:], right_state["joints"], beaker, obstacles,
                         ik, fk=right.forward, cfg=pour_cfg, arm_cfg=cfg, other_arm=other)
    except PourInfeasible as error:
        outcome["plan_rejected"] = error.rejected
        raise
    outcome["plan"] = {"lip_xyz": plan.lip_xyz, "lip_above_rim_mm": plan.lip_above_rim_mm,
                       "theta_pre_deg": plan.theta_pre_deg, "theta_max_deg": plan.theta_max_deg,
                       "yaw_deg": plan.yaw_deg, "min_clearance": plan.min_clearance, "samples": len(plan.steps),
                       "ik_queries": ik.calls, "seconds": round(time.monotonic() - started, 1)}
    (directory / "pour_plan.json").write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    (directory / "pour_side.svg").write_text(side_view_svg(plan, right_state["pose"][:3], right_state["pose"][3:],
                                                           beaker, obstacles), encoding="utf-8")
    print("倒液规划（只读查询）可行：", json.dumps(outcome["plan"], ensure_ascii=False), flush=True)
    return plan


def execute_pour(right, sight, plan, beaker, *, until, dwell_s, jaw, jaw_change_limit, directory, outcome,
                 front) -> bool:
    """Run the planned pour on an arm whose session is already open (enabled)."""
    import cv2

    from chem_demo.run_handoff import _joint_speed
    from chem_demo.run_neck_pick import _stop_key
    from task_skills.chemistry.driver import _reply_status
    from task_skills.run_cartesian_jog import _rotation_angle_deg

    _joint_speed(right)
    if _reply_status(right.rig.dashboard.CP(CP_RATIO)) != 0:
        raise RuntimeError("CP 设置未被控制器确认")
    forward = phases(plan, until)

    def run_phase(name: str, steps: Sequence[Mapping[str, Any]]) -> None:
        if _stop_key():
            raise RuntimeError(f"空格：在 {name} 之前停止")
        for s in steps:
            right.move_joint(s["xyz"], s["rpy"], s["joints"])
        last = steps[-1]
        right.wait(last["xyz"], last["rpy"], last["joints"], _stop_key, timeout_s=30.0 + 3.0 * len(steps))
        pose = right.read()["pose"]
        err_p = max(abs(pose[i] - last["xyz"][i]) for i in range(3))
        err_r = _rotation_angle_deg(pose[3:], last["rpy"])
        if err_p > ARRIVAL_POSITION_TOL_MM or err_r > ARRIVAL_ORIENTATION_TOL_DEG:
            raise RuntimeError(f"{name} 末点未到位（{err_p:.2f} mm, {err_r:.2f}°）")
        now = int(right.gripper_position())
        outcome.setdefault("phases", []).append({"phase": name, "measured": pose, "jaw": now,
                                                 "time": datetime.now().isoformat(timespec="milliseconds")})
        if abs(now - jaw) > jaw_change_limit:
            raise RuntimeError(f"{name} 后右爪读数 {jaw}→{now}，可能滑瓶，停止")
        print(f"{name} 完成，右爪 {now}", flush=True)

    for name, steps in segments(forward, continuous=until is None):
        run_phase(name, steps)
        if name == "transit":
            cv2.imwrite(str(directory / "top_side.png"), sight.top_frame())
            front("side")
        if name == "pretilt":
            # Where the lip should be vs the beaker, drawn on the calibrated top camera.
            try:
                top = sight.top_frame()
                cv2.imwrite(str(directory / "top_pretilt.png"), top)
                cv2.imwrite(str(directory / "top_pretilt_overlay.png"), lip_overlay(top, plan.lip_xyz, beaker))
            except Exception as error:  # noqa: BLE001 - a picture never blocks the return path
                print(f"预倾顶视取帧失败：{error}", flush=True)
    if until is None:
        end = time.monotonic() + dwell_s   # the photos are taken inside the dwell
        print(f"倾倒中，停留 {dwell_s:.1f} s", flush=True)
        front("pour")
        try:
            cv2.imwrite(str(directory / "top_pour.png"), sight.top_frame())
        except Exception as error:  # noqa: BLE001
            print(f"倾倒顶视取帧失败：{error}", flush=True)
        while time.monotonic() < end:
            if _stop_key():
                break
            time.sleep(0.05)
    elif until == "pretilt":
        front("pretilt")
    back = way_back(forward, plan.steps[0])
    run_phase("return", back)
    front("after")
    outcome["status"] = "poured_and_back" if until is None else f"stopped_after_{until}_and_back"
    return True


def add_pour_arguments(parser, *, defaults_measured: bool = False) -> None:
    """Pour options shared by run_pour and run_handoff --then-pour."""
    parser.add_argument("--target", default="left_beaker")
    parser.add_argument("--theta-max", type=float, default=120.0)
    parser.add_argument("--dwell-s", type=float, default=8.0 if defaults_measured else DEFAULT_DWELL_S)
    parser.add_argument("--jaw-change-limit", type=int, default=8)
    parser.add_argument("--no-front", action="store_true", help="不用正面 C920 取帧")
    # 2026-11-23: three handoffs measured (17.8,24.7,12.2), (17.6,26.6,18.3), (16.4,25.3,19.5) mm with
    # chem_demo.grasp_offset; the last one put ~40 ml of water into the beaker.
    m = MEASURED_GRASP_OFFSET_MM if defaults_measured else (0.0, 0.0, 0.0)
    parser.add_argument("--grasp-shift-mm", type=float, default=m[0],
                        help="瓶子在爪中比模型沿工具轴多出的距离（chem_demo.grasp_offset 测得）")
    parser.add_argument("--grasp-up-mm", type=float, default=m[1], help="瓶子在爪中比模型高出的距离")
    parser.add_argument("--grasp-pour-mm", type=float, default=m[2], help="瓶子在爪中比模型沿倾倒方向多出的距离")
    parser.add_argument("--lip-back-mm", type=float, default=PourConfig.lip_back_mm,
                        help="唇边目标点离杯轴往回收的距离（沿倾倒方向）")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="右臂持瓶绕瓶口唇边倾倒到左天平烧杯（顶视重新定位，控制器逆解规划）")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="只读：定位烧杯并规划，不动机械臂")
    mode.add_argument("--live", action="store_true", help="现场执行")
    parser.add_argument("--until", choices=("side", "pretilt"), default=None,
                        help="与 --live 一起用：到侧悬停（side）或预倾（pretilt）后原路返回，不出液")
    parser.add_argument("--path-clear", action="store_true", help="确认倒液区域没有规划之外的器材（--live 必须）")
    parser.add_argument("--right-hold-window", type=int, nargs=2, default=(150, 210), metavar=("MIN", "MAX"),
                        help="右爪持瓶读数范围（11-23 实测 180）")
    add_pour_arguments(parser)
    parser.add_argument("--scene-image", type=Path, default=None,
                        help="用这张顶视图定位烧杯（如交接开始时两臂在拍照位拍的 01_top.png）；持瓶时瓶子会挡住目标烧杯")
    args = parser.parse_args(argv)
    if args.until and not args.live:
        parser.error("--until 只能和 --live 一起用")
    if args.live and not args.path_clear:
        parser.error("--live 需要 --path-clear")
    directory = POUR_DATA / (datetime.now().strftime("run-%Y%m%d_%H%M%S") + "-" + uuid4().hex[:10])
    directory.mkdir(parents=True)
    outcome: dict[str, Any] = {"mode": "preflight" if args.preflight else f"live:{args.until or 'full'}"}
    print(f"倒液 | 记录：{directory}", flush=True)
    started = time.perf_counter()
    try:
        passed = _run(args, directory, outcome)
    except (Exception, KeyboardInterrupt) as exc:
        outcome["error"] = str(exc) or type(exc).__name__
        print(f"已停止：{outcome['error']}", flush=True)
        passed = False
    outcome.update(success=passed, elapsed_s=round(time.perf_counter() - started, 3))
    (directory / "result.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2, default=str),
                                           encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
