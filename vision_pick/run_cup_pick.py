"""One-command vision cup pick: locate from the top camera, rim pinch, lift, put back.

Run in the robot env (dobot-xtrainer). The camera only works from .venv-calibration, so the
locate step runs there as a subprocess and hands back a JSON report.

Sequence, each motion step after a typed confirmation (Enter alone aborts and disables):
  enable + home (arm out of the camera view) -> snap tool down -> photo + locate (overlay window)
  -> pre-close jaws -> hover above the rim -> straight down into the cup -> close -> lift
  -> operator records the outcome -> straight down -> open to the pre-close width -> up
  -> back to the start pose -> disable.
Every straight leg is one continuous MovL, checked every 10 mm by the jog program's checks (IK/FK round trip, joint jump,
wrist margin, soft limits) before anything moves; space during a move stops it and the program.
The wrist never turns (see cup_locate.rim_grasp_points). Each trial is written to its own
folder plus one line in calibration-data/vision-pick/trials.jsonl.

    C:/software/anaconda3/envs/dobot-xtrainer/python.exe -X utf8 -m vision_pick.run_cup_pick --live
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from task_skills.run_cartesian_jog import (SERVO_PERIOD_S, DEFAULT_PROFILE, Jog, JogRefused, JogStopped,
                                           Limits, LiveArm, SimArm, down_tilt_deg, snap_down_rpy,
                                           start_refusal)

ROOT = Path(__file__).resolve().parents[1]
TRIALS_DIR = ROOT / "calibration-data" / "vision-pick"
VISION_PYTHON = ROOT / ".venv-calibration" / "Scripts" / "python.exe"
DEFAULTS = {"table_z": 240.18, "pre_close": 100, "depth": 15.0, "hover": 30.0, "lift": 30.0}
TRACK_MAX_AGE_S = 0.5  # an older tracking frame means "not visible now"
SANITY = {"iou_min": 0.85, "height_mm": (30.0, 250.0), "rim_radius_mm": (15.0, 80.0)}


class Aborted(RuntimeError):
    """The operator declined a step."""


class OneShotLink:
    """--no-viewer: one photo + locate subprocess, overlay in a blocking window, no live view."""

    def __init__(self, out_dir: Path, args: argparse.Namespace):
        self.out_dir, self.args = out_dir, args

    def locate(self, rpy: Sequence[float]) -> dict[str, Any]:
        args = self.args
        command = [str(args.vision_python), "-X", "utf8", "-m", "vision_pick.cup_locate",
                   "--table-z", str(args.table_z), "--arm", args.arm, "--out", str(self.out_dir),
                   "--rpy", *[str(v) for v in rpy], "--show"]
        command += ["--image", str(args.image)] if args.image else ["--camera", str(args.camera)]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"定位程序没有输出结果（退出码 {completed.returncode}）：{completed.stderr[-800:]}") from error
        if "error" in report:
            raise RuntimeError(f"定位失败：{report['error']}")
        return report

    def relocate(self, rpy: Sequence[float], reference_xy: Sequence[float] | None = None) -> dict[str, Any] | None:
        return None  # needs the live viewer (it holds the start photo and the camera)

    def start_tracking(self, table_z: float, height: float, top_radius: float,
                       bottom_radius: float | None = None) -> None:
        raise RuntimeError("跟随模式需要实时画面窗口，不能和 --no-viewer 一起用")

    def latest_track(self) -> dict[str, Any] | None:
        return None

    def stop_tracking(self) -> None:
        pass

    def status(self, step: str, tcp: Sequence[float] | None = None, plan: dict | None = None) -> None:
        pass

    def stop_requested(self) -> bool:
        return False

    def close(self) -> None:
        pass


class ViewerLink:
    """Live window process (vision_pick.viewer) that owns the camera; JSON lines both ways."""

    def __init__(self, out_dir: Path, args: argparse.Namespace, out: Callable[[str], None]):
        import queue
        import threading

        command = [str(args.vision_python), "-X", "utf8", "-m", "vision_pick.viewer", "--out", str(out_dir),
                   "--arm", args.arm]
        command += ["--image", str(args.image)] if args.image else ["--camera", str(args.camera)]
        if args.viewer_headless:
            command.append("--headless")
        self.out, self.table_z, self.arm = out, args.table_z, args.arm
        self.process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.messages: "queue.Queue[dict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.track: dict[str, Any] | None = None
        self.broken = False
        threading.Thread(target=self._read, daemon=True).start()
        self._wait("ready", 30.0)

    def _read(self) -> None:
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "stop":
                self.stop_event.set()
            elif message.get("type") == "track":
                self.track = message  # only the newest position matters; never queued
            else:
                self.messages.put(message)
        self.messages.put({"type": "error", "error": "画面程序已退出"})

    def _wait(self, kind: str, timeout: float) -> dict[str, Any]:
        import queue

        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.messages.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty as error:
                raise RuntimeError(f"画面程序 {timeout:.0f} s 内没有回应（{kind}）") from error
            if message.get("type") == "error":
                tail = ""
                if self.process.poll() is not None:
                    tail = "；" + self.process.stderr.read()[-600:]
                raise RuntimeError(f"画面程序：{message['error']}{tail}")
            if message.get("type") == kind:
                return message

    def _send(self, message: dict[str, Any]) -> None:
        if self.broken:
            return
        try:
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.broken = True
            self.out("注意：画面窗口已关闭，流程继续，但不再显示，也不能在窗口里按空格停止")

    def locate(self, rpy: Sequence[float]) -> dict[str, Any]:
        self._send({"cmd": "locate", "table_z": self.table_z, "arm": self.arm, "rpy": list(rpy)})
        return self._wait("locate", 60.0)["report"]

    def relocate(self, rpy: Sequence[float], reference_xy: Sequence[float] | None = None) -> dict[str, Any] | None:
        message = {"cmd": "relocate", "table_z": self.table_z, "arm": self.arm, "rpy": list(rpy)}
        if reference_xy is not None:
            message["reference_xy"] = [float(v) for v in reference_xy]
        self._send(message)
        return self._wait("relocate", 60.0)

    def start_tracking(self, table_z: float, height: float, top_radius: float,
                       bottom_radius: float | None = None) -> None:
        self.track = None
        self._send({"cmd": "track", "on": True, "table_z": table_z, "height": height, "top_radius": top_radius,
                    "bottom_radius": top_radius if bottom_radius is None else bottom_radius})

    def latest_track(self) -> dict[str, Any] | None:
        track = self.track
        return None if track is None else {**track, "age": time.time() - track["t"]}

    def stop_tracking(self) -> None:
        self._send({"cmd": "track", "on": False})

    def status(self, step: str, tcp: Sequence[float] | None = None, plan: dict | None = None) -> None:
        message: dict[str, Any] = {"cmd": "status", "step": step}
        if tcp is not None:
            message["tcp"] = [round(float(v), 2) for v in tcp]
        if plan is not None:
            message["plan"] = plan
        self._send(message)

    def stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def close(self) -> None:
        self._send({"cmd": "quit"})
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


TRACK_SMOOTHING = 0.5  # weight of a new clean sighting in the filtered cup position


def follow_step(command: Sequence[float], velocity: Sequence[float], goal: Sequence[float], top_speed: float,
                accel: float, gain: float, period: float) -> tuple[list[float], list[float]]:
    """One tick of a smooth follower: desired velocity = gain x error, capped by top_speed; the change
    of velocity per tick is capped by accel, so the arm neither jerks nor overshoots back and forth.
    Near the goal the speed falls off with the error and it settles exactly on the goal."""
    error = [goal[i] - command[i] for i in range(3)]
    distance = math.sqrt(sum(e * e for e in error))
    if distance < 0.02 and math.sqrt(sum(v * v for v in velocity)) < accel * period:
        return list(goal), [0.0, 0.0, 0.0]
    desired_speed = min(top_speed, gain * distance, math.sqrt(2.0 * accel * distance))
    desired = [e / distance * desired_speed for e in error] if distance > 1e-9 else [0.0, 0.0, 0.0]
    change = [desired[i] - velocity[i] for i in range(3)]
    size = math.sqrt(sum(c * c for c in change))
    limit = accel * period
    if size > limit:
        change = [c * limit / size for c in change]
    new_velocity = [velocity[i] + change[i] for i in range(3)]
    step = [new_velocity[i] * period for i in range(3)]
    if math.sqrt(sum(v * v for v in step)) >= distance:  # never step past the goal
        return list(goal), [0.0, 0.0, 0.0]
    return [round(command[i] + step[i], 4) for i in range(3)], new_velocity


def sanity(report: dict[str, Any]) -> str | None:
    fit = report["fit"]
    if fit["iou"] < SANITY["iou_min"]:
        return f"轮廓吻合度 {fit['iou']:.2f} < {SANITY['iou_min']}，分割或拟合不可信"
    low, high = SANITY["height_mm"]
    if not low <= fit["height"] <= high:
        return f"算出的杯高 {fit['height']:.1f} mm 不在 {low}-{high} 之间"
    low, high = SANITY["rim_radius_mm"]
    if not low <= fit["top_radius"] <= high:
        return f"算出的杯口半径 {fit['top_radius']:.1f} mm 不在 {low}-{high} 之间"
    return None


class CupPick:
    def __init__(self, arm: Any, args: argparse.Namespace, out_dir: Path, *,
                 ask: Callable[[str], str], out: Callable[[str], None],
                 stop_pressed: Callable[[], bool], link: Any,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.arm, self.args, self.out_dir = arm, args, out_dir
        self.ask, self.out, self.link = ask, out, link
        self.clock, self.sleep = clock, sleep
        # Space in the console or space/Esc in the viewer window: E-stop (StopScript, then disable).
        self.stop_pressed = lambda: stop_pressed() or link.stop_requested()
        self.auto = bool(getattr(args, "auto", False))
        self.log: dict[str, Any] = {"start": datetime.now().isoformat(timespec="seconds"),
                                    "arm": args.arm, "auto": bool(getattr(args, "auto", False)),
                                    "params": {k: getattr(args, k) for k in DEFAULTS},
                                    "steps": []}
        self.jog: Jog | None = None
        self.plan: dict[str, Any] | None = None

    def step(self, name: str, **fields: Any) -> None:
        self.log["steps"].append({"t": round(time.monotonic(), 2), "step": name, **fields})

    def show(self, text: str) -> None:
        tcp = self.arm.read()["pose"][:3] if self.jog is not None else None
        self.link.status(("AUTO | " if self.auto else "") + text, tcp=tcp, plan=self.plan)

    def checkpoint(self) -> None:
        """Between steps nothing is moving, so the stop key is polled here too, not only in motion."""
        if self.stop_pressed():
            self.arm.stop()
            self.show("E-STOP")
            raise JogStopped("急停：已停止，程序结束后下使能")

    def confirm(self, prompt: str, view: str) -> None:
        self.checkpoint()
        if self.auto:
            self.step("auto_confirm", prompt=prompt)
            return
        self.show(f"WAITING for y in terminal: {view}")
        if self.ask(f"{prompt}（输入 y 继续，回车中止）: ").strip().lower() != "y":
            raise Aborted(prompt)

    def pause(self, seconds: float, view: str) -> None:
        """Auto mode: a visible countdown during which the stop key still works."""
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            self.show(f"{view} in {left:.1f} s - SPACE = STOP")
            self.checkpoint()
            time.sleep(min(0.1, left))

    def travel(self, legs: list[list[float]], label: str, view: str,
               finals: list[list[float]] | None = None) -> None:
        """Each straight leg is one continuous MovL; all legs are checked before the first moves."""
        if finals is None:
            finals = self.jog.check_route(legs)
        self.checkpoint()
        self.show(view)
        self.jog.run_route(legs, finals, progress=lambda state: self.link.status(
            view, tcp=state["pose"][:3], plan=self.plan))
        self.show(view)
        state = self.arm.read()
        self.step(label, target=legs[-1], measured=[round(v, 3) for v in state["pose"][:3]])
        self.out(f"{label}：到达 {[round(v, 2) for v in legs[-1]]}")

    def vertical(self, z: float, label: str, view: str) -> None:
        self.travel([[self.jog.xyz[0], self.jog.xyz[1], round(z, 4)]], label, view)

    def gripper(self, position: int, label: str, view: str) -> None:
        self.checkpoint()
        self.show(view)
        feedback = self.arm.set_gripper(position)
        self.step(label, target=position, feedback=feedback)
        self.out(f"{label}：夹爪 {feedback.get('start')} -> {feedback.get('final')}（目标 {position}）")

    def run(self) -> str:
        args = self.args
        start = self.arm.read()
        xyz = [float(v) for v in start["pose"][:3]]
        limits =Limits(xyz, args.radius, xyz[2] - 400.0, xyz[2] + 150.0)
        self.jog = Jog(self.arm, limits, record_path=self.out_dir / "unused.csv", ask=self.ask,
                       out=self.out, stop_pressed=self.stop_pressed)
        if down_tilt_deg(self.jog.rpy) > 0.05:
            self.jog.snap_down()
        home_xyz, rpy = list(self.jog.xyz), list(self.jog.rpy)
        self.step("ready", pose=[*home_xyz, *rpy])

        self.out("拍照并定位（机械臂在初始位，不挡相机）……")
        self.show("locating cup")
        report = self.link.locate(rpy)
        self.log["vision"] = report
        fit, grasp = report["fit"], report["grasp"]
        self.out(f"杯子：中轴 ({fit['x']:.1f}, {fit['y']:.1f})，高 {fit['height']:.1f}，杯口直径 "
                 f"{2 * fit['top_radius']:.1f}，轮廓吻合度 {fit['iou']:.3f}；定位图 {report['overlay']}")
        problem = sanity(report)
        if problem:
            self.show("ABORT: implausible fit")
            raise Aborted(f"定位结果不合理：{problem}")
        rim = grasp["rim_point"]
        hover = [rim[0], rim[1], round(rim[2] + args.hover, 3)]
        grasp_z = round(rim[2] - args.depth, 3)
        grasp_floor = grasp_z - 5.0
        limits.z_min = grasp_floor  # nothing below the planned grasp depth
        lifted = round(grasp_z + args.lift, 3)
        self.plan = {"hover": hover, "grasp": [rim[0], rim[1], grasp_z]}
        self.log["plan"] = {"hover": hover, "grasp_z": grasp_z, "lift_z": lifted, "z_floor": grasp_floor}
        self.out(f"计划：悬停 {hover} -> 下降到 Z={grasp_z}（杯口下 {args.depth:.0f} mm）-> 夹取 -> 抬到 Z={lifted}")
        if self.auto:
            self.out(f"自动模式：{args.auto_delay:.0f} 秒后开始动作，看画面里红圈是否套在杯口上；"
                     "不对就按空格（终端或画面窗口）")
            self.pause(args.auto_delay, "AUTO start")

        self.confirm(f"看画面里红圈是否套在杯口上；夹爪先合拢到 {args.pre_close}", "check red rim circle, pre-close jaws")
        if self.arm.gripper_position() < args.pre_close:
            if self.auto:
                self.out(f"自动模式：夹爪将从 {self.arm.gripper_position()} 张开到 {args.pre_close}")
            self.confirm(f"夹爪当前比 {args.pre_close} 更合拢，要张开到 {args.pre_close}，确认手里没有东西",
                         "open jaws to pre-close")
        self.gripper(args.pre_close, "预合拢", "pre-closing jaws")

        grasp_centre = [fit["x"], fit["y"]]
        if args.follow:
            offset = [rim[0] - fit["x"], rim[1] - fit["y"]]  # fixed side of the rim, on the jaw axis
            follow_z = round(rim[2] + args.follow_height, 3)
            self.confirm(f"开始跟随：机械臂直接从当前位置移到抓取点外 {args.standoff:.0f} mm、杯口上方 "
                         f"{args.follow_height:.0f} mm（路线上不要有物体）", "start following")
            grasp_xy = self.follow_and_descend(report, offset, follow_z, grasp_z)
            grasp_centre = [grasp_xy[0] - offset[0], grasp_xy[1] - offset[1]]
            hover = [grasp_xy[0], grasp_xy[1], round(rim[2] + args.hover, 3)]
            self.plan = {"hover": hover, "grasp": [grasp_xy[0], grasp_xy[1], grasp_z]}
            self.log["plan"]["followed_to"] = {"rim_point": grasp_xy, "cup_centre": grasp_centre}
        else:
            legs, finals, travel_z = self.jog._plan_travel(hover, f"悬停点 {hover}", "-")
            self.confirm(f"移到杯口上方 {args.hover:.0f} mm（先在原地{'下降' if travel_z < self.jog.xyz[2] else '抬升'}"
                         f"到 Z={travel_z:.0f} 再平移，路线上不要有物体）", "move above the rim")
            self.travel(legs, "悬停", "moving to hover", finals)

            self.confirm("从侧面看：两指是否一根在杯壁内、一根在杯壁外？对了就下降", "fingers straddle the wall?")
            self.vertical(grasp_z, "下降到抓取高度", "descending into the cup")
        self.confirm("夹取", "close jaws")
        self.gripper(0, "夹取", "closing jaws")
        self.vertical(lifted, f"抬起 {args.lift:.0f} mm", "lifting")

        outcome = ""
        if self.auto:
            self.pause(args.hold, "HOLDING, look at the cup; putting back")  # result is asked at the end
        else:
            outcome = self.ask_outcome()

        self.confirm("放回：下降到原抓取高度", "lower the cup back")
        self.vertical(grasp_z, "放回", "putting back")
        self.confirm(f"张开到 {args.pre_close} 松开杯子", "release")
        self.gripper(args.pre_close, "松开", "releasing")
        self.vertical(hover[2], "抬到悬停高度", "leaving the cup")
        legs, finals, _ = self.jog._plan_travel(home_xyz, "初始位", "-")
        self.travel(legs, "回到起点", "returning", finals)
        self.measure_put_back(rpy, grasp_centre if args.follow else None)
        if not outcome:
            outcome = self.ask_outcome()
        self.show(f"DONE: {'success' if outcome == 's' else 'failure'}")
        return "成功" if outcome == "s" else "失败"

    def standoff(self, report: dict[str, Any], centre: Sequence[float], offset: Sequence[float]) -> list[float]:
        """Where to wait while following: args.standoff mm out from the grasp point, away from the camera.

        The camera looks down at an angle; anything further from its foot point projects further out
        in the image than the cup, so the waiting arm does not hide the cup it is following (it did
        when hovering right above the rim, 2026-11-04). Measured from the grasp point, not the cup
        centre: 110 mm from the centre made the arm wait ~70 mm past the grasp point and come back.
        The gripper-overlap model (checked live: predicted 17.4 %, measured 17 % above the rim) gives
        0 % at 40 mm from the grasp point for all six tested cup positions; 45 mm keeps a margin.
        """
        rim = [centre[0] + offset[0], centre[1] + offset[1]]
        foot = report.get("camera_xy")
        away = [centre[0] - foot[0], centre[1] - foot[1]] if foot else [1.0, 0.0]
        norm = math.hypot(*away) or 1.0
        return [rim[0] + self.args.standoff * away[0] / norm, rim[1] + self.args.standoff * away[1] / norm]

    def follow_and_descend(self, report: dict[str, Any], offset: Sequence[float], follow_z: float,
                           grasp_z: float) -> list[float]:
        """Follow the moving cup from beside it; approach and descend only once it has been still.

        Phases: follow (go to the standoff point beside the last clean sighting) -> hands_off (cup
        still, warn) -> approach (cross to above the grasp point at follow height) -> descend.
        Clean frame = tracking ok, fresh, and the gripper not over the cup in the image. Only clean
        frames move the target or count as "still"; a move counts only if abort_frames consecutive
        clean frames see it beyond abort_mm (one noisy or arm-biased frame made the arm cycle on
        2026-11-04). Once decided, approach and descent go on while the arm hides the cup.
        Motion: velocity follower (gain, speed cap, acceleration cap) streamed with ServoJ every
        SERVO_PERIOD_S; every point passes the jog checks; a refused point means stop and hold.
        Starts straight from where the arm is, so it reacts to the cup from the first tick.
        """
        args, jog = self.args, self.jog
        fit = report["fit"]
        self.link.start_tracking(fit["table_z"], fit["height"], fit["top_radius"], fit["bottom_radius"])
        command, joints = list(jog.xyz), list(jog.joints)
        velocity = [0.0, 0.0, 0.0]
        centre = [fit["x"], fit["y"]]            # filtered cup axis from clean frames
        history: list[tuple[float, list[float]]] = []
        phase, phase_until, target_xy = "follow", 0.0, None
        moving_frames, shifted_frames, last_track_t = 0, 0, None
        started, ticks, last_refusal = self.clock(), 0, ""
        telemetry: list[list[Any]] = []
        self.out(f"跟随中（≤{args.follow_speed:.0f} mm/s）：机械臂停在抓取点外 {args.standoff:.0f} mm、杯口上方 "
                 f"{args.follow_height:.0f} mm；杯子静止 {args.settle:.1f} 秒后提示把手拿开，{args.hands_off:.1f} 秒后过去抓取")
        try:
            while True:
                if self.stop_pressed():
                    self.arm.stop()
                    raise JogStopped("急停：跟随已停止")
                now = self.clock()
                if now - started > args.follow_timeout:
                    raise Aborted(f"跟随 {args.follow_timeout:.0f} 秒仍未完成下降")
                track = self.link.latest_track()
                new = bool(track) and track.get("t") != last_track_t
                clean = bool(track and track.get("ok") and not track.get("arm_overlap")
                             and track.get("age", 99.0) < TRACK_MAX_AGE_S)
                if new:
                    last_track_t = track.get("t")
                covered = bool(new and track.get("ok") and track.get("arm_overlap")
                               and track.get("age", 99.0) < TRACK_MAX_AGE_S)
                if covered:
                    # Gripper over the cup: the position is biased by a few mm, so only a big shift counts.
                    raw = list(track["xy"])
                    if target_xy is not None:
                        far = math.dist([raw[0] + offset[0], raw[1] + offset[1]], target_xy) > args.abort_mm_covered
                        moving_frames = moving_frames + 1 if far else 0
                    # A big, persistent shift also moves the cup estimate. Otherwise a cup pushed next to
                    # the waiting gripper is never seen clean again and the arm waits beside its old
                    # place forever (found with the 45 mm standoff).
                    shifted_frames = shifted_frames + 1 if math.dist(raw, centre) > args.abort_mm_covered else 0
                    if shifted_frames >= args.abort_frames:
                        centre, history, shifted_frames = raw, [], 0
                elif new:
                    shifted_frames = 0
                if clean and new:
                    raw = list(track["xy"])
                    if math.dist(raw, centre) > args.deadband:  # ignore sensor noise when the cup is still
                        centre = [centre[i] + TRACK_SMOOTHING * (raw[i] - centre[i]) for i in range(2)]
                    history.append((now, [centre[0] + offset[0], centre[1] + offset[1]]))
                    history = [h for h in history if now - h[0] <= args.settle + 1.0]
                    if target_xy is not None:
                        far = math.dist([raw[0] + offset[0], raw[1] + offset[1]], target_xy) > args.abort_mm
                        moving_frames = moving_frames + 1 if far else 0
                rim_xy = [centre[0] + offset[0], centre[1] + offset[1]]
                moved = moving_frames >= args.abort_frames
                top_speed = args.follow_speed
                if phase == "follow":
                    goal = [*self.standoff(report, centre, offset), follow_z]
                    arrived = math.dist(goal, command) < 2.0
                    if arrived and self.settled(history) and math.dist(history[-1][1], rim_xy) < 1.0:
                        phase, phase_until, target_xy, moving_frames = "hands_off", now + args.hands_off, list(rim_xy), 0
                        self.out(f"杯子已静止，{args.hands_off:.1f} 秒后过去抓取，请把手拿开")
                elif phase == "hands_off":
                    goal = list(command)
                    if moved:
                        phase, target_xy, moving_frames = "follow", None, 0
                        self.out("杯子又动了，继续跟随")
                    elif now >= phase_until:
                        phase = "approach"
                elif phase == "approach":
                    goal = [*target_xy, follow_z]
                    if moved:
                        phase, target_xy, moving_frames = "follow", None, 0
                        self.out("杯子动了，回到旁边继续跟随")
                    elif math.dist(goal, command) < 0.1:
                        phase = "descend"
                elif phase == "descend":
                    goal, top_speed = [*target_xy, grasp_z], args.descend_speed
                    if moved:
                        phase, target_xy, moving_frames = "retreat", None, 0
                        self.out("下降时杯子动了，先抬回跟随高度")
                        goal = [command[0], command[1], follow_z]
                    elif abs(command[2] - grasp_z) < 0.1 and math.dist(command[:2], target_xy) < 0.1:
                        self.step("followed", target=[*target_xy, grasp_z], ticks=ticks,
                                  seconds=round(now - started, 1))
                        self.save_telemetry(telemetry)
                        return list(target_xy)
                else:  # retreat: straight up first, then back to following from the side
                    goal = [command[0], command[1], follow_z]
                    if abs(command[2] - follow_z) < 0.5:
                        phase, history = "follow", []
                point, velocity = follow_step(command, velocity, goal, top_speed, args.follow_accel,
                                              args.follow_gain, SERVO_PERIOD_S)
                if math.dist(point, command) > 1e-4:
                    try:
                        joints = jog.check(point, jog.rpy, joints)
                    except JogRefused as refusal:
                        velocity = [0.0, 0.0, 0.0]  # stop at the last reachable point
                        if str(refusal) != last_refusal:
                            self.out(f"跟随目标不可达，原地等待：{refusal}")
                            last_refusal = str(refusal)
                    else:
                        self.arm.servo(point, list(jog.rpy), joints, SERVO_PERIOD_S)
                        command, last_refusal = point, ""
                telemetry.append([round(now - started, 3), phase, [round(v, 2) for v in command],
                                  track.get("xy") if track else None, clean,
                                  track.get("quality") if track else None,
                                  track.get("overlap") if track else None, moving_frames])
                ticks += 1
                if ticks % 3 == 0:
                    seen = "" if clean else " (cup not clear - using last clear view)"
                    view = {"follow": "FOLLOWING beside the cup" + seen,
                            "hands_off": "HANDS OFF - grasping soon", "approach": "APPROACHING - hands off",
                            "descend": "DESCENDING - hands off", "retreat": "cup moved - lifting back"}[phase]
                    aim = target_xy or rim_xy
                    self.plan = {"hover": [aim[0], aim[1], follow_z], "grasp": [aim[0], aim[1], grasp_z]}
                    self.link.status(("AUTO | " if self.auto else "") + view, tcp=command, plan=self.plan)
                self.sleep(max(0.0, SERVO_PERIOD_S - (self.clock() - now)))
        finally:
            self.save_telemetry(telemetry)
            self.arm.servo_end()
            self.link.stop_tracking()
            state = self.arm.settle()
            jog.xyz = [round(v, 4) for v in state["pose"][:3]]
            jog.joints = list(state["joints"])

    def save_telemetry(self, rows: list[list[Any]]) -> None:
        """One line per control tick: t, phase, commanded xyz, tracked xy, clean, quality, overlap, moving."""
        if rows:
            with (self.out_dir / "follow_log.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(["t", "phase", "command", "track_xy", "clean", "quality", "overlap",
                                         "moving_frames"]) + "\n")
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def settled(self, history: list[tuple[float, list[float]]]) -> bool:
        """Still for args.settle seconds: the clean positions in that window stay within settle_mm.

        The window ends at the last clean frame, so if the rim is then hidden (e.g. by the gripper
        above it) an untouched cup can still be grasped.
        """
        if not history:
            return False
        last = history[-1][0]
        window = [(t, xy) for t, xy in history if t >= last - self.args.settle]
        if last - window[0][0] < 0.9 * self.args.settle or len(window) < 5:
            return False
        points = [xy for _, xy in window]
        centre = [sum(p[i] for p in points) / len(points) for i in range(2)]
        return max(math.dist(p, centre) for p in points) <= self.args.settle_mm

    def measure_put_back(self, rpy: Sequence[float], reference_xy: Sequence[float] | None = None) -> None:
        """Arm back at home: photograph again and log how far the cup moved from where it was grasped."""
        self.show("measuring where the cup was put back")
        try:
            result = self.link.relocate(rpy, reference_xy)
        except RuntimeError as error:  # a fallen or hidden cup must not fail the trial record
            self.log["put_back"] = {"error": str(error)}
            self.out(f"放回位置没有测出来：{error}")
            return
        if result is None:
            return
        self.log["put_back"] = result["put_back"]
        self.log["vision_end"] = result["report"]
        error = result["put_back"]
        self.out(f"放回偏差：杯口中心移动了 {error['distance_mm']:.1f} mm（dx {error['dx']:+.1f}，dy {error['dy']:+.1f}）")

    def ask_outcome(self) -> str:
        self.show("WAITING: type s / f in terminal")
        outcome = ""
        while outcome not in ("s", "f"):
            outcome = self.ask("结果：s = 夹稳抬起 / f = 失败（滑落、夹歪、碰倒等）: ").strip().lower()
        note = self.ask("备注（可空，例如：偏左 3 mm、杯子被推动）: ").strip()
        self.log["outcome"] = {"success": outcome == "s", "note": note}
        return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视觉定位杯子并抓取（杯口一指内一指外），抬起后放回")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--sim", action="store_true", help="离线模拟，需配合 --image")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--radius", type=float, default=330.0)
    parser.add_argument("--table-z", type=float, default=DEFAULTS["table_z"], help="桌面在该臂基座中的 Z（mm）")
    parser.add_argument("--pre-close", type=int, default=DEFAULTS["pre_close"], help="下降前夹爪合拢到的值 0-255")
    parser.add_argument("--depth", type=float, default=DEFAULTS["depth"], help="指尖伸进杯口的深度 mm")
    parser.add_argument("--hover", type=float, default=DEFAULTS["hover"], help="悬停在杯口上方 mm")
    parser.add_argument("--lift", type=float, default=DEFAULTS["lift"], help="夹取后抬起 mm")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=Path, help="用已有图像代替现场拍照")
    parser.add_argument("--auto", action="store_true",
                        help="自动模式：跳过每一步的 y 确认（启动时的安全确认仍保留）；空格 = 急停")
    parser.add_argument("--auto-delay", type=float, default=3.0, help="自动模式：定位后等几秒再动，期间可急停")
    parser.add_argument("--hold", type=float, default=2.0, help="自动模式：抬起后停留几秒再放回")
    parser.add_argument("--follow", action="store_true",
                        help="跟随模式：抓取前用手移动杯子，机械臂在杯口上方跟随；杯子静止后才下降")
    parser.add_argument("--follow-height", type=float, default=60.0, help="跟随时指尖在杯口上方的高度 mm")
    parser.add_argument("--follow-speed", type=float, default=100.0, help="跟随时的最大速度 mm/s")
    parser.add_argument("--follow-accel", type=float, default=500.0, help="跟随时的最大加速度 mm/s²")
    parser.add_argument("--follow-gain", type=float, default=4.0,
                        help="跟随响应：每秒追上剩余距离的倍数（越大越快，太大会抖）")
    parser.add_argument("--deadband", type=float, default=0.8, help="杯子位置变化小于这个值（mm）视为噪声")
    parser.add_argument("--abort-frames", type=int, default=6, help="连续几帧看到杯子被移动才算真的移动")
    parser.add_argument("--abort-mm-covered", type=float, default=20.0,
                        help="夹爪挡住杯子时，杯子移动超过这个值（mm）才算真的移动（被挡时位置有几毫米偏差）")
    parser.add_argument("--standoff", type=float, default=45.0,
                        help="跟随时停在抓取点外多远（远离相机一侧，避免挡住杯子）mm")
    parser.add_argument("--descend-speed", type=float, default=30.0, help="跟随模式下降速度 mm/s")
    parser.add_argument("--settle", type=float, default=1.2, help="杯子静止多少秒才过去抓取")
    parser.add_argument("--settle-mm", type=float, default=3.0, help="静止判定：这段时间内位置变化不超过 mm")
    parser.add_argument("--abort-mm", type=float, default=8.0, help="过去抓取途中杯子移动超过 mm 就回去跟随")
    parser.add_argument("--hands-off", type=float, default=0.8, help="过去抓取前提示把手拿开的秒数")
    parser.add_argument("--follow-timeout", type=float, default=120.0, help="跟随最长秒数")
    parser.add_argument("--no-viewer", action="store_true",
                        help="不开实时画面窗口（改为拍一张照、弹出定位图，按任意键关闭）")
    parser.add_argument("--viewer-headless", action="store_true", help=argparse.SUPPRESS)  # offline smoke runs
    parser.add_argument("--vision-python", type=Path, default=VISION_PYTHON)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=TRIALS_DIR)
    return parser


def main(argv: Sequence[str] | None = None, *, ask: Callable[[str], str] = input,
         out: Callable[[str], None] = print, arm: Any = None, link: Any = None,
         stop_pressed: Callable[[], bool] | None = None,
         clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> int:
    args = build_parser().parse_args(argv)
    if args.sim and args.image is None and arm is None:
        out("--sim 需要 --image")
        return 2
    trial_dir = args.out / datetime.now().strftime("trial-%Y%m%d_%H%M%S")
    trial_dir.mkdir(parents=True, exist_ok=True)
    if stop_pressed is None:
        if sys.platform == "win32":
            from task_skills.run_cartesian_jog import _windows_keys
            stop_pressed = _windows_keys()[1]
        else:
            stop_pressed = lambda: False  # noqa: E731
    pick: CupPick | None = None
    result, code = "未开始", 1
    try:
        if arm is None:
            if args.sim:
                arm = SimArm([-117.6, -344.8, 415.5], snap_down_rpy([180.0, 0.0, 89.97]))
            else:
                profile = json.loads(args.profile.read_text(encoding="utf-8"))
                arm = LiveArm(profile, args.arm, args.speed, log=out)
                problem = start_refusal(arm.read().get("mode"))
                if problem:
                    out(f"拒绝启动，没有发送任何指令：{problem}")
                    return 5
                if ask(f"确认 {args.arm} 臂周围无人、急停在手边、另一只臂停在初始位不挡相机，输入 {args.arm} 继续: "
                       ).strip() != args.arm:
                    out("未确认，已退出，没有发送任何指令")
                    return 1
                out(arm.open_session(home=True))  # home = layout.physical_nova_home_degrees
        if link is None:
            if args.no_viewer:
                link = OneShotLink(trial_dir, args)
            else:
                out("打开实时画面窗口（相机预热约 1 秒）……窗口里按空格 = 停止机械臂")
                link = ViewerLink(trial_dir, args, out)
        pick = CupPick(arm, args, trial_dir, ask=ask, out=out, stop_pressed=stop_pressed, link=link,
                       clock=clock, sleep=sleep)
        result = pick.run()
        code = 0
    except Aborted as reason:
        result, code = f"中止：{reason}", 1
        out(f"已中止（{reason}），机械臂停在当前位置")
    except JogRefused as refusal:
        result, code = f"拒绝：{refusal}", 4
        out(f"拒绝，未发送这一步的运动：{refusal}")
    except JogStopped as stopped:
        result, code = "空格停止", 3
        out(str(stopped))
    except RuntimeError as error:
        result, code = f"错误：{error}", 4
        out(f"运行错误，已停止：{error}")
    finally:
        if arm is not None:
            arm.close()
        if link is not None:
            try:
                link.status("END: " + {0: "trial finished", 1: "aborted", 3: "stopped (space)",
                                       4: "refused or error"}.get(code, "ended") + " - arm disabled")
            except Exception:  # noqa: BLE001 - the viewer may already be gone
                pass
            link.close()
        if pick is not None:
            pick.log.update({"end": datetime.now().isoformat(timespec="seconds"), "result": result})
            (trial_dir / "trial.json").write_text(json.dumps(pick.log, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
            with (args.out / "trials.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"dir": trial_dir.name, "result": result,
                                         "outcome": pick.log.get("outcome"),
                                         "put_back_mm": (pick.log.get("put_back") or {}).get("distance_mm"),
                                         "fit": (pick.log.get("vision") or {}).get("fit")},
                                        ensure_ascii=False) + "\n")
    out(f"本次结果：{result}；记录在 {trial_dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
