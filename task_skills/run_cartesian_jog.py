"""Keyboard Cartesian jog for one Nova arm, with the tool orientation locked.

Why this exists: touching chessboard corners for the camera-to-base calibration
needs small, repeatable XYZ moves with the tool pointing straight down. A tap is
one MovL of one step; the RPY is copied unchanged, so the orientation never
drifts. Movement keys held together combine into one diagonal step (each active
axis moves by the step size): the key event comes from the console, which proves
it has focus, and the other held keys are then sampled with GetAsyncKeyState.
Nothing is sent until the key is either released (a tap: one step) or still held
after HOLD_DELAY_S (a hold). A single-axis hold uses the controller's own
Cartesian MoveJog, as DobotStudio does, and MoveJog() stops it the moment the key
is released, the axis changes, the soft limit comes within the predicted stopping
distance, an alarm appears or the program fails. The controller keeps the
orientation during a Cartesian jog; the drift is measured afterwards and reported.
A multi-axis hold has no native jog, so it streams ServoJ at SERVO_PERIOD_S, the
same follow command the X-Trainer teleop uses on this rig: every tick the next
Cartesian point (locked RPY) is solved and checked like a key target, and the
stream simply ends on release, so the arm stops within about one tick.

The joints are never commanded directly. Each target is solved by the
controller's InverseSolution, seeded with the previous solution, and accepted
only if the forward solution reproduces the target, no joint jumps by more than
MAX_JOINT_STEP_DEG (an IK branch switch) and J5 keeps WRIST_MARGIN_DEG from the
wrist singularity. A refused key sends nothing.

Live use reuses LiveDriver's session without its ClearError: device lease ->
EnableRobot -> speed -> Tool/User. An alarm is never cleared here. The program
refuses to start unless RobotMode is 4 (disabled) or 5 (enabled idle), so an alarm
has to be read and cleared at the pendant first, and an alarm during a move ends
the program at once. Power-on is never sent. After enabling, the arm JointMovJ's
to the calibrated home in layout.physical_nova_home_degrees (the only source of
that target, shared with the atoms; its tool Z points down), unless it is already
there. Every exit after the session opened ends with DisableRobot. The gripper
moves only on o/c/g, through the same command and feedback wait as the atom, and
anything that opens it needs a typed y; exiting leaves it as it is. Space during
a move stops it and ends the program. Keys
typed while the arm moves are discarded, so a held key cannot queue extra steps.

    python -X utf8 -m task_skills.run_cartesian_jog --arm right --sim
    python -X utf8 -m task_skills.run_cartesian_jog --arm right --live
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from .chemistry import geometry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = Path(__file__).resolve().parent / "chemistry" / "profile.json"
DEFAULT_RECORD_DIR = ROOT / "calibration-data" / "jog-points"

STEPS_MM = {"1": 0.1, "2": 1.0, "3": 5.0, "4": 10.0}
MOVES = {  # key -> (axis index, sign) in the arm's base frame
    "d": (0, +1), "a": (0, -1), "RIGHT": (0, +1), "LEFT": (0, -1),
    "w": (1, +1), "s": (1, -1), "UP": (1, +1), "DOWN": (1, -1),
    "r": (2, +1), "f": (2, -1), "PGUP": (2, +1), "PGDN": (2, -1),
}
AXIS_NAMES = "XYZ"
VIRTUAL_KEYS = {"d": 0x44, "a": 0x41, "w": 0x57, "s": 0x53, "r": 0x52, "f": 0x46,
                "RIGHT": 0x27, "LEFT": 0x25, "UP": 0x26, "DOWN": 0x28,
                "PGUP": 0x21, "PGDN": 0x22}

MAX_JOINT_STEP_DEG = 15.0
WRIST_MARGIN_DEG = 5.0
FK_POSITION_TOL_MM = 0.1
FK_ORIENTATION_TOL_DEG = 0.1
SNAP_LIMIT_DEG = 15.0
ARRIVAL_TOL_MM = 0.5
ARRIVAL_SAMPLES = 3
ARRIVAL_TIMEOUT_S = 30.0
POLL_S = 0.1
SPEED_RANGE = (1, 30)
ENABLE_TIMEOUT_S = 5.0
DISABLE_TIMEOUT_S = 5.0
DEFAULT_Z_DROP_MM = 400.0   # home is ~415 mm up; the table is unknown, so use the z key
FLOOR_BELOW_TOUCH_MM = 0.5
HOLD_DELAY_S = 0.3
SERVO_PERIOD_S = 0.03       # ServoJ t, as in dobot_control/robots/dobot.py
SERVO_MM_S_PER_PERCENT = 2.0  # diagonal speed: --speed 10 -> 20 mm/s
HOLD_POLL_S = 0.03
JOG_STOP_HORIZON_S = 0.25   # look this far ahead at the measured speed before a limit
JOG_STOP_MARGIN_MM = 3.0    # and never closer than this
SETTLE_TOL_MM = 0.05
ORIENTATION_DRIFT_WARN_DEG = 0.5
GOTO_HOVER_MM = 20.0        # t key: stop this far above the predicted corner
# Travel heights above the hover point, tried in order. Home is ~415 mm up, where the far
# corners are out of reach, so t first changes height over the start XY, crosses at
# hover + clearance, then lowers onto the hover point. Lower crossings are tried only
# when a higher one is unreachable.
GOTO_CLEARANCES_MM = (60.0, 40.0, 20.0, 0.0)
GOTO_SEGMENT_MM = 20.0      # and travel in checked MovL pieces no longer than this
GOTO_TILT_LIMIT_DEG = 1.0   # t needs the tool already snapped down (v)

# RobotMode values from the Dobot TCP/IP protocol (V3 controller).
ROBOT_MODES = {1: "初始化", 2: "抱闸松开", 3: "未上电", 4: "未使能", 5: "使能空闲", 6: "拖动",
               7: "运行中", 8: "轨迹录制", 9: "报警", 10: "暂停", 11: "点动"}
STARTABLE_MODES = (4, 5)
ALARM_MODE = 9


def start_refusal(mode: Any) -> str | None:
    """Why the arm may not be taken over in this RobotMode, or None."""
    if mode in STARTABLE_MODES:
        return None
    label = f"RobotMode={mode}（{ROBOT_MODES.get(mode, '未知')}）"
    if mode == ALARM_MODE:
        return f"{label}：请先在示教器上查看报警原因并手动清除，本程序不会清报警"
    if mode == 3:
        return f"{label}：请先用示教器或 DobotStudio 上电，本程序不会上电"
    return f"{label}：只在未使能(4)或使能空闲(5)时接管"

HELP = """\
按键（基座坐标系，先用 2 档 1 mm 试出方向）:
  d/a 或 →/←   X+ / X-        w/s 或 ↑/↓   Y+ / Y-
  r/f 或 PgUp/PgDn  Z+ / Z-（r 抬起，f 下降）
  同时按住多个方向键走斜线，例如 w+a = Y+ 且 X-（每个轴各走一个步长）
  轻点走一步；按住超过 0.3 s 连续移动，松手即停（单轴用控制器点动，斜线用 ServoJ 跟随）
  o / c        夹爪张开（需确认）/ 闭合      g  夹爪设到指定值 0-255（0 闭合，255 张开）
  1/2/3/4      步长 0.1 / 1 / 5 / 10 mm
  v            把工具摆成竖直朝下（偏差 ≤ 15° 才允许）
  p            记录当前 TCP 为标定角点（写入 CSV）
  t            输入角点编号，移到拟合预测位置正上方 20 mm（需 --session；原地升降到棋盘上方约 80 mm 再平移再下降）
               到了以后必须自己看准真实角点、平移修正后再下降，不能直接按预测位置记录
  z            把 Z 下限设到当前高度下 0.5 mm（尖端刚碰到棋盘时按）
  i            显示当前状态        h  显示本帮助
  q 或 Esc     退出（下使能；不会断电）
  运动中按空格  StopScript 并退出"""


class JogRefused(RuntimeError):
    """A key whose target failed a check; nothing was sent."""


class JogStopped(RuntimeError):
    """The operator pressed space during a move."""


def _rotation_angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    """Geodesic angle between two controller RPY orientations."""
    a = geometry.euler_zyx_to_matrix(first)
    b = geometry.euler_zyx_to_matrix(second)
    trace = sum(a[row][col] * b[row][col] for row in range(3) for col in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def down_tilt_deg(rpy: Sequence[float]) -> float:
    """Angle between the tool Z axis and straight down."""
    return geometry.angle_between_degrees(geometry.tool_axis(rpy, "Z"), (0.0, 0.0, -1.0))


def snap_down_rpy(rpy: Sequence[float]) -> list[float]:
    """Nearest orientation with tool Z straight down, keeping the tool X heading."""
    x = geometry.tool_axis(rpy, "X")
    norm = math.hypot(x[0], x[1])
    if norm < 1e-6:
        raise JogRefused("tool X is vertical, so the heading is undefined")
    x = (x[0] / norm, x[1] / norm, 0.0)
    z = (0.0, 0.0, -1.0)
    y = (z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0])
    matrix = tuple((x[row], y[row], z[row]) for row in range(3))
    return [round(value, 6) for value in geometry.matrix_to_euler_zyx(matrix)]


class Limits:
    """Soft box around the start position; the controller's own limits still apply."""

    def __init__(self, start_xyz: Sequence[float], radius_mm: float, z_min: float, z_max: float):
        self.center = (float(start_xyz[0]), float(start_xyz[1]))
        self.radius = float(radius_mm)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def violation(self, xyz: Sequence[float]) -> str | None:
        distance = math.hypot(xyz[0] - self.center[0], xyz[1] - self.center[1])
        if distance > self.radius:
            return f"水平距起点 {distance:.1f} mm，超过半径 {self.radius:.0f} mm"
        if xyz[2] < self.z_min:
            return f"Z={xyz[2]:.2f} 低于下限 {self.z_min:.2f}"
        if xyz[2] > self.z_max:
            return f"Z={xyz[2]:.2f} 高于上限 {self.z_max:.2f}"
        return None

    def describe(self) -> str:
        return (f"水平半径 {self.radius:.0f} mm（圆心 X={self.center[0]:.1f}, Y={self.center[1]:.1f}），"
                f"Z ∈ [{self.z_min:.1f}, {self.z_max:.1f}] mm")


class SimArm:
    """Offline stand-in: instant moves and a deterministic fake IK."""

    name = "sim"

    def __init__(self, xyz: Sequence[float], rpy: Sequence[float]):
        self.pose = [*map(float, xyz), *map(float, rpy)]
        self.joints = self.solve(xyz, rpy, None)
        self.sent: list[tuple[list[float], list[float]]] = []
        self.jogs: list[str] = []
        self.jogging: tuple[int, int] | None = None
        self.jog_mm_per_read = 0.5
        self.servos: list[Any] = []
        self.gripper_pos = 255

    def read(self) -> dict[str, Any]:
        if self.jogging is not None:  # a native jog advances a little per look
            axis, sign = self.jogging
            self.pose[axis] = round(self.pose[axis] + sign * self.jog_mm_per_read, 6)
            self.joints = self.solve(self.pose[:3], self.pose[3:], None)
        return {"joints": list(self.joints), "pose": list(self.pose), "mode": 5}

    def jog(self, label: str) -> None:
        self.jogs.append(label)
        self.jogging = ("XYZ".index(label[0]), +1 if label[1] == "+" else -1)

    def jog_stop(self) -> None:
        self.jogs.append("stop")
        self.jogging = None

    def settle(self) -> dict[str, Any]:
        return self.read()

    def servo(self, xyz, rpy, joints, period: float) -> None:
        self.servos.append(list(xyz))
        self.pose = [*xyz, *rpy]
        self.joints = list(joints)

    def servo_end(self) -> None:
        self.servos.append("end")

    def gripper_position(self) -> int:
        return self.gripper_pos

    def set_gripper(self, position: int) -> dict[str, Any]:
        before, self.gripper_pos = self.gripper_pos, position
        return {"start": before, "target": position, "final": position}

    def solve(self, xyz, rpy, seed) -> list[float]:
        return [xyz[0] / 10.0, xyz[1] / 10.0, xyz[2] / 10.0, 0.0, 90.0, float(rpy[2])]

    def forward(self, joints) -> dict[str, list[float]]:
        # The fake IK cannot be inverted for RPY, so echo the last solved target.
        return {"xyz": [joints[0] * 10.0, joints[1] * 10.0, joints[2] * 10.0],
                "rpy": list(self._last_rpy)}

    def plan(self, xyz, rpy, seed) -> list[float]:
        self._last_rpy = list(rpy)
        return self.solve(xyz, rpy, seed)

    def move(self, xyz, rpy, joints) -> None:
        self.sent.append((list(xyz), list(rpy)))
        self.pose = [*xyz, *rpy]  # an instant arm
        self.joints = list(joints)

    def wait(self, xyz, rpy, joints, stop_pressed: Callable[[], bool]) -> dict[str, Any]:
        self.pose = [*xyz, *rpy]
        self.joints = list(joints)
        return self.read()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class LiveArm:
    """One arm through LiveDriver's lease, strict echo check and state reads."""

    name = "live"

    def __init__(self, profile: dict[str, Any], arm: str, speed: int,
                 log: Callable[[str], None] = print):
        from .chemistry.driver import LiveDriver

        self.profile = profile
        self.arm = arm
        self.speed = speed
        self.log = log
        self.session_open = False
        self.driver = LiveDriver(profile, allow_live=True, confirmation=lambda: True, log=log)
        self.rig = self.driver._rig(arm, profile, with_motion=False)

    def read(self) -> dict[str, Any]:
        return self.driver._read_state(self.rig)

    def _wait_mode(self, wanted: int, timeout: float, what: str) -> None:
        deadline = time.time() + timeout
        while True:
            mode = self.read()["mode"]
            if mode == wanted:
                return
            if (mode == ALARM_MODE and wanted != ALARM_MODE) or time.time() > deadline:
                raise RuntimeError(f"{what}后 RobotMode={mode}（{ROBOT_MODES.get(mode, '未知')}），"
                                   f"未进入 {wanted}（{ROBOT_MODES[wanted]}）")
            time.sleep(POLL_S)

    def open_session(self, *, home: bool = True) -> str:
        """Enable without clearing alarms, then park at the calibrated home."""
        from .chemistry.driver import ARRIVAL_TOL_DEG, _reply_status, angle_delta
        from .layout import physical_nova_home_degrees

        # Re-read right before enabling: the operator may have taken time to confirm.
        problem = start_refusal(self.read()["mode"])
        if problem:
            raise RuntimeError(problem)
        self.rig = self.driver._ensure_session(self.profile, self.arm, self.speed,
                                               clear_errors=False)
        self.session_open = True
        self._wait_mode(5, ENABLE_TIMEOUT_S, "使能")
        note = "跳过回初始位（--no-home）"
        if home:
            target = list(physical_nova_home_degrees(self.arm))
            joints = self.read()["joints"]
            worst = max(abs(angle_delta(joints[i], target[i])) for i in range(6))
            if worst <= ARRIVAL_TOL_DEG:
                note = f"已在初始位 {target}（最大关节偏差 {worst:.3f}°），未发送运动"
            else:
                self.log(f"{self.arm}: JointMovJ 回初始位 {target}")
                self.driver._configure_speed(self.rig, self.speed, linear=False)
                self.driver._move_to_home(self.rig, self.arm, None)
                note = f"已回到初始位 {target}"
        self.driver._configure_speed(self.rig, self.speed, linear=True)
        return note

    def disable(self) -> str:
        from .chemistry.driver import _reply_status

        # Let a stopped move settle first, so the brakes do not catch a moving arm.
        deadline = time.time() + 3.0
        while self.read()["mode"] == 7 and time.time() < deadline:
            time.sleep(POLL_S)
        if _reply_status(self.rig.dashboard.DisableRobot()) != 0:
            raise RuntimeError("DisableRobot 未得到 status 0 应答")
        self._wait_mode(4, DISABLE_TIMEOUT_S, "下使能")
        return "已下使能（RobotMode=4）"

    def plan(self, xyz, rpy, seed) -> list[float]:
        return self.driver._inverse_solution(self.rig, xyz, rpy, seed, self.profile)

    def forward(self, joints) -> dict[str, list[float]]:
        return self.driver._forward_solution(self.rig, joints, self.profile)

    def move(self, xyz, rpy, joints) -> None:
        from .chemistry.driver import linear_frame

        self.driver._send_motion(self.rig, linear_frame(xyz, rpy), "jog MovL")

    def wait(self, xyz, rpy, joints, stop_pressed: Callable[[], bool]) -> dict[str, Any]:
        deadline = time.time() + ARRIVAL_TIMEOUT_S
        stable = 0
        while time.time() < deadline:
            if stop_pressed():
                self.stop()
                raise JogStopped("空格：已发送 StopScript")
            state = self.read()
            if state["mode"] == ALARM_MODE:
                raise RuntimeError("运动中控制器报警（RobotMode=9），请到示教器查看原因；本程序不会清报警")
            error = max(abs(state["pose"][axis] - xyz[axis]) for axis in range(3))
            stable = stable + 1 if state["mode"] == 5 and error <= ARRIVAL_TOL_MM else 0
            if stable >= ARRIVAL_SAMPLES:
                return state
            time.sleep(POLL_S)
        self.stop()
        raise RuntimeError(f"{ARRIVAL_TIMEOUT_S:.0f} s 内未到达目标，已发送 StopScript")

    def jog(self, label: str) -> None:
        # Default CoordType=1 (user frame), User0/Tool0 per the Dobot SDK docstring.
        self.driver._send_motion(self.rig, f"MoveJog({label})", f"MoveJog {label}")

    def jog_stop(self) -> None:
        self.driver._send_motion(self.rig, "MoveJog()", "MoveJog stop")

    def servo(self, xyz, rpy, joints, period: float) -> None:
        # Fire-and-forget like the teleop follower; servo_end drains the replies.
        self.rig.motion.ServoJ(*[float(value) for value in joints], float(period))

    def servo_end(self) -> None:
        """Drain the ServoJ replies so the next strict-echo command reads only its own."""
        import socket

        sock = self.rig.motion.socket_dobot
        previous = sock.gettimeout()
        time.sleep(0.15)
        try:
            sock.settimeout(0.05)
            while True:
                try:
                    if not sock.recv(65536):
                        break
                except socket.timeout:
                    break
        finally:
            sock.settimeout(previous)

    def gripper_position(self) -> int:
        return int(self.driver._gripper(self.rig, self.profile).get_current_position())

    def set_gripper(self, position: int) -> dict[str, Any]:
        """Same command and feedback wait as the gripper.set_position atom."""
        from .chemistry.choreography import GRIPPER_SPEED

        before = self.gripper_position()
        self.driver._gripper_move(self.rig, self.profile, position, GRIPPER_SPEED)
        return self.driver._wait_for_gripper(self.rig, self.profile, before, position)

    def settle(self) -> dict[str, Any]:
        """Wait until a stopped jog is enabled-idle and no longer moving."""
        deadline = time.time() + ARRIVAL_TIMEOUT_S
        previous: list[float] | None = None
        stable = 0
        while time.time() < deadline:
            state = self.read()
            if state["mode"] == ALARM_MODE:
                raise RuntimeError("点动中控制器报警（RobotMode=9），请到示教器查看原因；本程序不会清报警")
            moved = math.dist(state["pose"][:3], previous) if previous else math.inf
            stable = stable + 1 if state["mode"] == 5 and moved <= SETTLE_TOL_MM else 0
            if stable >= ARRIVAL_SAMPLES:
                return state
            previous = state["pose"][:3]
            time.sleep(POLL_S)
        self.stop()
        raise RuntimeError(f"点动停止后 {ARRIVAL_TIMEOUT_S:.0f} s 内未静止，已发送 StopScript")

    def stop(self) -> None:
        self.driver.stop([self.arm])

    def close(self) -> None:
        """Disable on every exit once enabled, then release the lease."""
        try:
            if self.session_open:
                try:
                    self.log(self.disable())
                except Exception as error:  # noqa: BLE001 - the lease must still be released
                    self.log(f"下使能失败，请在示教器上手动下使能：{type(error).__name__}: {error}")
        finally:
            self.driver.close()


class Jog:
    """Key handling and the per-target checks; the arm backend does the I/O."""

    def __init__(self, arm: Any, limits: Limits, *, record_path: Path,
                 ask: Callable[[str], str] = input, out: Callable[[str], None] = print,
                 stop_pressed: Callable[[], bool] = lambda: False,
                 discard_keys: Callable[[], None] = lambda: None,
                 held_moves: Callable[[], set[str]] = set,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 servo_speed: float = 10 * SERVO_MM_S_PER_PERCENT, targets: Any = None):
        state = arm.read()
        self.targets = targets
        self.held_moves = held_moves
        self.clock = clock
        self.sleep = sleep
        self.hold_delay = HOLD_DELAY_S
        self.servo_speed = servo_speed
        self.arm = arm
        self.limits = limits
        self.xyz = [float(value) for value in state["pose"][:3]]
        self.rpy = [float(value) for value in state["pose"][3:]]
        self.joints = [float(value) for value in state["joints"]]
        self.step = STEPS_MM["2"]
        self.record_path = record_path
        self.ask = ask
        self.out = out
        self.stop_pressed = stop_pressed
        self.discard_keys = discard_keys
        self.recorded = 0

    def status(self) -> str:
        x, y, z = self.xyz
        return (f"目标 X={x:.2f} Y={y:.2f} Z={z:.2f} mm | RPY={[round(v, 3) for v in self.rpy]} | "
                f"工具Z偏离竖直向下 {down_tilt_deg(self.rpy):.2f}° | 步长 {self.step} mm")

    def check(self, xyz: Sequence[float], rpy: Sequence[float],
              seed: Sequence[float] | None = None) -> list[float]:
        seed = list(self.joints if seed is None else seed)
        problem = self.limits.violation(xyz)
        if problem:
            raise JogRefused(problem)
        try:
            joints = self.arm.plan(list(xyz), list(rpy), seed)
            forward = self.arm.forward(joints)
        except RuntimeError as error:
            raise JogRefused(f"控制器逆解/正解失败（可能不可达）：{error}") from error
        jump = max(abs(math.remainder(joints[i] - seed[i], 360.0)) for i in range(6))
        if jump > MAX_JOINT_STEP_DEG:
            raise JogRefused(f"逆解关节变化 {jump:.1f}° > {MAX_JOINT_STEP_DEG}°，可能换了解支")
        wrist = abs(math.remainder(joints[4], 180.0))
        if wrist < WRIST_MARGIN_DEG:
            raise JogRefused(f"J5={joints[4]:.2f}° 离腕部奇异只剩 {wrist:.2f}°")
        position_error = max(abs(forward["xyz"][i] - xyz[i]) for i in range(3))
        orientation_error = _rotation_angle_deg(forward["rpy"], rpy)
        if position_error > FK_POSITION_TOL_MM or orientation_error > FK_ORIENTATION_TOL_DEG:
            raise JogRefused(f"正解复核不一致：{position_error:.3f} mm / {orientation_error:.3f}°")
        return joints

    def go(self, xyz: Sequence[float], rpy: Sequence[float], *, report: bool = True) -> None:
        joints = self.check(xyz, rpy)
        self.arm.move(list(xyz), list(rpy), joints)
        try:
            state = self.arm.wait(list(xyz), list(rpy), joints, self.stop_pressed)
        finally:
            self.discard_keys()
        self.xyz, self.rpy, self.joints = list(xyz), list(rpy), list(joints)
        error = max(abs(state["pose"][i] - xyz[i]) for i in range(3))
        if report:
            self.out(f"到达，实测偏差 {error:.3f} mm | {self.status()}")

    def travel_path(self, target: Sequence[float], travel_z: float) -> list[list[float]]:
        """Vertical to travel_z over the start, across at travel_z, then vertical onto target."""
        start = list(self.xyz)
        path: list[list[float]] = []
        previous = start
        for waypoint in ([start[0], start[1], travel_z], [target[0], target[1], travel_z], list(target)):
            pieces = math.ceil(math.dist(previous, waypoint) / GOTO_SEGMENT_MM - 1e-9)
            for k in range(1, pieces + 1):
                path.append([round(previous[i] + (waypoint[i] - previous[i]) * k / pieces, 4)
                             for i in range(3)])
            previous = waypoint
        return path

    def goto_corner(self) -> None:
        """t: hover above the fit's predicted corner; the operator still aligns to the real one."""
        if self.targets is None:
            raise JogRefused("启动时没有给 --session，无法计算角点位置")
        tilt = down_tilt_deg(self.rpy)
        if tilt > GOTO_TILT_LIMIT_DEG:
            raise JogRefused(f"工具偏离竖直 {tilt:.2f}°，先按 v 摆正再用 t")
        text = self.ask("移到哪个角点上方（编号，回车取消）: ").strip()
        if not text:
            self.out("已取消")
            return
        if not text.isdigit():
            raise JogRefused("角点编号必须是非负整数")
        corner = int(text)
        try:
            predicted = self.targets.corner_base_mm(corner)
        except RuntimeError as error:
            raise JogRefused(str(error)) from error
        if corner in self.targets.fit_ids:
            self.out(f"注意：角点 {corner} 参与了本臂拟合，不能当独立验收点")
        hover = [predicted[0], predicted[1], round(predicted[2] + GOTO_HOVER_MM, 4)]
        failures = []
        for clearance in GOTO_CLEARANCES_MM:
            travel_z = round(hover[2] + clearance, 4)
            path = self.travel_path(hover, travel_z)
            seed = list(self.joints)
            try:
                for point in path:  # every piece is checked before anything moves
                    seed = self.check(point, self.rpy, seed)
            except JogRefused as refusal:
                failures.append(f"Z={travel_z:.1f} 平移：在 {[round(v, 1) for v in point]} 处{refusal}")
                continue
            break
        else:
            raise JogRefused(f"角点 {corner}（预测 {[round(v, 1) for v in predicted]}）各平移高度都不可达；"
                             + "；".join(failures) + "。可先手动移近一些再按 t")
        self.out(f"角点 {corner} 预测位置 X={predicted[0]:.2f} Y={predicted[1]:.2f} Z={predicted[2]:.2f}；"
                 f"先在原地{'下降' if travel_z < self.xyz[2] else '抬升'}到 Z={travel_z:.1f}，平移后下降到"
                 f"其上方 {GOTO_HOVER_MM:.0f} mm（{len(path)} 段直线）。平移路线上不要有杯子等物体")
        for point in path:
            self.go(point, self.rpy, report=False)
        self.out(f"已到角点 {corner} 上方 | {self.status()}\n"
                 "先看指尖和真实角点是否对齐，对不齐就平移修正，再下降触碰")

    def handle(self, key: str) -> bool:
        """Process one key; False means quit."""
        if key in ("q", "ESC"):
            return False
        try:
            if key in MOVES:
                self.move_keys({key} | self.held_moves())
            elif len(key) > 1 and all(char in MOVES for char in key):
                self.move_keys(set(key))  # line mode: "wa" + Enter
            elif key in STEPS_MM:
                self.step = STEPS_MM[key]
                self.out(f"步长 {self.step} mm")
            elif key == "v":
                self.snap_down()
            elif key == "p":
                self.record()
            elif key == "t":
                self.goto_corner()
            elif key in ("o", "c", "g"):
                self.gripper(key)
            elif key == "z":
                self.set_floor()
            elif key == "i":
                self.out(self.status())
            elif key == "h":
                self.out(HELP)
        except JogRefused as refusal:
            self.out(f"拒绝，未发送任何运动：{refusal}")
        return True

    @staticmethod
    def direction(keys: set[str]) -> list[int]:
        """Per-axis sign of the held movement keys; opposite keys cancel, duplicates count once."""
        signs: list[set[int]] = [set(), set(), set()]
        for key in keys:
            if key in MOVES:
                axis, sign = MOVES[key]
                signs[axis].add(sign)
        return [sum(axis_signs) for axis_signs in signs]  # {+1, -1} sums to 0

    def move_keys(self, keys: set[str]) -> None:
        """Release before HOLD_DELAY_S is a one-step tap; a longer hold jogs or streams."""
        direction = self.direction(keys)
        if not any(direction):
            self.out("相反方向抵消，未移动")
            return
        pressed_at = self.clock()
        held = direction
        while self.clock() - pressed_at < self.hold_delay:
            held = self.direction(self.held_moves())
            if not any(held):
                break
            self.sleep(HOLD_POLL_S)
        else:
            held = self.direction(self.held_moves())
        try:
            if not any(held):
                self.step_once(direction)
            elif sum(map(abs, held)) == 1:
                self.jog_native(held)
            else:
                self.servo_stream(held)
        finally:
            self.discard_keys()

    @staticmethod
    def _label(direction: Sequence[int], step: float | None = None) -> str:
        unit = "" if step is None else f"{step}"
        return " ".join(f"{AXIS_NAMES[axis]}{'+' if direction[axis] > 0 else '-'}{unit}"
                        for axis in range(3) if direction[axis])

    def step_once(self, direction: Sequence[int]) -> None:
        self.out(self._label(direction, self.step) + " mm")
        target = [round(self.xyz[axis] + direction[axis] * self.step, 4) for axis in range(3)]
        self.go(target, self.rpy)

    def jog_native(self, direction: Sequence[int]) -> None:
        """Controller MoveJog while exactly this one axis stays held."""
        axis = next(index for index in range(3) if direction[index])
        sign = direction[axis]
        label = self._label(direction)
        stopped = False
        reason = "松手"
        self.out(f"连续点动 {label}（松手即停）")
        try:
            # Inside the try: a rejected echo may still have started the jog.
            self.arm.jog(label)
            previous: tuple[float, list[float]] | None = None
            while True:
                if self.stop_pressed():
                    stopped, reason = True, "空格"
                    break
                if self.direction(self.held_moves()) != list(direction):
                    break
                state = self.arm.read()
                if state["mode"] == ALARM_MODE:
                    raise RuntimeError("点动中控制器报警（RobotMode=9），请到示教器查看原因；本程序不会清报警")
                now, position = self.clock(), list(state["pose"][:3])
                speed = 0.0
                if previous is not None and now > previous[0]:
                    speed = abs(position[axis] - previous[1][axis]) / (now - previous[0])
                previous = (now, position)
                ahead = list(position)
                ahead[axis] += sign * max(JOG_STOP_MARGIN_MM, speed * JOG_STOP_HORIZON_S)
                problem = self.limits.violation(ahead)
                if problem:
                    reason = f"接近软限位（{problem}）"
                    break
                self.sleep(HOLD_POLL_S)
        finally:
            self.arm.jog_stop()
        state = self.arm.settle()
        drift = _rotation_angle_deg(state["pose"][3:], self.rpy)
        self.xyz = [round(value, 4) for value in state["pose"][:3]]
        self.joints = list(state["joints"])
        self.out(f"点动停止（{reason}）| {self.status()}")
        if drift > ORIENTATION_DRIFT_WARN_DEG:
            self.out(f"注意：点动后姿态偏离锁定值 {drift:.2f}°，可按 v 摆正")
        if stopped:
            raise JogStopped("空格：点动已停止")

    def servo_stream(self, direction: Sequence[int]) -> None:
        """Multi-axis hold: ServoJ follow points along the diagonal until release."""
        norm = math.sqrt(sum(value * value for value in direction))
        unit = [value / norm for value in direction]
        label = self._label(direction)
        self.out(f"连续斜线移动 {label}，{self.servo_speed:.0f} mm/s（松手即停）")
        last, last_joints = list(self.xyz), list(self.joints)
        ticks, stopped, reason = 0, False, "松手"
        previous = self.clock()
        try:
            while True:
                if self.stop_pressed():
                    stopped, reason = True, "空格"
                    break
                held = self.direction(self.held_moves())
                if held != list(direction):
                    reason = "松手" if not any(held) else "方向改变"
                    break
                now = self.clock()
                period = min(max(now - previous, SERVO_PERIOD_S), 0.1)
                previous = now
                target = [round(last[i] + unit[i] * self.servo_speed * period, 4) for i in range(3)]
                try:
                    joints = self.check(target, self.rpy, last_joints)
                except JogRefused as refusal:
                    reason = f"下一点被拒绝：{refusal}"
                    break
                self.arm.servo(target, list(self.rpy), joints, period)
                last, last_joints = target, joints
                ticks += 1
                if ticks % 5 == 0 and self.arm.read()["mode"] == ALARM_MODE:
                    raise RuntimeError("运动中控制器报警（RobotMode=9），请到示教器查看原因；本程序不会清报警")
                self.sleep(max(0.0, SERVO_PERIOD_S - (self.clock() - now)))
        finally:
            self.arm.servo_end()
        state = self.arm.settle()
        drift = _rotation_angle_deg(state["pose"][3:], self.rpy)
        self.xyz = [round(value, 4) for value in state["pose"][:3]]
        self.joints = list(state["joints"])
        self.out(f"连续移动停止（{reason}，{ticks} 个跟随点）| {self.status()}")
        if drift > ORIENTATION_DRIFT_WARN_DEG:
            self.out(f"注意：移动后姿态偏离锁定值 {drift:.2f}°，可按 v 摆正")
        if stopped:
            raise JogStopped("空格：连续移动已停止")

    def set_floor(self) -> None:
        """Make the touch height the floor, so the tip cannot be driven into the board."""
        z = float(self.arm.read()["pose"][2])
        self.limits.z_min = round(z - FLOOR_BELOW_TOUCH_MM, 4)
        self.out(f"Z 下限已设为 {self.limits.z_min:.2f} mm（当前 Z={z:.2f}）；下次启动可加 "
                 f"--z-min {self.limits.z_min:.1f} --no-home")

    def gripper(self, key: str) -> None:
        """o opens, c closes, g sets a value; opening needs a typed y because it releases."""
        current = self.arm.gripper_position()
        if key == "c":
            target = 0
        elif key == "o":
            target = 255
        else:
            text = self.ask(f"夹爪目标 0-255（当前 {current}，0 闭合 / 255 张开，回车取消）: ").strip()
            if not text:
                self.out("已取消")
                return
            if not text.isdigit() or not 0 <= int(text) <= 255:
                raise JogRefused("夹爪目标必须是 0-255 的整数")
            target = int(text)
        if target > current:
            answer = self.ask(f"张开到 {target} 会松开夹着的东西，输入 y 确认: ").strip().lower()
            if answer != "y":
                self.out("未确认，夹爪未动作")
                return
        feedback = self.arm.set_gripper(target)
        self.out(f"夹爪 {feedback.get('start')} -> {feedback.get('final')}（目标 {target}；"
                 "位置反馈不能证明夹住了东西）")

    def snap_down(self) -> None:
        tilt = down_tilt_deg(self.rpy)
        if tilt > SNAP_LIMIT_DEG:
            raise JogRefused(f"工具偏离竖直 {tilt:.1f}° > {SNAP_LIMIT_DEG}°，请先用示教器摆到大致朝下")
        target = snap_down_rpy(self.rpy)
        self.out(f"摆正朝下：转动 {_rotation_angle_deg(self.rpy, target):.2f}°，位置不变")
        self.go(self.xyz, target)

    def _rows(self) -> list[dict[str, str]]:
        if not self.record_path.exists():
            return []
        with self.record_path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _write_rows(self, rows: list[dict[str, str]], event: str, old: dict[str, str] | None,
                    new: dict[str, str] | None) -> None:
        """Rewrite the CSV whole, and keep every replaced or deleted row in a history file."""
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.record_path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["corner_id", "split", "x", "y", "z"])
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.record_path)
        history = self.record_path.with_name(self.record_path.stem + "_history.jsonl")
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                                     "event": event, "old": old, "new": new},
                                    ensure_ascii=False) + "\n")

    def record(self) -> None:
        """p: record the TCP for a corner; an id already recorded can be overwritten or deleted."""
        pose = self.arm.read()["pose"]
        corner = self.ask("角点编号（回车取消）: ").strip()
        if not corner:
            self.out("已取消记录")
            return
        if not corner.isdigit():
            raise JogRefused("角点编号必须是非负整数")
        rows = self._rows()
        existing = next((row for row in rows if int(row["corner_id"]) == int(corner)), None)
        if existing is not None:
            choice = self.ask(f"角点 {corner} 已记录为 {existing['split']}（X={existing['x']} Y={existing['y']} "
                              f"Z={existing['z']}）。y=用当前位置覆盖 / d=删除这条 / 回车=取消: ").strip().lower()
            if choice == "d":
                rows.remove(existing)
                self._write_rows(rows, "delete", existing, None)
                self.recorded = len(rows)
                self.out(f"已删除角点 {corner} 的记录（旧值保存在历史文件里）")
                return
            if choice != "y":
                self.out("未修改")
                return
        default = "t" if existing is None else existing["split"][0]
        split = self.ask(f"用途 t=train / v=val [{default}]: ").strip().lower() or default
        if split not in ("t", "v"):
            raise JogRefused("用途只能是 t 或 v")
        row = {"corner_id": str(int(corner)), "split": "train" if split == "t" else "val",
               "x": f"{pose[0]:.4f}", "y": f"{pose[1]:.4f}", "z": f"{pose[2]:.4f}"}
        if existing is not None:
            rows[rows.index(existing)] = row
        else:
            rows.append(row)
        self._write_rows(rows, "overwrite" if existing else "record", existing, row)
        self.recorded = len(rows)
        self.out(f"{'已覆盖' if existing else '已记录'}角点 {corner} 为 {row['split']}"
                 f"（实测 X={pose[0]:.3f} Y={pose[1]:.3f} Z={pose[2]:.3f}）-> {self.record_path}"
                 f"（共 {len(rows)} 个）")


KeySource = tuple[Callable[[], str], Callable[[], bool], Callable[[], None], Callable[[], set[str]]]


def _windows_keys() -> KeySource:
    import ctypes
    import msvcrt

    get_state = ctypes.windll.user32.GetAsyncKeyState
    special = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT", "I": "PGUP", "Q": "PGDN"}

    def read_key() -> str:
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            return special.get(msvcrt.getwch(), "")
        if char == "\x1b":
            return "ESC"
        return char.lower()

    def stop_pressed() -> bool:
        pressed = False
        while msvcrt.kbhit():
            pressed = pressed or msvcrt.getwch() == " "
        return pressed

    def discard() -> None:
        while msvcrt.kbhit():
            msvcrt.getwch()

    def held() -> set[str]:
        # Only consulted right after a console key event, so a key held in
        # another window can never start a move on its own.
        return {key for key, code in VIRTUAL_KEYS.items() if get_state(code) & 0x8000}

    return read_key, stop_pressed, discard, held


def _line_keys() -> KeySource:
    """Fallback without msvcrt: type keys and press Enter; "wa" moves diagonally."""
    def read_key() -> str:
        return input("> ").strip().lower()
    return read_key, (lambda: False), (lambda: None), set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="键盘控制单臂 XYZ 点动，姿态锁定")
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sim", action="store_true", help="离线模拟，不连接机械臂")
    mode.add_argument("--live", action="store_true", help="连接真机")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--tool", type=int, help="工具坐标系编号，默认取 profile.frame.tool")
    parser.add_argument("--user", type=int, help="用户坐标系编号，默认取 profile.frame.user")
    parser.add_argument("--speed", type=int, default=10, help="速度百分比 1-30，默认 10")
    parser.add_argument("--radius", type=float, default=200.0, help="水平活动半径 mm，默认 200")
    parser.add_argument("--z-min", type=float, help="Z 下限 mm（基座系），默认起点 Z-400；运行中可按 z 设到当前高度")
    parser.add_argument("--z-max", type=float, help="Z 上限 mm（基座系），默认起点 Z+150")
    parser.add_argument("--no-home", action="store_true",
                        help="使能后不回初始位（例如标定中途重启程序时）")
    parser.add_argument("--servo-speed", type=float,
                        help="斜线连续移动速度 mm/s，默认 --speed × 2（上限 60）")
    parser.add_argument("--record", type=Path, help="角点 CSV 路径，默认 calibration-data/jog-points/")
    parser.add_argument("--session", type=Path,
                        help="标定向导会话目录；给了才能用 t 键移到拟合预测的角点上方")
    return parser


def main(argv: Sequence[str] | None = None, *, keys: Any = None,
         ask: Callable[[str], str] = input, out: Callable[[str], None] = print) -> int:
    args = build_parser().parse_args(argv)
    if not SPEED_RANGE[0] <= args.speed <= SPEED_RANGE[1]:
        out(f"--speed 必须在 {SPEED_RANGE[0]}-{SPEED_RANGE[1]} 之间")
        return 2
    servo_speed = (args.servo_speed if args.servo_speed is not None
                   else args.speed * SERVO_MM_S_PER_PERCENT)
    if not 0 < servo_speed <= 60:
        out("--servo-speed 必须在 (0, 60] mm/s 之间")
        return 2
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if args.tool is not None:
        profile["frame"]["tool"] = args.tool
    if args.user is not None:
        profile["frame"]["user"] = args.user
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = args.record or DEFAULT_RECORD_DIR / f"{args.arm}-{stamp}.csv"
    targets = None
    if args.session is not None:
        from .calibration_targets import TargetError, load_arm_fit

        try:
            targets = load_arm_fit(args.session, args.arm)
        except (TargetError, OSError, KeyError, ValueError) as error:
            out(f"无法读取标定会话：{error}")
            return 2
        out(f"t 键目标：{targets.robot_id} 拟合 {targets.job_id}（TCP {targets.tcp_name}，"
            f"拟合用过的角点 {sorted(targets.fit_ids)}）")

    arm: Any = None
    try:
        if args.sim:
            arm = SimArm([-150.0, -350.0, 200.0], snap_down_rpy([178.0, 3.0, 90.0]))
        else:
            arm = LiveArm(profile, args.arm, args.speed, log=out)
        start = arm.read()
        frame = profile["frame"]
        out(f"[{arm.name}] {args.arm} 臂 | Tool{frame['tool']} User{frame['user']} | 速度 {args.speed}%")
        out(f"当前 pose={[round(v, 3) for v in start['pose']]} "
            f"joints={[round(v, 3) for v in start['joints']]} RobotMode={start.get('mode')}")
        if args.live:
            from .layout import physical_nova_home_degrees

            problem = start_refusal(start.get("mode"))
            if problem:
                out(f"拒绝启动，没有发送任何指令：{problem}")
                return 5
            home = list(physical_nova_home_degrees(args.arm))
            out("将执行：设备租约 -> EnableRobot -> 速度 -> Tool/User"
                + ("" if args.no_home else f" -> JointMovJ 回初始位 {home}（关节空间运动，注意周围）")
                + "。退出时 DisableRobot 下使能。"
                "不会清报警、不会上电或断电；夹爪只在按 o/c/g 时动作。")
            if ask(f"确认 {args.arm} 臂周围无人无障碍、急停在手边，输入 {args.arm} 继续: ").strip() != args.arm:
                out("未确认，已退出，没有发送任何指令")
                return 1
            out(arm.open_session(home=not args.no_home))
            start = arm.read()
        xyz = start["pose"][:3]
        limits = Limits(xyz, args.radius,
                        args.z_min if args.z_min is not None else xyz[2] - DEFAULT_Z_DROP_MM,
                        args.z_max if args.z_max is not None else xyz[2] + 150.0)
        out(f"起点 pose={[round(v, 3) for v in start['pose']]}；工具 Z 偏离竖直向下 "
            f"{down_tilt_deg(start['pose'][3:]):.2f}°；软限位：{limits.describe()}")
        read_key, stop_pressed, discard, held = keys or (
            _windows_keys() if sys.platform == "win32" else _line_keys())
        jog = Jog(arm, limits, record_path=record, ask=ask, out=out,
                  stop_pressed=stop_pressed, discard_keys=discard, held_moves=held,
                  servo_speed=servo_speed, targets=targets)
        out(HELP)
        out(jog.status())
        while jog.handle(read_key()):
            pass
        out(f"退出。本次记录 {jog.recorded} 个角点" + (f" -> {record}" if jog.recorded else ""))
        return 0
    except JogStopped as stopped:
        out(str(stopped))
        return 3
    except RuntimeError as error:
        out(f"运行错误，已停止：{error}")
        return 4
    finally:
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    sys.exit(main())
