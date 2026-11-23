"""Small, lease-guarded Cartesian interface for a single Dobot arm.

The live adapter deliberately takes the shared device lease before opening any
controller sockets. It then reuses ``run_cartesian_jog``'s LiveArm and Jog
checks, while keeping this interface limited to finite XYZ moves and short
ServoJ microtargets.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .chemistry.driver import LiveDriver
from .device_lease import DeviceLease
from .run_cartesian_jog import (
    DEFAULT_PROFILE,
    ARRIVAL_TOL_MM,
    Jog,
    JogRefused,
    JogStopped,
    Limits,
    LiveArm,
    SimArm,
    start_refusal,
    SERVO_PERIOD_S,
)


SPEED_PERCENT = 5
# The UI should describe this as a 100 mm horizontal radius and Z within
# +/-60 mm of the enable-time position. The existing Limits helper enforces a
# conservative cylinder, which is inside the corresponding square XY box.
WORKSPACE_RADIUS_MM = 200.0
WORKSPACE_Z_HALF_HEIGHT_MM = 150.0
ALLOWED_STEP_MM = (0.5, 1.0, 5.0)
MAX_DRIVE_SPEED_MM_S = 12.0
# Operator speed slider: thumb-drive speed at full deflection (mm/s). Preset
# moves (home, photo) scale their controller speed percentage from it.
MIN_DRIVE_SPEED_MM_S = 5.0
MAX_DRIVE_SPEED_MM_S_LIMIT = 150.0
DEFAULT_DRIVE_SPEED_MM_S = 30.0
PRESET_PERCENT_RANGE = (5, 50)
PHOTO_ROUTE_RADIUS_MM = 2000.0  # route planning only; every 10 mm is IK/FK-checked
PHOTO_POSES = Path(__file__).resolve().parent.parent / "chem_demo" / "poses.json"
PRESET_TIMEOUT_S = 90.0
PRESET_POLL_S = 0.1
GRIPPER_OPEN = 255
GRIPPER_CLOSED = 0
MAX_SERVO_PERIOD_S = 0.1
CLEAR_TIMEOUT_S = 5.0
BUSY_MODES = (7, 10)  # running or paused queue
POWER_ON_TIMEOUT_S = 40.0  # controller needs ~10-20 s after PowerOn
MAX_STREAM_DT_S = 0.2       # longest ServoJ duration / setpoint advance per point
DRIVE_DRIFT_TOL_MM = 5.0    # measured pose may trail the setpoint by 2 steps + this
STATE_STABILITY_TOL_MM = 0.5
STATE_STABILITY_TOL_DEG = 0.5
LIVE_LOCK_PURPOSE = "chemistry-atomic-skills"
BUSY_LEASE_MESSAGE = "机械臂正在被占用；请在当前 jog 窗口输入 q 退出后重试。"


class MobileArmError(RuntimeError):
    """The adapter could not safely perform the requested operation."""


class MobileCleanupError(MobileArmError):
    """Shutdown failed; the caller must treat physical state as uncertain."""


class MobileArm:
    """Lease-guarded, one-arm XYZ control for a sequential server worker.

    Construction is inert. A live ``open`` acquires the shared device lease
    before creating dashboard, feedback, or motion connections. ``step`` sends
    one checked MovL; ``drive`` follows fresh joystick samples with finite
    ServoJ microtargets and settles on release.
    """

    def __init__(self, arm: str, live: bool):
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        if not isinstance(live, bool):
            raise TypeError("live must be a bool")
        self.arm = arm
        self.live = live
        self._lock = threading.RLock()
        self._backend: Any = None
        self._driver: LiveDriver | None = None
        self._jog: Jog | None = None
        self._limits: Limits | None = None
        self._opened = False
        self._motion_pending = False
        self._stop_sent = False
        self._clock: Callable[[], float] = time.monotonic
        self.drive_speed = DEFAULT_DRIVE_SPEED_MM_S
        self.gripper_position: int | None = None

    def set_speed(self, mm_s: float) -> None:
        """Safe to call from another thread: the drive loop reads it per tick."""
        if isinstance(mm_s, bool) or not isinstance(mm_s, (int, float)) or not math.isfinite(mm_s) \
                or not MIN_DRIVE_SPEED_MM_S <= mm_s <= MAX_DRIVE_SPEED_MM_S_LIMIT:
            raise ValueError("speed must be 5-150 mm/s")
        self.drive_speed = float(mm_s)

    @property
    def preset_percent(self) -> int:
        low, high = PRESET_PERCENT_RANGE
        share = (self.drive_speed - MIN_DRIVE_SPEED_MM_S) / (
            MAX_DRIVE_SPEED_MM_S_LIMIT - MIN_DRIVE_SPEED_MM_S)
        return int(round(low + share * (high - low)))

    @property
    def workspace(self) -> dict[str, Any]:
        """Bounds for UI display; Z offsets are relative to the enable-time pose."""
        return {"horizontal_radius_mm": WORKSPACE_RADIUS_MM,
                "z_from_enable_mm": [-WORKSPACE_Z_HALF_HEIGHT_MM,
                                     WORKSPACE_Z_HALF_HEIGHT_MM]}

    def open(self, cancel: threading.Event | None = None,
             clear_errors: bool = False) -> dict[str, Any]:
        """Connect (or create a simulator) and return the current snapshot.

        ``clear_errors`` is the operator's explicit "clear alarm and re-enable":
        ClearError is sent under the lease before the usual start checks.
        """
        with self._lock:
            if self._opened:
                return self.snapshot()
            try:
                self._check_cancel(cancel)
                if self.live:
                    self._open_live_backend()
                    if clear_errors:
                        self._clear_alarm(cancel)
                    before = self._read_state()
                    problem = start_refusal(before["mode"])
                    if problem:
                        raise MobileArmError(problem)
                    self._check_cancel(cancel)
                    self._open_live_session(cancel)
                    state = self._read_state()
                    if state["mode"] != 5:
                        raise MobileArmError(
                            f"机械臂未进入使能空闲模式 (RobotMode={state['mode']})。")
                else:
                    self._backend = SimArm(
                        [-150.0, -350.0, 200.0], [180.0, 0.0, 90.0])
                    state = self._read_state()
                    if state["mode"] != 5:
                        raise MobileArmError(
                            f"模拟机械臂状态不安全 (RobotMode={state['mode']})。")

                xyz = state["pose"][:3]
                self._limits = Limits(
                    xyz,
                    WORKSPACE_RADIUS_MM,
                    xyz[2] - WORKSPACE_Z_HALF_HEIGHT_MM,
                    xyz[2] + WORKSPACE_Z_HALF_HEIGHT_MM,
                )
                # Jog owns the existing IK/FK, branch-jump, wrist, and soft-limit
                # checks. Its record path is unused by this adapter.
                self._jog = Jog(self._backend, self._limits, record_path=Path("."))
                latest = self._read_state()
                if latest["mode"] != 5:
                    raise MobileArmError(
                        f"open ended outside enabled idle mode (RobotMode={latest['mode']})")
                self._require_same_pose(state, latest)
                state = latest
                self._sync_jog(state)
                self._opened = True
                return self._snapshot_from_state(state)
            except Exception as original:
                try:
                    self.close()
                except Exception as cleanup_error:
                    raise MobileCleanupError(
                        f"open failed ({original}); cleanup also failed ({cleanup_error})") from original
                raise

    def snapshot(self) -> dict[str, Any]:
        """Read a fresh state when connected, without sending a command."""
        with self._lock:
            if not self._opened or self._backend is None:
                raise MobileArmError("MobileArm is not open.")
            return self._snapshot_from_state(self._read_state())

    def step(self, axis: str, direction: int, mm: float,
             cancel: threading.Event) -> dict[str, Any]:
        """Move once along base-frame X, Y, or Z by 0.5, 1, or 5 mm."""
        with self._lock:
            self._require_open()
            axis_index = self._axis_index(axis)
            if (isinstance(direction, bool) or not isinstance(direction, int)
                    or direction not in (-1, 1)):
                raise ValueError("direction must be -1 or +1")
            distance = self._step_size(mm)
            self._require_event(cancel)
            if cancel.is_set():
                raise JogStopped("move cancelled before dispatch")

            state = self._idle_state()
            self._sync_jog(state)
            target = list(state["pose"][:3])
            target[axis_index] = round(target[axis_index] + direction * distance, 4)
            joints = self._jog.check(target, state["pose"][3:], state["joints"])

            # Planning talks to the controller for IK/FK. Re-read mode and pose
            # after that work so a stale idle snapshot cannot authorize a move.
            dispatch_state = self._idle_state()
            self._require_same_pose(state, dispatch_state)
            if cancel.is_set():
                raise JogStopped("move cancelled before dispatch")

            self._motion_pending = True
            self._stop_sent = False
            try:
                self._backend.move(target, state["pose"][3:], joints)
                arrived = self._wait_for_step(
                    target, state["pose"][3:], joints, cancel)
                if cancel.is_set():
                    self._safe_stop()
                    raise JogStopped("move cancelled during arrival wait")
                result = self._normalize_state(arrived)
                if result["mode"] != 5:
                    raise MobileArmError(
                        f"move ended outside enabled idle mode (RobotMode={result['mode']})")
                self._sync_jog(result)
                self._motion_pending = False
                return self._snapshot_from_state(result)
            except JogStopped:
                if not self._stop_sent:
                    self._safe_stop()
                self._motion_pending = False
                raise
            except Exception:
                self._safe_stop()
                raise
            finally:
                if self._stop_sent:
                    self._motion_pending = False

    def drive(self, sample: Callable[[], Sequence[float] | None],
              cancel: threading.Event) -> dict[str, Any]:
        """Follow a normalized XYZ joystick until release, timeout, or cancel.

        Smooth streaming: the commanded setpoint advances from the previous
        commanded point (not the lagging measured pose) by speed x elapsed time,
        and each ServoJ is given the measured loop time as its duration, so the
        arm is still travelling when the next point arrives instead of
        stop-starting. Every point is still IK/FK-checked against the soft
        workspace, joint jumps and the wrist, and the measured pose must stay
        close to the commanded one. Zero input holds; ``None`` ends the stream
        and settles; cancellation sends StopScript.
        """
        if not callable(sample):
            raise TypeError("sample must be callable")
        with self._lock:
            self._require_open()
            self._require_event(cancel)
            self._motion_pending = False
            self._stop_sent = False
            stream_sent = False
            command_xyz: list[float] | None = None
            command_joints: list[float] | None = None
            rpy: list[float] | None = None
            last_dispatch: float | None = None
            loop_time = SERVO_PERIOD_S
            last_step_mm = 0.0
            try:
                while True:
                    if cancel.is_set():
                        raise JogStopped("drive cancelled")
                    tick_start = self._clock()
                    state = self._drive_state(stream_sent)
                    if rpy is None:
                        rpy = list(state["pose"][3:])
                    if command_xyz is None:
                        command_xyz = list(state["pose"][:3])
                        command_joints = list(state["joints"])
                    else:
                        drift = math.dist(state["pose"][:3], command_xyz)
                        if drift > 2.0 * last_step_mm + DRIVE_DRIFT_TOL_MM:
                            raise MobileArmError(
                                f"实际位置落后指令 {drift:.1f} mm，已停止拖曳")
                    self._sync_jog(state)

                    intent = sample()
                    if intent is None:
                        break
                    vector = self._drive_vector(intent)
                    magnitude = math.sqrt(sum(value * value for value in vector))
                    if magnitude <= 1e-9:
                        # Holding still: re-anchor on the measured pose once the
                        # arm has caught up, and restart timing on the next push.
                        command_xyz = list(state["pose"][:3])
                        command_joints = list(state["joints"])
                        last_dispatch = None
                        last_step_mm = 0.0
                        self._wait_tick(cancel, SERVO_PERIOD_S, tick_start)
                        continue

                    now = self._clock()
                    elapsed = SERVO_PERIOD_S if last_dispatch is None else now - last_dispatch
                    elapsed = min(MAX_STREAM_DT_S, max(SERVO_PERIOD_S, elapsed))
                    speed = self.drive_speed
                    target = [round(command_xyz[i] + vector[i] * speed * elapsed, 4)
                              for i in range(3)]
                    if self._limits.violation(target):
                        # Stop at the envelope; the next drag can steer back in.
                        if stream_sent:
                            self._servo_end()
                            state = self._normalize_state(self._backend.settle())
                        else:
                            state = self._idle_state()
                        if state["mode"] != 5:
                            raise MobileArmError(
                                f"soft-limit stop did not settle to idle "
                                f"(RobotMode={state['mode']})")
                        self._sync_jog(state)
                        self._motion_pending = False
                        return self._snapshot_from_state(state)
                    joints = self._jog.check(target, rpy, command_joints)

                    latest = sample()
                    if latest is None:
                        break
                    if cancel.is_set():
                        raise JogStopped("drive cancelled before dispatch")

                    # The ServoJ duration is the expected time until the next
                    # point, so consecutive points blend into one motion.
                    duration = min(MAX_STREAM_DT_S, max(SERVO_PERIOD_S, loop_time))
                    self._motion_pending = True
                    self._stop_sent = False
                    stream_sent = True
                    self._send_servo(target, rpy, joints, duration)
                    sent_at = self._clock()
                    if last_dispatch is not None:
                        loop_time = 0.7 * loop_time + 0.3 * (sent_at - last_dispatch)
                    last_dispatch = sent_at
                    last_step_mm = math.dist(target, command_xyz)
                    command_xyz, command_joints = target, list(joints)
                    self._wait_tick(cancel, SERVO_PERIOD_S, tick_start)

                if stream_sent:
                    self._servo_end()
                    settled = self._backend.settle()
                    state = self._normalize_state(settled)
                    if state["mode"] != 5:
                        raise MobileArmError(
                            f"drive did not settle in enabled idle mode (RobotMode={state['mode']})")
                else:
                    state = self._idle_state()
                self._sync_jog(state)
                self._motion_pending = False
                return self._snapshot_from_state(state)
            except JogStopped:
                if stream_sent:
                    self._end_servo_and_stop()
                raise
            except Exception:
                if stream_sent:
                    self._end_servo_and_stop()
                raise

    def _clear_alarm(self, cancel: threading.Event | None) -> None:
        from .chemistry.driver import _reply_status

        dashboard = self._backend.rig.dashboard
        # Power off (3), e.g. after an emergency stop: the operator asked for
        # the phone to power the arm back on instead of using the pendant.
        if self._read_state()["mode"] == 3:
            self._power_on(cancel)
        # A motion queue left running (7) or paused (10) by the fault keeps
        # the controller from returning to idle, so abort it first.
        if self._read_state()["mode"] in BUSY_MODES:
            dashboard.StopScript()
        if _reply_status(dashboard.ClearError()) != 0:
            raise MobileArmError("ClearError 未得到 status 0 应答")
        self._reset_stop_flags()
        deadline = self._clock() + CLEAR_TIMEOUT_S
        stop_resent = False
        while True:
            mode = self._read_state()["mode"]
            if mode in (4, 5):
                return
            if mode in BUSY_MODES and not stop_resent:
                dashboard.StopScript()
                stop_resent = True
            if self._clock() > deadline:
                if mode == 9:
                    raise MobileArmError(
                        "清除后仍在报警（RobotMode=9）：报警原因可能未排除，请到示教器查看")
                raise MobileArmError(f"清除后控制器未回到空闲（RobotMode={mode}），请到示教器查看")
            self._check_cancel(cancel)
            time.sleep(0.1)

    def _power_on(self, cancel: threading.Event | None) -> None:
        from .chemistry.driver import _reply_status

        reply = self._backend.rig.dashboard.PowerOn()
        if _reply_status(reply) != 0:
            raise MobileArmError(f"PowerOn 被控制器拒绝（{reply!r}）：请检查急停是否已复位")
        deadline = self._clock() + POWER_ON_TIMEOUT_S
        while True:
            mode = self._read_state()["mode"]
            if mode not in (1, 2, 3):  # initialising / brake release / unpowered
                return
            if self._clock() > deadline:
                raise MobileArmError(
                    f"上电 {POWER_ON_TIMEOUT_S:.0f} s 后仍未就绪（RobotMode={mode}）："
                    "请检查急停是否已复位、控制柜是否通电")
            self._check_cancel(cancel)
            time.sleep(0.5)

    def _reset_stop_flags(self) -> None:
        """Forget an acknowledged stop so later moves and the gripper run again."""
        self._motion_pending = False
        self._stop_sent = False
        if self._driver is not None:
            self._driver._stop_requested = False

    def _live_disable(self) -> None:
        """Abort any queue, DisableRobot, and wait for disabled (4).

        An alarm (9) also counts as released: the controller has already
        stopped the arm and refuses DisableRobot until the alarm is cleared,
        so treating it as a cleanup failure would lock the remote out.
        """
        from .chemistry.driver import _reply_status

        dashboard = self._backend.rig.dashboard
        mode = self._read_state()["mode"]
        if mode in BUSY_MODES:
            dashboard.StopScript()
            deadline = self._clock() + 3.0
            while self._read_state()["mode"] in BUSY_MODES and self._clock() < deadline:
                time.sleep(0.1)
        status = _reply_status(dashboard.DisableRobot())
        deadline = self._clock() + CLEAR_TIMEOUT_S
        while True:
            mode = self._read_state()["mode"]
            if mode in (4, 9):
                self._backend.session_open = False
                return
            if self._clock() > deadline:
                raise RuntimeError(
                    f"DisableRobot 应答 status={status}，{CLEAR_TIMEOUT_S:.0f} s 后 RobotMode={mode}")
            time.sleep(0.1)

    def recover(self, cancel: threading.Event) -> dict[str, Any]:
        """Clear an alarm and re-enable while this adapter holds the arm."""
        from .chemistry.driver import _reply_status

        with self._lock:
            self._require_open()
            self._require_event(cancel)
            if self.live:
                self._clear_alarm(cancel)
                dashboard = self._backend.rig.dashboard
                if self._read_state()["mode"] != 5 and \
                        _reply_status(dashboard.EnableRobot()) != 0:
                    raise MobileArmError("EnableRobot 未得到 status 0 应答")
                deadline = self._clock() + CLEAR_TIMEOUT_S
                while self._read_state()["mode"] != 5:
                    if self._clock() > deadline:
                        raise MobileArmError("重新使能后未进入使能空闲（RobotMode=5）")
                    self._check_cancel(cancel)
                    time.sleep(0.1)
                self._driver._configure_speed(self._backend.rig, SPEED_PERCENT, linear=True)
            state = self._idle_state()
            self._sync_jog(state)
            return self._snapshot_from_state(state)

    def go_home(self, cancel: threading.Event) -> dict[str, Any]:
        """JointMovJ to the calibrated home joints of this arm."""
        from .layout import physical_nova_home_degrees

        return self.move_joints(list(physical_nova_home_degrees(self.arm)), "复位", cancel)

    def move_joints(self, joints: Sequence[float], label: str,
                    cancel: threading.Event) -> dict[str, Any]:
        """One joint-space move to a stored pose, cancellable while it runs.

        The soft workspace is re-centred on the arrival pose, so thumb control
        continues around the preset instead of around the enable-time pose.
        """
        with self._lock:
            self._require_open()
            self._require_event(cancel)
            target = [float(value) for value in joints]
            if len(target) != 6 or not all(math.isfinite(value) for value in target):
                raise ValueError("preset needs six finite joint angles")
            if cancel.is_set():
                raise JogStopped("move cancelled before dispatch")
            self._idle_state()
            self._motion_pending = True
            self._stop_sent = False
            try:
                if self.live:
                    from .chemistry.driver import joint_frame

                    rig = self._backend.rig
                    self._driver._configure_speed(rig, self.preset_percent, linear=False)
                    self._driver._send_motion(rig, joint_frame(target), f"mobile {label}")
                    state = self._wait_for_joints(target, label, cancel)
                    self._driver._configure_speed(rig, SPEED_PERCENT, linear=True)
                else:
                    # The simulator's fake IK maps joints[:3] * 10 to XYZ; keep
                    # its joints self-consistent so later jog checks agree.
                    sim = self._backend
                    sim.pose = [target[0] * 10.0, target[1] * 10.0, target[2] * 10.0,
                                *sim.pose[3:]]
                    sim.joints = sim.solve(sim.pose[:3], sim.pose[3:], None)
                    state = self._normalize_state(sim.read())
                if cancel.is_set():
                    self._safe_stop()
                    raise JogStopped(f"{label} cancelled")
            except JogStopped:
                if not self._stop_sent:
                    self._safe_stop()
                self._motion_pending = False
                raise
            except Exception:
                self._safe_stop()
                raise
            self._motion_pending = False
            if state["mode"] != 5:
                raise MobileArmError(
                    f"{label} ended outside enabled idle mode (RobotMode={state['mode']})")
            xyz = state["pose"][:3]
            self._limits = Limits(xyz, WORKSPACE_RADIUS_MM,
                                  xyz[2] - WORKSPACE_Z_HALF_HEIGHT_MM,
                                  xyz[2] + WORKSPACE_Z_HALF_HEIGHT_MM)
            self._jog.limits = self._limits
            self._sync_jog(state)
            return self._snapshot_from_state(state)

    def go_photo(self, cancel: threading.Event) -> dict[str, Any]:
        """Travel to the verified top-camera photo pose in chem_demo/poses.json.

        Same route as the jog tool's x command that verified the pose: rise
        vertically, cross at the higher Z, then descend, each leg a straight
        MovL whose every 10 mm was IK/FK-checked first. Orientation is kept, so
        the wrist cannot flip to another IK branch (a single JointMovJ to a
        freshly solved pose did exactly that on the left arm).
        """
        from .run_cartesian_jog import GOTO_TILT_LIMIT_DEG, down_tilt_deg

        pose = json.loads(PHOTO_POSES.read_text(encoding="utf-8"))["photo"][self.arm]
        target = [float(v) for v in pose["xyz"]]
        with self._lock:
            self._require_open()
            self._require_event(cancel)
            state = self._idle_state()
            self._sync_jog(state)
            tilt = down_tilt_deg(state["pose"][3:])
            if tilt > GOTO_TILT_LIMIT_DEG:
                raise MobileArmError(
                    f"夹爪偏离竖直 {tilt:.1f}°，先点「复位」再去拍摄位")
            route_limits = Limits(state["pose"][:3], PHOTO_ROUTE_RADIUS_MM, -1000.0, 1000.0)
            self._jog.limits = route_limits
            try:
                travel_z = max(state["pose"][2], target[2])
                legs = self._jog.travel_legs(target, travel_z)
                try:
                    finals = self._jog.check_route(legs)
                except JogRefused as refusal:
                    raise MobileArmError(f"拍摄位路线检查未通过：{refusal}") from refusal
            finally:
                self._jog.limits = self._limits
            rpy = list(state["pose"][3:])
            arrived = state
            self._motion_pending = True
            self._stop_sent = False
            try:
                if self.live:
                    self._driver._configure_speed(
                        self._backend.rig, self.preset_percent, linear=True)
                for leg, joints in zip(legs, finals):
                    if cancel.is_set():
                        raise JogStopped("拍摄位 cancelled")
                    self._backend.move(list(leg), rpy, list(joints))
                    arrived = self._wait_for_step(leg, rpy, joints, cancel)
                    if cancel.is_set():
                        self._safe_stop()
                        raise JogStopped("拍摄位 cancelled")
                if self.live:
                    self._driver._configure_speed(self._backend.rig, SPEED_PERCENT, linear=True)
            except JogStopped:
                if not self._stop_sent:
                    self._safe_stop()
                self._motion_pending = False
                raise
            except Exception:
                self._safe_stop()
                raise
            self._motion_pending = False
            state = self._normalize_state(arrived)
            if state["mode"] != 5:
                raise MobileArmError(f"拍摄位到达后不在使能空闲（RobotMode={state['mode']}）")
            xyz = state["pose"][:3]
            self._limits = Limits(xyz, WORKSPACE_RADIUS_MM,
                                  xyz[2] - WORKSPACE_Z_HALF_HEIGHT_MM,
                                  xyz[2] + WORKSPACE_Z_HALF_HEIGHT_MM)
            self._jog.limits = self._limits
            self._sync_jog(state)
            return self._snapshot_from_state(state)

    def move_pose(self, xyz: Sequence[float], rpy: Sequence[float], label: str,
                  cancel: threading.Event) -> dict[str, Any]:
        """Solve IK near the current joints (controller-checked), then JointMovJ."""
        from .run_cartesian_jog import (FK_POSITION_TOL_MM, WRIST_MARGIN_DEG)

        with self._lock:
            self._require_open()
            state = self._idle_state()
            try:
                joints = self._backend.plan(list(xyz), list(rpy), state["joints"])
                forward = self._backend.forward(joints)
            except RuntimeError as error:
                raise MobileArmError(f"{label} 逆解失败（可能不可达）：{error}") from error
            error_mm = max(abs(forward["xyz"][i] - xyz[i]) for i in range(3))
            if error_mm > FK_POSITION_TOL_MM:
                raise MobileArmError(f"{label} 正解复核偏差 {error_mm:.3f} mm，未运动")
            if abs(math.remainder(joints[4], 180.0)) < WRIST_MARGIN_DEG:
                raise MobileArmError(f"{label} 的 J5 接近腕部奇异，未运动")
            return self.move_joints(joints, label, cancel)

    def _wait_for_joints(self, target: Sequence[float], label: str,
                         cancel: threading.Event) -> dict[str, Any]:
        from .chemistry.driver import ARRIVAL_TOL_DEG, angle_delta

        deadline = self._clock() + PRESET_TIMEOUT_S
        stable = 0
        while self._clock() < deadline:
            if cancel.is_set():
                self._safe_stop()
                raise JogStopped(f"{label} stopped by operator")
            state = self._read_state()
            if state["mode"] == 9:
                raise MobileArmError(f"{label} 途中控制器报警（RobotMode=9），请到示教器查看")
            worst = max(abs(angle_delta(state["joints"][i], target[i])) for i in range(6))
            stable = stable + 1 if state["mode"] == 5 and worst <= ARRIVAL_TOL_DEG else 0
            if stable >= 3:
                return state
            cancel.wait(PRESET_POLL_S)
        self._safe_stop()
        raise MobileArmError(f"{label} 在 {PRESET_TIMEOUT_S:.0f} s 内未到位，已发送 StopScript")

    def set_gripper(self, position: int, cancel: threading.Event) -> dict[str, Any]:
        """Command the jaws and wait for position feedback (not proof of a grasp)."""
        with self._lock:
            self._require_open()
            self._require_event(cancel)
            if isinstance(position, bool) or position not in (GRIPPER_OPEN, GRIPPER_CLOSED):
                raise ValueError("gripper target must be fully open or fully closed")
            if cancel.is_set():
                raise JogStopped("gripper command cancelled")
            feedback = self._backend.set_gripper(int(position))
            final = feedback.get("final") if isinstance(feedback, dict) else None
            self.gripper_position = int(final) if final is not None else int(position)
            return self._snapshot_from_state(self._read_state())

    def close(self) -> None:
        """Stop an unfinished motion, then disable only our leased live session."""
        with self._lock:
            backend, driver = self._backend, self._driver
            failures: list[str] = []
            try:
                if self._motion_pending:
                    try:
                        self._safe_stop()
                    except Exception as error:
                        failures.append(f"StopScript: {type(error).__name__}: {error}")
                if self.live and backend is not None and getattr(backend, "session_open", False):
                    try:
                        # Errors are reported to the server caller, not swallowed.
                        self._live_disable()
                    except Exception as error:
                        failures.append(f"DisableRobot: {type(error).__name__}: {error}")
                if self.live and driver is not None:
                    had_lease = driver._lease is not None
                    try:
                        report = driver.close()
                    except Exception as error:
                        failures.append(f"driver close: {type(error).__name__}: {error}")
                    else:
                        if had_lease and not report.get("lease_released"):
                            failures.append("device lease was not confirmed released")
                        failures.extend(
                            f"{key}: {value}" for key, value in report.items()
                            if key == "lease_error" or key.endswith("_close_error"))
                elif backend is not None:
                    try:
                        backend.close()
                    except Exception as error:
                        failures.append(f"backend close: {type(error).__name__}: {error}")
            finally:
                self._backend = None
                self._driver = None
                self._jog = None
                self._limits = None
                self._opened = False
                self._motion_pending = False
                self._stop_sent = False
            if failures:
                raise MobileCleanupError("shutdown failed: " + "; ".join(failures))

    def _open_live_backend(self) -> None:
        profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
        driver = LiveDriver(
            profile, allow_live=True, confirmation=lambda: True, log=lambda _: None)
        self._driver = driver

        # LiveArm's regular constructor opens sockets before _ensure_session()
        # takes its lease. The adapter takes the same shared lease first, then
        # initializes LiveArm's fields and asks LiveDriver for the rig.
        lease = DeviceLease(path=driver._lock_path, purpose=LIVE_LOCK_PURPOSE)
        try:
            lease.acquire()
        except RuntimeError as error:
            raise MobileArmError(BUSY_LEASE_MESSAGE) from error
        driver._lease = lease

        backend = LiveArm.__new__(LiveArm)
        backend.profile = profile
        backend.arm = self.arm
        backend.speed = SPEED_PERCENT
        backend.log = lambda _: None
        backend.session_open = False
        backend.driver = driver
        self._backend = backend
        backend.rig = driver._rig(self.arm, profile, with_motion=False)

    def _open_live_session(self, cancel: threading.Event | None = None) -> None:
        backend = self._backend
        dashboard = getattr(getattr(backend, "rig", None), "dashboard", None)
        if dashboard is None:
            backend.open_session(home=False)
            return

        # A dashboard command may take effect even when its reply/configuration
        # later fails. Record the exact EnableRobot attempt so open() cleanup
        # disables only after a session command was actually attempted.
        had_override = "EnableRobot" in getattr(dashboard, "__dict__", {})
        previous_override = getattr(dashboard, "__dict__", {}).get("EnableRobot")
        original = dashboard.EnableRobot

        def tracked_enable(*args: Any, **kwargs: Any) -> Any:
            state = self._read_state()
            problem = start_refusal(state["mode"])
            if problem:
                raise MobileArmError(problem)
            self._check_cancel(cancel)
            self._enable_attempted = True
            return original(*args, **kwargs)

        self._enable_attempted = False
        dashboard.EnableRobot = tracked_enable
        try:
            backend.open_session(home=False)
        except Exception:
            if self._enable_attempted:
                backend.session_open = True
            raise
        finally:
            if had_override:
                dashboard.EnableRobot = previous_override
            else:
                try:
                    delattr(dashboard, "EnableRobot")
                except AttributeError:
                    pass

    def _read_state(self) -> dict[str, Any]:
        if self._backend is None:
            raise MobileArmError("MobileArm has no connected backend.")
        state = self._normalize_state(self._backend.read())
        if self.live and state.get("source") != "dashboard":
            raise MobileArmError(
                "fresh dashboard state is unavailable; no command was sent")
        return state

    def _idle_state(self) -> dict[str, Any]:
        state = self._read_state()
        if state["mode"] != 5:
            raise MobileArmError(
                f"move requires enabled idle mode (RobotMode=5); got {state['mode']}")
        return state

    def _drive_state(self, stream_sent: bool) -> dict[str, Any]:
        state = self._read_state()
        allowed = (5, 7) if stream_sent else (5,)
        if state["mode"] not in allowed:
            raise MobileArmError(
                f"drive requires enabled idle mode before dispatch, then its own running "
                f"mode (allowed {allowed}); got RobotMode={state['mode']}")
        return state

    def _normalize_state(self, state: Any) -> dict[str, Any]:
        if not isinstance(state, dict):
            raise MobileArmError("controller state is missing or malformed")
        pose = state.get("pose")
        joints = state.get("joints")
        mode = state.get("mode")
        if not isinstance(pose, (list, tuple)) or len(pose) != 6:
            raise MobileArmError("controller state is missing a six-value pose")
        if not isinstance(joints, (list, tuple)) or len(joints) != 6:
            raise MobileArmError("controller state is missing six joint values")
        try:
            clean_pose = [float(value) for value in pose]
            clean_joints = [float(value) for value in joints]
        except (TypeError, ValueError) as error:
            raise MobileArmError("controller state contains non-numeric values") from error
        if not all(math.isfinite(value) for value in (*clean_pose, *clean_joints)):
            raise MobileArmError("controller state contains stale or non-finite values")
        source = state.get("source")
        if self.live and source != "dashboard":
            raise MobileArmError(
                "fresh dashboard state is unavailable; no command was sent")
        if isinstance(mode, bool):
            raise MobileArmError("controller state is missing a valid RobotMode")
        try:
            clean_mode = int(mode)
        except (TypeError, ValueError, OverflowError) as error:
            raise MobileArmError("controller state is missing a valid RobotMode") from error
        if clean_mode != mode or not 1 <= clean_mode <= 11:
            raise MobileArmError(f"controller returned an invalid RobotMode: {mode!r}")
        return {"pose": clean_pose, "joints": clean_joints, "mode": clean_mode,
                "source": source}

    @staticmethod
    def _snapshot_from_state(state: dict[str, Any]) -> dict[str, Any]:
        return {"pose": list(state["pose"]),
                "joints": list(state["joints"]),
                "mode": int(state["mode"])}

    def _sync_jog(self, state: dict[str, Any]) -> None:
        if self._jog is not None:
            self._jog.xyz = list(state["pose"][:3])
            self._jog.rpy = list(state["pose"][3:])
            self._jog.joints = list(state["joints"])

    @staticmethod
    def _require_same_pose(before: dict[str, Any], after: dict[str, Any]) -> None:
        position_delta = max(abs(before["pose"][i] - after["pose"][i])
                             for i in range(3))
        joint_delta = max(abs(before["joints"][i] - after["joints"][i])
                          for i in range(6))
        if position_delta > STATE_STABILITY_TOL_MM or joint_delta > STATE_STABILITY_TOL_DEG:
            raise MobileArmError(
                "arm state changed while checking the target; refusing a stale move")

    @staticmethod
    def _require_drive_pose_consistency(before: dict[str, Any],
                                        after: dict[str, Any],
                                        in_flight_mm: float = 0.0) -> None:
        # During our own ServoJ stream a single previous microtarget can still
        # be in flight. Allow that bounded displacement, but reject larger
        # drift before adding another target.
        position_delta = max(abs(before["pose"][i] - after["pose"][i])
                             for i in range(3))
        max_delta = max(MAX_DRIVE_SPEED_MM_S * MAX_SERVO_PERIOD_S, in_flight_mm) + ARRIVAL_TOL_MM
        if position_delta > max_delta:
            raise MobileArmError(
                "arm moved farther than one bounded joystick microtarget; stopping drive")

    @staticmethod
    def _axis_index(axis: str) -> int:
        if not isinstance(axis, str) or axis.upper() not in "XYZ" or len(axis) != 1:
            raise ValueError("axis must be X, Y, or Z")
        return "XYZ".index(axis.upper())

    @staticmethod
    def _step_size(mm: float) -> float:
        if isinstance(mm, bool):
            raise ValueError("mm must be 0.5, 1, or 5")
        try:
            value = float(mm)
        except (TypeError, ValueError) as error:
            raise ValueError("mm must be 0.5, 1, or 5") from error
        if not math.isfinite(value) or not any(math.isclose(value, allowed, abs_tol=1e-9)
                                               for allowed in ALLOWED_STEP_MM):
            raise ValueError("mm must be 0.5, 1, or 5")
        return value

    @staticmethod
    def _drive_vector(sample: Sequence[float]) -> tuple[float, float, float]:
        if not isinstance(sample, (list, tuple)) or len(sample) != 3:
            raise ValueError("joystick sample must contain three normalized XYZ values")
        try:
            vector = tuple(float(value) for value in sample)
        except (TypeError, ValueError) as error:
            raise ValueError("joystick sample contains non-numeric values") from error
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("joystick sample contains non-finite values")
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude > 1.000001:
            raise ValueError("joystick sample must be normalized inside the unit sphere")
        return vector  # type: ignore[return-value]

    @staticmethod
    def _require_event(cancel: threading.Event) -> None:
        if not callable(getattr(cancel, "is_set", None)):
            raise TypeError("cancel must be a threading.Event")

    def _wait_tick(self, cancel: threading.Event, period: float = SERVO_PERIOD_S,
                   tick_start: float | None = None) -> None:
        remaining = max(0.0, period - (self._clock() - tick_start)) if tick_start is not None else period
        cancel.wait(remaining)
        if cancel.is_set():
            raise JogStopped("drive cancelled")

    def _send_servo(self, xyz: Sequence[float], rpy: Sequence[float],
                    joints: Sequence[float], period: float) -> None:
        if not self.live:
            self._backend.servo(xyz, rpy, joints, period)
            return
        if self._driver is None or self._backend is None:
            raise MobileArmError("live ServoJ requested without a leased driver")
        # Match DobotApiMove.ServoJ's exact six-decimal frame and wait for the
        # strict status-0 echo. _send_motion does not retry ambiguous writes.
        fields = [*(float(value) for value in joints), float(period)]
        command = "ServoJ({},{},gain=500)".format(
            ",".join(f"{value:f}" for value in fields[:-1]), f"{fields[-1]:f}")
        self._driver._send_motion(self._backend.rig, command, "mobile arm ServoJ")

    def _wait_for_step(self, xyz: Sequence[float], rpy: Sequence[float],
                       joints: Sequence[float], cancel: threading.Event) -> dict[str, Any]:
        if not self.live:
            return self._backend.wait(xyz, rpy, joints, cancel.is_set)
        backend = self._backend
        had_override = "stop" in getattr(backend, "__dict__", {})
        previous_override = getattr(backend, "__dict__", {}).get("stop")
        backend.stop = self._safe_stop
        try:
            return backend.wait(xyz, rpy, joints, cancel.is_set)
        finally:
            if had_override:
                backend.stop = previous_override
            else:
                try:
                    delattr(backend, "stop")
                except AttributeError:
                    pass

    def _servo_end(self) -> None:
        # Each live ServoJ consumed its own strict echo, so no reply drain is
        # needed. SimArm records stream completion for deterministic tests.
        if not self.live:
            self._backend.servo_end()

    def _end_servo_and_stop(self) -> None:
        try:
            self._servo_end()
        except Exception:
            pass
        self._safe_stop()

    def _safe_stop(self) -> None:
        if not self._motion_pending or self._stop_sent or self._backend is None:
            return
        try:
            if self.live:
                if self._driver is None:
                    raise MobileArmError("StopScript requested without a live driver")
                report = self._driver.stop([self.arm])
                entry = report.get("stopped", {}).get(self.arm, {})
                reply = entry.get("stop_script")
                from .chemistry.driver import _reply_status

                if _reply_status(reply) != 0:
                    raise RuntimeError(
                        f"StopScript on {self.arm} was not acknowledged with status 0: {reply!r}")
            else:
                self._backend.stop()
        finally:
            self._stop_sent = True
            # driver.stop() latches a flag that also blocks the gripper; this
            # adapter tracks its own stop, so clear the driver's latch.
            if self._driver is not None:
                self._driver._stop_requested = False

    def _require_open(self) -> None:
        if not self._opened or self._backend is None or self._jog is None:
            raise MobileArmError("MobileArm is not open.")

    @staticmethod
    def _check_cancel(cancel: threading.Event | None) -> None:
        if cancel is None:
            return
        MobileArm._require_event(cancel)
        if cancel.is_set():
            raise JogStopped("operation cancelled")

