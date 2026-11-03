"""Live SDK driver for the registered chemistry atoms.

Why this module exists: catalog.json names
task_skills.chemistry.driver.LiveDriver.execute as the implementation of every
atom, and cli.py imports it only under --live. Importing this module has no side
effects, opens nothing, and does not require pyserial; the gripper module is
imported lazily by the one atom that needs it.

Rules inherited from the validated field scripts in 2026-10-15/left-arm-tube and
deliberately not re-invented here:

* Session: device lease -> ClearError -> EnableRobot -> speed/accel -> Tool0/User0.
  The arm is never homed and never disabled, and a failure stops the queue while
  keeping whatever the jaws hold.
* Frames on port 30003 are accepted only when the controller echoes back the
  exact frame that was requested. An empty reply, a timeout, a mismatched echo or
  an extra frame is not success. Only the two documented retryable statuses
  (-10000 busy/dropped, -2 not in TCP remote mode) get a single resend.
* The gripper convention is measured, not assumed: 0 is the closed target and
  255 is open. Position feedback never proves that an object is held, so no atom
  ever emits an object-held fact.
* Preflight is read-only: it may read state, request inverse/forward solutions
  and open the 30004 broadcast, but it must not clear errors, enable, configure,
  move or touch the gripper. Opening the Feetech port is not read-only - the
  driver writes a torque limit while constructing it - so the gripper release
  gate is enforced at execution instead, before anything moves.
"""

from __future__ import annotations

import math
import os
import re
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..device_lease import DeviceLease
from ..layout import physical_nova_home_degrees
from . import geometry


DASHBOARD_PORT = 29999
MOTION_PORT = 30003
FEEDBACK_PORT = 30004
FEEDBACK_PACKET = 1440
FEEDBACK_MAGIC = struct.pack("<Q", 0x123456789ABCDEF)
FEEDBACK_FIELDS = {
    "robot_mode": ("<Q", 24),
    "q_actual": ("<6d", 432),
    "tool_vector_actual": ("<6d", 624),
    "enable_status": ("<B", 1026),
    "running_status": ("<B", 1028),
    "error_status": ("<B", 1029),
}

# The controller answers these while it is busy or while TCP remote mode is off.
RETRYABLE_STATUS = (-10000, -2)
DASHBOARD_ATTEMPTS = 3
DASHBOARD_RETRY_DELAY_S = 1.2
# Port 30003 is connected with a 0.05 s socket timeout by the SDK; a motion frame
# can legitimately arrive later than that, so the strict reader widens the window
# instead of treating the first empty read as an answer.
MOTION_ACK_TIMEOUT_S = 2.0

ARRIVAL_TOL_DEG = 0.5
ARRIVAL_SAMPLES = 3
ARRIVAL_TIMEOUT_S = 180.0
POLL_INTERVAL_S = 0.2
STALL_S = 12.0

POSITION_TOL_MM = 0.5
ANGLE_TOL_DEG = 0.5
SOURCE_JOINT_TOL_DEG = 1.0
SOURCE_POSE_TOL_MM = 0.5
SOURCE_ANGLE_TOL_DEG = 0.5
J5_MARGIN_DEG = 0.25
FK_POSITION_TOLERANCE_MM = 0.1
FK_ORIENTATION_TOLERANCE_DEG = 0.1
LINEAR_SCAN_SAMPLES = 21

ROTATE_START_TOOL_X_TILT_TOL_DEG = 1.0
ROTATE_START_JAW_HORIZONTAL_TOL_DEG = 0.05
ROTATE_JAW_GRAVITY_TOL_DEG = 0.1

GRIPPER_TOLERANCE = 3
GRIPPER_MIN_TRAVEL = 10
GRIPPER_STABLE_SAMPLES = 5
GRIPPER_TIMEOUT_S = 20.0
GRIPPER_POLL_S = 0.2
GRIPPER_OPEN_MINIMUM = 250

def default_lock_path() -> str:
    """Shared rig owner path, kept identical to the field scripts on Windows."""
    override = os.environ.get("DOBOT_DEVICE_LOCK_PATH")
    if override:
        return override
    if os.name == "nt":
        return r"C:\DobotRuntime\device-owner"
    return str(Path(__file__).resolve().parent / "artifacts" / "device-owner")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _six(values: Any, label: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 6:
        raise ValueError(f"{label} must contain exactly six values")
    return [_finite(value, f"{label}[{index}]") for index, value in enumerate(values)]


def angle_delta(current: float, target: float) -> float:
    """Signed shortest difference in degrees, so a +/-360 wrap is not a jump."""
    return math.remainder(current - target, 360.0)


def j5_margin(joint: float) -> float:
    """Distance of J5 from the 180 deg wrist singularity."""
    return abs(math.remainder(joint, 180.0))


def _reply_status(reply: Any) -> int | None:
    match = re.match(r"^\s*([+-]?\d+)", str(reply))
    return int(match.group(1)) if match else None


def parse_frame(reply: Any) -> tuple[int | None, tuple[float, ...] | None, str]:
    """Tolerant dashboard frame parser -> (status, six values or None, message)."""
    match = re.match(r"^\s*([+-]?\d+)\s*,\s*\{([^{}]*)\}", str(reply))
    if not match:
        return None, None, f"unparsable reply: {reply!r}"
    status = int(match.group(1))
    body = match.group(2).strip()
    if status != 0:
        return status, None, body or "(empty)"
    try:
        values = tuple(float(value.strip()) for value in body.split(","))
    except ValueError:
        return status, None, f"non-numeric body: {body!r}"
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        return status, None, f"expected six finite values, got {body!r}"
    return status, values, body


def parse_single(reply: Any) -> int | None:
    """Read the payload of a single-value reply such as 0,{5},RobotMode();."""
    match = re.search(r"\{([^{}]*)\}", str(reply))
    if not match:
        return None
    values = [value.strip() for value in match.group(1).split(",") if value.strip()]
    if len(values) != 1:
        return None
    try:
        return int(float(values[0]))
    except ValueError:
        return None


def joint_frame(joints: Iterable[float]) -> str:
    return "JointMovJ({})".format(",".join(f"{value:.6f}" for value in joints))


def linear_frame(xyz: Iterable[float], rpy: Iterable[float]) -> str:
    fields = ",".join(f"{value:.6f}" for value in (*xyz, *rpy))
    return f"MovL({fields})"


def inverse_frame(xyz: Iterable[float], rpy: Iterable[float], seed: Iterable[float],
                  user: int, tool: int) -> str:
    fields = ",".join(f"{value:.9f}" for value in (*xyz, *rpy))
    seed_fields = ",".join(f"{value:.9f}" for value in seed)
    return f"InverseSolution({fields},{user},{tool},1,{{{seed_fields}}})"


def forward_frame(joints: Iterable[float], user: int, tool: int) -> str:
    fields = ",".join(f"{value:.9f}" for value in joints)
    return f"PositiveSolution({fields},{user},{tool})"


class _Feedback:
    """Read-only 30004 broadcast; needs no dashboard permission and never commands."""

    def __init__(self, ip: str, timeout: float = 5.0, samples: int = 2):
        self.sock = socket.create_connection((ip, FEEDBACK_PORT), timeout=timeout)
        self.sock.settimeout(timeout)
        self.samples = samples

    def read(self) -> dict[str, Any]:
        values: dict[str, Any] | None = None
        for _ in range(self.samples):
            values = self._read_packet()
        return values  # type: ignore[return-value]

    def _read_packet(self) -> dict[str, Any]:
        """The stream is a run of 1440-byte packets, so search for the magic.

        While the controller is not in TCP remote mode the stream still flows but
        every payload is zero; those packets are rejected here so a zeroed stream
        is reported as unusable instead of being mistaken for a real state.
        """
        buffer = b""
        deadline = time.time() + 5.0
        zeroed = False
        while time.time() < deadline:
            chunk = self.sock.recv(8192)
            if not chunk:
                raise RuntimeError("feedback socket closed")
            buffer += chunk
            index = buffer.find(FEEDBACK_MAGIC)
            while index >= 0 and len(buffer) - index >= FEEDBACK_PACKET:
                record = buffer[index:index + FEEDBACK_PACKET]
                if record.startswith(FEEDBACK_MAGIC):
                    values = self._decode(record)
                    if values is not None:
                        return values
                    zeroed = True
                index = buffer.find(FEEDBACK_MAGIC, index + 1)
            buffer = buffer[-FEEDBACK_PACKET:]
        raise RuntimeError(
            "30004 feedback carries no usable packet"
            + (" (all payloads were zero - the controller is most likely not in TCP remote mode)"
               if zeroed else ""))

    @staticmethod
    def _decode(record: bytes) -> dict[str, Any] | None:
        values: dict[str, Any] = {}
        for name, (fmt, offset) in FEEDBACK_FIELDS.items():
            size = struct.calcsize(fmt)
            values[name] = struct.unpack(fmt, record[offset:offset + size])
        joints = [round(value, 6) for value in values["q_actual"]]
        pose = [round(value, 6) for value in values["tool_vector_actual"]]
        mode = int(values["robot_mode"][0])
        enabled = int(values["enable_status"][0])
        if not 1 <= mode <= 10:
            return None
        if enabled not in (0, 1):
            return None
        if not all(math.isfinite(value) and abs(value) <= 360.0 for value in joints):
            return None
        if not all(math.isfinite(value) and abs(value) <= 3000.0 for value in pose):
            return None
        values["q_actual"] = joints
        values["pose_actual"] = pose
        return values

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class _ArmRig:
    """Lazily opened connections for one physical arm."""

    def __init__(self, arm: str, ip: str):
        self.arm = arm
        self.ip = ip
        self.dashboard: Any = None
        self.motion: Any = None
        self.feedback: _Feedback | None = None
        self.gripper: Any = None
        self.configured = False


class LiveDriver:
    """Strict, lease-guarded live implementation of the five registered atoms.

    Construction is inert. Every hardware action additionally requires
    allow_live=True and a confirmation callable that returns True, so a caller
    cannot reach the rig by accident.
    """

    def __init__(self, profile: dict[str, Any] | None = None, *, allow_live: bool = False,
                 confirmation: Callable[[], bool] | None = None,
                 lock_path: str | os.PathLike[str] | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self._profile = profile
        self._allow_live = bool(allow_live)
        self._confirmation = confirmation
        self._lock_path = Path(lock_path).expanduser() if lock_path is not None else Path(default_lock_path())
        self._log_fn = log
        self._rigs: dict[str, _ArmRig] = {}
        self._lease: DeviceLease | None = None
        self._stop_requested = False
        self._finalized = False
        # Two arms may be under command at once, so the shared rig bookkeeping
        # needs a lock: the lease is one resource for the whole machine and two
        # threads racing to take it would make one of them fail on a rig that is
        # free. Each arm also gets its own gate so a single arm can never have
        # two frames in flight - validation should already prevent that, and this
        # is the second line of defence.
        self._session_lock = threading.Lock()
        self._arm_locks: dict[str, threading.Lock] = {}

    def _arm_lock(self, arm: str) -> threading.Lock:
        return self._arm_locks.setdefault(arm, threading.Lock())

    # ------------------------------------------------------------------ plumbing

    def _log(self, text: str) -> None:
        if self._log_fn is not None:
            try:
                self._log_fn(str(text))
            except Exception:  # noqa: BLE001 - a broken logger must not stop the driver
                pass

    def _sleep(self, seconds: float) -> None:
        """Sleep that honours a stop request instead of blocking through it."""
        deadline = time.time() + max(0.0, float(seconds))
        while time.time() < deadline:
            if self._stop_requested:
                raise RuntimeError("stop requested during dwell")
            time.sleep(min(POLL_INTERVAL_S, max(0.0, deadline - time.time())))

    def _profile_for(self, profile: dict[str, Any] | None) -> dict[str, Any]:
        selected = profile if profile is not None else self._profile
        if not isinstance(selected, dict) or "poses" not in selected:
            raise RuntimeError("LiveDriver needs the chemistry profile (poses/arms/pour)")
        return selected

    def _require_live(self, action: str) -> None:
        if not self._allow_live:
            raise RuntimeError(
                f"{action} refused: LiveDriver was constructed without allow_live=True")

    def _rig(self, arm: str, profile: dict[str, Any], *, with_motion: bool = True) -> _ArmRig:
        rig = self._rigs.get(arm)
        if rig is None:
            arms = profile.get("arms")
            if not isinstance(arms, dict) or arm not in arms:
                raise RuntimeError(f"profile has no binding for the {arm} arm")
            ip = arms[arm].get("ip")
            if not isinstance(ip, str) or not ip:
                raise RuntimeError(f"profile.arms.{arm}.ip must be a non-empty string")
            rig = _ArmRig(arm, ip)
            self._rigs[arm] = rig
        if rig.dashboard is None:
            from dobot_control.robots.dobot_api import DobotApiDashboard

            rig.dashboard = DobotApiDashboard(rig.ip, DASHBOARD_PORT)
        if with_motion and rig.motion is None:
            from dobot_control.robots.dobot_api import DobotApiMove

            rig.motion = DobotApiMove(rig.ip, MOTION_PORT)
        if rig.feedback is None:
            try:
                rig.feedback = _Feedback(rig.ip)
            except Exception as error:  # noqa: BLE001
                self._log(f"note: 30004 feedback unavailable on {arm} ({type(error).__name__}: {error})")
        return rig

    def _gripper(self, rig: _ArmRig, profile: dict[str, Any]) -> Any:
        if rig.gripper is None:
            # Imported here on purpose: this module pulls in pyserial and the
            # Feetech SDK, and importing driver.py must stay dependency-light.
            from dobot_control.gripper.dobot_gripper import DobotGripper

            binding = self._profile_for(profile)["arms"][rig.arm]
            rig.gripper = DobotGripper(port=binding["gripper_port"],
                                       id_name=int(binding["gripper_id"]),
                                       servo_pos=list(binding["servo_pos"]))
        return rig.gripper

    # ------------------------------------------------------------- state reading

    def _dashboard_command(self, rig: _ArmRig, command: str, label: str) -> tuple[str, list[str]]:
        with self._arm_lock(rig.arm):
            return self._dashboard_command_unlocked(rig, command, label)

    def _dashboard_command_unlocked(self, rig: _ArmRig, command: str,
                                    label: str) -> tuple[str, list[str]]:
        replies: list[str] = []
        for attempt in range(DASHBOARD_ATTEMPTS):
            reply = str(rig.dashboard.sendRecvMsg(command))
            replies.append(reply)
            if _reply_status(reply) not in RETRYABLE_STATUS:
                return reply, replies
            if attempt < DASHBOARD_ATTEMPTS - 1:
                self._sleep(DASHBOARD_RETRY_DELAY_S)
        return replies[-1], replies

    def _read_six(self, rig: _ArmRig, command: str, label: str) -> list[float]:
        reply, _ = self._dashboard_command(rig, command, label)
        status, values, message = parse_frame(reply)
        if status != 0 or values is None:
            hint = " (controller is not in TCP remote mode)" if status == -2 else ""
            raise RuntimeError(f"{label} on {rig.arm}: status {status} ({message}){hint}")
        return list(values)

    def _read_state(self, rig: _ArmRig) -> dict[str, Any]:
        """Dashboard first, 30004 broadcast second, and an explicit error if neither."""
        reason: str | None = None
        if rig.dashboard is not None:
            try:
                joints = self._read_six(rig, "GetAngle()", "GetAngle")
                mode_reply, _ = self._dashboard_command(rig, "RobotMode()", "RobotMode")
                if _reply_status(mode_reply) != 0:
                    raise RuntimeError(f"RobotMode() status {_reply_status(mode_reply)} ({mode_reply!r})")
                mode = parse_single(mode_reply)
                if mode is None:
                    raise RuntimeError(f"RobotMode() reply carries no single value: {mode_reply!r}")
                pose = self._read_six(rig, "GetPose()", "GetPose")
                return {"source": "dashboard", "joints": [round(value, 6) for value in joints],
                        "pose": [round(value, 6) for value in pose], "mode": mode}
            except Exception as error:  # noqa: BLE001
                reason = f"{type(error).__name__}: {error}"
        if rig.feedback is not None:
            try:
                values = rig.feedback.read()
                return {"source": "30004 broadcast", "joints": list(values["q_actual"]),
                        "pose": list(values["pose_actual"]), "mode": int(values["robot_mode"][0]),
                        "enabled": int(values["enable_status"][0]),
                        "error_status": int(values["error_status"][0])}
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(
                    f"no robot state available for the {rig.arm} arm: dashboard -> {reason}; "
                    f"30004 -> {type(error).__name__}: {error}. Put that arm into TCP/IP remote "
                    "control mode (and out of DobotStudio jog) first.") from error
        raise RuntimeError(f"no robot state available for the {rig.arm} arm: dashboard -> {reason}")

    def _forward_solution(self, rig: _ArmRig, joints: Sequence[float],
                          profile: dict[str, Any]) -> dict[str, Any]:
        frame = profile["frame"]
        command = forward_frame(joints, int(frame["user"]), int(frame["tool"]))
        reply, _ = self._dashboard_command(rig, command, "PositiveSolution")
        status, values, message = parse_frame(reply)
        if status != 0 or values is None:
            raise RuntimeError(f"PositiveSolution on {rig.arm}: status {status} ({message})")
        return {"xyz": list(values[:3]), "rpy": list(values[3:])}

    def _inverse_solution(self, rig: _ArmRig, xyz: Sequence[float], rpy: Sequence[float],
                          seed: Sequence[float], profile: dict[str, Any]) -> list[float]:
        frame = profile["frame"]
        command = inverse_frame(xyz, rpy, seed, int(frame["user"]), int(frame["tool"]))
        reply, _ = self._dashboard_command(rig, command, "InverseSolution")
        status, values, message = parse_frame(reply)
        if status != 0 or values is None:
            hint = " (controller is not in TCP remote mode)" if status == -2 else ""
            raise RuntimeError(f"InverseSolution on {rig.arm}: status {status} ({message}){hint}")
        return list(values)

    # ---------------------------------------------------------------- motion I/O

    def _send_motion(self, rig: _ArmRig, command: str, label: str) -> dict[str, Any]:
        """Send one frame and accept only the controller's exact echo of it.

        Port 30003 frames are queue instructions; the 0.05 s SDK timeout can
        expire before the echo arrives, so the read window is widened here rather
        than treating the first empty read as an answer.
        """
        with self._arm_lock(rig.arm):
            return self._send_motion_unlocked(rig, command, label)

    def _send_motion_unlocked(self, rig: _ArmRig, command: str, label: str) -> dict[str, Any]:
        self._require_live(label)
        expected = f"0,{{}},{command};"
        sock = rig.motion.socket_dobot
        previous_timeout = sock.gettimeout()
        try:
            sock.settimeout(MOTION_ACK_TIMEOUT_S)
            rig.motion.send_data(command)
            buffer = ""
            deadline = time.time() + MOTION_ACK_TIMEOUT_S
            while time.time() < deadline:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                if expected in buffer:
                    break
        finally:
            sock.settimeout(previous_timeout)
        if buffer.strip() != expected:
            raise RuntimeError(
                f"{label} on {rig.arm}: expected the exact status-0 echo {expected!r}, got {buffer!r}; "
                "no blind retry")
        return {"command": command, "expected_ack": expected, "raw_reply": buffer,
                "result": "acknowledged"}

    def _wait_for_joints(self, rig: _ArmRig, target: Sequence[float], label: str,
                         timeout: float | None = None) -> dict[str, Any]:
        allowed = ARRIVAL_TIMEOUT_S if timeout is None else float(timeout)
        deadline = time.time() + allowed
        stable = 0
        last_progress = time.time()
        previous: float | None = None
        samples: list[dict[str, Any]] = []
        while time.time() < deadline:
            state = self._read_state(rig)
            worst = max(abs(angle_delta(state["joints"][index], target[index])) for index in range(6))
            samples.append({"worst_joint_error_deg": round(worst, 4), "mode": state["mode"],
                            "source": state["source"]})
            if previous is not None and abs(worst - previous) > 0.01:
                last_progress = time.time()
            previous = worst
            if state["mode"] == 5 and worst <= ARRIVAL_TOL_DEG:
                stable += 1
                if stable >= ARRIVAL_SAMPLES:
                    return {"mode": state["mode"], "worst_joint_error_deg": round(worst, 4),
                            "samples": samples[-ARRIVAL_SAMPLES:]}
            else:
                stable = 0
            if time.time() - last_progress > STALL_S:
                raise RuntimeError(
                    f"{label} on {rig.arm} made no progress for {STALL_S:.0f} s "
                    f"(last worst joint error {worst:.4f} deg, RobotMode {state['mode']}"
                    + (", 10 = paused" if state["mode"] == 10 else "") + ")")
            time.sleep(POLL_INTERVAL_S)
        raise RuntimeError(f"{label} on {rig.arm} did not reach the joint target within "
                           f"{allowed:.0f} s")

    def _wait_for_pose(self, rig: _ArmRig, xyz: Sequence[float], rpy: Sequence[float],
                       label: str) -> dict[str, Any]:
        deadline = time.time() + ARRIVAL_TIMEOUT_S
        stable = 0
        last_progress = time.time()
        previous: float | None = None
        samples: list[dict[str, Any]] = []
        while time.time() < deadline:
            state = self._read_state(rig)
            pose = state["pose"]
            position_error = max(abs(pose[index] - xyz[index]) for index in range(3))
            orientation_error = geometry.angle_between_degrees(pose[3:], rpy)
            samples.append({"position_error_mm": round(position_error, 4),
                            "orientation_error_deg": round(orientation_error, 4),
                            "mode": state["mode"], "source": state["source"]})
            metric = position_error + orientation_error
            if previous is not None and abs(metric - previous) > 0.01:
                last_progress = time.time()
            previous = metric
            if state["mode"] == 5 and position_error <= POSITION_TOL_MM \
                    and orientation_error <= ANGLE_TOL_DEG:
                stable += 1
                if stable >= ARRIVAL_SAMPLES:
                    return {"mode": state["mode"], "position_error_mm": round(position_error, 4),
                            "orientation_error_deg": round(orientation_error, 4),
                            "samples": samples[-ARRIVAL_SAMPLES:]}
            else:
                stable = 0
            if time.time() - last_progress > STALL_S:
                raise RuntimeError(
                    f"{label} on {rig.arm} made no progress for {STALL_S:.0f} s "
                    f"(last error {position_error:.4f} mm / {orientation_error:.4f} deg, "
                    f"RobotMode {state['mode']})")
            time.sleep(POLL_INTERVAL_S)
        raise RuntimeError(f"{label} on {rig.arm} did not reach the Cartesian target within "
                           f"{ARRIVAL_TIMEOUT_S:.0f} s")

    # ------------------------------------------------------------------ gripper

    def _gripper_move(self, rig: _ArmRig, profile: dict[str, Any], position: int, speed: int) -> str:
        gripper = self._gripper(rig, profile)
        result = gripper.move(position, speed, 0)
        comm_result, servo_error = (list(result) + [None, None])[:2]
        if comm_result != 0 or servo_error != 0:
            raise RuntimeError(f"gripper on {rig.arm} rejected the move: {result!r}")
        return f"comm_result={comm_result} servo_error={servo_error}"

    def _wait_for_gripper(self, rig: _ArmRig, profile: dict[str, Any], start: int,
                          target: int) -> dict[str, Any]:
        """Require travel-or-target plus stable samples.

        The direction is verified by travel rather than by a hard-coded sign, so
        the measured 255-open / 0-closed convention cannot silently invert.
        """
        gripper = self._gripper(rig, profile)
        deadline = time.time() + GRIPPER_TIMEOUT_S
        previous: int | None = None
        stable = 0
        samples: list[int] = []
        while time.time() < deadline:
            if self._stop_requested:
                raise RuntimeError("stop requested while waiting for the gripper")
            last = int(gripper.get_current_position())
            samples.append(last)
            moved = abs(last - start)
            if previous is not None and abs(last - previous) <= GRIPPER_TOLERANCE:
                stable += 1
            else:
                stable = 0
            if (moved >= GRIPPER_MIN_TRAVEL or abs(last - target) <= GRIPPER_TOLERANCE) \
                    and stable >= GRIPPER_STABLE_SAMPLES:
                return {"start": start, "target": target, "final": last,
                        "samples": samples[-GRIPPER_STABLE_SAMPLES:]}
            previous = last
            time.sleep(GRIPPER_POLL_S)
        raise RuntimeError(f"gripper on {rig.arm} did not reach {target} and stabilise; "
                           f"last {samples[-1] if samples else None}")

    # ------------------------------------------------------------------ preflight

    def preflight(self, graph: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read-only preflight for a whole graph. Never commands the rig.

        Each node is checked against a *simulated* per-arm state that starts from
        the live read and is advanced by every node before it in topological
        order. Checking all nodes against one frozen pre-run snapshot instead
        would demand that an arm sit at several different taught poses at once,
        which no reachable start configuration can satisfy. The declared start
        state is still enforced: a node whose reference no earlier node reaches
        is rejected, because the simulated state never gets there either.
        """
        selected = self._profile_for(profile)
        report: dict[str, Any] = {
            "driver": "LiveDriver",
            "live_enabled": self._allow_live,
            "hardware_commands_sent": False,
            "checks": [],
        }
        if not self._allow_live:
            report["success"] = False
            report["note"] = ("constructed without allow_live=True: no connection was opened and "
                              "no check was performed")
            return report

        nodes = {node["id"]: node for node in graph["nodes"]}
        order = graph.get("topological_order") or [node["id"] for node in graph["nodes"]]
        arms = sorted({nodes[node_id]["arm"] for node_id in order})
        checks: list[dict[str, Any]] = []
        failure: str | None = None
        states: dict[str, dict[str, Any]] = {}
        for arm in arms:
            try:
                rig = self._rig(arm, selected, with_motion=False)
                state = self._read_state(rig)
                states[arm] = state
                checks.append({"node": f"<{arm} state>", "success": True, "mode": state["mode"],
                               "source": state["source"],
                               "joints": [round(value, 4) for value in state["joints"]]})
            except Exception as error:  # noqa: BLE001
                failure = f"{arm} arm state: {type(error).__name__}: {error}"
                break
        if failure is None:
            for node_id in order:
                node = nodes[node_id]
                try:
                    entry = self._preflight_node(node, selected, states[node["arm"]])
                    entry.setdefault("node", node_id)
                except Exception as error:  # noqa: BLE001
                    entry = {"node": node_id, "skill": node["skill"], "arm": node["arm"],
                             "success": False, "error": f"{type(error).__name__}: {error}"}
                checks.append(entry)
                if entry.get("success") is False:
                    failure = f"{node_id}: {entry.get('error') or 'check failed'}"
                    break
                self._advance_state(node, selected, states[node["arm"]], entry)
        report["checks"] = checks
        report["success"] = failure is None
        if failure is not None:
            report["error"] = failure
        return report

    def _advance_state(self, node: dict[str, Any], profile: dict[str, Any],
                       state: dict[str, Any], entry: dict[str, Any]) -> None:
        """Move the simulated arm state to where this node leaves the arm.

        A gripper action and a dwell move nothing. A joint or linear move ends
        exactly on its target reference. A pivot ends on the last waypoint the
        controller actually solved, which is the tipped pose and not the launch
        pose; using the launch pose would silently excuse a graph that never
        untips the beaker.
        """
        skill = node["skill"]
        if skill in {"gripper.set_position", "control.hold"}:
            return
        if skill in {"arm.move_joint", "arm.move_linear"}:
            target_ref = node["parameters"]["target_ref"]
            target = self._pose(profile, target_ref)
            state["joints"] = [round(value, 6)
                               for value in _six(target.get("joints"), f"poses[{target_ref}].joints")]
            state["pose"] = [round(value, 6) for value in list(target["xyz"]) + list(target["rpy"])]
            return
        if skill == "arm.rotate_about_pivot":
            final = (entry.get("checks") or {}).get("final_waypoint")
            if not isinstance(final, dict):
                raise RuntimeError("rotate check did not report the waypoint it ends on")
            state["joints"] = [round(value, 6) for value in final["joints"]]
            state["pose"] = [round(value, 6) for value in final["pose"]]
            return
        if skill == "arm.move_to_home":
            final = (entry.get("checks") or {}).get("home_pose")
            if not isinstance(final, dict):
                raise RuntimeError("home check did not report the pose it ends on")
            state["joints"] = [round(value, 6) for value in final["joints"]]
            state["pose"] = [round(value, 6) for value in list(final["xyz"]) + list(final["rpy"])]
            return
        raise RuntimeError(f"cannot advance a simulated state for skill {skill!r}")

    def _preflight_node(self, node: dict[str, Any], profile: dict[str, Any],
                        state: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "arm.move_joint": self._check_move_joint,
            "arm.move_linear": self._check_move_linear,
            "arm.move_to_home": self._check_move_to_home,
            "gripper.set_position": self._check_gripper,
            "arm.rotate_about_pivot": self._check_rotate,
        }
        skill = node["skill"]
        if skill == "control.hold":
            return {"skill": skill, "arm": node["arm"], "success": True,
                    "checks": {"dwell_seconds": node["parameters"]["seconds"]}}
        if skill not in handlers:
            raise RuntimeError(f"unregistered chemistry skill: {skill!r}")
        return handlers[skill](node, profile, state)

    def _pose(self, profile: dict[str, Any], reference: str) -> dict[str, Any]:
        poses = profile.get("poses")
        if not isinstance(poses, dict) or reference not in poses:
            raise RuntimeError(f"pose reference {reference!r} is missing from profile.poses")
        return poses[reference]

    def _observe_source(self, rig: _ArmRig, profile: dict[str, Any], state: dict[str, Any],
                        source_ref: str | None) -> dict[str, Any]:
        """Compare a state against the declared start reference.

        Preflight feeds this the simulated state at that point in the graph;
        execution feeds it the live read, so a graph that skipped its
        precondition is refused rather than silently raced through.
        """
        if source_ref is None:
            return {"source_ref": None, "observed": False}
        expected = self._pose(profile, source_ref)
        joints = _six(expected.get("joints"), f"poses[{source_ref}].joints")
        travel = [round(angle_delta(state["joints"][index], joints[index]), 4) for index in range(6)]
        worst = max(abs(value) for value in travel)
        position_error = max(abs(state["pose"][index] - expected["xyz"][index]) for index in range(3))
        orientation_error = geometry.angle_between_degrees(state["pose"][3:], expected["rpy"])
        result = {"source_ref": source_ref, "observed": True, "max_joint_error_deg": round(worst, 4),
                  "position_error_mm": round(position_error, 4),
                  "orientation_error_deg": round(orientation_error, 4)}
        if worst > SOURCE_JOINT_TOL_DEG or position_error > SOURCE_POSE_TOL_MM \
                or orientation_error > SOURCE_ANGLE_TOL_DEG:
            raise RuntimeError(
                f"start state does not match {source_ref}: joint {worst:.4f} deg, "
                f"{position_error:.4f} mm, {orientation_error:.4f} deg")
        return result

    def _check_move_joint(self, node: dict[str, Any], profile: dict[str, Any],
                          state: dict[str, Any]) -> dict[str, Any]:
        rig = self._rig(node["arm"], profile, with_motion=False)
        parameters = node["parameters"]
        target_ref = parameters["target_ref"]
        target = self._pose(profile, target_ref)
        joints = _six(target.get("joints"), f"poses[{target_ref}].joints")
        tilt = geometry.tilt_from_vertical_deg(target["rpy"], "X")
        if tilt > parameters["tilt_limit_deg"] + 1e-9:
            raise RuntimeError(f"target tool X tilt {tilt:.4f} deg exceeds "
                               f"tilt_limit_deg {parameters['tilt_limit_deg']}")
        source = self._observe_source(rig, profile, state, parameters.get("source_ref"))
        forward = self._forward_solution(rig, joints, profile)
        position_error = max(abs(forward["xyz"][index] - target["xyz"][index]) for index in range(3))
        orientation_error = geometry.angle_between_degrees(forward["rpy"], target["rpy"])
        if position_error > POSITION_TOL_MM or orientation_error > ANGLE_TOL_DEG:
            raise RuntimeError(
                f"taught joints do not reproduce {target_ref}: forward solution differs by "
                f"{position_error:.4f} mm / {orientation_error:.4f} deg")
        travel = [round(angle_delta(state["joints"][index], joints[index]), 4) for index in range(6)]
        return {"skill": node["skill"], "arm": node["arm"], "target_ref": target_ref, "success": True,
                "checks": {"tool_x_tilt_deg": round(tilt, 6), "source": source,
                           "forward_solution": {"position_error_mm": round(position_error, 5),
                                                "orientation_error_deg": round(orientation_error, 5)},
                           "command_travel_deg": travel,
                           "max_command_travel_deg": round(max(abs(value) for value in travel), 4)}}

    def _check_move_to_home(self, node: dict[str, Any], profile: dict[str, Any],
                            state: dict[str, Any]) -> dict[str, Any]:
        """The home target is a library constant, so there is no taught pose to verify.

        What can still be checked is that the controller can solve the calibrated
        home joints at all - if the forward solution is refused, the arm cannot be
        parked there and the graph must not start. The end pose is reported so the
        simulated state of later nodes is the real home, not an assumption.
        """
        arm = node["arm"]
        rig = self._rig(arm, profile, with_motion=False)
        joints = [float(value) for value in physical_nova_home_degrees(arm)]
        source = self._observe_source(rig, profile, state, node["parameters"].get("source_ref"))
        forward = self._forward_solution(rig, joints, profile)
        travel = [round(angle_delta(state["joints"][index], joints[index]), 4) for index in range(6)]
        return {"skill": node["skill"], "arm": arm, "target": "layout.physical_nova_home_degrees",
                "success": True,
                "checks": {"source": source,
                           "home_pose": {"joints": [round(value, 6) for value in joints],
                                         "xyz": [round(value, 6) for value in forward["xyz"]],
                                         "rpy": [round(value, 6) for value in forward["rpy"]]},
                           "command_travel_deg": travel,
                           "max_command_travel_deg": round(max(abs(value) for value in travel), 4)}}

    def _check_move_linear(self, node: dict[str, Any], profile: dict[str, Any],
                           state: dict[str, Any]) -> dict[str, Any]:
        """Chained IK seeded by the previous solution, with a full FK round trip.

        The scan is a reference-path check only: controller blending under CP can
        still alter the executed path, so no claim about the physical trajectory
        follows from it.
        """
        rig = self._rig(node["arm"], profile, with_motion=False)
        parameters = node["parameters"]
        source_ref, target_ref = parameters["source_ref"], parameters["target_ref"]
        source_pose, target_pose = self._pose(profile, source_ref), self._pose(profile, target_ref)
        source = self._observe_source(rig, profile, state, source_ref)
        seed = list(state["joints"])
        waypoints: list[dict[str, Any]] = []
        for index in range(LINEAR_SCAN_SAMPLES):
            fraction = index / (LINEAR_SCAN_SAMPLES - 1)
            xyz = [source_pose["xyz"][axis]
                   + fraction * (target_pose["xyz"][axis] - source_pose["xyz"][axis])
                   for axis in range(3)]
            rpy = list(source_pose["rpy"]) if index < LINEAR_SCAN_SAMPLES - 1 else list(target_pose["rpy"])
            joints = self._inverse_solution(rig, xyz, rpy, seed, profile)
            margin = j5_margin(joints[4])
            if margin < J5_MARGIN_DEG:
                raise RuntimeError(f"linear sample {index} sits on the J5 singularity: "
                                   f"margin {margin:.3f} deg")
            forward = self._forward_solution(rig, joints, profile)
            position_error = max(abs(forward["xyz"][axis] - xyz[axis]) for axis in range(3))
            orientation_error = geometry.angle_between_degrees(forward["rpy"], rpy)
            tilt = geometry.tilt_from_vertical_deg(forward["rpy"], "X")
            waypoints.append({"index": index, "xyz": [round(value, 4) for value in xyz],
                              "j5_margin_deg": round(margin, 4),
                              "position_error_mm": round(position_error, 5),
                              "orientation_error_deg": round(orientation_error, 5),
                              "tool_x_tilt_deg": round(tilt, 6)})
            if position_error > POSITION_TOL_MM or orientation_error > ANGLE_TOL_DEG:
                raise RuntimeError(f"linear sample {index} round trip {position_error:.4f} mm / "
                                   f"{orientation_error:.4f} deg")
            if tilt > parameters["tilt_limit_deg"] + 1e-9:
                raise RuntimeError(f"linear sample {index} tool X tilt {tilt:.4f} deg exceeds "
                                   f"tilt_limit_deg {parameters['tilt_limit_deg']}")
            seed = joints
        return {"skill": node["skill"], "arm": node["arm"], "success": True,
                "checks": {"source": source, "samples": waypoints,
                           "max_position_error_mm": max(item["position_error_mm"] for item in waypoints),
                           "max_tool_x_tilt_deg": max(item["tool_x_tilt_deg"] for item in waypoints)}}

    def _check_gripper(self, node: dict[str, Any], profile: dict[str, Any],
                       state: dict[str, Any]) -> dict[str, Any]:
        """Validate what can be known without opening the Feetech port.

        Opening that port writes a torque limit, so it is not a read-only action;
        the measured opening and the release gate are therefore re-checked at
        execution, before anything moves.
        """
        parameters = node["parameters"]
        checks: dict[str, Any] = {"requested_position": parameters["position"],
                                  "release_authorised": bool(parameters["allow_release"]),
                                  "opening_checked_at_execution": True}
        at_ref = parameters.get("at_ref")
        if at_ref is not None:
            expected = self._pose(profile, at_ref)
            position_error = max(abs(state["pose"][index] - expected["xyz"][index]) for index in range(3))
            orientation_error = geometry.angle_between_degrees(state["pose"][3:], expected["rpy"])
            checks["at_ref"] = {"reference": at_ref, "position_error_mm": round(position_error, 4),
                                "orientation_error_deg": round(orientation_error, 4)}
            if position_error > SOURCE_POSE_TOL_MM or orientation_error > SOURCE_ANGLE_TOL_DEG:
                raise RuntimeError(f"arm is not at {at_ref}: {position_error:.4f} mm / "
                                   f"{orientation_error:.4f} deg")
        return {"skill": node["skill"], "arm": node["arm"], "success": True, "checks": checks}

    def _rotate_waypoints(self, node: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
        parameters = node["parameters"]
        source = self._pose(profile, parameters["source_ref"])
        return geometry.rotate_about_pivot(source["xyz"], source["rpy"], parameters["offset_mm"],
                                           parameters["tilt_deg"], parameters["segments"], "Y")

    def _check_rotate(self, node: dict[str, Any], profile: dict[str, Any],
                      state: dict[str, Any]) -> dict[str, Any]:
        """Every waypoint must solve, round-trip and keep the fingers vertical.

        One unreachable or twisting waypoint rejects the whole graph: the pour is
        a single continuous action and must not be silently shortened.
        """
        parameters = node["parameters"]
        rig = self._rig(node["arm"], profile, with_motion=False)
        waypoints = self._rotate_waypoints(node, profile)
        start = self._pose(profile, parameters["source_ref"])
        tool_x_tilt = geometry.tilt_from_vertical_deg(start["rpy"], "X")
        jaw_angle = geometry.angle_from_horizontal_deg(geometry.tool_axis(start["rpy"], "Y"))
        if tool_x_tilt > ROTATE_START_TOOL_X_TILT_TOL_DEG:
            raise RuntimeError(f"start tool X is {tool_x_tilt:.4f} deg from vertical; the pour must "
                               "start from the upright carry pose")
        if jaw_angle > ROTATE_START_JAW_HORIZONTAL_TOL_DEG:
            raise RuntimeError(f"start jaw-plate normal is {jaw_angle:.4f} deg out of horizontal; the "
                               "finger plates must be vertical at the start")
        source = self._observe_source(rig, profile, state, parameters["source_ref"])
        seed = list(state["joints"])
        samples: list[dict[str, Any]] = []
        final_waypoint: dict[str, Any] | None = None
        for waypoint in waypoints:
            joints = self._inverse_solution(rig, waypoint["xyz"], waypoint["reference_rpy"], seed, profile)
            forward = self._forward_solution(rig, joints, profile)
            position_error = max(abs(forward["xyz"][axis] - waypoint["xyz"][axis]) for axis in range(3))
            orientation_error = geometry.angle_between_degrees(forward["rpy"], waypoint["reference_rpy"])
            jaw_drift = geometry.angle_from_horizontal_deg(geometry.tool_axis(forward["rpy"], "Y"))
            samples.append({"step": waypoint["step"], "theta_deg": round(waypoint["theta_deg"], 4),
                            "position_error_mm": round(position_error, 5),
                            "orientation_error_deg": round(orientation_error, 5),
                            "jaw_normal_out_of_horizontal_deg": round(jaw_drift, 5),
                            "tool_x_tilt_deg": round(geometry.tilt_from_vertical_deg(forward["rpy"], "X"), 5)})
            if position_error > FK_POSITION_TOLERANCE_MM \
                    or orientation_error > FK_ORIENTATION_TOLERANCE_DEG:
                raise RuntimeError(f"pour waypoint {waypoint['step']} round trip "
                                   f"{position_error:.4f} mm / {orientation_error:.4f} deg")
            if jaw_drift > ROTATE_JAW_GRAVITY_TOL_DEG:
                raise RuntimeError(f"pour waypoint {waypoint['step']} jaw-plate normal is "
                                   f"{jaw_drift:.4f} deg out of horizontal")
            final_waypoint = {"joints": list(joints),
                              "pose": list(forward["xyz"]) + list(forward["rpy"])}
            seed = joints
        if final_waypoint is None:
            raise RuntimeError("the pour has no waypoints to solve")
        return {"skill": node["skill"], "arm": node["arm"], "success": True,
                "checks": {"source": source, "start_tool_x_tilt_deg": round(tool_x_tilt, 6),
                           "start_jaw_out_of_horizontal_deg": round(jaw_angle, 6),
                           "tips_deg": parameters["tilt_deg"], "segments": parameters["segments"],
                           "samples": samples,
                           "final_waypoint": final_waypoint,
                           "max_position_error_mm": max(item["position_error_mm"] for item in samples),
                           "max_jaw_out_of_horizontal_deg": max(
                               item["jaw_normal_out_of_horizontal_deg"] for item in samples)}}

    # ------------------------------------------------------------------- confirm

    def confirm(self) -> bool:
        """Fail closed: without an injected confirmation there is no consent."""
        if self._confirmation is None:
            self._log("no confirmation callable was injected; refusing to treat this as consent")
            return False
        return bool(self._confirmation())

    # ------------------------------------------------------------------ execution

    def _ensure_session(self, profile: dict[str, Any], arm: str, speed: int,
                        *, configure: bool = True, clear_errors: bool = True) -> _ArmRig:
        """Acquire the rig once, then bring the requested arm into remote control.

        ``configure=False`` opens the connections and takes the lease without
        clearing errors or enabling the arm, which is what a pure dwell needs: a
        five second wait must not power up a robot by itself.
        ``clear_errors=False`` skips ClearError, for callers that refuse to start
        on an alarm and leave clearing it to the operator at the pendant.
        """
        with self._session_lock:
            return self._open_session(profile, arm, speed, configure=configure,
                                      clear_errors=clear_errors)

    def _open_session(self, profile: dict[str, Any], arm: str, speed: int,
                      *, configure: bool = True, clear_errors: bool = True) -> _ArmRig:
        if self._lease is None:
            lease = DeviceLease(path=self._lock_path, purpose="chemistry-atomic-skills")
            lease.acquire()
            self._lease = lease
            self._log(f"device lease acquired: {lease.owner}")
        rig = self._rig(arm, profile, with_motion=True)
        if configure and not rig.configured:
            steps = [("EnableRobot", rig.dashboard.EnableRobot)]
            if clear_errors:
                steps.insert(0, ("ClearError", rig.dashboard.ClearError))
            self._log(f"{arm}: {' -> '.join(label for label, _ in steps)} -> "
                      f"speed/accel {speed}% -> Tool/User")
            for label, call in steps:
                if _reply_status(call()) != 0:
                    raise RuntimeError(f"{label} on {arm} was not acknowledged with status 0")
            for name in ("SpeedFactor", "AccJ", "SpeedJ", "AccL", "SpeedL"):
                if _reply_status(getattr(rig.dashboard, name)(speed)) != 0:
                    raise RuntimeError(f"{name} on {arm} was not acknowledged with status 0")
            frame = profile["frame"]
            if _reply_status(rig.dashboard.Tool(int(frame["tool"]))) != 0:
                raise RuntimeError(f"Tool{frame['tool']} on {arm} was not acknowledged with status 0")
            if _reply_status(rig.dashboard.User(int(frame["user"]))) != 0:
                raise RuntimeError(f"User{frame['user']} on {arm} was not acknowledged with status 0")
            rig.configured = True
        return rig

    def _configure_speed(self, rig: _ArmRig, speed: int, *, linear: bool) -> None:
        names = ("SpeedFactor", "SpeedL", "AccL") if linear else ("SpeedFactor", "SpeedJ", "AccJ")
        for name in names:
            if _reply_status(getattr(rig.dashboard, name)(speed)) != 0:
                raise RuntimeError(f"{name} on {rig.arm} was not acknowledged with status 0")

    def execute(self, node: dict[str, Any], profile: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run exactly one atom. Raises on failure; never claims physical success."""
        skill = node["skill"]
        self._require_live(str(skill))
        selected = self._profile_for(profile)
        arm = node["arm"]
        parameters = node["parameters"]
        handlers = {
            "arm.move_joint": self._run_move_joint,
            "arm.move_linear": self._run_move_linear,
            "arm.move_to_home": self._run_move_to_home,
            "gripper.set_position": self._run_gripper,
            "arm.rotate_about_pivot": self._run_rotate,
            "control.hold": self._run_hold,
        }
        if skill not in handlers:
            raise RuntimeError(f"unregistered chemistry skill: {skill!r}")
        result = handlers[skill](node, selected, arm, parameters)
        result.update({"skill": skill, "arm": arm, "parameters": parameters, "simulated": False,
                       "physical_success_claimed": False, "object_held_confirmed_emitted": False})
        return result

    def _run_move_joint(self, node: dict[str, Any], profile: dict[str, Any], arm: str,
                        parameters: dict[str, Any]) -> dict[str, Any]:
        target_ref = parameters["target_ref"]
        joints = _six(self._pose(profile, target_ref).get("joints"), f"poses[{target_ref}].joints")
        rig = self._ensure_session(profile, arm, parameters["speed"])
        self._configure_speed(rig, parameters["speed"], linear=False)
        command = joint_frame(joints)
        acknowledgment = self._send_motion(rig, command, f"JointMovJ -> {target_ref}")
        arrival = self._wait_for_joints(rig, joints, f"arrival at {target_ref}")
        self._log(f"{arm}: arrived at {target_ref}")
        return {"success": True, "command_trace": [command], "acknowledgments": [acknowledgment],
                "arrival": arrival}

    def _run_move_to_home(self, node: dict[str, Any], profile: dict[str, Any], arm: str,
                          parameters: dict[str, Any]) -> dict[str, Any]:
        speed = int(parameters["speed"])
        rig = self._ensure_session(profile, arm, speed)
        self._configure_speed(rig, speed, linear=False)
        result = self._move_to_home(rig, arm, None)
        self._log(f"{arm}: returned to the calibrated initial joint pose")
        return result

    def _run_move_linear(self, node: dict[str, Any], profile: dict[str, Any], arm: str,
                         parameters: dict[str, Any]) -> dict[str, Any]:
        target = self._pose(profile, parameters["target_ref"])
        rig = self._ensure_session(profile, arm, parameters["speed"])
        state = self._read_state(rig)
        observed = self._observe_source(rig, profile, state, parameters["source_ref"])
        self._configure_speed(rig, parameters["speed"], linear=True)
        command = linear_frame(target["xyz"], target["rpy"])
        acknowledgment = self._send_motion(rig, command, f"MovL -> {parameters['target_ref']}")
        arrival = self._wait_for_pose(rig, target["xyz"], target["rpy"],
                                      f"arrival at {parameters['target_ref']}")
        return {"success": True, "start_observation": observed, "command_trace": [command],
                "acknowledgments": [acknowledgment], "arrival": arrival}

    def _run_gripper(self, node: dict[str, Any], profile: dict[str, Any], arm: str,
                     parameters: dict[str, Any]) -> dict[str, Any]:
        position, speed = int(parameters["position"]), int(parameters["speed"])
        rig = self._ensure_session(profile, arm, speed)
        before = int(self._gripper(rig, profile).get_current_position())
        opening = before < position
        if opening and not parameters["allow_release"]:
            raise RuntimeError(
                f"gripper on {arm} is at {before} and the command opens it to {position}; that "
                "would release a held object. allow_release=True needs on-site confirmation.")
        notes: list[str] = []
        if not opening and before < GRIPPER_OPEN_MINIMUM:
            # Reported, not refused: the atom's declared contract has no such
            # precondition, and a partial re-close is a legal opening action. It
            # does mean a successful close here cannot be read as a fresh grasp.
            notes.append(f"jaws were only at {before} (< {GRIPPER_OPEN_MINIMUM}) before closing; "
                         "this is not a fresh grasp of an open gripper")
        # One arm owns one serial port and one gripper; the gate keeps the
        # command and its feedback read together even if two arms run at once.
        with self._arm_lock(arm):
            outcome = self._gripper_move(rig, profile, position, speed)
            feedback = self._wait_for_gripper(rig, profile, before, position)
        return {"success": True, "gripper_result": outcome, "opened": opening,
                "position_feedback": feedback, "notes": notes,
                "note": "position feedback only; it does not prove an object is held"}

    def _run_rotate(self, node: dict[str, Any], profile: dict[str, Any], arm: str,
                    parameters: dict[str, Any]) -> dict[str, Any]:
        if arm != "right":
            raise RuntimeError("arm.rotate_about_pivot is currently bound to the right arm")
        waypoints = self._rotate_waypoints(node, profile)
        rig = self._ensure_session(profile, arm, parameters["speed"])
        observed = self._observe_source(rig, profile, self._read_state(rig), parameters["source_ref"])
        self._configure_speed(rig, parameters["speed"], linear=True)
        if _reply_status(rig.dashboard.CP(int(parameters["cp"]))) != 0:
            raise RuntimeError(f"CP({parameters['cp']}) on {arm} was not acknowledged with status 0")
        commands: list[str] = []
        acknowledgments: list[dict[str, Any]] = []
        for waypoint in waypoints:
            command = linear_frame(waypoint["xyz"], waypoint["reference_rpy"])
            acknowledgments.append(self._send_motion(
                rig, command, f"pour segment {waypoint['step']}/{waypoints[-1]['step']}"))
            commands.append(command)
        last = waypoints[-1]
        arrival = self._wait_for_pose(rig, last["xyz"], last["reference_rpy"], "pour end point")
        return {"success": True, "start_observation": observed, "command_trace": commands,
                "acknowledgments": acknowledgments, "arrival": arrival,
                "reference_path": [{"step": item["step"],
                                    "xyz": [round(value, 4) for value in item["xyz"]],
                                    "reference_rpy": [round(value, 4) for value in item["reference_rpy"]]}
                                   for item in waypoints],
                "note": "the commanded reference path is not a measurement of the liquid stream"}

    def _run_hold(self, node: dict[str, Any], profile: dict[str, Any], arm: str,
                  parameters: dict[str, Any]) -> dict[str, Any]:
        self._ensure_session(profile, arm, 1, configure=False)
        started = time.time()
        self._sleep(float(parameters["seconds"]))
        return {"success": True, "dwell_seconds": float(parameters["seconds"]),
                "elapsed_seconds": round(time.time() - started, 3)}

    # --------------------------------------------------------------------- home

    def home(self, arms: Iterable[str] = ("left", "right"),
             profile: dict[str, Any] | None = None,
             *, timeout: float | None = None) -> dict[str, Any]:
        """Return the requested arms to the calibrated initial joint pose.

        This is a session utility, not a registered atom: no catalog entry, no
        recipe and no DAG refers to it, because homing is the operator's reset
        action rather than a step of a task. It reuses the atom session and frame
        rules unchanged - lease -> ClearError -> EnableRobot -> speed/accel ->
        Tool/User, then ``JointMovJ`` to the calibrated home joints from
        ``layout.physical_nova_home_degrees``, then arrival evidence.

        Arms are driven one at a time so an operator can abort between them, and
        the move is in joint space, so no intermediate Cartesian path is claimed.
        The jaws are never commanded and the arms are never disabled: whatever a
        gripper is holding stays held, and the evidence says only that feedback
        matched the home joints.

        One arm failing does not cancel the others - the arms are independent and
        an operator usually still wants the remaining arm parked - so every
        requested arm gets an entry, failed ones carry their error, and the
        report's ``success`` is False if any arm did not arrive.

        An arm that is already sitting at the calibrated joints in ``RobotMode``
        is reported with ``skipped`` and is not commanded at all: parked is a
        state, not a gesture, and re-issuing a joint move to the pose an arm is
        already holding is wear for nothing. That judgement is made by one
        read-only ``GetAngle`` against the same ``physical_nova_home_degrees``
        target and the same ``ARRIVAL_TOL_DEG`` the arrival wait uses, so
        "already parked" is exactly the evidence a fresh move would have to
        produce. The probe is the dashboard-only read the preflight uses: looking
        must not take the lease, clear errors, reconfigure or enable an arm.
        """
        self._require_live("home")
        selected = self._profile_for(profile)
        if not self.confirm():
            raise RuntimeError("homing was not confirmed on site; nothing was commanded")
        results: dict[str, Any] = {}
        for arm in dict.fromkeys(arms):
            try:
                observed = self._observe_home(selected, arm)
                if observed["at_home"]:
                    self._log(f"{arm}: already at the calibrated initial joint pose; no command sent")
                    results[arm] = {"arm": arm, "success": True, "skipped": True,
                                    "reason": "already at the calibrated home joints",
                                    "observed": observed}
                    continue
                results[arm] = self._home_arm(selected, arm, timeout)
            except Exception as error:  # noqa: BLE001 - one arm must not cancel the rest
                results[arm] = {"arm": arm, "success": False,
                                "error": f"{type(error).__name__}: {error}"}
                self._log(f"{arm}: homing failed: {type(error).__name__}: {error}")
        failed = sorted(arm for arm, entry in results.items() if entry.get("success") is not True)
        return {"driver": "LiveDriver", "operation": "home", "arms": results,
                "success": not failed, "failed_arms": failed,
                "grippers_commanded": False, "arms_disabled": False,
                "physical_success_claimed": False}

    def _observe_home(self, profile: dict[str, Any], arm: str) -> dict[str, Any]:
        """One read-only look at an arm, measured against its calibrated home.

        Never a command, and no session either - the same dashboard-only look the
        preflight uses. An arm that turns out to be parked is therefore left
        exactly as it was found: not enabled, not cleared, not reconfigured, and
        no lease taken for a question that may end in no motion at all.
        """
        rig = self._rig(arm, profile, with_motion=False)
        state = self._read_state(rig)
        target = list(physical_nova_home_degrees(arm))
        worst = max(abs(angle_delta(state["joints"][index], target[index])) for index in range(6))
        return {"source": state["source"], "mode": state["mode"], "joints": list(state["joints"]),
                "target_joints": target, "worst_joint_error_deg": round(worst, 4),
                "at_home": state["mode"] == 5 and worst <= ARRIVAL_TOL_DEG}

    def _home_arm(self, profile: dict[str, Any], arm: str,
                  timeout: float | None) -> dict[str, Any]:
        speed = int(profile["arms"][arm].get("speed_default", 20))
        rig = self._ensure_session(profile, arm, speed)
        self._configure_speed(rig, speed, linear=False)
        result = self._move_to_home(rig, arm, timeout)
        self._log(f"{arm}: returned to the calibrated initial joint pose")
        return {"arm": arm, "speed": speed, "simulated": False,
                "physical_success_claimed": False, **result}

    def _move_to_home(self, rig: _ArmRig, arm: str, timeout: float | None) -> dict[str, Any]:
        """The one implementation of "go to the calibrated initial pose".

        Both the ``arm.move_to_home`` atom and the ``home`` session utility call
        this, so the joint target can only ever come from
        ``layout.physical_nova_home_degrees`` and cannot drift between them.
        """
        joints = list(physical_nova_home_degrees(arm))
        command = joint_frame(joints)
        acknowledgment = self._send_motion(rig, command, f"home on {arm}")
        arrival = self._wait_for_joints(rig, joints, f"arrival at home on {arm}", timeout)
        return {"success": True, "target_joints": joints, "command_trace": [command],
                "acknowledgments": [acknowledgment], "arrival": arrival}

    # --------------------------------------------------------------------- stop

    def stop(self, used_arms: Iterable[str]) -> dict[str, Any]:
        """Stop the queue on the arms actually used; hold and stay enabled."""
        self._stop_requested = True
        results: dict[str, Any] = {}
        for arm in list(used_arms):
            rig = self._rigs.get(arm)
            if rig is None or rig.dashboard is None:
                continue
            entry: dict[str, Any] = {"gripper_held": True, "disable_robot_called": False}
            try:
                entry["stop_script"] = str(rig.dashboard.StopScript())
            except Exception as error:  # noqa: BLE001
                entry["stop_script"] = f"{type(error).__name__}: {error}"
            results[arm] = entry
        self._log(f"stop requested; arms {sorted(results)} held enabled")
        return {"stopped": results, "gripper_released": False, "disable_robot_called": False}

    # -------------------------------------------------------------------- close

    def close(self) -> dict[str, Any]:
        """Idempotent teardown; safe when nothing was ever opened."""
        report: dict[str, Any] = {"lease_released": False, "closed": []}
        if self._finalized and not self._rigs and self._lease is None:
            return report
        if self._lease is not None:
            try:
                self._lease.release()
                report["lease_released"] = True
            except Exception as error:  # noqa: BLE001
                report["lease_error"] = f"{type(error).__name__}: {error}"
            self._lease = None
        for arm, rig in list(self._rigs.items()):
            for attribute in ("gripper", "feedback", "motion", "dashboard"):
                connection = getattr(rig, attribute, None)
                if connection is None:
                    continue
                try:
                    connection.disconnect() if attribute == "gripper" else connection.close()
                    report["closed"].append(f"{arm}.{attribute}")
                except Exception as error:  # noqa: BLE001
                    report[f"{arm}.{attribute}_close_error"] = f"{type(error).__name__}: {error}"
                setattr(rig, attribute, None)
        self._rigs.clear()
        self._finalized = True
        return report
