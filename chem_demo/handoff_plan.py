"""Offline planner for the left-to-right bottle handoff (right gripper takes the body from the side).

Nothing here touches hardware. Kinematics arrive as injected callables (``left_ik`` / ``right_ik``
with the signature ``ik(xyz, rpy, seed_joints) -> joints`` that RAISE ``RuntimeError`` when the
pose is unreachable, and optional ``fk(joints) -> {"xyz", "rpy"}``). Every checked sample follows the
rules of ``task_skills.run_cartesian_jog.Jog.check``: joint jump versus the seed at most
``MAX_JOINT_STEP_DEG``, J5 at least ``WRIST_MARGIN_DEG`` from the wrist singularity, FK agreement.

Frames: each arm uses its own base frame (mm, degrees, RPY = Rz@Ry@Rx as in the controller).
``T_right_left`` (4x4, right base <- left base) is an input. Nothing is hard-coded to a bench
location: the handoff point is searched around the live left TCP and the right pose is derived
from the bottle position at that handoff.

ESTIMATES / UNVERIFIED (flagged because only TCP poses are available, not link geometry):
  * Tool0 is the FLANGE. Flange -> fingertips ~120 mm, pad centre 16 mm behind the tips (see
    chem_demo.gripper_geometry; the 240/215 mm used before 2026-11-22 assumed the table at the base
    plane and were wrong by ~120 mm). Measure with a ruler before trusting them.
  * Grippers, wrists and the hanging bottle are capsules around segments in the right base frame:
    left gripper from the flange DOWN to the fingertips, left wrist UP from the flange; right gripper
    body and fingers OUT along tool Z, right wrist BACK. Radii include the wrist cameras. This is a
    coarse proxy, not a collision check of the links (Nova 2 forearm/upper arm are not modelled).
  * The roll convention (see ``rpy_for_tool_axis``) fixes only where tool X/Y point; which of them
    is the jaw closing axis is NOT verified. The default rolls (0, 180: tool X horizontal; +-90:
    tool Y horizontal) try both, so the ranking does not depend on that assumption.
  * Bottle is a vertical cylinder hanging from the left TCP; the right base is assumed to be
    close to Z-aligned with the left base (the full rotation of ``T_right_left`` is applied).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from chem_demo.gripper_geometry import FINGERTIP_FROM_FLANGE_MM, bottle_hang, pad_offset_flange, table_z_real
from task_skills.chemistry.geometry import euler_zyx_to_matrix, matrix_to_euler_zyx
from task_skills.run_cartesian_jog import (
    FK_ORIENTATION_TOL_DEG,
    FK_POSITION_TOL_MM,
    MAX_JOINT_STEP_DEG,
    WRIST_MARGIN_DEG,
    _rotation_angle_deg,
)

_HANG = bottle_hang()
MARGIN_CAP_DEG = 180.0  # reported when no joint limits are injected
EPS = 1e-9


class HandoffInfeasible(RuntimeError):
    """No handoff candidate survived; ``histogram`` counts the rejection reasons."""

    def __init__(self, histogram: Mapping[str, int], message: str = "no feasible handoff", total: int = 0):
        self.histogram = dict(sorted(histogram.items(), key=lambda kv: (-kv[1], kv[0])))
        self.total_candidates = total
        top = ", ".join(f"{k}={v}" for k, v in list(self.histogram.items())[:6])
        super().__init__(f"{message}: {top}")


def _plain(value: Any) -> Any:
    """Recursively convert numpy / tuples / dataclasses into JSON-serialisable builtins."""
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


@dataclass(frozen=True)
class HandoffConfig:
    """Search space and guards. Defaults are bench estimates, not verified limits."""

    left_x_range_mm: tuple[float, float] = (100.0, 330.0)      # left TCP guard box (left base)
    left_y_range_mm: tuple[float, float] = (-480.0, -300.0)
    left_z_range_mm: tuple[float, float] = (350.0, 545.0)      # left flange; handoff at photo height 535
    right_z_range_mm: tuple[float, float] = (150.0, 560.0)     # right flange; pads ~200 mm out at bottle height
    right_radius_range_mm: tuple[float, float] = (150.0, 640.0)  # horizontal radius from right base origin (pads now ~104 mm out)
    left_dxy_offsets_mm: tuple[float, ...] = (-40.0, 0.0, 40.0)  # handoff XY offsets around current left TCP
    left_dz_offsets_mm: tuple[float, ...] = (0.0, -40.0)  # handoff Z offsets around current left Z
    approach_offsets_deg: tuple[float, ...] = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
    roll_candidates_deg: tuple[float, ...] = (0.0, 180.0, 90.0, -90.0)  # about the tool axis
    standoff_candidates_mm: tuple[float, ...] = (120.0,)  # back along the tool axis from contact
    right_pad_offset_tool_mm: tuple[float, float, float] = pad_offset_flange()  # flange -> pad centre (gripper_geometry)
    sample_step_mm: float = 10.0
    sample_step_deg: float = 10.0
    # Capsule proxies, ESTIMATES (Tool0 = flange; Nova 2 wrist: J5->flange 88 mm, J4->J5 120 mm).
    left_gripper_radius_mm: float = 50.0       # gripper body, open fingers and wrist camera
    left_gripper_length_mm: float = FINGERTIP_FROM_FLANGE_MM  # flange -> fingertips, downward (estimate ~120)
    left_wrist_length_mm: float = 150.0        # flange -> above J5, upward
    right_gripper_radius_mm: float = 40.0      # gripper body (wrist camera sits by J6, covered by the wrist capsule)
    # flange -> palm / start of the fingers, along tool Z (fingers look ~75 mm long; estimate)
    right_gripper_length_mm: float = FINGERTIP_FROM_FLANGE_MM - 75.0
    right_finger_radius_mm: float = 20.0       # fingers with the jaw axis horizontal: thin vertically
    # Open jaws: each finger's OUTER face this far from the tool axis along the jaw axis (tool Y).
    # ESTIMATE (~46 mm, 2026-11-22 e-stop analysis); ruler measurement pending. Until 2026-11-23 the
    # fingers were one capsule on the tool axis, so an open finger could sweep ~46 mm sideways into
    # the bottle unchecked (23:00 e-stop).
    right_finger_open_outer_mm: float = 46.0
    right_finger_end_mm: float = FINGERTIP_FROM_FLANGE_MM  # flange -> fingertips
    right_wrist_length_mm: float = 150.0       # back from the flange
    wrist_radius_mm: float = 45.0
    min_clearance_mm: float = 15.0
    pre_standoff_mm: float = 100.0             # wrist rotation finishes this far behind the standoff
    # Right tool axis tilted this far below horizontal (jaw axis stays horizontal). 2026-11-22: 35 deg
    # keeps the right forearm/wrist ~32 cm above the table instead of ~13 cm at the same bottle height.
    pitch_candidates_deg: tuple[float, ...] = (0.0,)
    left_corner_z_mm: tuple[float, ...] = (500.0, 470.0, 440.0)  # lower vertical-lift tops to try, highest first
    # Whole-arm checks with chem_demo.nova2_model (off for analytic fakes whose joints are not real).
    body_model: bool = False
    max_elbow_deg: float = 130.0               # |J3|: beyond this the forearm folds onto the upper arm
    self_clearance_mm: float = 25.0
    # White platform in controller coordinates: ~120 mm ABOVE the base plane (gripper_geometry). The
    # 0.0 used until 2026-11-22 let the right arm be planned ~13 cm "above the table" that was really
    # at its level. Right-base value; the left table is 3 mm higher, inside the 40 mm clearance.
    table_z_mm: float = table_z_real("right")
    table_clearance_mm: float = 40.0
    arm_clearance_mm: float = 30.0             # between the two arms' links
    joint_limit_margin_deg: float = 3.0                        # only used when joint limits are injected
    max_refine_mm: float = 25.0
    # Seed for the contact IK, which fixes the right arm's branch for the whole handoff and the
    # pour after it. Empty = the start joints (2026-11-23 runs: J4 ~120 mm BELOW the flange, the
    # forearm sloping down over the balances, so no pour path exists). WRIST_UP_SEED keeps J4 above
    # the flange; the joint move to it then turns J5 through 0, which is harmless in joint space.
    right_branch_seed: tuple[float, ...] = ()
    # Absolute left handoff points (left base frame). When given they replace the offset grid, so a
    # stored handoff station is checked as is instead of being searched for again.
    left_points_mm: tuple[tuple[float, float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


@dataclass(frozen=True)
class BottleHold:
    """Bottle held at the neck by the left TCP (values measured elsewhere)."""

    left_tcp_pose: tuple[float, ...]       # measured left flange pose, 6-vector (mm, deg)
    # From the commissioned neck grasp (gripper_geometry.bottle_hang): bottom ~236 mm below the
    # flange (115.7 measured + flange->tips ~120 estimated), rim-plane 134.3 mm above the bottom.
    # Until 2026-11-22 the table was taken at the base plane, which hung the bottle 120 mm too low.
    grasp_drop_mm: float = _HANG["grasp_drop_mm"]   # right pads: 40 mm above the bottle bottom
    body_diameter_mm: float = 70.0
    total_height_mm: float = _HANG["total_height_mm"]
    rim_below_flange_mm: float = _HANG["rim_below_flange_mm"]

    def __post_init__(self) -> None:
        if len(self.left_tcp_pose) != 6:
            raise ValueError("left_tcp_pose must be a 6-vector")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


@dataclass
class HandoffPlan:
    left_legs: list[list[float]]           # left TCP waypoints (xyz); orientation fixed at left_rpy
    left_rpy: list[float]
    left_handoff_xyz: list[float]
    right_waypoints: list[dict[str, Any]]  # park .. standoff .. contact: {name, xyz, rpy, joints, step_kind}
    contact_pose: dict[str, Any]           # right Tool0 pose when the pads are at the body centre
    standoff_pose: dict[str, Any]
    retreat_poses: list[dict[str, Any]]    # contact -> standoff -> park, each with joints
    step_margins: list[dict[str, Any]]     # per checked step: jump / J5 margin / joint-limit margin
    candidate: dict[str, Any]              # chosen search parameters
    rejected_summary: dict[str, int]       # reason -> number of (left, right) candidates rejected
    T_right_left: list[list[float]]
    bottle_right_xyz: list[float]          # body centre in the right base frame
    bottle: BottleHold

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


# ----------------------------------------------------------------------------- orientation

def rpy_for_tool_axis(axis_xyz: Sequence[float], roll_deg: float = 0.0) -> list[float]:
    """RPY (Rz@Ry@Rx, degrees) whose tool Z axis points along ``axis_xyz``.

    Roll 0 for a horizontal axis ``a``: tool X = a x up (horizontal), tool Y = straight down.
    ``roll_deg`` then rotates the frame about the tool axis (right-handed about +a). A vertical
    axis falls back to x0 = a x world-X. (180, 0, 90) is tool Z straight down in this convention.
    """
    a = np.asarray(axis_xyz, float)
    norm = float(np.linalg.norm(a))
    if norm < EPS:
        raise ValueError("axis must be non-zero")
    a = a / norm
    x0 = np.cross(a, [0.0, 0.0, 1.0])
    if np.linalg.norm(x0) < 1e-6:
        x0 = np.cross(a, [1.0, 0.0, 0.0])
    x0 = x0 / np.linalg.norm(x0)
    base = np.column_stack([x0, np.cross(a, x0), a])
    r = math.radians(roll_deg)
    spin = np.array([[math.cos(r), -math.sin(r), 0.0], [math.sin(r), math.cos(r), 0.0], [0.0, 0.0, 1.0]])
    return [float(v) for v in matrix_to_euler_zyx((base @ spin).tolist())]


def _rot(rpy: Sequence[float]) -> np.ndarray:
    return np.array(euler_zyx_to_matrix(rpy), float)


def _quat(R: np.ndarray) -> np.ndarray:
    t = float(np.trace(R))
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i], q[1 + j], q[1 + k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    q = np.array(q, float)
    return q / np.linalg.norm(q)


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 1 - 1e-9:
        return q0 + (q1 - q0) * t
    ang = math.acos(min(1.0, d))
    return (math.sin((1 - t) * ang) * q0 + math.sin(t * ang) * q1) / math.sin(ang)


def _interpolate(name: str, kind: str, xyz0: np.ndarray, rpy0: Sequence[float], xyz1: np.ndarray,
                 rpy1: Sequence[float], cfg: HandoffConfig) -> list[tuple[str, str, np.ndarray, list[float]]]:
    """Samples (excluding the start, including the end) at most sample_step_mm / sample_step_deg apart."""
    dist = float(np.linalg.norm(xyz1 - xyz0))
    angle = _rotation_angle_deg(rpy0, rpy1)
    n = max(1, math.ceil(dist / cfg.sample_step_mm - 1e-9), math.ceil(angle / cfg.sample_step_deg - 1e-9))
    q0, q1 = _quat(_rot(rpy0)), _quat(_rot(rpy1))
    out = []
    for k in range(1, n + 1):
        t = k / n
        rpy = list(rpy1) if k == n else [float(v) for v in matrix_to_euler_zyx(_quat_matrix(_slerp(q0, q1, t)).tolist())]
        out.append((f"{name}_{k}" if k < n else name, kind, xyz0 + (xyz1 - xyz0) * t, rpy))
    return out


# ----------------------------------------------------------------------------- checks

class _Reject(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Minimum distance between segments p1-q1 and p2-q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(r))
    if a <= EPS:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = float(d1 @ r)
        if e <= EPS:
            s, t = min(1.0, max(0.0, -c / a)), 0.0
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def _ik_step(ik: Callable, fk: Callable | None, side: str, xyz: Sequence[float], rpy: Sequence[float],
             seed: Sequence[float], limits: Sequence[Sequence[float]] | None, cfg: HandoffConfig) -> dict[str, Any]:
    """One IK sample with the Jog.check rules; raises _Reject(reason)."""
    try:
        joints = [float(v) for v in ik([float(v) for v in xyz], [float(v) for v in rpy], list(seed))]
    except RuntimeError:
        raise _Reject(f"{side}_unreachable") from None
    if len(joints) != 6 or not all(math.isfinite(v) for v in joints):
        raise _Reject(f"{side}_unreachable")
    jump = max(abs(math.remainder(joints[i] - seed[i], 360.0)) for i in range(6))
    if jump > MAX_JOINT_STEP_DEG:
        raise _Reject(f"{side}_joint_jump")
    j5 = abs(math.remainder(joints[4], 180.0))
    if j5 < WRIST_MARGIN_DEG:
        raise _Reject(f"{side}_wrist_singularity")
    margin = MARGIN_CAP_DEG
    if limits is not None:
        margin = min(MARGIN_CAP_DEG, min(min(j - lo, hi - j) for j, (lo, hi) in zip(joints, limits)))
        if margin < cfg.joint_limit_margin_deg:
            raise _Reject(f"{side}_joint_limit")
    if fk is not None:
        forward = fk(joints)
        err = max(abs(forward["xyz"][i] - xyz[i]) for i in range(3))
        if err > FK_POSITION_TOL_MM or _rotation_angle_deg(forward["rpy"], rpy) > FK_ORIENTATION_TOL_DEG:
            raise _Reject(f"{side}_fk_mismatch")
    return {"joints": joints, "jump_deg": jump, "j5_margin_deg": j5, "limit_margin_deg": margin}


# Right-arm IK seed for the wrist-up branch at the handoff grasp (J4 ~120 mm above the flange),
# from chem_demo.nova2_model.ik_numeric at the 2026-11-23 grasp (-141.7, -523.8, 291.5) rpy (0, 90, -84.8).
WRIST_UP_SEED = (-94.0, -52.0, -38.0, -90.0, -99.0, -90.0)

_TOOL_SEGMENTS = None


def _tool_segments(cfg: HandoffConfig) -> list[tuple[float, float, float, str]]:
    """Gripper body and fingers along tool Z from the flange (start, end, radius, name)."""
    return [(0.0, cfg.right_gripper_length_mm - cfg.right_gripper_radius_mm, cfg.right_gripper_radius_mm, "gripper"),
            (cfg.right_gripper_length_mm, cfg.right_finger_end_mm - cfg.right_finger_radius_mm,
             cfg.right_finger_radius_mm, "fingers")]


# Pairs that may touch by construction (adjacent links) are not checked against each other.
_SELF_PAIRS = [("gripper", "base"), ("gripper", "upper_arm"), ("fingers", "base"), ("fingers", "upper_arm"),
               ("wrist", "base"), ("wrist", "upper_arm"), ("wrist2_to_flange", "base"),
               ("wrist2_to_flange", "upper_arm"), ("forearm", "base"), ("gripper", "forearm"),
               ("fingers", "forearm")]


def body_reject(side: str, joints: Sequence[float], rpy: Sequence[float], cfg: HandoffConfig,
                others: Sequence[tuple[np.ndarray, np.ndarray, float, str]] = ()) -> str | None:
    """Whole-arm checks for one sample: elbow fold, self-collision, table, the other arm/bottle."""
    from chem_demo.nova2_model import capsules
    if abs(math.remainder(joints[2], 360.0)) > cfg.max_elbow_deg:
        return f"{side}_elbow_folded"
    caps = {c[3]: c for c in capsules(joints, rpy, tool_segments=_tool_segments(cfg))}
    for a, b in _SELF_PAIRS:
        ca, cb = caps[a], caps[b]
        if _segment_distance(ca[0], ca[1], cb[0], cb[1]) < ca[2] + cb[2] + cfg.self_clearance_mm:
            return f"{side}_self_{a}_{b}"
    floor = cfg.table_z_mm + cfg.table_clearance_mm
    for name, c in caps.items():
        if name != "base" and min(c[0][2], c[1][2]) - c[2] < floor:
            return f"{side}_table_{name}"
    for c in caps.values():
        for o in others:
            if _segment_distance(c[0], c[1], o[0], o[1]) < c[2] + o[2] + cfg.arm_clearance_mm:
                return f"{side}_{c[3]}_hits_{o[3]}"
    return None


def arm_capsules_in(T: np.ndarray, joints: Sequence[float], rpy: Sequence[float], cfg: HandoffConfig,
                    prefix: str) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """The other arm's capsules (links and gripper) mapped by the 4x4 ``T`` into this arm's frame."""
    from chem_demo.nova2_model import capsules
    R, t = T[:3, :3], T[:3, 3]
    return [(R @ a + t, R @ b + t, r, f"{prefix}_{n}")
            for a, b, r, n in capsules(joints, rpy, tool_segments=_tool_segments(cfg)) if n != "base"]


def _run_chain(poses: Sequence[tuple[str, str, np.ndarray, list[float]]], seed: Sequence[float], ik: Callable,
               fk: Callable | None, side: str, guard: Callable[[np.ndarray], str | None],
               clear: Callable[[str, np.ndarray, list[float]], str | None] | None,
               limits: Sequence[Sequence[float]] | None, cfg: HandoffConfig,
               others: Sequence[tuple[np.ndarray, np.ndarray, float, str]] = ()) -> list[dict[str, Any]]:
    """Check poses in order, each seeded from the previous solution; raises _Reject on the first failure."""
    steps, seed = [], list(seed)
    for name, kind, xyz, rpy in poses:
        reason = guard(xyz) or (clear(kind, xyz, rpy) if clear else None)
        if reason:
            raise _Reject(reason)
        step = _ik_step(ik, fk, side, xyz, rpy, seed, limits, cfg)
        if cfg.body_model:
            reason = body_reject(side, step["joints"], rpy, cfg, others)
            if reason:
                raise _Reject(reason)
        seed = step["joints"]
        steps.append({"name": name, "side": side, "step_kind": kind,
                      "xyz": [round(float(v), 4) for v in xyz], "rpy": [float(v) for v in rpy], **step})
    return steps


def _in_range(value: float, bounds: Sequence[float]) -> bool:
    return bounds[0] - EPS <= value <= bounds[1] + EPS


class _Geometry:
    """Guards and capsule clearance, evaluated in the right base frame."""

    def __init__(self, T: np.ndarray, bottle: BottleHold, cfg: HandoffConfig):
        self.T, self.bottle, self.cfg = T, bottle, cfg
        self._left: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def left_guard(self, xyz: np.ndarray) -> str | None:
        c = self.cfg
        ok = _in_range(xyz[0], c.left_x_range_mm) and _in_range(xyz[1], c.left_y_range_mm) \
            and _in_range(xyz[2], c.left_z_range_mm)
        return None if ok else "left_guard"

    def right_guard(self, xyz: np.ndarray) -> str | None:
        c = self.cfg
        ok = _in_range(xyz[2], c.right_z_range_mm) and _in_range(math.hypot(xyz[0], xyz[1]), c.right_radius_range_mm)
        return None if ok else "right_guard"

    def to_right(self, left_xyz: np.ndarray) -> np.ndarray:
        return self.T[:3, :3] @ left_xyz + self.T[:3, 3]

    def clearance(self, left_xyz: np.ndarray, right_xyz: np.ndarray, right_rpy: Sequence[float],
                  check_bottle: bool) -> str | None:
        """Right gripper/fingers/wrist capsules vs left gripper/wrist capsules and the hanging bottle.

        ``check_bottle`` False is the final approach, where the right fingers must reach around the
        bottle: then only the right gripper body and wrist are kept away from it.
        """
        c = self.cfg
        key = tuple(float(v) for v in left_xyz)
        if key not in self._left:  # left flange and its up axis in the right frame, once per handoff
            self._left[key] = (self.to_right(np.asarray(left_xyz, float)), self.T[:3, :3] @ np.array([0.0, 0.0, 1.0]))
        p, up = self._left[key]
        gap = c.min_clearance_mm
        b = self.bottle
        # Each segment is shortened by its radius so the capsule's round end stops at the physical end.
        left = [(p - up * (c.left_gripper_length_mm - c.left_gripper_radius_mm), p, c.left_gripper_radius_mm,
                 "left_gripper"),
                (p, p + up * c.left_wrist_length_mm, c.wrist_radius_mm, "left_wrist")]
        bottle = (p - up * b.rim_below_flange_mm, p - up * (b.rim_below_flange_mm + b.total_height_mm),
                  b.body_diameter_mm / 2, "bottle")
        a = _tool_axis(right_rpy)
        q = np.asarray(right_xyz, float)
        body = (q, q + a * (c.right_gripper_length_mm - c.right_gripper_radius_mm), c.right_gripper_radius_mm,
                "right_gripper")
        fingers = _open_fingers(q, a, _jaw_axis(right_rpy), c)
        wrist = (q - a * c.right_wrist_length_mm, q, c.wrist_radius_mm, "right_wrist")
        reach = c.left_gripper_length_mm + c.left_wrist_length_mm + b.rim_below_flange_mm + b.total_height_mm \
            + c.right_finger_end_mm + c.right_wrist_length_mm + 4 * max(c.left_gripper_radius_mm, c.right_gripper_radius_mm)
        if math.dist(p, q) > reach:  # cheap reject: the capsules cannot be this close
            return None
        for r_seg in (body, *fingers, wrist):
            for l_seg in left:
                if _segment_distance(l_seg[0], l_seg[1], r_seg[0], r_seg[1]) < l_seg[2] + r_seg[2] + gap:
                    return f"clearance_{r_seg[3]}_{l_seg[3]}"
            if check_bottle or not r_seg[3].startswith("right_finger"):
                if _segment_distance(bottle[0], bottle[1], r_seg[0], r_seg[1]) < bottle[2] + r_seg[2] + gap:
                    return f"clearance_{r_seg[3]}_bottle"
        return None


def _jaw_axis(rpy: Sequence[float]) -> np.ndarray:
    """Tool Y axis (the jaw opening direction)."""
    return _rot([float(v) for v in rpy])[:, 1]


def _open_fingers(q: np.ndarray, a: np.ndarray, y: np.ndarray, c: "HandoffConfig") -> tuple:
    """The two open fingers as capsules beside the tool axis, outer faces at +-right_finger_open_outer_mm."""
    side = max(c.right_finger_open_outer_mm - c.right_finger_radius_mm, 0.0)
    start, end = c.right_gripper_length_mm, c.right_finger_end_mm - c.right_finger_radius_mm
    return tuple((q + a * start + y * k * side, q + a * end + y * k * side, c.right_finger_radius_mm, f"right_finger_{n}")
                 for k, n in ((1.0, "a"), (-1.0, "b")))


def _tool_axis(rpy: Sequence[float]) -> np.ndarray:
    """Tool Z axis (third column of Rz@Ry@Rx) without building the full matrix."""
    rx, ry, rz = (math.radians(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return np.array([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr])


# ----------------------------------------------------------------------------- planning

def _state(state: Mapping[str, Any], label: str) -> tuple[np.ndarray, list[float], list[float]]:
    pose, joints = [float(v) for v in state["pose"]], [float(v) for v in state["joints"]]
    if len(pose) != 6 or len(joints) != 6:
        raise ValueError(f"{label} state needs 6-vector pose and joints")
    return np.array(pose[:3]), pose[3:], joints


def _left_path(start: np.ndarray, handoff: np.ndarray, rpy: list[float], cfg: HandoffConfig,
               corner_z: float | None = None):
    """Straight up over the start to ``corner_z`` (default the handoff Z), then one straight line
    to the handoff point (rising further when the corner is lower: the left arm reaches less far
    the higher it is, so it may only climb to the handoff height while moving inwards)."""
    corner = np.array([start[0], start[1], handoff[2] if corner_z is None else corner_z])
    legs = [("left_z", "left_vertical", start, corner), ("left_xy", "left_horizontal", corner, handoff)]
    poses, waypoints = [], [start]
    for name, kind, a, b in legs:
        if np.linalg.norm(b - a) > EPS:
            poses += _interpolate(name, kind, a, rpy, b, rpy, cfg)
            waypoints.append(b)
    return poses, waypoints


def plan_handoff(left_state: Mapping[str, Any], bottle: BottleHold, T_right_left: Sequence[Sequence[float]],
                 left_ik: Callable, right_ik: Callable, config: HandoffConfig | None = None, *,
                 fk_left: Callable | None = None, fk_right: Callable | None = None,
                 right_start_state: Mapping[str, Any],
                 joint_limits_deg: Sequence[Sequence[float]] | None = None) -> HandoffPlan:
    """Search the left handoff point, right approach azimuth, roll and standoff; return the best plan.

    ``left_state`` / ``right_start_state``: {"pose": [x,y,z,rx,ry,rz], "joints": [6]}. The right arm
    is assumed parked while the left one moves, and the left one stationary while the right one
    approaches. Rejection reasons are counted per (left, right) candidate combination and the
    counts sum to the number of combinations enumerated. Ranking (lexicographic, values bucketed so
    noise does not dominate): larger minimum J5 margin (5 deg buckets), larger minimum joint-limit
    margin (5 deg buckets; constant without ``joint_limits_deg``), smaller max joint jump (2 deg),
    shorter total motion (25 mm), smaller approach-azimuth offset (side facing the right arm), then
    enumeration order. Raises HandoffInfeasible when nothing survives.
    """
    cfg = config or HandoffConfig()
    T = np.asarray(T_right_left, float)
    if T.shape != (4, 4):
        raise ValueError("T_right_left must be 4x4")
    left_start, left_rpy, left_joints = _state(left_state, "left")
    park_xyz, park_rpy, park_joints = _state(right_start_state, "right")
    if np.linalg.norm(left_start - np.array(bottle.left_tcp_pose[:3], float)) > 5.0:
        raise ValueError("bottle.left_tcp_pose disagrees with the live left TCP by more than 5 mm")
    geo = _Geometry(T, bottle, cfg)

    if cfg.left_points_mm:
        lefts = [np.array(point, float) for point in cfg.left_points_mm]
    else:
        lefts = [left_start + np.array([dx, dy, dz]) for dz in cfg.left_dz_offsets_mm
                 for dx in cfg.left_dxy_offsets_mm for dy in cfg.left_dxy_offsets_mm]
    rights = list(itertools.product(cfg.approach_offsets_deg, cfg.roll_candidates_deg, cfg.standoff_candidates_mm,
                                    cfg.pitch_candidates_deg))
    histogram: Counter[str] = Counter()
    feasible = 0
    best: tuple[tuple, dict[str, Any]] | None = None

    parked_right = arm_capsules_in(np.linalg.inv(T), park_joints, park_rpy, cfg, "right") if cfg.body_model else []
    for li, handoff in enumerate(lefts):
        left_steps = None
        reason = geo.left_guard(handoff)
        if not reason:
            corners = [float(handoff[2])] + [z for z in cfg.left_corner_z_mm if left_start[2] <= z < handoff[2]]
            for corner_z in corners:
                poses, waypoints = _left_path(left_start, handoff, left_rpy, cfg, corner_z)
                # The right arm is parked while the left one moves: keep the grippers apart on the way.
                try:
                    left_steps = _run_chain(poses, left_joints, left_ik, fk_left, "left", geo.left_guard,
                                            lambda kind, xyz, rpy: geo.clearance(xyz, park_xyz, park_rpy, True),
                                            joint_limits_deg, cfg, parked_right)
                    break
                except _Reject as rejected:
                    reason = rejected.reason
        if left_steps is None:
            histogram[reason] += len(rights)
            continue

        holding_left = arm_capsules_in(T, left_steps[-1]["joints"], left_rpy, cfg, "left") if cfg.body_model else []
        body_right = geo.to_right(handoff - np.array([0.0, 0.0, bottle.grasp_drop_mm]))
        radial = np.array([body_right[0], body_right[1], 0.0])
        radial = radial / np.linalg.norm(radial) if np.linalg.norm(radial) > EPS else np.array([1.0, 0.0, 0.0])
        for ri, (az, roll, standoff, pitch) in enumerate(rights):
            c, s = math.cos(math.radians(az)), math.sin(math.radians(az))
            cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
            axis = np.array([(c * radial[0] - s * radial[1]) * cp, (s * radial[0] + c * radial[1]) * cp, -sp])
            rpy = rpy_for_tool_axis(axis, roll)
            R = _rot(rpy)
            contact = body_right - R @ np.array(cfg.right_pad_offset_tool_mm)   # Tool0 with pads at body centre
            stand = contact - axis * standoff
            try:
                if cfg.body_model:
                    steps, retreat = _right_joint_first(geo, handoff, park_joints, stand, contact, rpy, axis, cfg,
                                                        right_ik, fk_right, joint_limits_deg, holding_left)
                    raise _Done
                forward = _right_forward(geo, handoff, park_xyz, park_rpy, stand, contact, rpy, cfg)
                steps = _run_chain(forward, park_joints, right_ik, fk_right, "right", geo.right_guard,
                                   lambda kind, xyz, r: geo.clearance(handoff, xyz, r, not kind.startswith("final")),
                                   joint_limits_deg, cfg, holding_left)
                back = _right_backward(forward, park_xyz, park_rpy)
                retreat = _run_chain(back, steps[-1]["joints"], right_ik, fk_right, "right", geo.right_guard,
                                     lambda kind, xyz, r: geo.clearance(handoff, xyz, r, not kind.startswith("final")),
                                     joint_limits_deg, cfg, holding_left)
            except _Done:
                pass
            except _Reject as rejected:
                histogram[rejected.reason] += 1
                continue
            feasible += 1
            every = left_steps + steps + retreat
            motion = sum(float(np.linalg.norm(b - a)) for a, b in zip(waypoints, waypoints[1:])) \
                + sum(float(np.linalg.norm(np.array(b["xyz"]) - np.array(a["xyz"]))) for a, b in zip(steps, steps[1:]))
            # Real arms: first prefer the right forearm/wrist kept highest above the platform instruments.
            height = _arm_low_point(steps) if cfg.body_model else 0.0
            key = (-int(height // 20), -int(min(s_["j5_margin_deg"] for s_ in every) // 5),
                   -int(min(s_["limit_margin_deg"] for s_ in every) // 5),
                   int(max(s_["jump_deg"] for s_ in every) // 2), int(motion // 25), abs(az), li, ri)
            if best is None or key < best[0]:
                best = (key, {"li": li, "handoff": handoff, "waypoints": waypoints, "left_steps": left_steps,
                              "steps": steps, "retreat": retreat, "az": az, "roll": roll, "standoff": standoff,
                              "pitch": pitch,
                              "rpy": rpy, "contact": contact, "stand": stand, "axis": axis,
                              "body": body_right, "motion": motion})
    total = len(lefts) * len(rights)
    if best is None:
        raise HandoffInfeasible(histogram, total=total)
    best[1]["counts"] = {"candidates_enumerated": total, "candidates_feasible": feasible}
    return _assemble(best[1], bottle, T, park_xyz, park_rpy, park_joints, left_rpy, dict(histogram))


def _right_forward(geo: _Geometry, handoff: np.ndarray, park_xyz: np.ndarray, park_rpy: list[float],
                   stand: np.ndarray, contact: np.ndarray, rpy: list[float], cfg: HandoffConfig):
    """Park -> pre-standoff (position and orientation blended) -> standoff -> contact (both straight
    along the tool axis, orientation fixed). Samples are <=10 mm / <=10 deg apart; the cycle sends
    each of the three segments as one continuous MovL."""
    axis = _tool_axis(rpy)
    pre = stand - axis * cfg.pre_standoff_mm
    for xyz, bottle in ((pre, True), (stand, True), (contact, False)):  # cheap endpoint screen first
        reason = geo.right_guard(xyz) or geo.clearance(handoff, xyz, rpy, bottle)
        if reason:
            raise _Reject(reason)
    return (_interpolate("pre_approach", "orient_approach", park_xyz, park_rpy, pre, rpy, cfg)
            + _interpolate("approach", "straight_approach", pre, rpy, stand, rpy, cfg)
            + _interpolate("contact", "final_approach", stand, rpy, contact, rpy, cfg))


def _arm_low_point(steps: Sequence[Mapping[str, Any]]) -> float:
    """Lowest point (mm, capsule surface) of the right elbow, forearm and wrist over the steps."""
    from chem_demo.nova2_model import capsules
    names = ("elbow", "forearm", "wrist", "wrist2_to_flange")
    return min(min(a[2], b[2]) - r for st in steps for a, b, r, n in capsules(st["joints"]) if n in names)


class _Done(Exception):
    """Internal: the joint-first right path was built (skips the Cartesian-blend branch)."""


JOINT_SAMPLE_DEG = 5.0


def _right_joint_first(geo: _Geometry, handoff: np.ndarray, park_joints: Sequence[float], stand: np.ndarray,
                       contact: np.ndarray, rpy: list[float], axis: np.ndarray, cfg: HandoffConfig,
                       right_ik: Callable, fk_right: Callable | None, limits, others):
    """Real-arm right path: one joint-space move (JointMovJ) from the start to the pre-standoff,
    then straight along the tool axis to the standoff and the contact.

    The Cartesian blend from the tool-down start to a horizontal tool made the controller's IK fold
    the elbow onto the upper arm (2026-11-22 self-collision). Here the end configuration is solved
    backwards from the contact (seeded from the start), so the arm stays extended, and every joint
    sample of the move is checked with the whole-arm model.
    """
    if fk_right is None:
        raise _Reject("right_needs_fk")
    pre = stand - axis * cfg.pre_standoff_mm
    for xyz, bottle in ((pre, True), (stand, True), (contact, False)):
        reason = geo.right_guard(xyz) or geo.clearance(handoff, xyz, rpy, bottle)
        if reason:
            raise _Reject(reason)
    seed = list(cfg.right_branch_seed) if cfg.right_branch_seed else list(park_joints)
    solved = []
    for xyz in (contact, stand, pre):  # contact first: the far, extended configuration fixes the branch
        try:
            seed = [float(v) for v in right_ik([float(v) for v in xyz], [float(v) for v in rpy], seed)]
        except RuntimeError:
            raise _Reject("right_unreachable") from None
        solved.append(seed)
    pre_joints = solved[-1]
    reason = body_reject("right", pre_joints, rpy, cfg, others)
    if reason:
        raise _Reject(reason)
    q0, q1 = np.array(park_joints, float), np.array(pre_joints, float)
    n = max(1, math.ceil(float(np.abs(q1 - q0).max()) / JOINT_SAMPLE_DEG))
    joint_steps = []
    for k in range(1, n + 1):
        q = (q0 + (q1 - q0) * k / n).tolist()
        f = fk_right(q)
        xyz, r = np.array(f["xyz"], float), [float(v) for v in f["rpy"]]
        reason = geo.right_guard(xyz) or geo.clearance(handoff, xyz, r, True) or body_reject("right", q, r, cfg, others)
        if reason:
            raise _Reject(reason)
        j5 = abs(math.remainder(q[4], 180.0))
        if j5 < WRIST_MARGIN_DEG and (k == n or not cfg.right_branch_seed):
            raise _Reject("right_wrist_singularity")
        joint_steps.append({"name": "pre_approach" if k == n else f"pre_approach_{k}", "side": "right",
                            "step_kind": "joint_approach", "xyz": [round(float(v), 4) for v in xyz], "rpy": r,
                            "joints": q, "jump_deg": float(np.abs(q1 - q0).max()) / n, "j5_margin_deg": j5,
                            "limit_margin_deg": MARGIN_CAP_DEG})
    straight = (_interpolate("approach", "straight_approach", pre, rpy, stand, rpy, cfg)
                + _interpolate("contact", "final_approach", stand, rpy, contact, rpy, cfg))
    steps = joint_steps + _run_chain(straight, pre_joints, right_ik, fk_right, "right", geo.right_guard,
                                     lambda kind, xyz, r: geo.clearance(handoff, xyz, r, not kind.startswith("final")),
                                     limits, cfg, others)
    # Retreat = the same configurations in reverse (no new IK branch can sneak in), ending at the start.
    retreat = []
    for i in range(len(steps) - 2, -2, -1):
        src = steps[i] if i >= 0 else None
        kind_ahead = steps[i + 1]["step_kind"]
        new_kind = {"final_approach": "final_retreat", "straight_approach": "straight_retreat"}.get(kind_ahead,
                                                                                                    "joint_return")
        if src is None:
            f = fk_right(list(park_joints))
            src = {"name": "park", "xyz": [float(v) for v in f["xyz"]], "rpy": [float(v) for v in f["rpy"]],
                   "joints": list(park_joints), "jump_deg": 0.0, "j5_margin_deg": abs(math.remainder(park_joints[4], 180.0)),
                   "limit_margin_deg": MARGIN_CAP_DEG}
        retreat.append({**src, "name": f"retreat_{src['name']}", "side": "right", "step_kind": new_kind})
    return steps, retreat


def _right_backward(forward, park_xyz: np.ndarray, park_rpy: list[float]):
    """Reverse of the forward samples: contact -> standoff -> park (the same poses, opposite order)."""
    seq = [("park", "start", park_xyz, list(park_rpy))] + list(forward)
    back = []
    for i in range(len(seq) - 1, 0, -1):
        _, kind, _, _ = seq[i]
        name, _, xyz, rpy = seq[i - 1]
        back.append((f"retreat_{name}", "final_retreat" if kind == "final_approach" else "orient_return", xyz, rpy))
    return back


def _pose_dict(step: Mapping[str, Any]) -> dict[str, Any]:
    return {k: step[k] for k in ("name", "xyz", "rpy", "joints", "step_kind")}


def _assemble(b: dict[str, Any], bottle: BottleHold, T: np.ndarray, park_xyz: np.ndarray,
              park_rpy: list[float], park_joints: list[float], left_rpy: list[float],
              histogram: dict[str, int]) -> HandoffPlan:
    steps, retreat = b["steps"], b["retreat"]
    park = {"name": "park", "xyz": [round(float(v), 4) for v in park_xyz], "rpy": list(park_rpy),
            "joints": list(park_joints), "step_kind": "start"}
    margins = [{k: s[k] for k in ("name", "side", "step_kind", "jump_deg", "j5_margin_deg", "limit_margin_deg")}
               for s in b["left_steps"] + steps + retreat]
    return HandoffPlan(
        left_legs=[[round(float(v), 4) for v in w] for w in b["waypoints"]],
        left_rpy=list(left_rpy),
        left_handoff_xyz=[round(float(v), 4) for v in b["handoff"]],
        right_waypoints=[park] + [_pose_dict(s) for s in steps],
        contact_pose=_pose_dict(steps[-1]),
        standoff_pose=_pose_dict(next(s for s in steps if s["name"] == "approach")),
        retreat_poses=[_pose_dict(s) for s in retreat],
        step_margins=margins,
        candidate={"left_offset_mm": [round(float(v), 4) for v in b["handoff"] - np.array(bottle.left_tcp_pose[:3])],
                   "approach_offset_deg": b["az"], "pitch_down_deg": b.get("pitch", 0.0), "roll_deg": b["roll"], "standoff_mm": b["standoff"],
                   "tool_axis": [round(float(v), 6) for v in b["axis"]], "total_motion_mm": round(b["motion"], 2),
                   **b["counts"]},
        rejected_summary=histogram,
        T_right_left=T.tolist(),
        bottle_right_xyz=[round(float(v), 4) for v in b["body"]],
        bottle=bottle,
    )


def refine_contact(plan: HandoffPlan, tool_delta_mm: Sequence[float], right_ik: Callable,
                   config: HandoffConfig | None = None, *, fk_right: Callable | None = None,
                   joint_limits_deg: Sequence[Sequence[float]] | None = None) -> dict[str, Any]:
    """Shift standoff and contact by a measured error expressed in the right Tool0 axes.

    ``tool_delta_mm`` = (x, y, z) along the Tool0 X / Y / Z axes at the contact orientation (which
    of X/Y is the jaw axis is an unverified assumption; the caller maps its measurement). The
    shifted standoff is solved from the original standoff joints (jump <= MAX_JOINT_STEP_DEG),
    then the final approach is re-sampled to the shifted contact. Returns the shifted contact pose
    dict plus ``standoff``, ``approach`` (checked samples) and the applied deltas. Raises
    HandoffInfeasible for deltas above ``max_refine_mm`` or when any check fails.
    """
    cfg = config or HandoffConfig()
    delta = np.asarray(tool_delta_mm, float)
    if delta.shape != (3,):
        raise ValueError("tool_delta_mm must be a 3-vector")
    if float(np.linalg.norm(delta)) > cfg.max_refine_mm:
        raise HandoffInfeasible({"refine_delta_too_large": 1}, f"|delta| > {cfg.max_refine_mm:g} mm")
    rpy = list(plan.contact_pose["rpy"])
    shift = _rot(rpy) @ delta
    stand = np.array(plan.standoff_pose["xyz"]) + shift
    contact = np.array(plan.contact_pose["xyz"]) + shift
    geo = _Geometry(np.array(plan.T_right_left), plan.bottle, cfg)
    handoff = np.array(plan.left_handoff_xyz)
    poses = [("refined_standoff", "refine_standoff", stand, rpy)] \
        + _interpolate("refined_contact", "final_approach", stand, rpy, contact, rpy, cfg)
    try:
        steps = _run_chain(poses, plan.standoff_pose["joints"], right_ik, fk_right, "right", geo.right_guard,
                           lambda kind, xyz, r: geo.clearance(handoff, xyz, r, kind == "refine_standoff"),
                           joint_limits_deg, cfg)
    except _Reject as rejected:
        raise HandoffInfeasible({rejected.reason: 1}, "refined contact rejected") from None
    out = _pose_dict(steps[-1])
    out.update({"standoff": _pose_dict(steps[0]), "approach": [_pose_dict(s) for s in steps],
                "delta_tool_mm": [float(v) for v in delta], "delta_base_mm": [round(float(v), 4) for v in shift]})
    return out
