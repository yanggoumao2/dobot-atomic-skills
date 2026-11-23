"""Offline planner for the right-arm pour into a beaker (right gripper holds the bottle body).

Nothing here touches hardware. Start state: right Tool0 at the hold pose the handoff leaves (tool
axis horizontal, jaws on the body ~40 mm above the bottle bottom, bottle upright). Pour = tilt about
the bottle's lip, not about the tool axis: the J6 turn is compensated by a translation so the lip
stays put above the beaker while the bottle tips (a pure J6 turn would swing the mouth ~94 mm
round the grasp and walk the stream across a ~45 mm beaker mouth).

Phases (right base frame, controller conventions: Tool0 = flange, RPY = Rz@Ry@Rx in degrees):
  lift      straight up from the hold until the bottle bottom clears the target rim
  transit   horizontal, upright, to a side hover beside the beaker (bodies apart in XY)
  pretilt   lip slides from the side hover onto its pour point while the bottle tips to theta_pre
  pour      tip about the fixed lip theta_pre -> theta_max
then the same samples backwards to the hold pose (the existing handback/reset expects that pose).

Kinematics are injected like chem_demo.handoff_plan (``ik(xyz, rpy, seed) -> joints`` raising
RuntimeError); offline use chem_demo.nova2_model.ik_numeric, which matches the controller's
InverseSolution to ~0.2 deg on the handoff plans, but the live run re-checks with the controller.

ESTIMATES / UNVERIFIED (ruler measurements pending; every one is a named constant):
  * bottle: height 134.3 mm (measured: ring plane - table), body radius 35, body straight part
    95 mm, neck radius 22.5 (GL45 guess), grasp 40 mm above the bottom (handoff target, vision-set);
  * flange -> fingertips 180 mm (gripper_geometry, not a ruler value);
  * the balance under the beaker is a 110 mm-radius cylinder centred on the beaker up to the pan top;
  * where the bottle sits along the tool axis is known to ~+-10 mm (handoff vision), so every bottle
    clearance is also checked with the bottle shifted +-HOLD_SHIFT_MM along the tool axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from chem_demo.gripper_geometry import FINGERTIP_FROM_FLANGE_MM, pad_offset_flange, table_z_real
from chem_demo.handoff_plan import HandoffConfig, _ik_step, _Reject, _rot, _segment_distance, body_reject
from task_skills.chemistry.geometry import matrix_to_euler_zyx

# Bottle (ESTIMATES unless noted)
BOTTLE_HEIGHT_MM = 134.3        # measured: neck-ring plane minus table at the commissioned grasp
BODY_RADIUS_MM = 35.0
BODY_STRAIGHT_MM = 95.0         # bottom -> start of the shoulder
NECK_RADIUS_MM = 22.5           # GL45 neck outer radius (guess); the lip sits on it
NECK_START_MM = 112.0           # shoulder ends, neck begins
GRASP_ABOVE_BOTTOM_MM = 40.0
HOLD_SHIFT_MM = 10.0            # bottle position along the tool axis is known to about this
BALANCE_RADIUS_MM = 110.0       # balance footprint around the beaker (ESTIMATE)
PAN_TOP_CW = 339.73             # right base, fingertip touch on the right pan rim (2026-11-20), cw


@dataclass(frozen=True)
class Vessel:
    """Upright cylinder obstacle (right base frame, controller Z)."""

    name: str
    x: float
    y: float
    radius: float
    bottom_z: float
    top_z: float

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class PourConfig:
    lip_heights_mm: tuple[float, ...] = (25.0, 30.0, 35.0, 40.0, 50.0, 60.0)  # lip above the rim, lowest first
    pretilt_candidates_deg: tuple[float, ...] = (50.0, 40.0, 60.0)
    yaw_candidates_deg: tuple[float, ...] = (0.0, -20.0, 20.0, -40.0, 40.0, -60.0, 60.0)
    theta_max_deg: float = 120.0
    lip_back_mm: float = 5.0          # lip this far short of the beaker axis (the stream leans forward)
    side_gap_mm: float = 25.0         # bottle body to beaker wall at the side hover
    transit_clear_mm: float = 40.0    # bottle bottom above the highest rim on the way
    rim_clear_mm: float = 8.0         # bottle surface to the target beaker
    obstacle_clear_mm: float = 20.0   # bottle / gripper to other vessels and the balance
    # The hold pose is where the handoff left the arm (executed, so not re-judged against obstacles);
    # at 2026-11-23 17:30 its J4/J5 wrist hung ~2 mm (model radii) beside the centre beaker. The lift
    # only moves away, so it gets a smaller arm margin.
    lift_arm_clear_mm: float = 10.0   # or the hold pose's own gap if that is smaller
    step_mm: float = 10.0
    step_deg: float = 5.0
    hold_shift_mm: float = HOLD_SHIFT_MM
    # Measured bottle-in-hand correction: the bottle sits this far further along the tool axis than
    # the pad-offset model says (2026-11-23 pretilt top + front frames: ~+20 mm).
    grasp_along_tool_mm: float = 0.0
    # ... and this far higher / further along the pour direction (upright hold). Solved 2026-11-23 from
    # the blue mouth ring in two top frames (upright side hover + 60 deg pre-tilt): ~+26 up, ~+12 along d.
    grasp_up_mm: float = 0.0
    grasp_along_pour_mm: float = 0.0
    max_radius_mm: float = 640.0      # right flange horizontal radius guard (as the handoff planner)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class PourInfeasible(RuntimeError):
    def __init__(self, message: str, rejected: Mapping[str, int]):
        super().__init__(message)
        self.rejected = dict(rejected)


@dataclass
class PourPlan:
    steps: list[dict[str, Any]]           # forward samples hold -> theta_max, each with joints
    pour_start_index: int                 # first sample of the pour phase
    lip_xyz: list[float]
    lip_above_rim_mm: float
    theta_pre_deg: float
    theta_max_deg: float
    yaw_deg: float                        # hold turned about the vertical during the transit
    pour_dir: list[float]                 # horizontal unit vector, bottle -> beaker
    tilt_axis: list[float]
    beaker: dict[str, Any]
    min_clearance: dict[str, float]       # smallest margin per check over the whole path
    rejected: dict[str, int]
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ----------------------------------------------------------------------------- geometry

def _rotm(axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def bottle_profile() -> list[tuple[float, float]]:
    """(height above bottom, radius) rings describing the bottle surface, bottom to lip."""
    rings = [(0.0, BODY_RADIUS_MM * 0.6), (0.0, BODY_RADIUS_MM)]
    rings += [(h, BODY_RADIUS_MM) for h in np.linspace(10.0, BODY_STRAIGHT_MM, 6)]
    rings += [(h, BODY_RADIUS_MM + (NECK_RADIUS_MM - BODY_RADIUS_MM) * (h - BODY_STRAIGHT_MM)
               / (NECK_START_MM - BODY_STRAIGHT_MM)) for h in np.linspace(BODY_STRAIGHT_MM, NECK_START_MM, 4)[1:]]
    rings += [(h, NECK_RADIUS_MM) for h in np.linspace(NECK_START_MM, BOTTLE_HEIGHT_MM, 3)[1:]]
    return rings


def _bottle_points_local(n_around: int = 24) -> np.ndarray:
    """Surface points in a bottle frame: origin at the grasp point on the axis, columns (d, k, up)."""
    pts = []
    for h, r in bottle_profile():
        for i in range(n_around):
            a = 2 * math.pi * i / n_around
            pts.append((r * math.cos(a), r * math.sin(a), h - GRASP_ABOVE_BOTTOM_MM))
    return np.array(pts)


_LOCAL = _bottle_points_local()


def _cylinder_gap(points: np.ndarray, v: Vessel) -> float:
    """Smallest distance from points to a solid upright cylinder (negative inside)."""
    dxy = np.hypot(points[:, 0] - v.x, points[:, 1] - v.y) - v.radius
    dz = np.maximum(v.bottom_z - points[:, 2], points[:, 2] - v.top_z)
    outside = np.hypot(np.maximum(dxy, 0.0), np.maximum(dz, 0.0))
    inside = np.minimum(np.maximum(dxy, dz), 0.0)
    return float(np.min(outside + inside))


def _capsule_points(a: np.ndarray, b: np.ndarray, r: float, step: float = 10.0) -> tuple[np.ndarray, float]:
    n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / step)) + 1)
    return np.array([a + (b - a) * t for t in np.linspace(0.0, 1.0, n)]), r


class PourGeometry:
    """Rigid bottle + flange, from the hold pose.

    ``yaw_deg`` turns the hold about the vertical through the grasp point (the bottle stays upright,
    so this only changes where the arm is): the pour is planned for that turned hold, and the turn
    is blended in during the transit. Upright samples are given by the grasp point and the yaw
    fraction, tilt samples by the lip position and the tilt angle.
    """

    def __init__(self, hold_xyz: Sequence[float], hold_rpy: Sequence[float], beaker: Vessel, yaw_deg: float = 0.0,
                 grasp_along_tool_mm: float = 0.0, grasp_up_mm: float = 0.0, grasp_along_pour_mm: float = 0.0):
        self.hold_p = np.asarray(hold_xyz, float)
        self.hold_R = _rot([float(v) for v in hold_rpy])
        self.grasp = self.hold_p + self.hold_R @ np.asarray(pad_offset_flange(), float)
        tool = self.hold_R[:, 2].copy()
        tool[2] = 0.0
        if np.linalg.norm(tool) > 0.5:
            self.grasp = self.grasp + tool / np.linalg.norm(tool) * float(grasp_along_tool_mm)
        self.up = np.array([0.0, 0.0, 1.0])
        self.yaw = float(yaw_deg)
        Rz = _rotm(self.up, self.yaw)
        self.v_R = Rz @ self.hold_R                              # the turned ("virtual") hold
        self.v_p = self.grasp + Rz @ (self.hold_p - self.grasp)
        n = self.v_R[:, 2].copy()
        n[2] = 0.0
        if np.linalg.norm(n) < 0.5:
            raise ValueError("the hold pose must have a near-horizontal tool axis")
        self.n = n / np.linalg.norm(n)                       # tool axis, horizontal
        perp = np.cross(self.up, self.n)
        to_beaker = np.array([beaker.x, beaker.y, 0.0]) - np.array([self.grasp[0], self.grasp[1], 0.0])
        self.d = perp * (1.0 if float(perp @ to_beaker) >= 0 else -1.0)   # pour direction
        self.k = np.cross(self.up, self.d)                   # tilt axis: up -> d for positive angles
        # Measured bottle-in-hand offsets move the bottle model (and its lip) relative to the jaws.
        self.grasp = self.grasp + self.up * float(grasp_up_mm) + self.d * float(grasp_along_pour_mm)
        self.v_p = self.grasp + Rz @ (self.hold_p - self.grasp)  # yaw turns about the (corrected) bottle axis
        self.frame = np.column_stack([self.d, self.k, self.up])
        self.lip_offset = self.up * (BOTTLE_HEIGHT_MM - GRASP_ABOVE_BOTTOM_MM) + self.d * NECK_RADIUS_MM
        self.lip_hold = self.grasp + self.lip_offset
        self.beaker = beaker

    # upright (theta 0), any yaw fraction
    def upright(self, grasp_pos: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
        Rf = _rotm(self.up, self.yaw * frac)
        return grasp_pos + Rf @ (self.hold_p - self.grasp), Rf @ self.hold_R

    def upright_points(self, grasp_pos: np.ndarray, frac: float, shift: float = 0.0) -> np.ndarray:
        n_now = _rotm(self.up, self.yaw * frac) @ self.hold_R[:, 2]
        n_now[2] = 0.0
        n_now = n_now / np.linalg.norm(n_now)
        return (self.frame @ _LOCAL.T).T + grasp_pos + n_now * shift

    # tilted about the lip, full yaw
    def pose(self, theta: float, lip: np.ndarray) -> tuple[np.ndarray, list[float], np.ndarray]:
        """Flange xyz, flange rpy and the rotation for a tilt ``theta`` with the lip at ``lip``."""
        Rt = _rotm(self.k, theta)
        p = lip + Rt @ (self.v_p - self.lip_hold)
        R = Rt @ self.v_R
        return p, [float(v) for v in matrix_to_euler_zyx(R.tolist())], Rt

    def bottle_points(self, theta: float, lip: np.ndarray, shift: float = 0.0) -> np.ndarray:
        Rt = _rotm(self.k, theta)
        local = (self.frame @ _LOCAL.T).T + self.grasp + self.n * shift
        return lip + (Rt @ (local - self.lip_hold).T).T

    def sample(self, phase: str, theta: float, frac: float, point: np.ndarray, shift: float = 0.0):
        """(flange xyz, flange rpy, bottle points, lip) for one path sample."""
        if phase in TILT_PHASES:
            p, rpy, _ = self.pose(theta, point)
            return p, rpy, self.bottle_points(theta, point, shift), point
        p, R = self.upright(point, frac)
        rpy = [float(v) for v in matrix_to_euler_zyx(R.tolist())]
        lip = point + _rotm(self.up, self.yaw * frac) @ (_rotm(self.up, -self.yaw) @ self.lip_offset)
        return p, rpy, self.upright_points(point, frac, shift), lip


TILT_PHASES = ("pretilt", "pour")


# ----------------------------------------------------------------------------- planning

def _sample_path(geo: PourGeometry, cfg: PourConfig, lip_target: np.ndarray, theta_pre: float,
                 transit_dz: float, side_grasp: np.ndarray) -> list[tuple[str, float, float, np.ndarray]]:
    """(phase, theta, yaw fraction, point) samples hold -> theta_max; point = grasp (upright) or lip (tilt)."""
    out: list[tuple[str, float, float, np.ndarray]] = [("hold", 0.0, 0.0, geo.grasp.copy())]
    lifted = geo.grasp + np.array([0.0, 0.0, transit_dz])

    def line(phase: str, a: np.ndarray, b: np.ndarray, t0: float, t1: float, f0: float, f1: float) -> None:
        n = max(1, math.ceil(float(np.linalg.norm(b - a)) / cfg.step_mm - 1e-9),
                math.ceil(abs(t1 - t0) / cfg.step_deg - 1e-9), math.ceil(abs(geo.yaw * (f1 - f0)) / cfg.step_deg - 1e-9))
        for i in range(1, n + 1):
            s = i / n
            out.append((phase, t0 + (t1 - t0) * s, f0 + (f1 - f0) * s, a + (b - a) * s))

    line("lift", geo.grasp, lifted, 0.0, 0.0, 0.0, 0.0)
    line("transit", lifted, side_grasp, 0.0, 0.0, 0.0, 1.0)
    line("pretilt", side_grasp + geo.lip_offset, lip_target, 0.0, theta_pre, 1.0, 1.0)
    line("pour", lip_target, lip_target, theta_pre, cfg.theta_max_deg, 1.0, 1.0)
    return out


def plan_pour(hold_xyz: Sequence[float], hold_rpy: Sequence[float], hold_joints: Sequence[float],
              beaker: Vessel, obstacles: Sequence[Vessel], ik: Callable, *, fk: Callable | None = None,
              cfg: PourConfig = PourConfig(), arm_cfg: HandoffConfig | None = None,
              other_arm: Sequence[tuple[np.ndarray, np.ndarray, float, str]] = ()) -> PourPlan:
    """Search lip height (lowest first), hold yaw (smallest first) and pre-tilt; first fully checked path wins."""
    arm_cfg = arm_cfg or HandoffConfig(body_model=True)
    balance = Vessel("balance", beaker.x, beaker.y, BALANCE_RADIUS_MM, table_z_real("right"),
                     PAN_TOP_CW - FINGERTIP_FROM_FLANGE_MM)
    others = [balance, *obstacles]
    rejected: dict[str, int] = {}
    axis_xy = np.array([beaker.x, beaker.y, 0.0])
    highest = max([beaker.top_z] + [o.top_z for o in obstacles])

    for height in cfg.lip_heights_mm:
        for yaw in cfg.yaw_candidates_deg:
            geo = PourGeometry(hold_xyz, hold_rpy, beaker, yaw, cfg.grasp_along_tool_mm, cfg.grasp_up_mm,
                               cfg.grasp_along_pour_mm)
            # bottom above the highest rim on the way + margin
            transit_dz = max(0.0, highest + cfg.transit_clear_mm - (geo.grasp[2] - GRASP_ABOVE_BOTTOM_MM))
            # side hover: bottle beside the beaker along -d, lined up with it along the tool axis
            side_axis = axis_xy - geo.d * (BODY_RADIUS_MM + beaker.radius + cfg.side_gap_mm)
            side_grasp = np.array([side_axis[0], side_axis[1], geo.grasp[2] + transit_dz])
            lip_xy = axis_xy - geo.d * cfg.lip_back_mm
            lip_target = np.array([lip_xy[0], lip_xy[1], beaker.top_z + height])
            for theta_pre in cfg.pretilt_candidates_deg:
                samples = _sample_path(geo, cfg, lip_target, theta_pre, transit_dz, side_grasp)
                try:
                    steps, margins = _check(samples, geo, cfg, arm_cfg, beaker, others, ik, fk, hold_joints, other_arm)
                except _Reject as exc:
                    key = f"yaw{yaw:+.0f}:{exc.reason}"
                    rejected[key] = rejected.get(key, 0) + 1
                    continue
                start = next(i for i, s in enumerate(steps) if s["phase"] == "pour")
                return PourPlan(steps=steps, pour_start_index=start, lip_xyz=[round(float(v), 3) for v in lip_target],
                                lip_above_rim_mm=height, theta_pre_deg=theta_pre, theta_max_deg=cfg.theta_max_deg,
                                yaw_deg=yaw, pour_dir=[round(float(v), 5) for v in geo.d],
                                tilt_axis=[round(float(v), 5) for v in geo.k], beaker=beaker.to_dict(),
                                min_clearance=margins, rejected=rejected, config=cfg.to_dict())
    raise PourInfeasible(f"no pour path passed the checks: {rejected}", rejected)


def _check(samples, geo: PourGeometry, cfg: PourConfig, arm_cfg: HandoffConfig, beaker: Vessel,
           others: Sequence[Vessel], ik: Callable, fk: Callable | None, seed: Sequence[float],
           other_arm) -> tuple[list[dict[str, Any]], dict[str, float]]:
    from chem_demo.handoff_plan import _tool_segments
    from chem_demo.nova2_model import capsules
    steps: list[dict[str, Any]] = []
    margins: dict[str, float] = {}
    seed = list(seed)
    table = table_z_real("right")
    hold_gap: dict[tuple[str, str], float] = {}

    def note(name: str, value: float, limit: float, reason: str) -> None:
        margins[name] = min(margins.get(name, math.inf), round(value - limit, 2))
        if value < limit:
            raise _Reject(reason)

    for idx, (phase, theta, frac, point) in enumerate(samples):
        p, rpy, _, lip = geo.sample(phase, theta, frac, point)
        if math.hypot(p[0], p[1]) > cfg.max_radius_mm:
            raise _Reject(f"{phase}_radius")
        step = _ik_step(ik, fk, "right", p, rpy, seed, None, arm_cfg)
        reason = None if phase == "hold" else body_reject("right", step["joints"], rpy, arm_cfg, other_arm)
        if reason:
            raise _Reject(f"{phase}_{reason}")
        seed = step["joints"]
        for shift in ((0.0,) if phase == "hold" else (-cfg.hold_shift_mm, 0.0, cfg.hold_shift_mm)):
            pts = geo.sample(phase, theta, frac, point, shift)[2]
            note("bottle_table", float(pts[:, 2].min()) - table, cfg.obstacle_clear_mm, f"{phase}_bottle_table")
            note("bottle_target", _cylinder_gap(pts, beaker), cfg.rim_clear_mm, f"{phase}_bottle_hits_beaker")
            for o in others:
                note(f"bottle_{o.name}", _cylinder_gap(pts, o), cfg.obstacle_clear_mm, f"{phase}_bottle_hits_{o.name}")
        if phase == "hold":
            for a, b, r, name in capsules(step["joints"], rpy, tool_segments=_tool_segments(arm_cfg)):
                pts, rad = _capsule_points(a, b, r)
                for o in (beaker, *others):
                    key = (name, o.name)
                    hold_gap[key] = min(hold_gap.get(key, math.inf), _cylinder_gap(pts, o) - rad)
            steps.append({"index": idx, "phase": phase, "theta_deg": 0.0, "lip": [round(float(v), 3) for v in lip],
                          "xyz": [round(float(v), 4) for v in p], "rpy": rpy, **step})
            continue
        arm_clear = cfg.lift_arm_clear_mm if phase == "lift" else cfg.obstacle_clear_mm
        caps = capsules(step["joints"], rpy, tool_segments=_tool_segments(arm_cfg))
        for a, b, r, name in caps:
            if name in ("base", "upper_arm"):
                continue
            pts, rad = _capsule_points(a, b, r)
            for o in (beaker, *others):
                limit = arm_clear
                if phase == "lift":  # never closer than the executed hold pose already was
                    limit = min(arm_clear, hold_gap.get((name, o.name), math.inf) - 0.5)
                note(f"arm_{o.name}" if phase != "lift" else f"lift_arm_{o.name}", _cylinder_gap(pts, o) - rad,
                     limit, f"{phase}_{name}_hits_{o.name}")
        steps.append({"index": idx, "phase": phase, "theta_deg": round(theta, 3),
                      "lip": [round(float(v), 3) for v in lip],
                      "xyz": [round(float(v), 4) for v in p], "rpy": rpy, **step})
    return steps, margins


def reverse_steps(plan: PourPlan) -> list[dict[str, Any]]:
    """The way back: theta_max -> hold through the same checked samples."""
    return list(reversed(plan.steps[:-1]))


# ----------------------------------------------------------------------------- picture

def side_view_svg(plan: PourPlan, hold_xyz: Sequence[float], hold_rpy: Sequence[float], beaker: Vessel,
                  obstacles: Sequence[Vessel] = ()) -> str:
    """Section in the pour plane (horizontal d, vertical z) with bottle outlines along the path."""
    geo = PourGeometry(hold_xyz, hold_rpy, beaker, plan.yaw_deg, plan.config.get("grasp_along_tool_mm", 0.0),
                       plan.config.get("grasp_up_mm", 0.0), plan.config.get("grasp_along_pour_mm", 0.0))
    d = geo.d
    ox = float(np.array([beaker.x, beaker.y, 0.0]) @ d)

    def uv(pt: np.ndarray) -> tuple[float, float]:
        return float(pt @ d) - ox, float(pt[2])

    shown = [s for s in plan.steps if s["phase"] in ("pretilt", "pour")]
    shown = shown[::max(1, len(shown) // 8)] + [plan.steps[-1]]
    us, vs = [], []
    outlines = []
    for s in [plan.steps[0], *shown]:
        pts = geo.bottle_points(s["theta_deg"], np.asarray(s["lip"], float))
        flat = [uv(q) for q in pts]
        outlines.append((s, flat))
        us += [a for a, _ in flat]
        vs += [b for _, b in flat]
    table = table_z_real("right")
    pan = PAN_TOP_CW - FINGERTIP_FROM_FLANGE_MM
    us += [-BALANCE_RADIUS_MM, BALANCE_RADIUS_MM]
    vs += [table, beaker.top_z]
    u0, u1, v0, v1 = min(us) - 30, max(us) + 30, table - 10, max(vs) + 30
    scale = 3.0
    W, H = (u1 - u0) * scale, (v1 - v0) * scale

    def X(u: float) -> float:
        return (u - u0) * scale

    def Y(v: float) -> float:
        return (v1 - v) * scale

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H + 40:.0f}" width="{W:.0f}" height="{H + 40:.0f}" '
             f'font-family="sans-serif" font-size="13"><rect width="100%" height="100%" fill="white"/>']
    parts.append(f'<line x1="0" y1="{Y(table):.1f}" x2="{W:.0f}" y2="{Y(table):.1f}" stroke="#888"/>')
    parts.append(f'<rect x="{X(-BALANCE_RADIUS_MM):.1f}" y="{Y(pan):.1f}" width="{2 * BALANCE_RADIUS_MM * scale:.1f}" '
                 f'height="{(pan - table) * scale:.1f}" fill="#eee" stroke="#aaa"/>')
    parts.append(f'<rect x="{X(-beaker.radius):.1f}" y="{Y(beaker.top_z):.1f}" width="{2 * beaker.radius * scale:.1f}" '
                 f'height="{(beaker.top_z - beaker.bottom_z) * scale:.1f}" fill="#cfe8ff" stroke="#3478c4" stroke-width="2"/>')
    for s, flat in outlines:
        pts = np.array(flat)
        hull = _hull(pts)
        colour = "#999" if s["phase"] == "hold" else ("#e07b00" if s["phase"] == "pretilt" else "#c0262d")
        path = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in hull)
        parts.append(f'<polygon points="{path}" fill="none" stroke="{colour}" stroke-width="1.5"/>')
    lip = np.asarray(plan.lip_xyz, float)
    lu, lv = uv(lip)
    parts.append(f'<circle cx="{X(lu):.1f}" cy="{Y(lv):.1f}" r="5" fill="#c0262d"/>')
    parts.append(f'<text x="8" y="{H + 25:.0f}">pour plane section (mm, right base): lip {plan.lip_above_rim_mm:.0f} mm '
                 f'above rim, pre-tilt {plan.theta_pre_deg:.0f}°, max {plan.theta_max_deg:.0f}°; '
                 f'grey = hold, orange = pre-tilt, red = pour</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _hull(points: np.ndarray) -> list[tuple[float, float]]:
    pts = sorted(set((round(float(a), 2), round(float(b), 2)) for a, b in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
