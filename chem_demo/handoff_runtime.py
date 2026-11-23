"""Glue between the live arms, the reachability planner and the handoff cycle.

Everything that can be tested without hardware lives here: the shared-driver arm
adapter, the planner call with live IK queries (cached, timed), and the closed-loop
pose refinement used by the right wrist servo.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from chem_demo.stations import photo_pose
from chem_demo.handoff_plan import (BottleHold, HandoffConfig, HandoffInfeasible, _Reject, _ik_step, _rot,
                                    plan_handoff)

MAX_TOTAL_REFINE_MM = 40.0   # cap on the tool-X/Y shift from the standoff pose (alignment + down-check)
# Pads this far short of the bottle axis when the joint move ends. 100 mm until 2026-11-22: with the
# flange->tips length underestimated by ~60 mm and the open fingers not modelled, the fingers came
# within a few cm of the bottle at the end of the joint arc (23:00 e-stop). 180 mm leaves the open
# fingertips (pads + 16) ~130 mm from the 35 mm body; the last 180 mm are a straight tool-axis move.
STANDOFF_MM = 180.0
RANGE_SLACK_MM = 60.0
# Handoff happens between the arms: left guard box and right radius are widened from the planner's
# bench defaults because the middle of the platform is ~530 mm from each base. The right radius is the
# TCP radius: the horizontal gripper puts the wrist ~200 mm nearer the base, so it may exceed the 625 mm reach. The controller IK and
# the 15 degree joint-jump rule remain the real limits.
# Handoff at the platform photo height (Kunyu, 2026-11-22): instruments stand on the platform, so
# the bottle is carried as high as the photo pose. Left flange 535 mm -> bottle bottom ~176 mm.
# 2026-11-22 evening: Kunyu found 535 still low; 610 mm puts the bottle bottom ~25 cm above the table
# (cw arithmetic: 610 - 115.7 - 243.2; independent of the flange->tips estimate) and is about the highest the left arm reaches with a vertical gripper near its base.
# 2026-11-22 night, corrected gripper geometry (pads ~104 mm from the flange, not 215): 610 mm is out of
# reach (read-only IK scan: left or right unreachable at every point); 570 mm is the highest feasible
# level (bottle bottom ~21 cm above the table), found at r ~380 mm from the left base.
LEFT_HANDOFF_Z_LEVELS = (570.0,)
LEFT_DXY_OFFSETS = (0.0, -40.0, -80.0, -120.0, -160.0)


def handoff_points(wide: bool = False) -> tuple[tuple[float, float, float], ...]:
    """Absolute left flange candidates at the photo height, on arcs around the left base.

    At 535 mm the left arm reaches ~440 mm horizontally (controller IK, 2026-11-22), less than the
    bottle's own radius, so the handoff is searched between the left base and the middle of the
    platform (azimuth -90 deg points at the right arm) instead of around the lifted bottle.
    """
    radii = (300.0, 340.0, 380.0, 420.0) if not wide else (300.0, 340.0, 380.0, 420.0, 460.0)
    azimuths = (-90.0, -80.0, -70.0, -60.0) if not wide else (-100.0, -90.0, -80.0, -70.0, -60.0, -50.0)
    z = LEFT_HANDOFF_Z_LEVELS[0]
    return tuple((round(r * np.cos(np.radians(a)), 2), round(r * np.sin(np.radians(a)), 2), z)
                 for r in radii for a in azimuths)


def handoff_config(left_pose: Sequence[float], *, wide: bool = False) -> HandoffConfig:
    """Search space for the live left TCP. Offsets are relative to ``left_pose``."""
    z = float(left_pose[2])
    dz = tuple(level - z for level in LEFT_HANDOFF_Z_LEVELS)
    return HandoffConfig(
        left_points_mm=handoff_points(wide),
        body_model=True,
        pre_standoff_mm=0.0,   # the joint move ends at the standoff itself: backing further folds the elbow
        # Horizontal side grasp (Kunyu, 2026-11-22: needed for pouring by J6). With the real ~120 mm
        # gripper a pitched tool puts the palm into the bottle, so the 35/20 deg tilts are gone.
        pitch_candidates_deg=(0.0,),
        left_x_range_mm=(-60.0, 330.0), left_y_range_mm=(-560.0, -150.0), left_z_range_mm=(350.0, 620.0),
        right_z_range_mm=(150.0, 620.0), right_radius_range_mm=(150.0, 800.0),
        left_dxy_offsets_mm=LEFT_DXY_OFFSETS if wide else (0.0, -60.0, -120.0, -160.0),
        left_dz_offsets_mm=dz,
        approach_offsets_deg=(-45.0, -30.0, -15.0, 0.0, 15.0, 30.0) if wide else (-30.0, -15.0, 0.0, 15.0),
        # The jaw axis is tool Y and the bottle stands upright: tool Y must be horizontal, tool X vertical.
        roll_candidates_deg=(90.0,),   # +90: tool X down, so the wrist camera looks at the bottle from above the axis
        standoff_candidates_mm=(STANDOFF_MM,),
    )


class CachedIK:
    """Memoise controller IK queries (same pose and seed -> same answer) and count them."""

    def __init__(self, ik: Callable, label: str):
        self.ik, self.label = ik, label
        self.cache: dict[tuple, Any] = {}
        self.calls = self.hits = 0

    def __call__(self, xyz: Sequence[float], rpy: Sequence[float], seed: Sequence[float]) -> list[float]:
        key = (tuple(round(float(v), 2) for v in xyz), tuple(round(float(v), 2) for v in rpy),
               tuple(round(float(v), 0) for v in seed))
        if key in self.cache:
            self.hits += 1
            result = self.cache[key]
        else:
            self.calls += 1
            try:
                result = list(self.ik(list(xyz), list(rpy), list(seed)))
            except RuntimeError as error:
                result = error
            self.cache[key] = result
        if isinstance(result, RuntimeError):
            raise result
        return list(result)


def make_plan_fn(left: Any, right: Any, T_right_left: Sequence[Sequence[float]], *, wide: bool = False,
                 grasp_drop_mm: float | None = None, body_diameter_mm: float = 70.0,
                 station: Mapping[str, Any] | None = None, body_model: bool = True,
                 pitch_deg: float | None = None,
                 log: Callable[[str], None] = print) -> Callable[[Mapping[str, Any], Mapping[str, Any]], dict]:
    """Return ``plan_fn(left_state, right_state)`` doing the search with live (read-only) IK queries.

    With ``station`` only that handoff point and approach azimuth are checked (a few seconds)
    instead of the whole grid (about a minute of controller IK queries).
    """

    def plan_fn(left_state: Mapping[str, Any], right_state: Mapping[str, Any]) -> dict[str, Any]:
        left_ik, right_ik = CachedIK(left.plan, "left"), CachedIK(right.plan, "right")
        config = dataclasses.replace(handoff_config(left_state["pose"], wide=wide), body_model=body_model)
        if pitch_deg is not None:
            config = dataclasses.replace(config, pitch_candidates_deg=(float(pitch_deg),))
        if station is not None:
            config = dataclasses.replace(config, left_points_mm=(tuple(float(v) for v in station["left_tcp"]),),
                                         approach_offsets_deg=(float(station["approach_offset_deg"]),))
            if "pitch_down_deg" in station:
                config = dataclasses.replace(config, pitch_candidates_deg=(float(station["pitch_down_deg"]),))
        hold = {} if grasp_drop_mm is None else {"grasp_drop_mm": grasp_drop_mm}
        bottle = BottleHold(tuple(float(v) for v in left_state["pose"]), body_diameter_mm=body_diameter_mm, **hold)
        started = time.monotonic()
        try:
            plan = plan_handoff(left_state, bottle, T_right_left, left_ik, right_ik, config,
                                fk_left=left.forward, fk_right=right.forward, right_start_state=right_state)
        except HandoffInfeasible as error:
            log(f"handoff search found nothing in {time.monotonic() - started:.1f} s: {dict(error.histogram)}")
            raise
        log(f"handoff search {time.monotonic() - started:.1f} s, IK queries left {left_ik.calls}"
            f"+{left_ik.hits} cached, right {right_ik.calls}+{right_ik.hits} cached")
        result = plan.to_dict()
        result["search"] = {"seconds": round(time.monotonic() - started, 2), "left_ik_calls": left_ik.calls,
                            "right_ik_calls": right_ik.calls, "config": config.to_dict()}
        result["handoff_left_tcp"] = result["left_handoff_xyz"]
        return result

    return plan_fn


def make_refine_fn(right: Any, origin: Mapping[str, Any], config: HandoffConfig | None = None,
                   limits: Sequence[Sequence[float]] | None = None
                   ) -> Callable[[Mapping[str, Any], Sequence[float]], dict[str, Any]]:
    """Shift a right-arm waypoint by a measured error given along the Tool0 X/Y/Z axes.

    The result is IK/FK checked from the current joints with the same rules as every planned
    step (15 degree jump, wrist margin, forward check), stays inside the planner's right-arm
    guard box, and the total shift from ``origin`` (the standoff pose) is capped in the tool
    frame: at most ``MAX_TOTAL_REFINE_MM`` sideways (X/Y) and ``-RANGE_SLACK_MM .. STANDOFF_MM +
    RANGE_SLACK_MM`` along the tool axis, so a bad measurement cannot walk the hand away.
    """
    cfg = config or HandoffConfig()
    origin_xyz = np.array(origin["xyz"], float)
    origin_R = _rot([float(v) for v in origin["rpy"]])

    def refine(current: Mapping[str, Any], delta_tool_mm: Sequence[float]) -> dict[str, Any]:
        delta = np.asarray(delta_tool_mm, float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise HandoffInfeasible({"bad_delta": 1}, "tool delta must be three finite numbers")
        rpy = [float(v) for v in current["rpy"]]
        xyz = np.array(current["xyz"], float) + _rot(rpy) @ delta
        shift = origin_R.T @ (xyz - origin_xyz)
        if (abs(shift[0]) > MAX_TOTAL_REFINE_MM or abs(shift[1]) > MAX_TOTAL_REFINE_MM or
                not -RANGE_SLACK_MM <= shift[2] <= STANDOFF_MM + RANGE_SLACK_MM):
            raise HandoffInfeasible({"refine_walked_too_far": 1}, "refinement left the planned neighbourhood")
        radius = float(np.hypot(xyz[0], xyz[1]))
        if not (cfg.right_z_range_mm[0] <= xyz[2] <= cfg.right_z_range_mm[1] and
                cfg.right_radius_range_mm[0] <= radius <= cfg.right_radius_range_mm[1]):
            raise HandoffInfeasible({"right_guard": 1}, "refined right pose leaves the right-arm guard box")
        # A long refine (the smooth approach: one MovL over ~15 cm) is checked like a planned segment:
        # samples every cfg.sample_step_mm along the straight line, each seeded from the previous one,
        # so the 15 degree jump rule applies per sample, not to the whole move.
        start = np.array(current["xyz"], float)
        count = max(1, int(np.ceil(float(np.linalg.norm(xyz - start)) / cfg.sample_step_mm)))
        seed = list(current["joints"])
        try:
            for k in range(1, count + 1):
                point = start + (xyz - start) * (k / count)
                step = _ik_step(right.plan, right.forward, "right", point, rpy, seed, limits, cfg)
                seed = step["joints"]
        except _Reject as rejected:
            raise HandoffInfeasible({rejected.reason: 1}, "refined right pose rejected") from None
        return {"name": "refined", "xyz": [round(float(v), 4) for v in xyz], "rpy": rpy,
                "joints": step["joints"], "step_kind": "refine"}

    return refine
