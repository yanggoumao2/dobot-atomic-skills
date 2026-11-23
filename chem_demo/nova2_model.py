"""Nova 2 link geometry for collision checks (positions only, base frame, mm and degrees).

Dimensions from the Dobot Nova Series User Guide V1.5, figure 5.1: shoulder height 223.4, upper
arm 280, forearm 225, J4->J5 120, J5->flange 88, forearm/wrist plane offset 117.5 from the base
axis. Joint axes at zero (arm straight up): J1 +Z; J2, J3, J4 -Y; J5 +Z through y = -117.5.
Checked against the controller's own forward kinematics (``PositiveSolution``) on both arms,
see tests/test_nova2_model.py. Link radii are estimates from the drawing and photos.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

D1, A2, A3, D5, D6, OFFSET = 223.4, 280.0, 225.0, 120.0, 88.0, 117.5

# (axis, point on axis) at the zero configuration.
_AXES = [
    (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])),
    (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, D1])),
    (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, D1 + A2])),
    (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, D1 + A2 + A3])),
    (np.array([0.0, 0.0, 1.0]), np.array([0.0, -OFFSET, 0.0])),
]

# Zero-configuration points and the last joint that moves them.
_POINTS = {
    "shoulder": (np.array([0.0, 0.0, D1]), 1),
    "elbow_inner": (np.array([0.0, 0.0, D1 + A2]), 2),
    "elbow_outer": (np.array([0.0, -OFFSET, D1 + A2]), 2),
    "wrist1": (np.array([0.0, -OFFSET, D1 + A2 + A3]), 3),
    "wrist2": (np.array([0.0, -OFFSET, D1 + A2 + A3 + D5]), 4),
    "flange": (np.array([0.0, -OFFSET - D6, D1 + A2 + A3 + D5]), 5),
}

# Capsules (segment end names, radius mm). Radii are estimates: housings ~90-100 mm across.
LINKS = {
    "base": ("base_bottom", "shoulder", 60.0),
    "upper_arm": ("shoulder", "elbow_inner", 50.0),
    "elbow": ("elbow_inner", "elbow_outer", 50.0),
    "forearm": ("elbow_outer", "wrist1", 45.0),
    "wrist": ("wrist1", "wrist2", 45.0),
    "wrist2_to_flange": ("wrist2", "flange", 45.0),
}


def _rotate(point: np.ndarray, axis: np.ndarray, centre: np.ndarray, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    k = axis / np.linalg.norm(axis)
    v = point - centre
    return centre + v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * float(k @ v) * (1 - math.cos(a))


def link_points(joints_deg: Sequence[float]) -> dict[str, np.ndarray]:
    """Key points of the arm in its base frame for the given joint angles (degrees)."""
    q = [float(v) for v in joints_deg]
    out = {"base_bottom": np.array([0.0, 0.0, 0.0])}
    for name, (p0, last) in _POINTS.items():
        p = p0.copy()
        for i in range(last - 1, -1, -1):  # product of exponentials, innermost joint first
            axis, centre = _AXES[i]
            p = _rotate(p, axis, centre, q[i])
        out[name] = p
    return out


def capsules(joints_deg: Sequence[float], tool_rpy: Sequence[float] | None = None, *,
             tool_segments: Sequence[tuple[float, float, float, str]] = ()) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """Link capsules, plus optional tool capsules given as (start_mm, end_mm, radius, name) along tool Z."""
    pts = link_points(joints_deg)
    caps = [(pts[a], pts[b], r, name) for name, (a, b, r) in LINKS.items()]
    if tool_segments:
        if tool_rpy is None:
            raise ValueError("tool segments need the tool orientation")
        from chem_demo.handoff_plan import _rot
        z = _rot([float(v) for v in tool_rpy])[:, 2]
        f = pts["flange"]
        caps += [(f + z * s, f + z * e, r, name) for s, e, r, name in tool_segments]
    return caps

