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


# ----------------------------------------------------------------------------- full pose FK / numeric IK
# J6 turns about -Y through the flange at the zero configuration; TOOL_AT_ZERO is the Tool0 orientation
# there. Fitted 2026-11-23 against 3148 controller InverseSolution samples (preflight plans of the
# handoff runs): position <= 0.3 mm, orientation <= 0.07 deg. Offline only: the live runtime still
# asks the controller for every IK.
_J6_AXIS = np.array([0.0, -1.0, 0.0])
TOOL_AT_ZERO = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])


def _rotm(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def flange_matrix(joints_deg: Sequence[float]) -> np.ndarray:
    """Tool0 orientation (3x3) in the base frame."""
    R = np.eye(3)
    for i, (axis, _) in enumerate(_AXES):
        R = R @ _rotm(axis, float(joints_deg[i]))
    return R @ _rotm(_J6_AXIS, float(joints_deg[5])) @ TOOL_AT_ZERO


def fk_pose(joints_deg: Sequence[float]) -> dict[str, list[float]]:
    """Tool0 pose {"xyz", "rpy"} (controller RPY = Rz@Ry@Rx, degrees)."""
    from task_skills.chemistry.geometry import matrix_to_euler_zyx
    xyz = link_points(joints_deg)["flange"]
    return {"xyz": [float(v) for v in xyz],
            "rpy": [float(v) for v in matrix_to_euler_zyx(flange_matrix(joints_deg).tolist())]}


def _log_rotation(R: np.ndarray) -> np.ndarray:
    """Rotation vector (radians) of R; correct up to 180 deg (the skew part alone vanishes there)."""
    cos = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    angle = math.acos(cos)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-6:
        return 0.5 * w
    if math.pi - angle < 1e-3:
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(axis))
        axis = (R[:, i] + np.eye(3)[:, i]) / 2.0
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    return w * (angle / (2.0 * math.sin(angle)))


def ik_numeric(xyz: Sequence[float], rpy: Sequence[float], seed: Sequence[float], *,
               iterations: int = 200, tol_mm: float = 0.01, tol_deg: float = 0.005) -> list[float]:
    """Damped least-squares IK from ``seed`` (stays on the seed's branch). Raises RuntimeError."""
    from task_skills.chemistry.geometry import euler_zyx_to_matrix
    target_p = np.asarray(xyz, float)
    target_R = np.array(euler_zyx_to_matrix([float(v) for v in rpy]), float)
    q = np.array([float(v) for v in seed], float)

    def error(qq: np.ndarray) -> np.ndarray:
        p = link_points(qq)["flange"]
        R = flange_matrix(qq)
        return np.concatenate([target_p - p, np.degrees(_log_rotation(target_R @ R.T)) * 2.0])  # ~2 mm per degree

    for _ in range(iterations):
        e = error(q)
        if np.linalg.norm(e[:3]) < tol_mm and np.linalg.norm(e[3:]) / 2.0 < tol_deg:
            R = flange_matrix(q)
            if float(np.trace(target_R.T @ R)) < 3.0 - 1e-6:  # independent check: not 180 deg off
                raise RuntimeError("ik_numeric converged to a wrong orientation")
            return [float(v) for v in q]
        J = np.zeros((6, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-3
            J[:, j] = (e - error(q + dq)) / 1e-3
        step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q = q + np.clip(step, -10.0, 10.0)
    raise RuntimeError("ik_numeric did not converge")
