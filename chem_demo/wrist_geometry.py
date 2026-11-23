"""Geometry for the wrist-camera bottle pick: board-free hand-eye and the upright-cylinder fit.

Frames and units: mm, the arm's own base frame, Tool0 = flange. Heights use the same convention as
TABLE_Z in chem_demo.locate: the "table height" of an arm is the flange Z at which the closed
fingertips touch the white table. The solved camera pose is therefore relative to a flange frame
whose origin sits at the fingertips (the hand-eye cannot see the real flange-to-fingertip length,
only this combination), and every object height comes out in the same convention, so a target
"fingertips h mm above the table" is simply flange Z = table_z + h.

Hand-eye without a board (run_wrist_calib): the D405 sees the textured table with valid depth.
The arm holds the tool straight down and makes small moves (X/Y/Z translations and turns about
the tool axis). Static table features, matched between frames and lifted to 3D with the aligned
depth, must map to the same base point from every pose:
    T_base_flange(i) @ T_flange_camera @ x_i  ==  T_base_flange(0) @ T_flange_camera @ x_0
Translations fix the camera rotation, turns about the tool axis fix its horizontal offset, and the
one thing these motions cannot see (the camera height along the tool axis) is fixed by the known
table height. Nothing here touches hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from task_skills.chemistry.geometry import euler_zyx_to_matrix
from vision_pick.cup_locate import Camera, frustum_silhouette


# ----------------------------------------------------------------------------- poses

def pose_matrix(pose: Sequence[float]) -> np.ndarray:
    """4x4 base <- flange from a controller pose (x, y, z mm, rx, ry, rz deg, Rz@Ry@Rx)."""
    T = np.eye(4)
    T[:3, :3] = np.array(euler_zyx_to_matrix(pose[3:6]), float)
    T[:3, 3] = [float(v) for v in pose[:3]]
    return T


def rotvec_matrix(rotvec: Sequence[float]) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rotvec, np.float64).reshape(3, 1))
    return R


def matrix_rotvec(R: np.ndarray) -> np.ndarray:
    vec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    return vec.reshape(3)


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising sum |R s_i - t_i|^2 (no translation)."""
    H = source.T @ target
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


# ----------------------------------------------------------------------------- camera helpers

def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def backproject(K: np.ndarray, pixels: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
    """Pixels (N,2) with camera-Z depths (N,) -> camera-frame points (N,3) mm."""
    x = (pixels[:, 0] - K[0, 2]) / K[0, 0]
    y = (pixels[:, 1] - K[1, 2]) / K[1, 1]
    return np.column_stack([x * depth_mm, y * depth_mm, depth_mm])


def project(K: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    return np.column_stack([K[0, 0] * points_cam[:, 0] / points_cam[:, 2] + K[0, 2],
                            K[1, 1] * points_cam[:, 1] / points_cam[:, 2] + K[1, 2]])


def sample_depth(depth: np.ndarray, pixels: np.ndarray, window: int = 2) -> np.ndarray:
    """Median of the valid depths in a (2w+1)^2 window per pixel; 0 where none is valid."""
    h, w = depth.shape
    out = np.zeros(len(pixels))
    for k, (u, v) in enumerate(np.round(pixels).astype(int)):
        if not (0 <= u < w and 0 <= v < h):
            continue
        patch = depth[max(v - window, 0):v + window + 1, max(u - window, 0):u + window + 1]
        valid = patch[patch > 0]
        if len(valid) >= max(3, patch.size // 2) and valid.max() - valid.min() < 0.03 * np.median(valid) + 2.0:
            out[k] = float(np.median(valid))  # reject depth edges (mixed foreground/background)
    return out


def wrist_camera(K: np.ndarray, flange_pose: Sequence[float], T_flange_camera: np.ndarray) -> Camera:
    """The wrist camera as a cup_locate.Camera (distortion ignored: D405 colour coefficients are ~0)."""
    return Camera(np.asarray(K, float), np.zeros(5), pose_matrix(flange_pose) @ T_flange_camera)


# ----------------------------------------------------------------------------- feature matching

def match_features(gray_a: np.ndarray, gray_b: np.ndarray, exclude: np.ndarray | None = None,
                   ratio: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + ratio test + mutual check; exclude = mask of pixels to ignore (the gripper)."""
    sift = cv2.SIFT_create(nfeatures=4000)
    allowed = None if exclude is None else (exclude == 0).astype(np.uint8) * 255
    ka, da = sift.detectAndCompute(gray_a, allowed)
    kb, db = sift.detectAndCompute(gray_b, allowed)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(da, db, k=2)
    backward = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
    pa, pb = [], []
    for pair in forward:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance and backward.get(best.trainIdx) == best.queryIdx:
            pa.append(ka[best.queryIdx].pt)
            pb.append(kb[best.trainIdx].pt)
    return np.array(pa, float).reshape(-1, 2), np.array(pb, float).reshape(-1, 2)


def fit_plane(points: np.ndarray, iterations: int = 300, tolerance: float = 3.0,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane n.x = d (unit n, pointing towards the camera); returns n, d, inlier mask."""
    rng = rng or np.random.default_rng(0)
    best = np.zeros(len(points), bool)
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        if np.linalg.norm(normal) < 1e-6:
            continue
        normal /= np.linalg.norm(normal)
        inliers = np.abs(points @ normal - normal @ sample[0]) < tolerance
        if inliers.sum() > best.sum():
            best = inliers
    centroid = points[best].mean(axis=0)
    _, _, Vt = np.linalg.svd(points[best] - centroid)
    normal = Vt[2]
    if normal @ centroid > 0:  # the camera (origin) is on the positive side
        normal = -normal
    d = float(normal @ centroid)
    inliers = np.abs(points @ normal - d) < tolerance
    return normal, d, inliers


# ----------------------------------------------------------------------------- hand-eye

@dataclass
class PairObservation:
    """Matched static points seen from the reference pose (index 0) and from pose i."""
    T0: np.ndarray      # base <- flange at the reference pose
    Ti: np.ndarray      # base <- flange at pose i
    x0: np.ndarray      # (N,3) camera frame, reference
    xi: np.ndarray      # (N,3) camera frame, pose i
    label: str = ""


def _residuals(params: np.ndarray, pairs: Sequence[PairObservation], plane_T: np.ndarray,
               plane_points: np.ndarray, table_z: float, plane_weight: float,
               keep: Sequence[np.ndarray]) -> np.ndarray:
    R, t = rotvec_matrix(params[:3]), params[3:]
    parts = []
    for pair, k in zip(pairs, keep):
        a = (pair.x0[k] @ R.T + t) @ pair.T0[:3, :3].T + pair.T0[:3, 3]
        b = (pair.xi[k] @ R.T + t) @ pair.Ti[:3, :3].T + pair.Ti[:3, 3]
        parts.append((a - b).reshape(-1))
    z = ((plane_points @ R.T + t) @ plane_T[:3, :3].T + plane_T[:3, 3])[:, 2]
    parts.append(plane_weight * (z - table_z))
    return np.concatenate(parts)


def initial_rotation(pairs: Sequence[PairObservation]) -> np.ndarray:
    """R_flange_camera from the pure translations: R0 R (mean xi - mean x0) = -(p_i - p_0)."""
    source, target = [], []
    for pair in pairs:
        if rotation_angle_deg(pair.T0[:3, :3].T @ pair.Ti[:3, :3]) > 0.5 or len(pair.x0) < 5:
            continue
        shift = pair.Ti[:3, 3] - pair.T0[:3, 3]
        if np.linalg.norm(shift) < 5.0:
            continue
        source.append(np.median(pair.xi - pair.x0, axis=0))
        target.append(-pair.T0[:3, :3].T @ shift)
    if len(source) < 2:
        raise RuntimeError("至少需要两次方向不同的平移才能定出相机朝向")
    return kabsch(np.array(source), np.array(target))


def initial_translation(R: np.ndarray, pairs: Sequence[PairObservation], plane_T: np.ndarray,
                        plane_points: np.ndarray, table_z: float) -> np.ndarray:
    """Linear least squares for t given R (pairs + the table-height row)."""
    rows, rhs = [], []
    for pair in pairs:
        A = pair.T0[:3, :3] - pair.Ti[:3, :3]
        for x0, xi in zip(pair.x0, pair.xi):
            rows.append(A)
            rhs.append(pair.Ti[:3, :3] @ (R @ xi) + pair.Ti[:3, 3] - pair.T0[:3, :3] @ (R @ x0) - pair.T0[:3, 3])
    weight = math.sqrt(max(1, sum(len(p.x0) for p in pairs)) / max(1, len(plane_points)))
    for p in plane_points:
        rows.append(weight * plane_T[2:3, :3])
        rhs.append(weight * np.array([table_z - plane_T[2, :3] @ (R @ p) - plane_T[2, 3]]))
    A = np.vstack(rows)
    b = np.concatenate([np.atleast_1d(v) for v in rhs])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def solve_hand_eye(pairs: Sequence[PairObservation], plane_T: np.ndarray, plane_points: np.ndarray,
                   table_z: float, *, rounds: int = 4, reject_mm: float = 2.0) -> dict:
    """Levenberg-Marquardt on all pair residuals plus the table-height prior, with outlier rejection.

    Returns T_flange_camera and per-pair residual statistics (mm).
    """
    R0 = initial_rotation(pairs)
    t0 = initial_translation(R0, pairs, plane_T, plane_points, table_z)
    params = np.concatenate([matrix_rotvec(R0), t0])
    keep = [np.ones(len(p.x0), bool) for p in pairs]
    plane_weight = 1.0
    for round_index in range(rounds):
        damping = 1e-3
        current = _residuals(params, pairs, plane_T, plane_points, table_z, plane_weight, keep)
        for _ in range(100):
            J = np.empty((len(current), 6))
            for k in range(6):
                delta = np.zeros(6)
                delta[k] = 1e-4 if k < 3 else 1e-2
                J[:, k] = (_residuals(params + delta, pairs, plane_T, plane_points, table_z, plane_weight, keep)
                           - _residuals(params - delta, pairs, plane_T, plane_points, table_z, plane_weight, keep)
                           ) / (2 * delta[k])
            normal, gradient = J.T @ J, J.T @ current
            improved = False
            while damping < 1e8:
                step = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), gradient)
                trial = _residuals(params + step, pairs, plane_T, plane_points, table_z, plane_weight, keep)
                if trial @ trial < current @ current:
                    params, current, damping, improved = params + step, trial, max(damping / 3, 1e-9), True
                    break
                damping *= 4
            if not improved or np.abs(step).max() < 1e-7:
                break
        if round_index == rounds - 1:
            break
        R, t = rotvec_matrix(params[:3]), params[3:]
        new_keep = []
        for pair in pairs:
            a = transform(pair.T0, pair.x0 @ R.T + t)
            b = transform(pair.Ti, pair.xi @ R.T + t)
            error = np.linalg.norm(a - b, axis=1)
            limit = max(reject_mm, 3.0 * float(np.median(error)))
            new_keep.append(error <= limit)
        keep = new_keep
    R, t = rotvec_matrix(params[:3]), params[3:]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    stats = []
    for pair, k in zip(pairs, keep):
        error = np.linalg.norm(transform(pair.T0, pair.x0[k] @ R.T + t) - transform(pair.Ti, pair.xi[k] @ R.T + t),
                               axis=1)
        stats.append({"pose": pair.label, "points": int(k.sum()), "of": int(len(k)),
                      "median_mm": round(float(np.median(error)), 3) if len(error) else None,
                      "p90_mm": round(float(np.percentile(error, 90)), 3) if len(error) else None})
    z = transform(plane_T, plane_points @ R.T + t)[:, 2]
    return {"T_flange_camera": T, "pairs": stats,
            "table_z_residual_mm": {"median": round(float(np.median(z - table_z)), 3),
                                    "p90_abs": round(float(np.percentile(np.abs(z - table_z), 90)), 3)}}


# ----------------------------------------------------------------------------- gripper mask

def gripper_mask(gray_frames: Sequence[np.ndarray], depth: np.ndarray, table_depth: float,
                 near_share: float = 0.6, still: int = 12, min_share: float = 0.004) -> np.ndarray:
    """Pixels that belong to the gripper: unchanged while the tool turns and closer than the table
    (or without depth). The table and everything on it rotate in the image; the gripper does not."""
    reference = cv2.GaussianBlur(gray_frames[0], (5, 5), 0).astype(np.int16)
    static = np.ones(reference.shape, bool)
    for frame in gray_frames[1:]:
        static &= np.abs(cv2.GaussianBlur(frame, (5, 5), 0).astype(np.int16) - reference) < still
    near = (depth == 0) | (depth < near_share * table_depth)
    mask = (static & near).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] >= min_share * mask.size:
            keep[labels == k] = 1
    return keep


def jaw_gap(mask: np.ndarray, K: np.ndarray, tip_cam: np.ndarray) -> dict:
    """Free space between the two largest gripper blobs, scaled to mm at the fingertip depth.

    An estimate: it assumes the narrowest image gap is at the fingertips. Reported, and used only
    to refuse a bottle that clearly does not fit.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 3:
        return {"ok": False, "reason": "画面里夹爪不是两块（夹爪合拢或被遮挡？）"}
    order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA])
    blobs = []
    for k in order[:2]:
        contours, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        blobs.append(max(contours, key=len).reshape(-1, 2).astype(float))
    a, b = blobs
    distances = np.linalg.norm(a[:, None, :] - b[None, ::4, :], axis=2)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    gap_px = float(distances[i, j])
    depth = float(tip_cam[2])
    return {"ok": True, "gap_px": round(gap_px, 1), "gap_mm": round(gap_px * depth / K[0, 0], 1),
            "points": [a[i].tolist(), b[::4][j].tolist()], "tip_depth_mm": round(depth, 1)}


# ----------------------------------------------------------------------------- cylinder fit

def usable_contour(mask: np.ndarray, blocked: np.ndarray, count: int = 300, guard: int = 6) -> tuple[np.ndarray, float]:
    """Outline points of the object mask that are real object edges: not next to the gripper or the
    image border (those are occlusion edges and would drag the fit). Returns points, usable share."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2)), 0.0
    points = max(contours, key=len).reshape(-1, 2)
    h, w = mask.shape
    near_block = cv2.dilate(blocked.astype(np.uint8), np.ones((2 * guard + 1, 2 * guard + 1), np.uint8))
    ok = ((points[:, 0] >= guard) & (points[:, 0] < w - guard) & (points[:, 1] >= guard) & (points[:, 1] < h - guard))
    ok &= near_block[points[:, 1].clip(0, h - 1), points[:, 0].clip(0, w - 1)] == 0
    usable = points[ok].astype(np.float64)
    share = float(ok.mean())
    if len(usable) > count:
        usable = usable[np.linspace(0, len(usable) - 1, count).astype(int)]
    return usable, share


def fit_upright_cylinder(camera: Camera, points: np.ndarray, base_z: float, start: Sequence[float],
                         trim: float = 0.15, rounds: int = 3) -> tuple[np.ndarray, float, float]:
    """Upright cylinder (x, y, h, r) standing on z = base_z whose silhouette passes through the given
    outline points. Same model and robust LM as chem_demo.locate.fit_cylinder, but on a filtered
    point set so occlusion edges are left out."""
    params = np.array(start, float)
    keep = np.ones(len(points), bool)

    def residuals(p: np.ndarray, pts: np.ndarray) -> np.ndarray:
        if p[2] < 5 or p[3] < 5:
            return np.full(len(pts), 1e3)
        outline = frustum_silhouette(camera, [p[0], p[1], p[2], p[3], p[3]], base_z)
        outline = outline.astype(np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(outline, (float(a), float(b)), True) for a, b in pts])

    for round_index in range(rounds):
        pts = points[keep]
        damping, current = 1e-2, residuals(params, pts)
        for _ in range(60):
            jacobian = np.empty((len(pts), 4))
            for k in range(4):
                delta = np.zeros(4)
                delta[k] = 0.1
                jacobian[:, k] = (residuals(params + delta, pts) - residuals(params - delta, pts)) / 0.2
            normal, gradient = jacobian.T @ jacobian, jacobian.T @ current
            improved = False
            while damping < 1e6:
                step = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), gradient)
                trial = residuals(params + step, pts)
                if trial @ trial < current @ current:
                    params, current, damping, improved = params + step, trial, max(damping / 3, 1e-6), True
                    break
                damping *= 4
            if not improved or np.abs(step).max() < 1e-3:
                break
        if round_index < rounds - 1:
            full = np.abs(residuals(params, points))
            keep = full <= np.quantile(full, 1.0 - trim)
    final = residuals(params, points[keep])
    return params, float(np.sqrt(np.mean(final ** 2))), float(keep.mean())


def table_height_check(camera: Camera, depth: np.ndarray, exclude: np.ndarray, table_z: float,
                       stride: int = 8) -> dict:
    """Independent check of the calibration: lift the valid table depth to base frame and compare
    its height with the known table height. Returns the median offset (mm) and point count."""
    K = camera.K
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    us, vs = us.reshape(-1), vs.reshape(-1)
    d = depth[vs, us].astype(float)
    ok = (d > 0) & (exclude[vs, us] == 0)
    if ok.sum() < 50:
        return {"ok": False, "points": int(ok.sum())}
    cam = backproject(K, np.column_stack([us[ok], vs[ok]]).astype(float), d[ok])
    base = transform(camera.T_base_camera, cam)
    z = base[:, 2]
    near_table = np.abs(z - table_z) < 25.0  # objects on the table are well above it
    if near_table.sum() < 50:
        return {"ok": False, "points": int(near_table.sum()), "median_z": round(float(np.median(z)), 1)}
    return {"ok": True, "points": int(near_table.sum()),
            "offset_mm": round(float(np.median(z[near_table]) - table_z), 2)}
