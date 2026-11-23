"""Register partial right-wrist checkerboard detections to one reference grid.

The plain checker has no IDs, so local corner indices cannot safely be compared across views.
This module anchors the mapping with depth-backed SIFT features outside the repeated checker
pattern, restricted to the board plane. A robust PnP fit predicts where every current board
corner lies in the reference camera, including corners outside the reference image. Registration
is rejected when those independent anchors or the checker lattice do not agree.
"""
from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from chem_demo.check_wrist_board import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    DEFAULT_SQUARE_MM,
    PNP_RMS_LIMIT_PX,
    _metric_pnp,
    detect_checkerboard,
)


MIN_OFF_GRID_MATCHES = 6
MIN_PNP_INLIERS = 6
MIN_PNP_INLIER_SHARE = 0.45
PNP_REPROJECTION_LIMIT_PX = 2.5
PLANE_DEPTH_LIMIT_MM = 12.0
PLANE_NORMAL_LIMIT_DEG = 12.0
PLANE_OFFSET_LIMIT_MM = 20.0
GRID_RESIDUAL_LIMIT = 0.30
AUGMENTED_CHART_LIMIT_PX = 2.5
AUGMENTED_DEPTH_LIMIT_MM = 12.0
AUGMENTED_SADDLE_LIMIT = 22.0


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("images must be grayscale, BGR, or BGRA")
    if gray.dtype != np.uint8 or not gray.size:
        raise ValueError("images must be non-empty uint8 arrays")
    return gray


def _validate_camera(K: np.ndarray) -> np.ndarray:
    matrix = np.asarray(K, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or
            matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or
            not np.allclose(matrix[2], [0.0, 0.0, 1.0])):
        raise ValueError("K must be a finite 3x3 pinhole camera matrix")
    return matrix


def _detection(image: np.ndarray, depth: np.ndarray, K: np.ndarray,
               supplied: Mapping[str, Any] | None, cols: int, rows: int,
               square_mm: float) -> dict[str, Any]:
    found = (dict(supplied) if supplied is not None else
             detect_checkerboard(image, cols, rows, square_mm, K=K, depth_mm=depth))
    pixels = np.asarray(found.get("used_corner_pixels", []), dtype=np.float64).reshape(-1, 2)
    grid = np.asarray(found.get("used_grid_indices_col_row_local", []), dtype=np.float64).reshape(-1, 2)
    if len(pixels) < 6 or len(grid) != len(pixels) or not np.all(np.isfinite(pixels)):
        raise ValueError("checker detections need at least six aligned usable corner pixels and grid indices")
    if not np.allclose(grid, np.rint(grid), atol=1e-6):
        raise ValueError("checker grid indices must be integers")
    result = dict(found)
    result["used_corner_pixels"] = pixels
    result["used_grid_indices_col_row_local"] = np.rint(grid).astype(np.int32)
    return result


def _sample_depth(depth: np.ndarray, pixel: np.ndarray, radius: int = 2) -> float:
    h, w = depth.shape
    u, v = np.rint(pixel).astype(int)
    if not (0 <= u < w and 0 <= v < h):
        return 0.0
    patch = depth[max(0, v - radius):min(h, v + radius + 1),
                  max(0, u - radius):min(w, u + radius + 1)]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < max(3, patch.size // 3):
        return 0.0
    median = float(np.median(valid))
    if float(valid.max() - valid.min()) > max(20.0, median * 0.025):
        return 0.0
    return median


def _backproject(K: np.ndarray, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    return np.column_stack(((pixels[:, 0] - K[0, 2]) * depths / K[0, 0],
                            (pixels[:, 1] - K[1, 2]) * depths / K[1, 1], depths))


def _fit_board_plane(pixels: np.ndarray, depth: np.ndarray, K: np.ndarray
                     ) -> tuple[np.ndarray, float, int, np.ndarray] | None:
    depths = np.array([_sample_depth(depth, pixel) for pixel in pixels], dtype=np.float64)
    valid = depths > 0
    if valid.sum() < 6:
        return None
    points = _backproject(K, pixels[valid], depths[valid])
    if len(points) < 6:
        return None
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    threshold = max(4.0, float(np.median(points[:, 2])) * 0.006)
    for _ in range(min(500, max(100, len(points) * 8))):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        inliers = np.abs(points @ normal - float(normal @ sample[0])) <= threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < max(6, int(np.ceil(0.60 * len(points)))):
        return None
    centroid = points[best].mean(axis=0)
    _, _, vt = np.linalg.svd(points[best] - centroid, full_matrices=False)
    normal = vt[-1]
    d = float(normal @ centroid)
    if d < 0:
        normal, d = -normal, -d
    residuals = np.abs(points @ normal - d)
    inliers = residuals <= threshold
    if inliers.sum() < max(6, int(np.ceil(0.60 * len(points)))):
        return None
    return normal, d, int(inliers.sum()), depths


def _checker_exclusion(shape: tuple[int, int], detection: Mapping[str, Any],
                       supplied: np.ndarray | None) -> np.ndarray:
    h, w = shape
    excluded = np.zeros((h, w), dtype=np.uint8)
    pixels = np.asarray(detection["used_corner_pixels"], dtype=np.float32)
    hull = cv2.convexHull(pixels.reshape(-1, 1, 2))
    cv2.fillConvexPoly(excluded, np.rint(hull).astype(np.int32), 255)
    grid = detection["used_grid_indices_col_row_local"]
    by_site = {tuple(site): pixel for site, pixel in zip(grid, pixels)}
    spacings = []
    for c, r in by_site:
        for neighbor in ((c + 1, r), (c, r + 1)):
            if neighbor in by_site:
                spacings.append(float(np.linalg.norm(by_site[(c, r)] - by_site[neighbor])))
    # Internal corners stop half a square inside the printed pattern. Exclude at least one
    # checker cell plus a small guard so repeated squares cannot masquerade as a board anchor.
    margin = int(np.clip(round(0.62 * np.median(spacings)) if spacings else 8, 6, 24))
    if margin:
        size = 2 * margin + 1
        excluded = cv2.dilate(excluded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    if supplied is not None:
        mask = np.asarray(supplied)
        if mask.shape != (h, w):
            raise ValueError("exclude masks must match their image dimensions")
        excluded[mask != 0] = 255
    return excluded


def _off_grid_features(gray: np.ndarray, depth: np.ndarray, K: np.ndarray,
                       plane_normal: np.ndarray, plane_d: float,
                       detection: Mapping[str, Any], exclude_mask: np.ndarray | None
                       ) -> tuple[list[cv2.KeyPoint], np.ndarray, np.ndarray, np.ndarray]:
    allowed = 255 - _checker_exclusion(gray.shape, detection, exclude_mask)
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.008, edgeThreshold=12)
    keypoints, descriptors = sift.detectAndCompute(gray, allowed)
    if descriptors is None or not keypoints:
        return [], np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 128), np.float32)
    kept, points, pixels, desc = [], [], [], []
    distance_from_checker = cv2.distanceTransform(allowed, cv2.DIST_L2, 3)
    for index, keypoint in enumerate(keypoints):
        pixel = np.asarray(keypoint.pt, dtype=np.float64)
        u, v = np.rint(pixel).astype(int)
        # A SIFT descriptor reaches well beyond its center. Require the whole support region to
        # stay clear of the repeated pattern as well as requiring the keypoint center outside it.
        support_radius = max(4.0, 2.5 * float(keypoint.size))
        if distance_from_checker[v, u] <= support_radius:
            continue
        z = _sample_depth(depth, pixel)
        if z <= 0:
            continue
        xyz = _backproject(K, pixel[None, :], np.array([z]))[0]
        if abs(float(plane_normal @ xyz - plane_d)) > PLANE_DEPTH_LIMIT_MM:
            continue
        kept.append(keypoint)
        points.append(xyz)
        pixels.append(pixel)
        desc.append(descriptors[index])
    if not desc:
        return [], np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 128), np.float32)
    return kept, np.asarray(points), np.asarray(pixels), np.asarray(desc, dtype=np.float32)


def _mutual_ratio_matches(desc_ref: np.ndarray, desc_cur: np.ndarray, ratio: float = 0.76
                          ) -> tuple[np.ndarray, np.ndarray]:
    if len(desc_ref) < 2 or len(desc_cur) < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(desc_ref, desc_cur, k=2)
    backward = matcher.knnMatch(desc_cur, desc_ref, k=2)
    reverse_best = {}
    for pair in backward:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            reverse_best[pair[0].queryIdx] = pair[0].trainIdx
    ia, ib = [], []
    for pair in forward:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance and reverse_best.get(best.trainIdx) == best.queryIdx:
            ia.append(best.queryIdx)
            ib.append(best.trainIdx)
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def _fit_reference_grid(detection: Mapping[str, Any]) -> np.ndarray | None:
    grid = detection["used_grid_indices_col_row_local"].astype(np.float64)
    pixels = detection["used_corner_pixels"].astype(np.float64)
    homography, _ = cv2.findHomography(grid, pixels, 0)
    if homography is None or abs(float(np.linalg.det(homography))) < 1e-12:
        return None
    return np.linalg.inv(homography)


def _reject(reason: str, **details: Any) -> dict[str, Any]:
    return {"status": "REJECTED", "accepted": False, "reason": reason,
            "current_site_ids_in_reference": [], "matched_intersections": [], **details}


def register_checkerboard_sites(
        reference_image: np.ndarray,
        reference_depth_mm: np.ndarray,
        current_image: np.ndarray,
        current_depth_mm: np.ndarray,
        K: np.ndarray,
        *,
        reference_detection: Mapping[str, Any] | None = None,
        current_detection: Mapping[str, Any] | None = None,
        reference_exclude_mask: np.ndarray | None = None,
        current_exclude_mask: np.ndarray | None = None,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
        square_mm: float = DEFAULT_SQUARE_MM,
) -> dict[str, Any]:
    """Register current usable checker corners to the reference view's local integer lattice.

    ``current_site_ids_in_reference`` is an Nx2 integer array serialized as a list, aligned
    one-to-one with ``current_detection['used_corner_pixels']``. IDs are local to the reference
    detector grid and may be signed or outside the reference-visible patch. The reference grid's
    global physical board origin remains unknown because the printed checker has no IDs.

    Jaw/exclusion masks use nonzero pixels for ignored regions. Distortion should already be
    removed from the supplied images; K is the matching pinhole camera matrix.
    """
    K = _validate_camera(K)
    if (not isinstance(cols, int) or not isinstance(rows, int) or cols < 3 or rows < 3 or
            not np.isfinite(square_mm) or square_mm <= 0):
        raise ValueError("cols and rows must be at least three and square_mm must be positive")
    reference_gray, current_gray = _gray(reference_image), _gray(current_image)
    reference_depth = np.asarray(reference_depth_mm, dtype=np.float64)
    current_depth = np.asarray(current_depth_mm, dtype=np.float64)
    if reference_depth.shape != reference_gray.shape or current_depth.shape != current_gray.shape:
        raise ValueError("depth arrays must align with their corresponding color images")
    if (reference_gray.shape != current_gray.shape or
            not np.all(np.isfinite(reference_depth)) or not np.all(np.isfinite(current_depth))):
        raise ValueError("both images must share dimensions and depths must be finite")

    ref = _detection(reference_image, reference_depth, K, reference_detection, cols, rows, square_mm)
    cur = _detection(current_image, current_depth, K, current_detection, cols, rows, square_mm)
    ref_plane = _fit_board_plane(ref["used_corner_pixels"], reference_depth, K)
    cur_plane = _fit_board_plane(cur["used_corner_pixels"], current_depth, K)
    if ref_plane is None or cur_plane is None:
        return _reject("checker corners do not have enough consistent depth to fit both board planes")
    n_ref, d_ref, ref_plane_count, _ = ref_plane
    n_cur, d_cur, cur_plane_count, _ = cur_plane

    _, ref_xyz, ref_feature_pixels, ref_desc = _off_grid_features(
        reference_gray, reference_depth, K, n_ref, d_ref, ref, reference_exclude_mask)
    _, _, cur_feature_pixels, cur_desc = _off_grid_features(
        current_gray, current_depth, K, n_cur, d_cur, cur, current_exclude_mask)
    match_ref, match_cur = _mutual_ratio_matches(ref_desc, cur_desc)
    feature_count = len(match_ref)
    if feature_count < MIN_OFF_GRID_MATCHES:
        return _reject("too few unique off-grid SIFT matches lie on the checker plane",
                       off_grid_match_count=feature_count,
                       required_off_grid_matches=MIN_OFF_GRID_MATCHES,
                       reference_plane_features=len(ref_desc), current_plane_features=len(cur_desc))
    object_points = ref_xyz[match_ref].astype(np.float64)
    image_points = cur_feature_pixels[match_cur].astype(np.float64)
    if (np.linalg.matrix_rank(object_points - object_points.mean(axis=0), tol=1e-3) < 2 or
            np.linalg.matrix_rank(image_points - image_points.mean(axis=0), tol=1e-2) < 2):
        return _reject("off-grid matches do not span a two-dimensional board-plane region",
                       off_grid_match_count=feature_count)

    cv2.setRNGSeed(0)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, K, None, iterationsCount=3000,
            reprojectionError=PNP_REPROJECTION_LIMIT_PX, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP)
    except cv2.error as exc:
        return _reject(f"depth-backed board PnP/RANSAC failed: {exc}",
                       off_grid_match_count=feature_count)
    if (not ok or inliers is None or len(inliers) < MIN_PNP_INLIERS or
            len(inliers) / max(1, feature_count) < MIN_PNP_INLIER_SHARE):
        return _reject("off-grid board matches do not support a unique PnP/RANSAC fit",
                       off_grid_match_count=feature_count,
                       pnp_inlier_count=0 if inliers is None else int(len(inliers)),
                       pnp_inlier_share=(0.0 if inliers is None else len(inliers) / max(1, feature_count)),
                       required_pnp_inlier_share=MIN_PNP_INLIER_SHARE)
    inlier_idx = inliers.reshape(-1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(object_points[inlier_idx], image_points[inlier_idx],
                                         K, None, rvec, tvec)
    except cv2.error:
        pass
    rotation, _ = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    predicted = object_points[inlier_idx] @ rotation.T + translation
    if np.any(predicted[:, 2] <= 0):
        return _reject("PnP places off-grid board features behind the current camera",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)))
    projected, _ = cv2.projectPoints(object_points[inlier_idx], rvec, tvec, K, None)
    reprojection = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_idx], axis=1)
    reproj_median = float(np.median(reprojection))
    reproj_p90 = float(np.percentile(reprojection, 90))
    if reproj_median > PNP_REPROJECTION_LIMIT_PX or reproj_p90 > 4.0:
        return _reject("off-grid PnP reprojection error is too large",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)),
                       pnp_median_reprojection_px=reproj_median,
                       pnp_p90_reprojection_px=reproj_p90)

    predicted_normal = rotation @ n_ref
    predicted_d = float(d_ref + predicted_normal @ translation)
    if predicted_normal @ n_cur < 0:
        n_cur, d_cur = -n_cur, -d_cur
    normal_angle = float(np.degrees(np.arccos(np.clip(predicted_normal @ n_cur, -1.0, 1.0))))
    plane_offset = abs(predicted_d - d_cur)
    if normal_angle > PLANE_NORMAL_LIMIT_DEG or plane_offset > PLANE_OFFSET_LIMIT_MM:
        return _reject("depth-backed PnP disagrees with the independently fitted current checker plane",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)),
                       checker_plane_normal_error_deg=normal_angle,
                       checker_plane_offset_error_mm=plane_offset)

    current_depth_samples = np.array([_sample_depth(current_depth, image_points[i]) for i in inlier_idx])
    valid_depth = current_depth_samples > 0
    feature_depth_error = np.abs(predicted[valid_depth, 2] - current_depth_samples[valid_depth])
    if (valid_depth.sum() < MIN_PNP_INLIERS or
            float(np.median(feature_depth_error)) > PLANE_DEPTH_LIMIT_MM):
        return _reject("off-grid PnP inliers fail the aligned current-depth check",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)),
                       valid_current_depth_inlier_count=int(valid_depth.sum()),
                       pnp_depth_median_error_mm=(float(np.median(feature_depth_error))
                                                  if len(feature_depth_error) else None))

    inverse_grid_h = _fit_reference_grid(ref)
    if inverse_grid_h is None:
        return _reject("reference checker corners do not define an invertible grid chart",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)))
    current_pixels = cur["used_corner_pixels"]
    ray = np.column_stack(((current_pixels[:, 0] - K[0, 2]) / K[0, 0],
                           (current_pixels[:, 1] - K[1, 2]) / K[1, 1],
                           np.ones(len(current_pixels))))
    denominator = ray @ n_cur
    if np.any(np.abs(denominator) < 1e-8):
        return _reject("some checker corner rays are parallel to the fitted board plane",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)))
    current_xyz = ray * (d_cur / denominator)[:, None]
    reference_xyz = (current_xyz - translation) @ rotation
    if np.any(reference_xyz[:, 2] <= 0):
        return _reject("current checker corners map behind the reference camera",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)))
    reference_pixels = np.column_stack((K[0, 0] * reference_xyz[:, 0] / reference_xyz[:, 2] + K[0, 2],
                                        K[1, 1] * reference_xyz[:, 1] / reference_xyz[:, 2] + K[1, 2]))
    grid_h = np.column_stack((reference_pixels, np.ones(len(reference_pixels)))) @ inverse_grid_h.T
    if np.any(np.abs(grid_h[:, 2]) < 1e-10):
        return _reject("current checker corners map to infinity in the reference grid chart",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)))
    grid_coordinates = grid_h[:, :2] / grid_h[:, 2, None]
    site_ids = np.rint(grid_coordinates).astype(np.int32)
    grid_residuals = np.linalg.norm(grid_coordinates - site_ids, axis=1)
    if np.any(grid_residuals > GRID_RESIDUAL_LIMIT):
        return _reject("off-grid evidence does not place all current corners on one reference checker lattice",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)),
                       maximum_grid_residual=float(grid_residuals.max()))
    if len({tuple(site) for site in site_ids}) != len(site_ids):
        return _reject("distinct current checker corners collapse onto duplicate reference site IDs",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)))

    current_local = cur["used_grid_indices_col_row_local"].astype(np.float64)
    design = np.column_stack((current_local, np.ones(len(current_local))))
    if np.linalg.matrix_rank(design) < 3:
        return _reject("current checker corner indices do not span a two-dimensional lattice",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)))
    coefficients, _, _, _ = np.linalg.lstsq(design, grid_coordinates, rcond=None)
    basis = np.rint(coefficients[:2, :]).astype(np.int32).T
    offset = np.rint(coefficients[2, :]).astype(np.int32)
    signed_permutation = (np.all(np.sum(np.abs(basis), axis=0) == 1) and
                          np.all(np.sum(np.abs(basis), axis=1) == 1))
    lattice_fit = current_local @ basis.T + offset
    lattice_residual = np.linalg.norm(grid_coordinates - lattice_fit, axis=1)
    if not signed_permutation or np.any(lattice_residual > GRID_RESIDUAL_LIMIT):
        return _reject("current checker ordering is ambiguous or inconsistent with the reference lattice",
                       off_grid_match_count=feature_count, pnp_inlier_count=int(len(inlier_idx)),
                       maximum_lattice_residual=float(lattice_residual.max()),
                       current_to_reference_basis=basis.tolist())

    ref_site_to_pixel = {
        tuple(site): pixel.tolist()
        for site, pixel in zip(ref["used_grid_indices_col_row_local"], ref["used_corner_pixels"])
    }
    matched = []
    for site, pixel in zip(site_ids, current_pixels):
        site_key = tuple(int(value) for value in site)
        if site_key in ref_site_to_pixel:
            matched.append({"site_index_col_row": list(site_key),
                            "reference_pixel": ref_site_to_pixel[site_key],
                            "current_pixel": pixel.tolist()})
    return {
        "status": "PASS",
        "accepted": True,
        "reason": None,
        "board": {"internal_corners": [int(cols), int(rows)], "square_mm": float(square_mm)},
        "coordinate_convention": "reference detector-local checker grid; unknown absolute board origin",
        "current_site_ids_in_reference": site_ids.tolist(),
        "matched_intersections": matched,
        "evidence": {
            "anchor": "off-grid SIFT features with depth on the checker plane",
            "off_grid_match_count": feature_count,
            "pnp_inlier_count": int(len(inlier_idx)),
            "pnp_median_reprojection_px": reproj_median,
            "pnp_p90_reprojection_px": reproj_p90,
            "pnp_depth_median_error_mm": float(np.median(feature_depth_error)),
            "reference_plane_corner_inliers": ref_plane_count,
            "current_plane_corner_inliers": cur_plane_count,
            "checker_plane_normal_error_deg": normal_angle,
            "checker_plane_offset_error_mm": plane_offset,
            "maximum_grid_residual": float(grid_residuals.max()),
            "maximum_lattice_residual": float(lattice_residual.max()),
            "current_to_reference_basis": basis.tolist(),
            "current_to_reference_offset": offset.tolist(),
        },
    }


def _strict_d4_extension(source_sites: np.ndarray, target_sites: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the exact signed-permutation map source -> target, or reject the fit."""
    source = np.asarray(source_sites, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_sites, dtype=np.float64).reshape(-1, 2)
    if len(source) < 6 or len(source) != len(target):
        return None
    if len({tuple(point) for point in source}) != len(source):
        return None
    design = np.column_stack((source, np.ones(len(source))))
    if np.linalg.matrix_rank(design) < 3:
        return None
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    basis = np.rint(coefficients[:2, :]).astype(np.int32).T
    offset = np.rint(coefficients[2, :]).astype(np.int32)
    d4 = (np.all(np.sum(np.abs(basis), axis=0) == 1) and
          np.all(np.sum(np.abs(basis), axis=1) == 1))
    if not d4 or not np.array_equal(source.astype(np.int32) @ basis.T + offset,
                                    target.astype(np.int32)):
        return None
    return basis, offset


def register_augmented_checkerboard_sites(
        reference_image: np.ndarray,
        reference_depth_mm: np.ndarray,
        current_image: np.ndarray,
        current_depth_mm: np.ndarray,
        K: np.ndarray,
        *,
        reference_base_detection: Mapping[str, Any],
        current_base_detection: Mapping[str, Any],
        current_augmented_detection: Mapping[str, Any],
        reference_exclude_mask: np.ndarray | None = None,
        current_exclude_mask: np.ndarray | None = None,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
        square_mm: float = DEFAULT_SQUARE_MM,
) -> dict[str, Any]:
    """Extend accepted base-chart IDs to measured additions using a strict D4 lattice map.

    SIFT/PnP registration uses the unaugmented reference and current detections so expanded
    checker hulls cannot hide the unique off-grid anchor. Added current sites inherit IDs only
    through the exact signed-permutation transform established by those accepted base matches.
    The augmented detection must retain every base site and its own saddle, chart, depth, and
    metric-PnP evidence must still pass.
    """
    K = _validate_camera(K)
    base_registration = register_checkerboard_sites(
        reference_image, reference_depth_mm, current_image, current_depth_mm, K,
        reference_detection=reference_base_detection,
        current_detection=current_base_detection,
        reference_exclude_mask=reference_exclude_mask,
        current_exclude_mask=current_exclude_mask,
        cols=cols, rows=rows, square_mm=square_mm)
    if not base_registration.get("accepted"):
        result = dict(base_registration)
        result["phase_extension"] = {"status": "REJECTED", "reason": base_registration.get("reason")}
        return result

    current_depth = np.asarray(current_depth_mm, dtype=np.float64)
    base = _detection(current_image, current_depth, K, current_base_detection, cols, rows, square_mm)
    augmented = _detection(current_image, current_depth, K, current_augmented_detection,
                           cols, rows, square_mm)
    base_grid = base["used_grid_indices_col_row_local"].astype(np.int32)
    base_pixels = base["used_corner_pixels"].astype(np.float64)
    augmented_grid = augmented["used_grid_indices_col_row_local"].astype(np.int32)
    augmented_pixels = augmented["used_corner_pixels"].astype(np.float64)
    base_ids = np.asarray(base_registration["current_site_ids_in_reference"], dtype=np.int32)
    d4 = _strict_d4_extension(base_grid, base_ids)
    if d4 is None:
        return _reject("base registration does not define one exact D4 checker phase",
                       phase_extension={"status": "REJECTED", "reason": "base-to-reference map is not D4"},
                       base_registration=base_registration)
    basis, offset = d4

    base_by_site = {tuple(site): pixel for site, pixel in zip(base_grid, base_pixels)}
    augmented_by_site = {tuple(site): pixel for site, pixel in zip(augmented_grid, augmented_pixels)}
    if len(augmented_by_site) != len(augmented_grid):
        return _reject("augmented detection contains duplicate local checker sites",
                       phase_extension={"status": "REJECTED", "reason": "duplicate augmented local sites"},
                       base_registration=base_registration)
    if any(site not in augmented_by_site for site in base_by_site):
        return _reject("augmented detection dropped an original measured base site",
                       phase_extension={"status": "REJECTED", "reason": "base site missing from augmented set"},
                       base_registration=base_registration)
    preserved_error = max(float(np.linalg.norm(augmented_by_site[site] - pixel))
                          for site, pixel in base_by_site.items())
    if preserved_error > 0.25:
        return _reject("augmented detection changed original measured base corners",
                       phase_extension={"status": "REJECTED", "maximum_base_pixel_change_px": preserved_error},
                       base_registration=base_registration)

    additions = set(augmented_by_site) - set(base_by_site)
    augmentation = current_augmented_detection.get("corner_augmentation", {})
    reported_additions = {tuple(map(int, site)) for site in
                          augmentation.get("added_grid_indices_col_row_local", [])}
    if additions:
        phase_agreement = float(augmentation.get("phase_agreement", np.nan))
        if (augmentation.get("status") != "PASS" or
                int(augmentation.get("added_corner_count", -1)) != len(additions) or
                reported_additions != additions or
                int(augmentation.get("phase_support_count", 0)) < 8 or
                not np.isfinite(phase_agreement) or phase_agreement < 0.75):
            return _reject("augmented sites lack complete checker-saddle phase evidence",
                           phase_extension={"status": "REJECTED", "added_site_count": len(additions),
                                            "reported_added_site_count": len(reported_additions)},
                           base_registration=base_registration)
        accepted_candidates = {
            tuple(map(int, record.get("grid_index_col_row_local", []))): record
            for record in augmentation.get("candidates", []) if record.get("status") == "ACCEPTED"
        }
        if set(accepted_candidates) != additions:
            return _reject("each augmented site must have a separately accepted saddle measurement",
                           phase_extension={"status": "REJECTED", "accepted_candidate_count": len(accepted_candidates),
                                            "added_site_count": len(additions)},
                           base_registration=base_registration)
        for site, record in accepted_candidates.items():
            try:
                contrast = float(record.get("checker_saddle_contrast", np.nan))
                polarity = int(record.get("expected_saddle_polarity", 0))
                diagonal_error = float(record.get("checker_diagonal_pair_error", np.nan))
                bidirectional = float(record.get("checker_bidirectional_contrast", np.nan))
                saddle_values = np.array([contrast, diagonal_error, bidirectional], dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                saddle_values = np.array([np.nan])
                polarity = 0
            if (not np.all(np.isfinite(saddle_values)) or polarity not in (-1, 1) or
                    record.get("source") not in ("goodFeaturesToTrack+cornerSubPix", "cornerSubPix+corner-response") or
                    contrast * polarity < AUGMENTED_SADDLE_LIMIT or
                    diagonal_error > max(24.0, abs(contrast) * 0.40) or
                    bidirectional < AUGMENTED_SADDLE_LIMIT or
                    int(record.get("grid_neighbor_support_count", 0)) < 1):
                return _reject("augmented site failed the independent saddle or neighbor check",
                               phase_extension={"status": "REJECTED", "site": list(site)},
                               base_registration=base_registration)

        added_pixels = np.asarray(augmentation.get("added_corner_pixels", []), dtype=np.float64).reshape(-1, 2)
        if len(added_pixels) != len(additions):
            return _reject("augmented pixel list does not align with added checker sites",
                           phase_extension={"status": "REJECTED"}, base_registration=base_registration)
        for site, pixel in zip(augmentation.get("added_grid_indices_col_row_local", []), added_pixels):
            if float(np.linalg.norm(augmented_by_site[tuple(map(int, site))] - pixel)) > 0.25:
                return _reject("augmentation metadata pixels do not match the measured site list",
                               phase_extension={"status": "REJECTED", "site": list(site)},
                               base_registration=base_registration)

        object_base = np.column_stack((base_grid[:, 0] * square_mm,
                                       base_grid[:, 1] * square_mm,
                                       np.zeros(len(base_grid), dtype=np.float64)))
        base_pnp = _metric_pnp(object_base, base_pixels, K, None)
        if base_pnp.get("status") != "PASS":
            return _reject("original current chart does not independently support metric PnP",
                           phase_extension={"status": "REJECTED", "base_pnp": base_pnp},
                           base_registration=base_registration)
        rvec = np.asarray(base_pnp["rvec_rad_local_to_camera"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(base_pnp["tvec_mm_local_origin_in_camera"], dtype=np.float64).reshape(3, 1)
        added_grid = np.asarray(sorted(additions), dtype=np.float64)
        added_objects = np.column_stack((added_grid[:, 0] * square_mm,
                                         added_grid[:, 1] * square_mm,
                                         np.zeros(len(added_grid), dtype=np.float64)))
        measured_added = np.asarray([augmented_by_site[tuple(site.astype(int))]
                                     for site in added_grid], dtype=np.float64)
        predicted_added, _ = cv2.projectPoints(added_objects, rvec, tvec, K, None)
        pnp_error = np.linalg.norm(predicted_added.reshape(-1, 2) - measured_added, axis=1)
        if np.any(pnp_error > AUGMENTED_CHART_LIMIT_PX):
            return _reject("augmented measured sites disagree with the original chart PnP phase",
                           phase_extension={"status": "REJECTED", "maximum_added_pnp_error_px": float(pnp_error.max())},
                           base_registration=base_registration)
        camera_points = cv2.Rodrigues(rvec)[0] @ added_objects.T + tvec
        measured_depth = np.asarray([_sample_depth(current_depth, pixel) for pixel in measured_added])
        depth_error = np.abs(camera_points[2] - measured_depth)
        if np.any(measured_depth <= 0) or np.any(depth_error > AUGMENTED_DEPTH_LIMIT_MM):
            return _reject("augmented measured sites fail independent aligned-depth support",
                           phase_extension={"status": "REJECTED",
                                            "maximum_added_depth_error_mm": (float(depth_error.max())
                                                                              if len(depth_error) else None)},
                           base_registration=base_registration)
        chart_h, _ = cv2.findHomography(base_grid.astype(np.float64), base_pixels, 0)
        if chart_h is None:
            return _reject("original current chart does not define a projective lattice",
                           phase_extension={"status": "REJECTED"}, base_registration=base_registration)
        chart_predicted = cv2.perspectiveTransform(added_grid.astype(np.float32).reshape(1, -1, 2),
                                                   chart_h).reshape(-1, 2)
        chart_error = np.linalg.norm(chart_predicted - measured_added, axis=1)
        if np.any(chart_error > AUGMENTED_CHART_LIMIT_PX):
            return _reject("augmented measured sites leave the original detector chart",
                           phase_extension={"status": "REJECTED", "maximum_added_chart_error_px": float(chart_error.max())},
                           base_registration=base_registration)
        all_objects = np.column_stack((augmented_grid[:, 0] * square_mm,
                                       augmented_grid[:, 1] * square_mm,
                                       np.zeros(len(augmented_grid), dtype=np.float64)))
        augmented_pnp = _metric_pnp(all_objects, augmented_pixels, K, None)
        if (augmented_pnp.get("status") != "PASS" or
                float(augmented_pnp.get("rms_px", np.inf)) > PNP_RMS_LIMIT_PX):
            return _reject("full augmented site set fails metric camera PnP",
                           phase_extension={"status": "REJECTED", "augmented_pnp": augmented_pnp},
                           base_registration=base_registration)

    basis, offset = d4
    site_ids = augmented_grid @ basis.T + offset
    if len({tuple(site) for site in site_ids}) != len(site_ids):
        return _reject("D4 phase extension produces duplicate reference site IDs",
                       phase_extension={"status": "REJECTED"}, base_registration=base_registration)
    ref_detection = _detection(reference_image, reference_depth_mm, K, reference_base_detection,
                               cols, rows, square_mm)
    ref_site_to_pixel = {
        tuple(site): pixel.tolist()
        for site, pixel in zip(ref_detection["used_grid_indices_col_row_local"],
                               ref_detection["used_corner_pixels"])
    }
    matched = []
    for site, pixel in zip(site_ids, augmented_pixels):
        site_key = tuple(int(value) for value in site)
        if site_key in ref_site_to_pixel:
            matched.append({"site_index_col_row": list(site_key),
                            "reference_pixel": ref_site_to_pixel[site_key],
                            "current_pixel": pixel.tolist()})
    result = dict(base_registration)
    result["current_site_ids_in_reference"] = site_ids.tolist()
    result["matched_intersections"] = matched
    result["evidence"] = dict(base_registration.get("evidence", {}))
    result["evidence"]["phase_extension"] = "strict D4 map from off-grid registered base chart"
    result["phase_extension"] = {
        "status": "PASS",
        "method": "base chart off-grid registration plus strict D4 extension",
        "base_corner_count": int(len(base_grid)),
        "augmented_corner_count": int(len(augmented_grid)),
        "added_corner_count": int(len(additions)),
        "current_to_reference_basis": basis.tolist(),
        "current_to_reference_offset": offset.tolist(),
        "maximum_preserved_base_pixel_change_px": preserved_error,
        "maximum_added_pnp_error_px": float(pnp_error.max()) if additions else 0.0,
        "maximum_added_depth_error_mm": float(depth_error.max()) if additions else 0.0,
        "maximum_added_chart_error_px": float(chart_error.max()) if additions else 0.0,
    }
    return result

