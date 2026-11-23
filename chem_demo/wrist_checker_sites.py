"""Recover individually measured checker sites around a right-wrist detector patch.

The regular detector supplies a local integer grid and a projective chart.  This optional
augmenter uses that chart only to seed a small image search. Every added site must then be
subpixel-refined at a real image-corner response, show the alternating two-axis checker saddle,
have aligned board depth, and agree with a metric camera PnP fit.
"""
from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from chem_demo.check_wrist_board import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    DEFAULT_SQUARE_MM,
    MIN_HANDEYE_CORNERS,
    PNP_RMS_LIMIT_PX,
    _metric_pnp,
    _stereo_check,
)


MIN_PHASE_SITES = 8
MIN_SADDLE_CONTRAST = 22.0
QUADRANT_SAMPLE_OFFSETS = (0.12, 0.14, 0.16, 0.18)
PNP_SITE_REPROJECTION_LIMIT_PX = 2.5
DEPTH_SITE_ERROR_LIMIT_MM = 12.0
CHART_SITE_RESIDUAL_LIMIT_PX = 2.5
MIN_GRID_CORNER_RESPONSE_FRACTION = 0.04


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("image must be grayscale, BGR, or BGRA")
    if gray.dtype != np.uint8 or not gray.size:
        raise ValueError("image must be a non-empty uint8 array")
    return gray


def _camera_matrix(K: np.ndarray) -> np.ndarray:
    matrix = np.asarray(K, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or
            matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or
            not np.allclose(matrix[2], [0.0, 0.0, 1.0])):
        raise ValueError("K must be a finite 3x3 pinhole camera matrix")
    return matrix


def _project_grid(homography: np.ndarray, grid: np.ndarray) -> np.ndarray:
    points = np.asarray(grid, dtype=np.float64).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(points.astype(np.float32),
                                         np.asarray(homography, dtype=np.float64))
    return projected.reshape(-1, 2).astype(np.float64)


def _checker_quadrants(gray: np.ndarray, homography: np.ndarray, site: tuple[int, int],
                       center: np.ndarray, exclude_mask: np.ndarray | None
                       ) -> tuple[float, tuple[float, float, float, float], float, float,
                                  tuple[float, ...]] | None:
    """Sample four square interiors, shrinking the aperture near a jaw or image border."""
    base = np.asarray(site, dtype=np.float64)
    origin = _project_grid(homography, base.reshape(1, 2))[0]
    shift = np.asarray(center, dtype=np.float64) - origin
    sample_levels = [np.asarray(QUADRANT_SAMPLE_OFFSETS, dtype=np.float64),
                     np.array([0.08, 0.10, 0.12, 0.14], dtype=np.float64),
                     np.array([0.08, 0.10, 0.12], dtype=np.float64)]
    h, w = gray.shape
    used_offsets = None
    values = None
    tried_levels: set[tuple[float, ...]] = set()
    for offsets in sample_levels:
        key = tuple(float(value) for value in offsets)
        if not len(offsets) or key in tried_levels:
            continue
        tried_levels.add(key)
        samples: list[tuple[float, float]] = []
        for row_sign in (-1, 1):
            for col_sign in (-1, 1):
                for dx in offsets:
                    for dy in offsets:
                        grid_point = base + [col_sign * dx, row_sign * dy]
                        pixel = _project_grid(homography, grid_point.reshape(1, 2))[0] + shift
                        samples.append((float(pixel[0]), float(pixel[1])))
        xy = np.asarray(samples, dtype=np.float32)
        rounded = np.rint(xy).astype(int)
        if (np.any(rounded[:, 0] < 1) or np.any(rounded[:, 0] >= w - 1) or
                np.any(rounded[:, 1] < 1) or np.any(rounded[:, 1] >= h - 1)):
            continue
        if exclude_mask is not None and np.any(exclude_mask[rounded[:, 1], rounded[:, 0]] != 0):
            continue
        map_x = xy[:, 0].reshape(1, -1)
        map_y = xy[:, 1].reshape(1, -1)
        values = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1)
        used_offsets = offsets
        break
    if values is None or used_offsets is None:
        return None
    count = len(used_offsets) ** 2
    quadrants = [values[index * count:(index + 1) * count].astype(float)
                 for index in range(4)]  # NW, NE, SW, SE
    means = tuple(float(np.mean(group)) for group in quadrants)
    # Diagonally adjacent squares have the same color.  This saddle polarity flips at each
    # horizontal or vertical lattice step, unlike a one-axis edge or a jaw silhouette.
    contrast = ((means[0] + means[3]) - (means[1] + means[2])) / 2.0
    diagonal_pair_error = max(abs(means[0] - means[3]), abs(means[1] - means[2]))
    bidirectional_contrast = min(abs(means[0] - means[1]), abs(means[2] - means[3]),
                                 abs(means[0] - means[2]), abs(means[1] - means[3]))
    return (float(contrast), means, float(diagonal_pair_error),
            float(bidirectional_contrast), tuple(float(value) for value in used_offsets))


def _depth_at(depth: np.ndarray, pixel: np.ndarray, radius: int = 2) -> float:
    h, w = depth.shape
    u, v = np.rint(pixel).astype(int)
    if not (radius <= u < w - radius and radius <= v < h - radius):
        return 0.0
    patch = depth[v - radius:v + radius + 1, u - radius:u + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size < max(5, patch.size // 3):
        return 0.0
    median = float(np.median(valid))
    if float(valid.max() - valid.min()) > max(20.0, median * 0.025):
        return 0.0
    return median


def _find_image_corner(gray: np.ndarray, predicted: np.ndarray, spacing_px: float,
                       exclude_mask: np.ndarray | None
                       ) -> tuple[np.ndarray, float, float, str] | None:
    """Find and subpixel-refine a real corner near a chart prediction."""
    h, w = gray.shape
    radius = max(5.0, min(0.35 * spacing_px, 24.0))
    # Keep enough room for cornerSubPix itself.  The stricter checker-quadrant sampler below
    # separately rejects any saddle whose actual square samples reach the image border.
    if (predicted[0] < 6 or predicted[0] >= w - 6 or
            predicted[1] < 6 or predicted[1] >= h - 6):
        return None
    x0, y0 = np.floor(predicted - radius).astype(int)
    x1, y1 = np.ceil(predicted + radius + 1).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    roi = gray[y0:y1, x0:x1]
    roi_mask = None
    if exclude_mask is not None:
        roi_mask = np.where(exclude_mask[y0:y1, x0:x1] == 0, 255, 0).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(
        roi, maxCorners=16, qualityLevel=0.006,
        minDistance=max(3.0, 0.08 * spacing_px), mask=roi_mask,
        blockSize=5, useHarrisDetector=False)
    response_map = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
    choices = []
    if corners is not None and len(corners):
        corners[:, 0, 0] += x0
        corners[:, 0, 1] += y0
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
        refined = corners.reshape(-1, 2).astype(np.float64)
        distances = np.linalg.norm(refined - predicted, axis=1)
        within = np.flatnonzero(distances <= radius)
        for index in within:
            pixel = refined[index]
            px, py = np.rint(pixel).astype(int)
            if not (3 <= px < w - 3 and 3 <= py < h - 3):
                continue
            if (exclude_mask is not None and
                    np.any(exclude_mask[py - 2:py + 3, px - 2:px + 3] != 0)):
                continue
            response = float(response_map[py, px])
            choices.append((distances[index], -response, pixel, response,
                            "goodFeaturesToTrack+cornerSubPix"))
    # GFTT can suppress a clear saddle beside a stronger pad edge in the same ROI.  Refining the
    # chart seed directly is still a measurement only when the resulting pixel has real corner
    # response and passes the independent checker photometry, depth, and PnP checks downstream.
    if not choices:
        corner = np.asarray(predicted, dtype=np.float32).reshape(1, 1, 2)
        cv2.cornerSubPix(gray, corner, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
        pixel = corner.reshape(2).astype(np.float64)
        px, py = np.rint(pixel).astype(int)
        distance = float(np.linalg.norm(pixel - predicted))
        support_unmasked = (exclude_mask is None or not np.any(
            exclude_mask[py - 2:py + 3, px - 2:px + 3] != 0))
        if (distance <= radius and 3 <= px < w - 3 and 3 <= py < h - 3 and
                support_unmasked):
            response = float(response_map[py, px])
            choices.append((distance, -response, pixel, response,
                            "cornerSubPix+corner-response"))
    if not choices:
        return None
    distance, _, pixel, response, source = min(choices, key=lambda row: (row[0], row[1]))
    return pixel, float(distance), response, source


def _pnp_projection(pnp: Mapping[str, Any], object_points: np.ndarray,
                    K: np.ndarray, dist: np.ndarray | None
                    ) -> tuple[np.ndarray, np.ndarray] | None:
    if pnp.get("status") != "PASS":
        return None
    try:
        rvec = np.asarray(pnp["rvec_rad_local_to_camera"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(pnp["tvec_mm_local_origin_in_camera"], dtype=np.float64).reshape(3, 1)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        camera_points = cv2.Rodrigues(rvec)[0] @ object_points.T + tvec
    except (KeyError, ValueError, cv2.error):
        return None
    return projected.reshape(-1, 2), camera_points[2].reshape(-1)


def _candidate_sites(grid: np.ndarray, cols: int, rows: int,
                     rejected_detector_sites: np.ndarray | None = None) -> list[tuple[int, int]]:
    sites = {tuple(map(int, site)) for site in np.rint(grid).astype(int)}
    min_col = min(site[0] for site in sites)
    max_col = max(site[0] for site in sites)
    min_row = min(site[1] for site in sites)
    max_row = max(site[1] for site in sites)
    # The detector patch has no absolute board origin. Search the unobserved margin that could
    # still fit within the printed board's full internal-corner dimensions; PnP and local image
    # evidence must validate each real corner before it can extend the measured patch.
    extra_cols = max(0, cols - (max_col - min_col + 1))
    extra_rows = max(0, rows - (max_row - min_row + 1))
    candidates = {
        (col, row)
        for row in range(min_row - extra_rows, max_row + extra_rows + 1)
        for col in range(min_col - extra_cols, max_col + extra_cols + 1)
        if (col, row) not in sites and
        (col < min_col or col > max_col or row < min_row or row > max_row)
    }
    if rejected_detector_sites is not None:
        rejected = np.asarray(rejected_detector_sites, dtype=np.float64).reshape(-1, 2)
        if len(rejected) and np.allclose(rejected, np.rint(rejected), atol=1e-6):
            candidates.update(tuple(map(int, site)) for site in np.rint(rejected).astype(int)
                              if tuple(map(int, site)) not in sites)
    return sorted(candidates, key=lambda site: (site[1], site[0]))


def _site_support(site: tuple[int, int], known_sites: set[tuple[int, int]]) -> int:
    col, row = site
    direct = sum(neighbor in known_sites for neighbor in
                 ((col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1),
                  (col - 1, row - 1), (col - 1, row + 1),
                  (col + 1, row - 1), (col + 1, row + 1)))
    if direct:
        return direct
    # One occluded lattice row may separate a visible measured corner from the rectangular
    # detector patch. Such a point still has nearby site support, and is independently checked
    # against the chart phase, aligned depth, and actual camera PnP below.
    return int(any(abs(col - known_col) + abs(row - known_row) <= 2
                   for known_col, known_row in known_sites))


def _recompute_report(result: dict[str, Any], pixels: np.ndarray, grid: np.ndarray,
                      K: np.ndarray, dist: np.ndarray | None,
                      depth: np.ndarray, square_mm: float,
                      cols: int, rows: int,
                      base_detection: Mapping[str, Any]) -> None:
    object_points = np.column_stack((grid[:, 0] * square_mm,
                                     grid[:, 1] * square_mm,
                                     np.zeros(len(grid), dtype=np.float64)))
    pnp = _metric_pnp(object_points, pixels, K, dist)
    pnp["total_detected_corner_count"] = int(len(pixels))
    pnp["used_corner_count"] = int(len(pixels))

    grid_h, _ = cv2.findHomography(grid.astype(np.float64), pixels.astype(np.float64), 0)
    if grid_h is not None:
        projected = _project_grid(grid_h, grid)
        residuals = np.linalg.norm(projected - pixels, axis=1)
    else:
        residuals = np.full(len(grid), np.inf, dtype=np.float64)

    old_board = dict(result.get("board", {}))
    min_col, min_row = np.min(grid, axis=0).astype(int)
    max_col, max_row = np.max(grid, axis=0).astype(int)
    dimensions = [int(max_col - min_col + 1), int(max_row - min_row + 1)]
    unique_sites = {tuple(map(int, site)) for site in grid}
    full_rectangular_sites = {
        (col, row) for row in range(min_row, max_row + 1)
        for col in range(min_col, max_col + 1)
    }
    full_board_measured = (dimensions == [cols, rows] and
                           len(unique_sites) == cols * rows and
                           unique_sites == full_rectangular_sites)
    old_board.update({
        "detected_grid_shape": dimensions,
        "local_span_mm": [float((dimensions[0] - 1) * square_mm),
                          float((dimensions[1] - 1) * square_mm)],
        "full_board_detected": bool(full_board_measured),
    })
    result["board"] = old_board
    result.update({
        # Keep the detector's raw arrays, chart filter and rejection mask intact. Augmented
        # measurements are explicit in the `augmented_*` report and in the used_* arrays.
        "measured_corner_count": int(len(pixels)),
        "used_corner_count": int(len(pixels)),
        "flat_m_total": int(base_detection.get("flat_m_total", base_detection.get("corner_count", len(pixels)))),
        "flat_m_used": int(len(pixels)),
        "chart_support_count": int(len(pixels)),
        "used_grid_indices_col_row_local": grid.astype(int).tolist(),
        "used_corner_pixels": pixels.tolist(),
        "used_object_points_local_mm": object_points.tolist(),
        "metric_pnp": pnp,
        "augmented_chart_filter": {
            "method": "detector chart plus individually measured checker saddles",
            "threshold_px": CHART_SITE_RESIDUAL_LIMIT_PX,
            "status": "PASS" if len(pixels) >= 4 and
                     float(np.percentile(residuals, 90)) <= CHART_SITE_RESIDUAL_LIMIT_PX else "FAIL",
            "inlier_count": int(len(pixels)),
            "outlier_count": 0,
            "median_inlier_residual_px": float(np.median(residuals)),
            "p90_inlier_residual_px": float(np.percentile(residuals, 90)),
        },
        "stereo_check": _stereo_check(depth, pixels, pnp.get("predicted_corner_z_mm"), window=2),
    })
    hull = cv2.convexHull(pixels.astype(np.float32).reshape(-1, 1, 2))
    h, w = depth.shape
    hull_area = float(cv2.contourArea(hull))
    x_span = float(np.ptp(pixels[:, 0]))
    y_span = float(np.ptp(pixels[:, 1]))
    result["coverage"] = {
        "image_hull_fraction": float(hull_area / (w * h)),
        "image_bbox_fraction": float(x_span * y_span / (w * h)),
        "expected_corner_fraction": float(len(pixels) / (cols * rows)),
        "used_image_hull_fraction": float(hull_area / (w * h)),
        "bbox_px": [float(pixels[:, 0].min()), float(pixels[:, 1].min()),
                    float(pixels[:, 0].max()), float(pixels[:, 1].max())],
    }
    supported = len(pixels) >= MIN_HANDEYE_CORNERS and pnp.get("status") == "PASS"
    result["handeye_support"] = {
        "total_detected": int(len(pixels)), "used": int(len(pixels)),
        "minimum": MIN_HANDEYE_CORNERS, "passed": bool(supported),
        "status": "PASS" if supported else "INSUFFICIENT",
    }


def augment_right_capture(
        image: np.ndarray,
        depth_mm: np.ndarray,
        K: np.ndarray,
        detection: Mapping[str, Any],
        *,
        exclude_mask: np.ndarray | None = None,
        cols: int = DEFAULT_COLS,
        rows: int = DEFAULT_ROWS,
        square_mm: float = DEFAULT_SQUARE_MM,
        dist: np.ndarray | None = None,
) -> dict[str, Any]:
    """Optionally add measured checker corners around a detected rectangular patch.

    Local grid sites are relative to the supplied detection.  Newly returned pixels always come
    from subpixel refinement of a real corner response; the homography predicts search locations
    only. This also remeasures raw detector sites rejected by its original chart filter, without
    changing those original rejection flags. A site is retained only with alternating checker
    saddle photometry, local depth, nearby lattice support, and an actual-camera PnP fit.
    """
    if not isinstance(cols, int) or not isinstance(rows, int) or cols < 3 or rows < 3:
        raise ValueError("cols and rows must each be integer counts of at least three")
    if not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("square_mm must be positive and finite")
    gray = _gray(image)
    depth = np.asarray(depth_mm, dtype=np.float64)
    if depth.ndim != 2 or depth.shape != gray.shape:
        raise ValueError("depth_mm must be a single-channel image aligned with image")
    if not np.all(np.isfinite(depth)):
        raise ValueError("depth_mm must contain finite values")
    camera = _camera_matrix(K)
    distortion = None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
    if distortion is not None and not np.all(np.isfinite(distortion)):
        raise ValueError("distortion coefficients must be finite")
    ignored = None if exclude_mask is None else np.asarray(exclude_mask)
    if ignored is not None and ignored.shape != gray.shape:
        raise ValueError("exclude_mask must match image dimensions")

    result = dict(detection)
    result["corner_augmentation"] = {
        "status": "NO_ADDITIONS", "method": "measured subpixel checker saddle search",
        "base_corner_count": 0, "added_corner_count": 0,
        "candidate_count": 0, "phase_support_count": 0,
        "reason": "augmentation did not find a supported additional measured corner",
        "candidates": [],
    }
    pixels = np.asarray(detection.get("used_corner_pixels", []), dtype=np.float64).reshape(-1, 2)
    grid = np.asarray(detection.get("used_grid_indices_col_row_local", []), dtype=np.float64).reshape(-1, 2)
    if (len(pixels) < 6 or len(grid) != len(pixels) or not np.all(np.isfinite(pixels)) or
            not np.allclose(grid, np.rint(grid), atol=1e-6)):
        result["corner_augmentation"].update({
            "status": "SKIPPED", "reason": "base detection needs six aligned measured grid corners"})
        return result
    grid = np.rint(grid).astype(np.int32)
    if len({tuple(site) for site in grid}) != len(grid):
        result["corner_augmentation"].update({
            "status": "SKIPPED", "reason": "base detection contains duplicate local grid sites"})
        return result

    base_h, _ = cv2.findHomography(grid.astype(np.float64), pixels, 0)
    if base_h is None or not np.all(np.isfinite(base_h)):
        result["corner_augmentation"].update({
            "status": "SKIPPED", "reason": "base detection does not define a local homography"})
        return result
    base_chart_pixels = _project_grid(base_h, grid)
    base_chart_residuals = np.linalg.norm(base_chart_pixels - pixels, axis=1)
    if float(np.percentile(base_chart_residuals, 90)) > max(2.0, CHART_SITE_RESIDUAL_LIMIT_PX):
        result["corner_augmentation"].update({
            "status": "SKIPPED", "reason": "base corners do not support a regular local homography"})
        return result

    base_object_points = np.column_stack((grid[:, 0] * square_mm, grid[:, 1] * square_mm,
                                          np.zeros(len(grid), dtype=np.float64)))
    base_pnp = _metric_pnp(base_object_points, pixels, camera, distortion)
    base_pose = _pnp_projection(base_pnp, base_object_points, camera, distortion)
    if base_pose is None:
        result["corner_augmentation"].update({
            "status": "SKIPPED", "reason": "base measured corners do not support a passing metric camera PnP"})
        return result

    base_spacing: list[float] = []
    by_site = {tuple(map(int, site)): pixel for site, pixel in zip(grid, pixels)}
    for (col, row), pixel in by_site.items():
        for neighbor in ((col + 1, row), (col, row + 1)):
            if neighbor in by_site:
                base_spacing.append(float(np.linalg.norm(pixel - by_site[neighbor])))
    if not base_spacing:
        result["corner_augmentation"].update({
            "status": "SKIPPED", "reason": "base checker lattice has no adjacent measured sites"})
        return result
    median_spacing = float(np.median(base_spacing))

    # Estimate which diagonal should be bright/dark at each parity from actual measured sites.
    phase_evidence: list[float] = []
    phase_site_count = 0
    for site, pixel in zip(grid, pixels):
        reading = _checker_quadrants(gray, base_h, tuple(map(int, site)), pixel, ignored)
        if reading is None:
            continue
        contrast, _, diagonal_error, bidirectional, _ = reading
        if (abs(contrast) >= MIN_SADDLE_CONTRAST and
                diagonal_error <= max(24.0, abs(contrast) * 0.40) and
                bidirectional >= MIN_SADDLE_CONTRAST):
            phase_evidence.append(contrast * (-1.0 if int(site[0] + site[1]) & 1 else 1.0))
            phase_site_count += 1
    if phase_site_count < MIN_PHASE_SITES:
        result["corner_augmentation"].update({
            "status": "SKIPPED", "phase_support_count": int(phase_site_count),
            "reason": "too few measured corners show a two-axis checker saddle to establish chart phase"})
        return result
    phase = 1 if float(np.median(phase_evidence)) >= 0 else -1
    phase_agreement = float(np.mean(np.asarray(phase_evidence) * phase >= MIN_SADDLE_CONTRAST))
    if phase_agreement < 0.75:
        result["corner_augmentation"].update({
            "status": "SKIPPED", "phase_support_count": int(phase_site_count),
            "reason": "base checker saddles do not agree on alternating chart polarity"})
        return result

    # Baseline corner strength supplies a relative floor for real image-corner detections.
    response_map = cv2.cornerMinEigenVal(gray, blockSize=5, ksize=3)
    base_responses = []
    for pixel in pixels:
        x, y = np.rint(pixel).astype(int)
        if 2 <= x < gray.shape[1] - 2 and 2 <= y < gray.shape[0] - 2:
            base_responses.append(float(response_map[y, x]))
    response_floor = max(1e-8, (float(np.median(base_responses)) if base_responses else 0.0) *
                         MIN_GRID_CORNER_RESPONSE_FRACTION)

    raw_grid = np.asarray(detection.get("grid_indices_col_row_local", grid),
                          dtype=np.float64).reshape(-1, 2)
    raw_mask = np.asarray(detection.get("chart_inlier_mask", []), dtype=bool).reshape(-1)
    if len(raw_grid) == len(raw_mask):
        rejected_sites = raw_grid[~raw_mask]
    else:
        used_site_set = {tuple(map(int, site)) for site in grid}
        rejected_sites = np.asarray([site for site in raw_grid
                                     if tuple(map(int, np.rint(site))) not in used_site_set],
                                    dtype=np.float64).reshape(-1, 2)
    candidate_sites = _candidate_sites(grid, cols, rows, rejected_sites)
    candidate_records: list[dict[str, Any]] = []
    base_projected, base_z = base_pose
    pnp_by_site = {tuple(map(int, site)): (projected, float(z))
                   for site, projected, z in zip(grid, base_projected, base_z)}
    for site in candidate_sites:
        expected_pixel = _project_grid(base_h, np.asarray(site, dtype=np.float64).reshape(1, 2))[0]
        neighbor_pixels = []
        for neighbor in ((site[0] - 1, site[1]), (site[0] + 1, site[1]),
                         (site[0], site[1] - 1), (site[0], site[1] + 1)):
            if neighbor in by_site:
                neighbor_pixels.append(by_site[neighbor])
        if neighbor_pixels:
            spacing = float(np.median([np.linalg.norm(point - expected_pixel)
                                       for point in neighbor_pixels]))
        else:
            spacing = median_spacing
        found = _find_image_corner(gray, expected_pixel, spacing, ignored)
        if found is None:
            candidate_records.append({"grid_index_col_row_local": list(site),
                                      "status": "REJECTED", "reason": "no real subpixel corner near chart seed"})
            continue
        measured_pixel, seed_distance, response, source = found
        candidate: dict[str, Any] = {
            "grid_index_col_row_local": list(site), "pixel": measured_pixel.tolist(),
            "source": source,
            "chart_seed_pixel": expected_pixel.tolist(),
            "seed_offset_px": seed_distance, "corner_response": response,
        }
        if response < response_floor:
            candidate.update({"status": "REJECTED", "reason": "image-corner response is too weak"})
            candidate_records.append(candidate)
            continue
        reading = _checker_quadrants(gray, base_h, site, measured_pixel, ignored)
        if reading is None:
            candidate.update({"status": "REJECTED", "reason": "checker quadrants touch a border or excluded region"})
            candidate_records.append(candidate)
            continue
        contrast, means, diagonal_error, bidirectional, sample_offsets = reading
        expected_polarity = phase * (-1 if (site[0] + site[1]) & 1 else 1)
        signed_contrast = contrast * expected_polarity
        candidate.update({"checker_quadrant_means_nw_ne_sw_se": list(means),
                          "checker_saddle_contrast": contrast,
                          "checker_diagonal_pair_error": diagonal_error,
                          "checker_bidirectional_contrast": bidirectional,
                          "checker_sample_offsets": list(sample_offsets),
                          "expected_saddle_polarity": expected_polarity})
        if (signed_contrast < MIN_SADDLE_CONTRAST or
                diagonal_error > max(24.0, abs(contrast) * 0.40) or
                bidirectional < MIN_SADDLE_CONTRAST):
            candidate.update({"status": "REJECTED",
                              "reason": "checker saddle fails diagonal-pair consistency or bidirectional contrast"})
            candidate_records.append(candidate)
            continue

        measured_depth = _depth_at(depth, measured_pixel)
        object_point = np.array([[site[0] * square_mm, site[1] * square_mm, 0.0]], dtype=np.float64)
        candidate_pose = _pnp_projection(base_pnp, object_point, camera, distortion)
        if measured_depth <= 0 or candidate_pose is None:
            candidate.update({"status": "REJECTED", "reason": "corner has no reliable aligned board depth"})
            candidate_records.append(candidate)
            continue
        pnp_pixel, expected_z = candidate_pose[0][0], float(candidate_pose[1][0])
        pnp_error = float(np.linalg.norm(measured_pixel - pnp_pixel))
        depth_error = abs(measured_depth - expected_z)
        candidate.update({"aligned_depth_mm": measured_depth,
                          "pnp_predicted_pixel": pnp_pixel.tolist(),
                          "pnp_pixel_error_px": pnp_error,
                          "pnp_predicted_z_mm": expected_z,
                          "depth_error_mm": depth_error})
        if pnp_error > PNP_SITE_REPROJECTION_LIMIT_PX:
            candidate.update({"status": "REJECTED", "reason": "measured corner disagrees with camera PnP projection"})
            candidate_records.append(candidate)
            continue
        if depth_error > DEPTH_SITE_ERROR_LIMIT_MM:
            candidate.update({"status": "REJECTED", "reason": "aligned depth is inconsistent with checker-plane PnP"})
            candidate_records.append(candidate)
            continue

        min_distance = min(float(np.linalg.norm(measured_pixel - old)) for old in pixels)
        if min_distance < 0.45 * min(median_spacing, spacing):
            candidate.update({"status": "REJECTED", "reason": "measured corner duplicates an existing site"})
            candidate_records.append(candidate)
            continue
        chart_offset = float(np.linalg.norm(measured_pixel - expected_pixel))
        if chart_offset > CHART_SITE_RESIDUAL_LIMIT_PX:
            candidate.update({"status": "REJECTED", "reason": "measured corner is too far from the regular local chart"})
            candidate_records.append(candidate)
            continue

        contrast_score = min(1.0, signed_contrast / 100.0)
        pnp_score = max(0.0, 1.0 - pnp_error / PNP_SITE_REPROJECTION_LIMIT_PX)
        seed_score = max(0.0, 1.0 - seed_distance / max(1.0, 0.35 * spacing))
        corner_score = min(1.0, response / max(response_floor * 10.0, 1e-8))
        candidate["rank_score"] = float(0.50 * contrast_score + 0.25 * pnp_score +
                                        0.15 * seed_score + 0.10 * corner_score)
        candidate["status"] = "MEASURED_CANDIDATE"
        candidate_records.append(candidate)

    candidates_by_site = {
        tuple(record["grid_index_col_row_local"]): record
        for record in candidate_records if record.get("status") == "MEASURED_CANDIDATE"
    }
    ranked = sorted(candidates_by_site.values(), key=lambda record: record["rank_score"], reverse=True)
    for rank, record in enumerate(ranked, 1):
        record["rank"] = rank
    accepted_pixels: list[np.ndarray] = []
    accepted_sites: list[tuple[int, int]] = []
    accepted_records: list[dict[str, Any]] = []
    current_grid = grid.copy()
    current_pixels = pixels.copy()
    known_sites = {tuple(map(int, site)) for site in grid}

    # Defer diagonal shell corners until one of their ring neighbors is accepted.  Every accepted
    # site therefore joins a measured checker lattice instead of forming an isolated prediction.
    pending = list(ranked)
    while pending:
        progressed = False
        for record in list(pending):
            site = tuple(record["grid_index_col_row_local"])
            support = _site_support(site, known_sites)
            record["grid_neighbor_support_count"] = int(support)
            if support < 1:
                continue
            measured_pixel = np.asarray(record["pixel"], dtype=np.float64)
            if any(np.linalg.norm(measured_pixel - old) < 0.45 * median_spacing
                   for old in np.vstack((pixels, np.asarray(accepted_pixels).reshape(-1, 2)))):
                record.update({"status": "REJECTED", "reason": "measured corner duplicates a supported site"})
                pending.remove(record)
                continue
            proposed_grid = np.vstack((current_grid, np.asarray(site, dtype=np.int32)))
            proposed_pixels = np.vstack((current_pixels, measured_pixel))
            proposed_span = np.ptp(proposed_grid, axis=0) + 1
            if proposed_span[0] > cols or proposed_span[1] > rows:
                record.update({"status": "REJECTED", "reason": "candidate would exceed the full checkerboard dimensions"})
                pending.remove(record)
                continue
            proposed_objects = np.column_stack((proposed_grid[:, 0] * square_mm,
                                                proposed_grid[:, 1] * square_mm,
                                                np.zeros(len(proposed_grid), dtype=np.float64)))
            proposed_pnp = _metric_pnp(proposed_objects, proposed_pixels, camera, distortion)
            proposed_projection = _pnp_projection(proposed_pnp, proposed_objects, camera, distortion)
            if proposed_projection is None or proposed_pnp.get("rms_px", np.inf) > PNP_RMS_LIMIT_PX:
                record.update({"status": "REJECTED", "reason": "augmented measured sites fail metric camera PnP"})
                pending.remove(record)
                continue
            reprojection_errors = np.linalg.norm(proposed_projection[0] - proposed_pixels, axis=1)
            if (reprojection_errors[-1] > PNP_SITE_REPROJECTION_LIMIT_PX or
                    np.any(reprojection_errors[len(pixels):] > PNP_SITE_REPROJECTION_LIMIT_PX)):
                record.update({"status": "REJECTED", "reason": "augmented site set has a camera PnP outlier"})
                pending.remove(record)
                continue
            if any(np.linalg.norm(measured_pixel - old) < 0.45 * median_spacing
                   for old in accepted_pixels):
                record.update({"status": "REJECTED", "reason": "measured corner duplicates an accepted site"})
                pending.remove(record)
                continue

            record.update({"status": "ACCEPTED", "pnp_rms_px_after_addition": float(proposed_pnp["rms_px"])})
            accepted_records.append(record)
            accepted_sites.append(site)
            accepted_pixels.append(measured_pixel)
            known_sites.add(site)
            current_grid, current_pixels = proposed_grid, proposed_pixels
            pending.remove(record)
            progressed = True
        if not progressed:
            break

    for record in pending:
        if record.get("status") == "MEASURED_CANDIDATE":
            record.update({"status": "REJECTED", "reason": "no measured lattice-neighbor support"})
    for rank, record in enumerate(sorted(accepted_records, key=lambda item: item["rank_score"], reverse=True), 1):
        record["accepted_rank"] = rank

    if accepted_pixels:
        added_pixels = np.asarray(accepted_pixels, dtype=np.float64).reshape(-1, 2)
        added_grid = np.asarray(accepted_sites, dtype=np.int32).reshape(-1, 2)
        combined_grid = np.vstack((grid, added_grid))
        combined_pixels = np.vstack((pixels, added_pixels))
        # Retain detector row-major ordering while allowing signed local coordinates outside its
        # original patch; this also keeps each pixel aligned with its integer lattice coordinate.
        order = np.lexsort((combined_grid[:, 0], combined_grid[:, 1]))
        combined_grid, combined_pixels = combined_grid[order], combined_pixels[order]
        _recompute_report(result, combined_pixels, combined_grid, camera, distortion,
                          depth, square_mm, cols, rows, detection)
        result["corner_augmentation"] = {
            "status": "PASS", "method": "subpixel image corner, alternating checker saddle, depth, and metric PnP",
            "base_corner_count": int(len(pixels)), "added_corner_count": int(len(added_pixels)),
            "candidate_count": int(len(candidate_sites)), "phase_support_count": int(phase_site_count),
            "phase_agreement": phase_agreement, "checker_phase": int(phase),
            "candidates": candidate_records,
            "added_grid_indices_col_row_local": added_grid.tolist(),
            "added_corner_pixels": added_pixels.tolist(),
            "augmented_grid_indices_col_row_local": combined_grid.tolist(),
            "augmented_corner_pixels": combined_pixels.tolist(),
            "augmented_object_points_local_mm": result["used_object_points_local_mm"],
            "ranked_added_sites": [record["grid_index_col_row_local"] for record in
                                   sorted(accepted_records, key=lambda item: item["rank_score"], reverse=True)],
            "reason": None,
        }
    else:
        result["corner_augmentation"].update({
            "candidate_count": int(len(candidate_sites)),
            "phase_support_count": int(phase_site_count), "phase_agreement": phase_agreement,
            "checker_phase": int(phase), "candidates": candidate_records,
            "reason": "no candidate passed measured-corner, saddle, depth, lattice, and camera-PnP checks",
        })
    return result


__all__ = ["augment_right_capture"]
