"""Offline diagnostic for a plain checkerboard seen by the wrist camera.

This tool does not open devices or write an active robot calibration. It detects the largest
visible rectangular checker grid (including partial views), reports its image coverage and, when
camera intrinsics are supplied, estimates a *local* metric board pose. A plain checkerboard has
no IDs: even a full view is ambiguous under a 180-degree flip. Local corner coordinates are never
claimed to be absolute physical board IDs.

Example (11 x 8 inner corners, 30 mm squares):
    python -m chem_demo.check_wrist_board snapshot.png --metadata acquisition.json --depth depth.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


DEFAULT_COLS = 11
DEFAULT_ROWS = 8
DEFAULT_SQUARE_MM = 30.0
PNP_RMS_LIMIT_PX = 1.5
STEREO_MEDIAN_LIMIT_MM = 3.0
STEREO_P90_LIMIT_MM = 6.0
CHART_RANSAC_THRESHOLD_PX = 0.9
MIN_HANDEYE_CORNERS = 30


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("image must be grayscale, BGR, or BGRA")


def _candidate_sizes(cols: int, rows: int) -> list[tuple[int, int]]:
    # Largest areas first. Trying smaller patterns is what allows useful reports when the camera
    # sees only part of a larger, untagged board.
    return sorted(((w, h) for w in range(3, cols + 1) for h in range(3, rows + 1)),
                  key=lambda size: (size[0] * size[1], min(size), size[0]), reverse=True)


def _detect_grids(gray: np.ndarray, cols: int, rows: int, *, search_all: bool = False
                  ) -> list[tuple[int, int, np.ndarray, str]]:
    flags = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY |
             cv2.CALIB_CB_NORMALIZE_IMAGE)
    candidates: list[tuple[int, int, np.ndarray, str]] = []
    # SBWithMeta + LARGER is a useful second path for a large board clipped by the camera frame.
    # A 8x6 seed detected 48 local intersections in the live 848x480 view even though the complete
    # 11x8 pattern was not visible. The returned meta dimensions describe the detected local grid.
    larger = getattr(cv2, "CALIB_CB_LARGER", 0)
    finder = getattr(cv2, "findChessboardCornersSBWithMeta", None)
    larger_count = 0
    if larger and finder is not None:
        seed = (min(cols, 8), min(rows, 6))
        try:
            found, corners, meta = finder(gray, seed, flags=flags | larger)
            if found and corners is not None and meta is not None:
                height, width = meta.shape[:2]
                pixels = np.asarray(corners, np.float32).reshape(-1, 2)
                if 3 <= width <= cols and 3 <= height <= rows and len(pixels) == width * height:
                    candidates.append((width, height, pixels, "findChessboardCornersSBWithMeta_LARGER"))
                    larger_count = len(pixels)
        except cv2.error:
            # Keep the plain SB path available on OpenCV builds with a different Meta overload.
            pass

    # Compare exact-grid candidates with at least the support returned by LARGER. This catches a
    # neighbouring grid interpretation without paying for every smaller sub-pattern on every
    # snapshot; when LARGER cannot run or finds nothing, fall back until the largest exact grid.
    found_area = 0
    for width, height in _candidate_sizes(cols, rows):
        area = width * height
        if larger_count and not search_all and area < larger_count:
            continue
        found, corners = cv2.findChessboardCornersSB(gray, (width, height), flags=flags)
        if found and corners is not None and len(corners) == area:
            candidates.append((width, height, np.asarray(corners, np.float32).reshape(-1, 2),
                               "findChessboardCornersSB"))
            found_area = max(found_area, area)
        if not search_all and not larger_count and found_area and area < found_area:
            break

    unique: list[tuple[int, int, np.ndarray, str]] = []
    for candidate in candidates:
        width, height, pixels, _ = candidate
        if any(width == old_w and height == old_h and len(pixels) == len(old_pixels) and
               np.allclose(pixels, old_pixels, atol=1.5) for old_w, old_h, old_pixels, _ in unique):
            continue
        unique.append(candidate)
    return unique


def _chart_inliers(grid_indices: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reject detections inconsistent with one planar projective checkerboard chart."""
    grid = np.asarray(grid_indices, dtype=np.float64).reshape(-1, 2)
    image_points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    mask = np.zeros(len(grid), dtype=bool)
    residuals = np.full(len(grid), np.inf, dtype=np.float64)
    if len(grid) < 4:
        return mask, residuals
    cv2.setRNGSeed(0)
    homography, raw_mask = cv2.findHomography(
        grid, image_points, cv2.RANSAC, CHART_RANSAC_THRESHOLD_PX,
        maxIters=3000, confidence=0.999)
    if homography is None or raw_mask is None:
        return mask, residuals
    mask = raw_mask.reshape(-1).astype(bool)
    if mask.sum() < 4:
        return np.zeros(len(grid), dtype=bool), residuals
    refined, _ = cv2.findHomography(grid[mask], image_points[mask], 0)
    if refined is not None:
        homography = refined
    projected = cv2.perspectiveTransform(grid.astype(np.float32).reshape(1, -1, 2),
                                         homography).reshape(-1, 2)
    residuals = np.linalg.norm(image_points - projected, axis=1)
    mask &= residuals <= CHART_RANSAC_THRESHOLD_PX
    return mask, residuals


def _metric_pnp(object_points: np.ndarray, pixels: np.ndarray, K: np.ndarray | None,
                distortion: np.ndarray | None) -> dict[str, Any]:
    if K is None:
        return {"status": "NOT_RUN", "reason": "camera intrinsics were not supplied"}
    if len(pixels) < 6:
        return {"status": "INSUFFICIENT", "reason": "at least six corners are required for PnP"}
    D = np.zeros(5, dtype=np.float64) if distortion is None else np.asarray(distortion, np.float64).reshape(-1)
    try:
        result = cv2.solvePnPGeneric(object_points.astype(np.float64), pixels.astype(np.float64),
                                     np.asarray(K, np.float64), D,
                                     flags=cv2.SOLVEPNP_IPPE)
        if not result[0]:
            return {"status": "FAIL", "reason": "planar PnP returned no solution"}
        candidates = []
        for rvec, tvec in zip(result[1], result[2]):
            camera_points = cv2.Rodrigues(rvec)[0] @ object_points.T + np.asarray(tvec).reshape(3, 1)
            if np.any(camera_points[2] <= 0):
                continue
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
            residual = projected.reshape(-1, 2) - pixels
            rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            candidates.append((rms, np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3),
                               camera_points.T))
        if not candidates:
            return {"status": "FAIL", "reason": "PnP solutions put board points behind the camera"}
        rms, rvec, tvec, camera_points = min(candidates, key=lambda candidate: candidate[0])
        try:
            refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
                object_points.astype(np.float64), pixels.astype(np.float64), K, D,
                rvec.reshape(3, 1), tvec.reshape(3, 1))
            projected, _ = cv2.projectPoints(object_points, refined_rvec, refined_tvec, K, D)
            residual = projected.reshape(-1, 2) - pixels
            refined_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            refined_points = cv2.Rodrigues(refined_rvec)[0] @ object_points.T + refined_tvec.reshape(3, 1)
            if refined_rms <= rms and np.all(refined_points[2] > 0):
                rms, rvec, tvec, camera_points = (refined_rms, refined_rvec.reshape(3),
                                                   refined_tvec.reshape(3), refined_points.T)
        except cv2.error:
            pass
        return {
            "status": "PASS" if rms <= PNP_RMS_LIMIT_PX else "FAIL",
            "check": "reprojection_fit_only",
            "rms_px": rms,
            "limit_px": PNP_RMS_LIMIT_PX,
            "rvec_rad_local_to_camera": rvec.tolist(),
            "tvec_mm_local_origin_in_camera": tvec.tolist(),
            "mean_corner_z_mm": float(np.mean(camera_points[:, 2])),
            "predicted_corner_z_mm": camera_points[:, 2].tolist(),
            "reason": None if rms <= PNP_RMS_LIMIT_PX else "reprojection RMS exceeds diagnostic limit",
        }
    except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
        return {"status": "FAIL", "reason": f"PnP failed: {exc}"}


def _stereo_check(depth_mm: np.ndarray | None, pixels: np.ndarray, predicted_z: Sequence[float] | None,
                  *, window: int = 2) -> dict[str, Any]:
    if depth_mm is None:
        return {"status": "NOT_RUN", "reason": "aligned depth image was not supplied"}
    if depth_mm.ndim != 2:
        return {"status": "FAIL", "reason": "depth image must be a single-channel array"}
    if predicted_z is None:
        return {"status": "NOT_RUN", "reason": "metric PnP is required for stereo comparison"}
    if depth_mm.shape[0] <= 0 or depth_mm.shape[1] <= 0:
        return {"status": "FAIL", "reason": "empty depth image"}
    observed, expected = [], []
    h, w = depth_mm.shape
    for (u, v), z in zip(pixels, predicted_z):
        x, y = np.rint([u, v]).astype(int)
        if not (0 <= x < w and 0 <= y < h):
            continue
        patch = depth_mm[max(0, y - window):min(h, y + window + 1),
                         max(0, x - window):min(w, x + window + 1)]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size:
            observed.append(float(np.median(valid)))
            expected.append(float(z))
    if len(observed) < 6:
        return {"status": "INSUFFICIENT", "valid_corner_count": len(observed),
                "total_corner_count": len(pixels), "reason": "fewer than six corners have valid depth"}
    error = np.asarray(observed) - np.asarray(expected)
    absolute = np.abs(error)
    median = float(np.median(absolute))
    p90 = float(np.percentile(absolute, 90))
    passed = median <= STEREO_MEDIAN_LIMIT_MM and p90 <= STEREO_P90_LIMIT_MM
    return {
        "status": "PASS" if passed else "FAIL",
        "comparison": "aligned_stereo_camera_z_vs_metric_pnp_camera_z",
        "valid_corner_count": len(observed),
        "total_corner_count": len(pixels),
        "median_abs_error_mm": median,
        "p90_abs_error_mm": p90,
        "signed_median_error_mm": float(np.median(error)),
        "limits_mm": {"median_abs": STEREO_MEDIAN_LIMIT_MM, "p90_abs": STEREO_P90_LIMIT_MM},
    }


def detect_checkerboard(image_bgr: np.ndarray, cols: int = DEFAULT_COLS, rows: int = DEFAULT_ROWS,
                        square_mm: float = DEFAULT_SQUARE_MM, *, K: np.ndarray | None = None,
                        dist: np.ndarray | None = None,
                        depth_mm: np.ndarray | None = None) -> dict[str, Any]:
    """Detect the largest visible checker sub-grid and return local metric correspondences.

    `cols` and `rows` are the full board's internal-corner dimensions. Returned coordinates are
    local to the detected patch's first corner because an untagged board cannot supply absolute IDs.
    The row/column order itself remains 180-degree ambiguous even for a full-board image.
    """
    if not isinstance(cols, int) or not isinstance(rows, int) or cols < 3 or rows < 3:
        raise ValueError("cols and rows must each be integer counts of at least three internal corners")
    if not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("square_mm must be a positive finite length")
    gray = _gray(np.asarray(image_bgr))
    if not gray.size:
        raise ValueError("image must not be empty")
    if gray.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")
    h, w = gray.shape
    if depth_mm is not None:
        depth_mm = np.asarray(depth_mm, dtype=np.float64)
        if depth_mm.ndim != 2 or depth_mm.shape != gray.shape:
            raise ValueError("aligned depth dimensions must exactly match the color image")
    if K is not None:
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("K must be a finite 3x3 camera matrix with positive focal lengths")
    found = _detect_grids(gray, cols, rows)
    if not found:
        return {
            "detected": False, "diagnostic_only": True, "calibration_accepted": False,
            "board": {"internal_corners": [cols, rows], "square_mm": float(square_mm)},
            "corner_count": 0,
            "coverage": {"image_hull_fraction": 0.0, "expected_corner_fraction": 0.0},
            "reason": "no rectangular checker grid of at least 3 by 3 corners was detected",
            "metric_pnp": {"status": "NOT_RUN", "reason": "no detected corners"},
            "stereo_check": _stereo_check(depth_mm, np.empty((0, 2)), None),
        }

    def evaluate(candidates):
        evaluated = []
        for observed_cols, observed_rows, candidate_pixels, strategy in candidates:
            candidate_grid = np.stack(
                np.meshgrid(np.arange(observed_cols), np.arange(observed_rows)), axis=-1
            ).reshape(-1, 2)
            candidate_object_points = np.column_stack((candidate_grid[:, 0] * square_mm,
                                                       candidate_grid[:, 1] * square_mm,
                                                       np.zeros(len(candidate_grid)))).astype(np.float64)
            chart_mask, chart_residuals = _chart_inliers(candidate_grid, candidate_pixels)
            used_pixels = candidate_pixels[chart_mask]
            used_grid = candidate_grid[chart_mask]
            used_object_points = candidate_object_points[chart_mask]
            candidate_pnp = _metric_pnp(used_object_points, used_pixels.astype(np.float64), K, dist)
            candidate_pnp["total_detected_corner_count"] = int(len(candidate_pixels))
            candidate_pnp["used_corner_count"] = int(len(used_pixels))
            rms = candidate_pnp.get("rms_px")
            chart_count = int(chart_mask.sum())
            pnp_pass = candidate_pnp.get("status") == "PASS"
            # True hand-eye candidates require both the geometric support floor and a useful PnP
            # fit when intrinsics are available. This is a support gate, not a calibration accept.
            handeye_supported = chart_count >= MIN_HANDEYE_CORNERS and (K is None or pnp_pass)
            evaluated.append((handeye_supported, pnp_pass, chart_count,
                              observed_cols == cols and observed_rows == rows,
                              len(candidate_pixels), -(float(rms) if rms is not None else float("inf")),
                              observed_cols, observed_rows, candidate_pixels, candidate_grid,
                              candidate_object_points, chart_mask, chart_residuals,
                              used_pixels, used_grid, used_object_points, candidate_pnp, strategy))
        return evaluated

    hypotheses = evaluate(found)
    viable = lambda items: any(row[0] for row in items)
    # If the larger-board hypothesis failed its 30-corner/geometric gate, include smaller exact
    # sub-patterns too. A misleading large grid must not hide a smaller, coherent usable chart.
    if not viable(hypotheses):
        expanded = _detect_grids(gray, cols, rows, search_all=True)
        found_keys = {(width, height, len(pixels)) for width, height, pixels, _ in found}
        additional = [candidate for candidate in expanded
                      if (candidate[0], candidate[1], len(candidate[2])) not in found_keys]
        if additional:
            hypotheses.extend(evaluate(additional))
    # Prefer the ≥30-corner support gate, then PnP consistency, then the greatest independent
    # corner support. This rejects the 56-corner / 1.8 px live hypothesis in favour of 48 usable.
    (handeye_supported, pnp_pass, chart_count, _, _, _, observed_cols, observed_rows,
     pixels, grid, object_points, chart_mask, chart_residuals, used_pixels, used_grid,
     used_object_points, pnp, detection_strategy) = max(
         hypotheses, key=lambda candidate: candidate[:6])
    hull = cv2.convexHull(pixels.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    x_span = float(np.ptp(pixels[:, 0]))
    y_span = float(np.ptp(pixels[:, 1]))
    used_hull_area = (float(cv2.contourArea(cv2.convexHull(used_pixels.astype(np.float32))))
                      if len(used_pixels) >= 3 else 0.0)
    stereo = _stereo_check(depth_mm, used_pixels, pnp.get("predicted_corner_z_mm"), window=2)
    full = observed_cols == cols and observed_rows == rows
    return {
        "detected": True,
        "diagnostic_only": True,
        "calibration_accepted": False,
        "board": {"internal_corners": [cols, rows], "square_mm": float(square_mm),
                  "full_board_detected": full, "detected_grid_shape": [observed_cols, observed_rows],
                  "local_span_mm": [float((observed_cols - 1) * square_mm),
                                    float((observed_rows - 1) * square_mm)]},
        "corner_count": int(len(pixels)),
        "used_corner_count": int(len(used_pixels)),
        "flat_m_total": int(len(pixels)),
        "flat_m_used": int(len(used_pixels)),
        "chart_support_count": chart_count,
        "handeye_support": {"total_detected": int(len(pixels)), "used": chart_count,
                            "minimum": MIN_HANDEYE_CORNERS,
                            "passed": bool(handeye_supported),
                            "status": "PASS" if handeye_supported else "INSUFFICIENT"},
        "chart_filter": {
            "method": "grid-to-image homography RANSAC",
            "threshold_px": CHART_RANSAC_THRESHOLD_PX,
            "rng_seed": 0,
            "status": "PASS" if chart_count >= MIN_HANDEYE_CORNERS else "INSUFFICIENT",
            "inlier_count": chart_count,
            "outlier_count": int(len(pixels) - chart_count),
            "median_inlier_residual_px": (float(np.median(chart_residuals[chart_mask]))
                                           if chart_count else None),
            "p90_inlier_residual_px": (float(np.percentile(chart_residuals[chart_mask], 90))
                                        if chart_count else None),
        },
        "coverage": {
            "image_hull_fraction": float(hull_area / (w * h)),
            "image_bbox_fraction": float((x_span * y_span) / (w * h)),
            "expected_corner_fraction": float(len(pixels) / (cols * rows)),
            "used_image_hull_fraction": float(used_hull_area / (w * h)),
            "bbox_px": [float(pixels[:, 0].min()), float(pixels[:, 1].min()),
                        float(pixels[:, 0].max()), float(pixels[:, 1].max())],
        },
        "coordinate_convention": "detector-local row-major grid; no absolute board IDs",
        "orientation_ambiguous": True,
        "detection_strategy": detection_strategy,
        "grid_indices_col_row_local": grid.tolist(),
        "corner_pixels": pixels.astype(float).tolist(),
        "object_points_local_mm": object_points.tolist(),
        "chart_inlier_mask": chart_mask.tolist(),
        "chart_residual_px": [float(value) if np.isfinite(value) else None
                               for value in chart_residuals],
        "used_grid_indices_col_row_local": used_grid.tolist(),
        "used_corner_pixels": used_pixels.astype(float).tolist(),
        "used_object_points_local_mm": used_object_points.tolist(),
        "metric_pnp": pnp,
        "stereo_check": stereo,
        "thresholds": {"pnp_reprojection_rms_px": PNP_RMS_LIMIT_PX,
                       "stereo_median_abs_mm": STEREO_MEDIAN_LIMIT_MM,
                       "stereo_p90_abs_mm": STEREO_P90_LIMIT_MM},
        "note": "PnP reprojection and stereo consistency are independent diagnostics; neither accepts a robot calibration.",
    }


def _camera_model(metadata_path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    camera = metadata.get("camera", metadata)
    intrinsics = camera.get("intrinsics", {}) if isinstance(camera, dict) else {}
    raw_K = intrinsics.get("matrix_px", metadata.get("K"))
    if raw_K is None:
        raise ValueError(f"{metadata_path}: no camera intrinsics matrix_px or K")
    raw_dist = intrinsics.get("distortion_coefficients", metadata.get("distortion_coefficients"))
    if intrinsics.get("distortion_removed_from_output", False):
        raw_dist = None
    return np.asarray(raw_K, np.float64), (None if raw_dist is None else np.asarray(raw_dist, np.float64))


def _load_depth(path: Path, scale_mm: float) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise ValueError(f"cannot read single-channel depth image: {path}")
    return np.asarray(depth, dtype=np.float64) * scale_mm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="saved wrist-camera color snapshot")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="full board internal corners across (default: 11)")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="full board internal corners down (default: 8)")
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                        help="printed checker square edge in millimeters (default: 30)")
    parser.add_argument("--metadata", type=Path, help="capture JSON with camera.intrinsics.matrix_px (or top-level K)")
    parser.add_argument("--depth", type=Path, help="optional depth aligned to the input color image; PNG values default to mm")
    parser.add_argument("--depth-scale-mm", type=float, default=1.0,
                        help="multiply depth samples by this scale (default: 1 for saved calibration mm images)")
    parser.add_argument("--out", type=Path, help="optional path for a diagnostic JSON report")
    return parser


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Keep console output useful; full corner arrays belong in the optional JSON report."""
    pnp = dict(report.get("metric_pnp", {}))
    pnp.pop("predicted_corner_z_mm", None)
    keys = ("detected", "diagnostic_only", "calibration_accepted", "board", "corner_count",
            "used_corner_count", "flat_m_total", "flat_m_used", "handeye_support", "coverage",
            "chart_filter", "metric_pnp", "stereo_check", "detection_strategy", "reason", "error")
    summary = {key: report[key] for key in keys if key in report}
    if "metric_pnp" in summary:
        summary["metric_pnp"] = pnp
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cols < 3 or args.rows < 3 or not np.isfinite(args.square_mm) or args.square_mm <= 0:
        parser.error("board requires at least 3 by 3 internal corners and a positive --square-mm")
    if not np.isfinite(args.depth_scale_mm) or args.depth_scale_mm <= 0:
        parser.error("--depth-scale-mm must be a positive finite value")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    try:
        K = dist = None
        if args.metadata:
            K, dist = _camera_model(args.metadata)
        depth = _load_depth(args.depth, args.depth_scale_mm) if args.depth else None
        result = detect_checkerboard(image, args.cols, args.rows, args.square_mm,
                                     K=K, dist=dist, depth_mm=depth)
    except (OSError, ValueError, cv2.error) as exc:
        result = {"detected": False, "diagnostic_only": True, "calibration_accepted": False,
                  "error": str(exc)}
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        forbidden = {args.image.resolve()}
        if args.metadata:
            forbidden.add(args.metadata.resolve())
        if args.depth:
            forbidden.add(args.depth.resolve())
        resolved_out = args.out.resolve()
        if (resolved_out in forbidden or
                resolved_out.name.casefold() in {"left_wrist.json", "right_wrist.json"}):
            parser.error("--out must be a separate diagnostic path and cannot target an active wrist calibration")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
        print(args.out)
    print(json.dumps(_console_summary(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("detected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
