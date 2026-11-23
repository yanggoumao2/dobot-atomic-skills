"""Offline RGB-D feature recovery from a naturally printed wrist-camera target.

This module uses the printed SF Express parcel in the saved left-wrist capture as a
stationary textured target. It never opens a camera or robot connection. SIFT matches
are restricted to manually bounded printed regions, checked by a planar homography,
then given depths from local fits to inverse depth on the dense temporal-median depth
image. A second, robust disparity-plane support mask excludes nearby jaws, bottles,
background surfaces and paper boundaries.

The recovered pairs can be passed to the existing hand-eye solver. Optional
``current_to_reference`` transforms are useful when a separate measurement model has
already estimated camera movement relative to the wrist; they are deliberately not
estimated here so this target extractor remains independent of those measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from chem_demo.wrist_geometry import PairObservation, backproject, pose_matrix, transform


# Coordinates are for the 848 x 480 D405 color stream used by the capture. The broad
# paper polygon supports a depth plane; feature extraction is limited to three
# separated printed areas, not the unprinted tabletop or nearby equipment.
_REFERENCE_SIZE = (848, 480)
PAPER_POLYGON = np.array(
    [[390, 90], [847, 90], [847, 475], [384, 442]], dtype=np.float32
)
PRINTED_REGIONS: dict[str, np.ndarray] = {
    "logo": np.array([[410, 110], [670, 110], [670, 325], [410, 325]], dtype=np.float32),
    "lower_band": np.array([[390, 345], [705, 345], [705, 460], [390, 460]], dtype=np.float32),
    "right_label": np.array([[700, 125], [847, 125], [847, 465], [700, 465]], dtype=np.float32),
}

SIFT_FEATURES = 16_000
SIFT_CONTRAST_THRESHOLD = 0.0015
SIFT_EDGE_THRESHOLD = 20
SIFT_RATIO = 0.82
HOMOGRAPHY_REPROJECTION_PX = 2.0
MIN_UNIQUE_INLIERS = 30
GLOBAL_PLANE_SUPPORT_MM = 10.0
LOCAL_RADIUS_PX = 10
LOCAL_EDGE_GUARD_PX = 6
LOCAL_MIN_VALID_PIXELS = 100
LOCAL_MIN_INLIERS = 70
LOCAL_MAX_MEDIAN_RESIDUAL_MM = 2.5
LOCAL_MAX_P90_RESIDUAL_MM = 5.0
LOCAL_CENTER_TOLERANCE_MM = 4.0


@dataclass
class TargetPair:
    """One reference-to-pose target correspondence set, in camera coordinates."""

    label: str
    T_base_flange_ref: np.ndarray
    T_base_flange_pose: np.ndarray
    pixels_ref_px: np.ndarray
    pixels_pose_px: np.ndarray
    bearings_ref: np.ndarray
    bearings_pose: np.ndarray
    points_ref_camera_mm: np.ndarray
    points_pose_camera_mm: np.ndarray
    raw_depths_ref_mm: np.ndarray
    raw_depths_pose_mm: np.ndarray
    region_ids: np.ndarray
    homography_ref_to_pose: np.ndarray
    homography_errors_px: np.ndarray
    diagnostics: dict

    @property
    def unique_inliers(self) -> int:
        return int(len(self.pixels_ref_px))


@dataclass
class TargetCapture:
    """Saved RGB-D capture with target-only 3D feature pairs and audit metadata."""

    folder: Path
    arm: str
    K: np.ndarray
    table_z_reference_mm: float
    reference_pose: np.ndarray
    poses: dict[str, np.ndarray]
    pairs: dict[str, TargetPair]
    plane_points_ref_mm: np.ndarray
    plane_models: dict[str, dict]
    report: dict


def _scaled_polygon(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    sx, sy = width / _REFERENCE_SIZE[0], height / _REFERENCE_SIZE[1]
    return np.rint(poly * np.array([sx, sy])).astype(np.int32)


def _polygon_mask(polygons: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    scaled = [_scaled_polygon(poly, w, h) for poly in polygons]
    cv2.fillPoly(mask, scaled, 255)
    return mask


def _region_name(point: Sequence[float]) -> str:
    p = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    for name, polygon in PRINTED_REGIONS.items():
        scaled = _scaled_polygon(polygon, *_REFERENCE_SIZE)
        if cv2.pointPolygonTest(scaled, (float(p[0, 0, 0]), float(p[0, 0, 1])), False) >= 0:
            return name
    return "other_print"


def _fit_disparity_plane(depth_mm: np.ndarray, region_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit inverse depth = a*u + b*v + c over the target and return its support mask."""
    if depth_mm.shape != region_mask.shape:
        raise ValueError("target support mask and depth image sizes differ")
    valid = (region_mask > 0) & np.isfinite(depth_mm) & (depth_mm >= 180.0) & (depth_mm <= 1200.0)
    h, w = depth_mm.shape
    vs, us = np.nonzero(valid)
    if len(us) < 500:
        raise ValueError("too little valid dense depth inside the printed target region")

    # Fit a deterministic spatially subsampled set, then score the full dense region.
    sample = (us % 3 == 0) & (vs % 3 == 0)
    if sample.sum() >= 300:
        u_fit, v_fit = us[sample], vs[sample]
        z_fit = depth_mm[v_fit, u_fit].astype(np.float64)
    else:
        u_fit, v_fit = us, vs
        z_fit = depth_mm[v_fit, u_fit].astype(np.float64)
    X = np.column_stack((u_fit, v_fit, np.ones(len(u_fit))))
    inv_z = 1.0 / z_fit
    keep = np.ones(len(z_fit), dtype=bool)
    beta = np.zeros(3, dtype=np.float64)
    for _ in range(10):
        if keep.sum() < 200:
            raise ValueError("target disparity plane has too few inliers")
        beta = np.linalg.lstsq(X[keep], inv_z[keep], rcond=None)[0]
        predicted = 1.0 / np.maximum(X @ beta, 1e-12)
        residual = z_fit - predicted
        center = float(np.median(residual[keep]))
        sigma = 1.4826 * float(np.median(np.abs(residual[keep] - center)))
        new_keep = np.abs(residual - center) <= max(6.0, 3.0 * sigma)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    vv, uu = np.mgrid[:h, :w]
    prediction_inv = beta[0] * uu + beta[1] * vv + beta[2]
    prediction = np.divide(1.0, prediction_inv, out=np.full((h, w), np.nan), where=prediction_inv > 0)
    residual_full = depth_mm.astype(np.float64) - prediction
    support = valid & np.isfinite(prediction) & (np.abs(residual_full) <= GLOBAL_PLANE_SUPPORT_MM)
    fit_residual = residual[keep]
    info = {
        "dense_candidates": int(valid.sum()),
        "fit_samples": int(len(z_fit)),
        "fit_inliers": int(keep.sum()),
        "support_pixels": int(support.sum()),
        "support_share": float(support.sum() / max(1, valid.sum())),
        "fit_median_abs_residual_mm": float(np.median(np.abs(fit_residual))),
        "fit_p90_abs_residual_mm": float(np.percentile(np.abs(fit_residual), 90)),
        "support_threshold_mm": GLOBAL_PLANE_SUPPORT_MM,
    }
    return beta, support.astype(np.uint8), info


def _raw_center_depth(depth: np.ndarray, u: float, v: float) -> float | None:
    h, w = depth.shape
    x, y = int(round(u)), int(round(v))
    patch = depth[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)].astype(np.float64)
    valid = patch[np.isfinite(patch) & (patch >= 180.0) & (patch <= 1200.0)]
    if len(valid) < 3:
        return None
    return float(np.median(valid))


def _local_disparity_depth(depth: np.ndarray, support: np.ndarray, paper_edge_distance: np.ndarray,
                           u: float, v: float) -> tuple[float, float, dict] | None:
    """Estimate feature depth from a robust local planar fit in inverse-depth space."""
    h, w = depth.shape
    x0, y0 = int(round(u)), int(round(v))
    r = LOCAL_RADIUS_PX
    if x0 < r or y0 < r or x0 >= w - r or y0 >= h - r:
        return None
    if paper_edge_distance.shape != depth.shape or support.shape != depth.shape:
        raise ValueError("depth, paper edge-distance and plane support mask sizes differ")
    if paper_edge_distance[y0, x0] < LOCAL_EDGE_GUARD_PX:
        return None
    patch = depth[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].astype(np.float64)
    allowed = support[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1] > 0
    valid = allowed & np.isfinite(patch) & (patch >= 180.0) & (patch <= 1200.0)
    if int(valid.sum()) < LOCAL_MIN_VALID_PIXELS:
        return None
    vv, uu = np.mgrid[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
    z = patch[valid]
    X = np.column_stack((uu[valid], vv[valid], np.ones(int(valid.sum()))))
    inverse = 1.0 / z
    keep = np.ones(len(z), dtype=bool)
    beta = np.zeros(3, dtype=np.float64)
    for _ in range(8):
        if keep.sum() < LOCAL_MIN_INLIERS:
            return None
        beta = np.linalg.lstsq(X[keep], inverse[keep], rcond=None)[0]
        predicted_inv = X @ beta
        predicted_z = 1.0 / np.maximum(predicted_inv, 1e-12)
        residual = z - predicted_z
        center = float(np.median(residual[keep]))
        sigma = 1.4826 * float(np.median(np.abs(residual[keep] - center)))
        new_keep = np.abs(residual - center) <= max(4.0, 3.0 * sigma)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    z_local = float(1.0 / np.dot(np.array([u, v, 1.0]), beta))
    if not np.isfinite(z_local) or not (180.0 <= z_local <= 1200.0) or keep.sum() < LOCAL_MIN_INLIERS:
        return None
    abs_residual = np.abs(residual[keep])
    median_residual = float(np.median(abs_residual))
    p90_residual = float(np.percentile(abs_residual, 90))
    if median_residual > LOCAL_MAX_MEDIAN_RESIDUAL_MM or p90_residual > LOCAL_MAX_P90_RESIDUAL_MM:
        return None
    raw = _raw_center_depth(depth, u, v)
    if raw is None or abs(raw - z_local) > LOCAL_CENTER_TOLERANCE_MM:
        return None
    return z_local, raw, {
        "local_support_pixels": int(valid.sum()),
        "local_plane_inliers": int(keep.sum()),
        "local_median_abs_residual_mm": median_residual,
        "local_p90_abs_residual_mm": p90_residual,
        "raw_center_depth_mm": raw,
        "local_depth_mm": z_local,
    }


def _sift() -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=SIFT_FEATURES,
                           contrastThreshold=SIFT_CONTRAST_THRESHOLD,
                           edgeThreshold=SIFT_EDGE_THRESHOLD)


def _mutual_ratio_matches(descriptors_a: np.ndarray | None, descriptors_b: np.ndarray | None,
                          ratio: float = SIFT_RATIO) -> list[cv2.DMatch]:
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(descriptors_b, descriptors_a)}
    return [best for pair in forward if len(pair) == 2
            for best, second in [pair]
            if best.distance < ratio * second.distance and reverse.get(best.trainIdx) == best.queryIdx]


def _homography_matches(keypoints_ref, descriptors_ref, keypoints_pose, descriptors_pose
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    matches = _mutual_ratio_matches(descriptors_ref, descriptors_pose)
    if len(matches) < 8:
        raise ValueError("too few mutual SIFT matches for a planar homography")
    pixels_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in matches])
    pixels_pose = np.float32([keypoints_pose[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pixels_ref, pixels_pose, cv2.RANSAC,
                                 HOMOGRAPHY_REPROJECTION_PX, maxIters=10_000, confidence=0.999)
    if H is None or mask is None or not np.all(np.isfinite(H)):
        raise ValueError("printed target correspondences do not support a stable homography")
    inlier_ids = np.flatnonzero(mask.reshape(-1).astype(bool))
    errors = np.linalg.norm(cv2.perspectiveTransform(pixels_ref[:, None, :], H)[:, 0, :] - pixels_pose,
                            axis=1)
    # SIFT may emit multiple orientations/scales at the same pixel. Keep one match per
    # source pixel and select the lowest reprojection-error representative.
    unique_ids: list[int] = []
    seen: set[tuple[int, int]] = set()
    for idx in sorted(inlier_ids.tolist(), key=lambda i: (float(errors[i]), float(matches[i].distance))):
        key = tuple(np.rint(pixels_ref[idx]).astype(int).tolist())
        if key not in seen:
            seen.add(key)
            unique_ids.append(idx)
    selected = np.asarray(unique_ids, dtype=int)
    selected_errors = errors[selected]
    if len(selected) < 8:
        raise ValueError("homography has too few unique target inliers")
    return pixels_ref[selected].astype(np.float64), pixels_pose[selected].astype(np.float64), H, selected_errors, {
        "mutual_ratio_matches": int(len(matches)),
        "homography_inliers": int(len(inlier_ids)),
        "homography_unique_source_pixels": int(len(selected)),
        "homography_median_reprojection_px": float(np.median(selected_errors)),
        "homography_p90_reprojection_px": float(np.percentile(selected_errors, 90)),
    }


def _pose_from_group(group: dict) -> np.ndarray:
    captures = group.get("captures", [])
    if not captures:
        raise ValueError("pose group has no RGB-D capture")
    return pose_matrix(captures[0]["robot_before"]["tcp_pose_mm_deg"])


def _camera_parameters(metadata: dict) -> tuple[np.ndarray, float]:
    camera = metadata["camera"]
    intrinsics = camera["intrinsics"]
    K = np.asarray(intrinsics["matrix_px"], dtype=np.float64)
    if (K.shape != (3, 3) or not np.all(np.isfinite(K)) or np.linalg.det(K) == 0
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-9)):
        raise ValueError("invalid camera intrinsics in acquisition metadata")
    table_z = float(metadata["table_z_reference_mm"])
    if not np.isfinite(table_z):
        raise ValueError("table_z_reference_mm must be finite")
    return K, table_z


def load_target_capture(folder: str | Path, *, labels: Sequence[str] | None = None,
                       minimum_inliers: int = MIN_UNIQUE_INLIERS) -> TargetCapture:
    """Load a saved capture and recover static printed-target 3D feature matches.

    The returned arrays are in millimeters and pixel coordinates. A pose with fewer
    than ``minimum_inliers`` geometrically checked, locally depth-supported unique
    matches raises ``ValueError``; the threshold is not silently weakened.
    """
    if minimum_inliers < MIN_UNIQUE_INLIERS:
        raise ValueError(f"minimum inliers cannot be lower than {MIN_UNIQUE_INLIERS}")
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"capture folder does not exist: {root}")
    metadata_path = root / "acquisition.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("arm") != "left":
        raise ValueError("the natural printed-target profile is defined for the left wrist capture")
    if metadata.get("status") not in ("complete", "acquired") or metadata.get("capture_problems"):
        raise ValueError("capture is incomplete or has acquisition problems")
    if int(metadata.get("capture_burst", 0)) != 15:
        raise ValueError("expected the complete 15-frame burst depth capture")
    K, table_z = _camera_parameters(metadata)
    pose_groups = metadata["pose_groups"]
    if "ref" not in pose_groups:
        raise ValueError("capture has no reference pose")
    if len(pose_groups) != 10 or any(len(group.get("captures", [])) != 15
                                     for group in pose_groups.values()):
        raise ValueError("expected the complete 10-pose by 15-frame RGB-D capture")
    requested = list(labels) if labels is not None else [name for name in pose_groups if name != "ref"]
    if any(label not in pose_groups for label in requested):
        raise ValueError("one or more requested pose labels are missing")
    poses = {name: _pose_from_group(group) for name, group in pose_groups.items()}
    reference_pose = poses["ref"]

    ref_color = cv2.imread(str(root / "ref.png"), cv2.IMREAD_COLOR)
    ref_depth = np.load(root / "ref_depth_temporal_median_mm.npy", allow_pickle=False)
    if ref_color is None or ref_depth.ndim != 2 or ref_color.shape[:2] != ref_depth.shape:
        raise ValueError("reference RGB and temporal-median depth are missing or misaligned")
    h, w = ref_depth.shape
    if (w, h) != _REFERENCE_SIZE:
        raise ValueError(f"expected {_REFERENCE_SIZE[0]}x{_REFERENCE_SIZE[1]} reference RGB-D, got {w}x{h}")

    paper_mask_ref = _polygon_mask([PAPER_POLYGON], ref_depth.shape)
    paper_edge_distance_ref = cv2.distanceTransform((paper_mask_ref > 0).astype(np.uint8), cv2.DIST_L2, 3)
    print_mask_ref = _polygon_mask(list(PRINTED_REGIONS.values()), ref_depth.shape)
    plane_ref, support_ref, plane_info_ref = _fit_disparity_plane(ref_depth, paper_mask_ref)
    ref_feature_mask = cv2.bitwise_and(print_mask_ref, support_ref * 255)
    gray_ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    detector = _sift()
    keypoints_ref, descriptors_ref = detector.detectAndCompute(gray_ref, ref_feature_mask)
    if descriptors_ref is None or len(keypoints_ref) < 20:
        raise ValueError("too few SIFT features on the printed reference target")

    plane_models: dict[str, dict] = {"ref": {
        "inverse_depth_coefficients": plane_ref.tolist(), "support": plane_info_ref,
    }}
    pair_reports: dict[str, dict] = {}
    pairs: dict[str, TargetPair] = {}

    for label in requested:
        color = cv2.imread(str(root / f"{label}.png"), cv2.IMREAD_COLOR)
        depth_path = root / f"{label}_depth_temporal_median_mm.npy"
        if color is None or not depth_path.is_file():
            raise ValueError(f"{label}: missing color frame or dense temporal-median depth")
        depth = np.load(depth_path, allow_pickle=False)
        if depth.ndim != 2 or color.shape[:2] != depth.shape or depth.shape != ref_depth.shape:
            raise ValueError(f"{label}: RGB and temporal-median depth arrays are not aligned")
        if not np.all(np.isfinite(depth)) or np.any(depth < 0):
            raise ValueError(f"{label}: depth contains invalid values")

        # First establish a broad planar registration, use it to transfer the paper
        # support, then rematch only inside dense depth that agrees with that surface.
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        initial_keypoints, initial_descriptors = detector.detectAndCompute(gray, None)
        _, _, H_initial, _, _ = _homography_matches(
            keypoints_ref, descriptors_ref, initial_keypoints, initial_descriptors)
        paper_mask_pose = cv2.warpPerspective(paper_mask_ref, H_initial, (w, h),
                                              flags=cv2.INTER_NEAREST)
        plane_pose, support_pose, plane_info_pose = _fit_disparity_plane(depth, paper_mask_pose)
        print_mask_pose = cv2.warpPerspective(print_mask_ref, H_initial, (w, h),
                                              flags=cv2.INTER_NEAREST)
        pose_feature_mask = cv2.bitwise_and(print_mask_pose, support_pose * 255)
        keypoints_pose, descriptors_pose = detector.detectAndCompute(gray, pose_feature_mask)
        if descriptors_pose is None:
            raise ValueError(f"{label}: no printed target features survive the dense plane support mask")
        pix_ref, pix_pose, H, homography_errors, match_info = _homography_matches(
            keypoints_ref, descriptors_ref, keypoints_pose, descriptors_pose)

        # Refit the pose target plane with the final homography before depth lifting.
        paper_mask_pose = cv2.warpPerspective(paper_mask_ref, H, (w, h), flags=cv2.INTER_NEAREST)
        paper_edge_distance_pose = cv2.distanceTransform((paper_mask_pose > 0).astype(np.uint8),
                                                          cv2.DIST_L2, 3)
        plane_pose, support_pose, plane_info_pose = _fit_disparity_plane(depth, paper_mask_pose)
        plane_models[label] = {
            "inverse_depth_coefficients": plane_pose.tolist(), "support": plane_info_pose,
        }
        raw_plane_residual_ref: list[float] = []
        raw_plane_residual_pose: list[float] = []
        kept_ref: list[np.ndarray] = []
        kept_pose: list[np.ndarray] = []
        kept_z_ref: list[float] = []
        kept_z_pose: list[float] = []
        kept_raw_ref: list[float] = []
        kept_raw_pose: list[float] = []
        kept_errors: list[float] = []
        kept_regions: list[str] = []
        local_reports: list[dict] = []
        for p_ref, p_pose, reprojection_error in zip(pix_ref, pix_pose, homography_errors):
            u0, v0 = map(float, p_ref)
            u1, v1 = map(float, p_pose)
            z0 = _local_disparity_depth(ref_depth, support_ref, paper_edge_distance_ref, u0, v0)
            z1 = _local_disparity_depth(depth, support_pose, paper_edge_distance_pose, u1, v1)
            if z0 is None or z1 is None:
                continue
            global_z0 = 1.0 / float(np.dot([u0, v0, 1.0], plane_ref))
            global_z1 = 1.0 / float(np.dot([u1, v1, 1.0], plane_pose))
            # Dense global plane support is a surface identity check; local fits still
            # supply the 3D depth so modest sheet warp is preserved.
            if abs(z0[0] - global_z0) > GLOBAL_PLANE_SUPPORT_MM:
                continue
            if abs(z1[0] - global_z1) > GLOBAL_PLANE_SUPPORT_MM:
                continue
            kept_ref.append(p_ref)
            kept_pose.append(p_pose)
            kept_z_ref.append(z0[0])
            kept_z_pose.append(z1[0])
            kept_raw_ref.append(z0[1])
            kept_raw_pose.append(z1[1])
            kept_errors.append(float(reprojection_error))
            kept_regions.append(_region_name(p_ref))
            local_reports.append({"ref": z0[2], "pose": z1[2]})
            raw_plane_residual_ref.append(abs(z0[1] - z0[0]))
            raw_plane_residual_pose.append(abs(z1[1] - z1[0]))

        if len(kept_ref) < minimum_inliers:
            raise ValueError(f"{label}: only {len(kept_ref)} unique matches survive homography, paper-plane, "
                             f"edge and local-depth filters; need {minimum_inliers}")
        pix_ref_array = np.asarray(kept_ref, dtype=np.float64)
        pix_pose_array = np.asarray(kept_pose, dtype=np.float64)
        z_ref_array = np.asarray(kept_z_ref, dtype=np.float64)
        z_pose_array = np.asarray(kept_z_pose, dtype=np.float64)
        raw_ref_array = np.asarray(kept_raw_ref, dtype=np.float64)
        raw_pose_array = np.asarray(kept_raw_pose, dtype=np.float64)
        points_ref = backproject(K, pix_ref_array, z_ref_array)
        points_pose = backproject(K, pix_pose_array, z_pose_array)
        rays_ref = np.column_stack(((pix_ref_array[:, 0] - K[0, 2]) / K[0, 0],
                                    (pix_ref_array[:, 1] - K[1, 2]) / K[1, 1],
                                    np.ones(len(pix_ref_array))))
        rays_pose = np.column_stack(((pix_pose_array[:, 0] - K[0, 2]) / K[0, 0],
                                     (pix_pose_array[:, 1] - K[1, 2]) / K[1, 1],
                                     np.ones(len(pix_pose_array))))
        region_ids = np.asarray(kept_regions, dtype=object)
        counts_by_region = {name: int(np.sum(region_ids == name)) for name in PRINTED_REGIONS}
        diagnostics = {
            **match_info,
            "unique_local_depth_supported_inliers": int(len(points_ref)),
            "target_regions": counts_by_region,
            "raw_vs_local_depth": {
                "ref_median_abs_mm": float(np.median(raw_plane_residual_ref)),
                "ref_p90_abs_mm": float(np.percentile(raw_plane_residual_ref, 90)),
                "pose_median_abs_mm": float(np.median(raw_plane_residual_pose)),
                "pose_p90_abs_mm": float(np.percentile(raw_plane_residual_pose, 90)),
            },
            "local_plane_residual_median_mm": float(np.median([
                max(row["ref"]["local_median_abs_residual_mm"],
                    row["pose"]["local_median_abs_residual_mm"]) for row in local_reports])),
            "local_plane_residual_p90_mm": float(np.percentile([
                max(row["ref"]["local_p90_abs_residual_mm"],
                    row["pose"]["local_p90_abs_residual_mm"]) for row in local_reports], 90)),
            "filters": {
                "sift_ratio": SIFT_RATIO,
                "homography_reprojection_px": HOMOGRAPHY_REPROJECTION_PX,
                "global_plane_support_mm": GLOBAL_PLANE_SUPPORT_MM,
                "local_radius_px": LOCAL_RADIUS_PX,
                "local_edge_guard_px": LOCAL_EDGE_GUARD_PX,
                "local_median_residual_mm_max": LOCAL_MAX_MEDIAN_RESIDUAL_MM,
                "local_p90_residual_mm_max": LOCAL_MAX_P90_RESIDUAL_MM,
                "local_center_depth_tolerance_mm": LOCAL_CENTER_TOLERANCE_MM,
                "minimum_unique_inliers": int(minimum_inliers),
            },
        }
        pair_reports[label] = diagnostics
        pairs[label] = TargetPair(
            label=label,
            T_base_flange_ref=reference_pose.copy(),
            T_base_flange_pose=poses[label].copy(),
            pixels_ref_px=pix_ref_array,
            pixels_pose_px=pix_pose_array,
            bearings_ref=rays_ref,
            bearings_pose=rays_pose,
            points_ref_camera_mm=points_ref,
            points_pose_camera_mm=points_pose,
            raw_depths_ref_mm=raw_ref_array,
            raw_depths_pose_mm=raw_pose_array,
            region_ids=region_ids,
            homography_ref_to_pose=H,
            homography_errors_px=np.asarray(kept_errors, dtype=np.float64),
            diagnostics=diagnostics,
        )

    # Use unique reference pixels only to build the table-height anchor plane cloud.
    cloud_by_pixel: dict[tuple[int, int], np.ndarray] = {}
    for pair in pairs.values():
        for pixel, point in zip(pair.pixels_ref_px, pair.points_ref_camera_mm):
            cloud_by_pixel.setdefault(tuple(np.rint(pixel).astype(int).tolist()), point)
    plane_points = np.asarray(list(cloud_by_pixel.values()), dtype=np.float64)
    if len(plane_points) < minimum_inliers:
        raise ValueError("reference printed target supplies too few distinct plane points")
    report = {
        "capture": root.name,
        "arm": metadata["arm"],
        "target": "natural printed SF Express parcel",
        "depth_source": "dense temporal median of 15 aligned RGB-D frames per pose",
        "camera_intrinsics_px": K.tolist(),
        "table_z_reference_mm": table_z,
        "reference_plane": plane_models["ref"],
        "pairs": pair_reports,
        "reference_plane_feature_count": int(len(plane_points)),
        "support_policy": "printed ROI AND dense camera-derived disparity-plane support within 10 mm; "
                          "then local disparity-plane fit with center-depth/edge checks",
    }
    return TargetCapture(root, metadata["arm"], K, table_z, reference_pose, poses,
                         pairs, plane_points, plane_models, report)


def as_pair_observations(capture: TargetCapture,
                         camera_corrections: Mapping[str, np.ndarray] | None = None,
                         *, region: str | None = None) -> list[PairObservation]:
    """Adapt recovered pairs to the existing solver, optionally correcting camera frames.

    Each correction must map points in ``label`` camera coordinates into the reference
    camera coordinates, matching the convention returned by the offline near-jaw PnP
    diagnostic. Corrections are external inputs and are never inferred by this module.
    """
    if region is not None and region not in (*PRINTED_REGIONS, "other_print"):
        raise ValueError(f"unknown printed-target region: {region}")
    corrections = camera_corrections or {}
    if camera_corrections is not None:
        missing = sorted(set(capture.pairs) - set(corrections))
        if missing:
            raise ValueError(f"camera corrections are missing for poses: {', '.join(missing)}")
    result: list[PairObservation] = []
    for label, pair in capture.pairs.items():
        keep = np.ones(pair.unique_inliers, dtype=bool) if region is None else pair.region_ids == region
        if int(keep.sum()) < 5:
            continue
        xi = pair.points_pose_camera_mm[keep]
        correction = corrections.get(label) if camera_corrections is not None else None
        if camera_corrections is not None:
            correction = np.asarray(correction, dtype=np.float64)
            if correction.shape != (4, 4) or not np.all(np.isfinite(correction)):
                raise ValueError(f"{label}: camera correction must be a finite 4x4 matrix")
            xi = transform(correction, xi)
        result.append(PairObservation(pair.T_base_flange_ref, pair.T_base_flange_pose,
                                      pair.points_ref_camera_mm[keep], xi, label))
    return result


def to_jsonable(capture: TargetCapture) -> dict:
    """Return only serializable target diagnostics, leaving the large arrays in memory."""
    return capture.report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="saved 10-pose left wrist RGB-D capture folder")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--pose", action="append", dest="labels", help="analyze one pose label; repeatable")
    args = parser.parse_args(argv)
    try:
        capture = load_target_capture(args.folder, labels=args.labels)
        payload = json.dumps(to_jsonable(capture), indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
            print(args.output)
        else:
            print(payload)
        return 0
    except (OSError, KeyError, ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        print(json.dumps({"capture": args.folder.name, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
