"""Offline, per-frame registration of a wrist camera against its visible open fingers.

The measurement only uses the near jaw features and aligned depth. It does not inspect table or
scene texture, robot poses, or pose labels. ``T_reference_current`` maps XYZ points from the current
camera into the camera frame in which the reference jaws were observed.

This is an offline calibration/diagnostic helper. It does not load an active robot calibration or
touch live hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from chem_demo.wrist_geometry import backproject, sample_depth
from chem_demo.wrist_landmarks import CameraTemplate, measure_open_pad_tips


MIN_JAW_DEPTH_MM = 45.0
MAX_JAW_DEPTH_MM = 175.0
MIN_INLIERS = 8
MIN_INLIERS_PER_JAW = 2
Arm = Literal["left", "right"]

# The right wrist sees its pads enter at the bottom of the image. Flipping only the color image
# makes the existing silhouette detector inspect their upper tips. Its pixel results are only a
# jaw-state gate; all metric processing continues to use the original, unflipped image and K.
_RIGHT_LANDMARK_TEMPLATE = CameraTemplate(
    left_center_x=(0.20, 0.50), right_center_x=(0.50, 0.90),
    max_component_top_y=0.04, min_aspect_ratio=0.15,
)


def _validate_arm(arm: str) -> Arm:
    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    return arm  # type: ignore[return-value]


def _open_jaws(color: np.ndarray, arm: Arm):
    if arm == "left":
        return measure_open_pad_tips(color)
    return measure_open_pad_tips(cv2.flip(color, 0), template=_RIGHT_LANDMARK_TEMPLATE)


def _right_jaw_region(shape: tuple[int, int]) -> np.ndarray:
    """Spatial prior for the two bottom-entering right pads in the captured camera view."""
    height, width = shape
    region = np.zeros((height, width), np.uint8)
    region[round(0.35 * height):, round(0.18 * width):round(0.46 * width)] = 1
    region[round(0.35 * height):, round(0.57 * width):round(0.95 * width)] = 1
    return region


@dataclass(frozen=True)
class WristMountMeasurement:
    """Result for one current frame.

    ``jaw_mask`` is an 8-bit image mask with nonzero pixels covering current near-finger regions.
    A failed result keeps the mask for diagnostics but has no transform.
    """

    ok: bool
    T_reference_current: np.ndarray | None
    jaw_mask: np.ndarray
    matches: int
    inliers: int
    left_inliers: int
    right_inliers: int
    median_residual_px: float | None
    p90_residual_px: float | None
    reason: str | None = None


def _validate_observation(color: np.ndarray, depth_mm: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    if not isinstance(color, np.ndarray) or color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color must be a uint8 BGR image")
    if (not isinstance(depth_mm, np.ndarray) or depth_mm.ndim != 2 or
            color.shape[:2] != depth_mm.shape or not np.issubdtype(depth_mm.dtype, np.number)):
        raise ValueError("depth_mm must be a numeric image aligned with color")
    if not np.all(np.isfinite(depth_mm)) or np.any(depth_mm < 0):
        raise ValueError("depth_mm must be finite and nonnegative")
    intrinsics = np.asarray(K, dtype=float)
    if (intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)) or
            min(intrinsics[0, 0], intrinsics[1, 1]) <= 0):
        raise ValueError("K must be a finite 3x3 pinhole matrix with positive focal lengths")
    return depth_mm.shape


def current_jaw_mask(depth_mm: np.ndarray, arm: Arm = "left") -> np.ndarray:
    """Make a conservative depth-supported mask for both open fingers.

    Near depth rejects distant scene texture. The per-arm spatial prior removes the central gap and
    visible surroundings; a 13-pixel dilation bridges depth holes near finger edges. The mask is
    0/255.
    """
    if (not isinstance(depth_mm, np.ndarray) or depth_mm.ndim != 2 or
            not np.issubdtype(depth_mm.dtype, np.number)):
        raise ValueError("depth_mm must be a numeric 2D image")
    if not np.all(np.isfinite(depth_mm)) or np.any(depth_mm < 0):
        raise ValueError("depth_mm must be finite and nonnegative")
    arm = _validate_arm(arm)
    height, width = depth_mm.shape
    if arm == "left":
        mask = ((depth_mm > 0) & (depth_mm < MAX_JAW_DEPTH_MM)).astype(np.uint8)
        mask[:, round(0.38 * width):round(0.61 * width)] = 0
        mask[round(0.6 * height):] = 0
        return cv2.dilate(mask, np.ones((13, 13), np.uint8)) * 255

    # In the right view the pads enter from below and sit at either side of the central gap.
    # Cropping before and after dilation keeps near objects in the gap and outer scene out.
    region = _right_jaw_region(depth_mm.shape)
    near = ((depth_mm > MIN_JAW_DEPTH_MM) & (depth_mm < MAX_JAW_DEPTH_MM)).astype(np.uint8)
    mask = cv2.dilate(near * region, np.ones((13, 13), np.uint8))
    return mask * region * 255


def _features(color: np.ndarray, depth_mm: np.ndarray, arm: Arm = "left") -> tuple[np.ndarray, np.ndarray]:
    allowed = (current_jaw_mask(depth_mm, arm) > 0).astype(np.uint8)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.SIFT_create(
        nfeatures=5000, contrastThreshold=0.005, edgeThreshold=15,
    ).detectAndCompute(gray, allowed)
    if descriptors is None or len(keypoints) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    return np.float32([key.pt for key in keypoints]), descriptors


def _matches(first: tuple[np.ndarray, np.ndarray],
             second: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    first_points, first_descriptors = first
    second_points, second_descriptors = second
    if min(len(first_descriptors), len(second_descriptors)) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher()
    reverse = {match.queryIdx: match.trainIdx
               for match in matcher.match(second_descriptors, first_descriptors)}
    kept = []
    for pair in matcher.knnMatch(first_descriptors, second_descriptors, k=2):
        if len(pair) < 2:
            continue
        best, runner_up = pair
        if best.distance < 0.8 * runner_up.distance and reverse.get(best.trainIdx) == best.queryIdx:
            kept.append(best)
    pa = first_points[[match.queryIdx for match in kept]]
    pb = second_points[[match.trainIdx for match in kept]]
    _, distinct = np.unique(np.round(pa), axis=0, return_index=True)
    return pa[distinct], pb[distinct]


@dataclass(frozen=True)
class WristMountReference:
    """Jaw-only features and depth from one settled, open-gripper reference frame."""

    K: np.ndarray
    depth_mm: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray
    jaw_mask: np.ndarray
    arm: Arm = "left"

    @classmethod
    def from_frame(cls, color: np.ndarray, depth_mm: np.ndarray, K: np.ndarray,
                   arm: Arm = "left") -> "WristMountReference":
        """Build a reusable reference from an aligned RGB-D image with both open pads visible."""
        arm = _validate_arm(arm)
        _validate_observation(color, depth_mm, K)
        landmarks = _open_jaws(color, arm)
        if not landmarks.ok or landmarks.state != "open":
            raise ValueError(f"reference requires a visible open gripper: {landmarks.reason or landmarks.state}")
        points, descriptors = _features(color, depth_mm, arm)
        if descriptors is None or len(points) < MIN_INLIERS:
            raise ValueError(f"reference has only {len(points)} depth-supported jaw features")
        return cls(np.asarray(K, float).copy(), np.asarray(depth_mm, float).copy(),
                   points, descriptors, current_jaw_mask(depth_mm, arm), arm)

    def measure(self, color: np.ndarray, depth_mm: np.ndarray, *, arm: Arm | None = None) -> WristMountMeasurement:
        """Estimate a per-frame 6D correction using reference jaw depth and current jaw pixels.

        This intentionally matches the validated PnP path in ``check_wrist_calibration``: the
        source jaw pixels are lifted with reference depth, robust PnP finds their current image pose,
        and support from both fingers is required before the transform is returned.
        """
        empty_mask = np.zeros(depth_mm.shape, np.uint8) if isinstance(depth_mm, np.ndarray) and depth_mm.ndim == 2 else np.zeros((0, 0), np.uint8)
        match_count = inlier_count = left_count = right_count = 0
        try:
            if arm is not None and _validate_arm(arm) != self.arm:
                raise ValueError("current frame arm differs from the wrist reference")
            _validate_observation(color, depth_mm, self.K)
            if depth_mm.shape != self.depth_mm.shape:
                raise ValueError("current frame resolution differs from the reference")
            mask = current_jaw_mask(depth_mm, self.arm)
            landmarks = _open_jaws(color, self.arm)
            if not landmarks.ok or landmarks.state != "open":
                raise ValueError(f"current frame has no confidently visible open gripper: {landmarks.reason or landmarks.state}")

            current_features = _features(color, depth_mm, self.arm)
            pa, pb = _matches((self.points, self.descriptors), current_features)
            match_count = len(pa)
            if len(pa) < MIN_INLIERS:
                raise ValueError(f"jaw registration has only {len(pa)} mutual feature matches")
            z = sample_depth(self.depth_mm, pa, window=1)
            keep = (z > MIN_JAW_DEPTH_MM) & (z < MAX_JAW_DEPTH_MM)
            ref_pixels = pa[keep]
            current_pixels = pb[keep].astype(float)
            xyz = backproject(self.K, ref_pixels, z[keep])
            if len(xyz) < MIN_INLIERS:
                raise ValueError(f"jaw registration has only {len(xyz)} matches with reliable reference depth")

            ok, rotation, translation, inliers = cv2.solvePnPRansac(
                xyz, current_pixels, self.K, None, iterationsCount=500,
                reprojectionError=1.8, flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) < MIN_INLIERS:
                count = 0 if inliers is None else len(inliers)
                raise ValueError(f"jaw registration is ambiguous ({count} PnP inliers)")
            indices = inliers.ravel()
            left = int((ref_pixels[indices, 0] < depth_mm.shape[1] / 2).sum())
            right = len(indices) - left
            if min(left, right) < MIN_INLIERS_PER_JAW:
                raise ValueError("jaw registration lacks support on both fingers")
            left_count, right_count, inlier_count = left, right, len(indices)

            # Reject spatially degenerate point sets and invalid/behind-camera solutions before
            # exposing a correction. Both fingers normally provide a broad, non-coplanar baseline.
            centered = xyz[indices] - xyz[indices].mean(axis=0)
            spread = np.linalg.svd(centered, compute_uv=False)
            if spread[-1] < 1.0 or spread[-1] / spread[0] < 0.01:
                raise ValueError("jaw PnP support is geometrically degenerate")
            rotation, translation = cv2.solvePnPRefineLM(
                xyz[indices], current_pixels[indices], self.K, None, rotation, translation,
            )
            current_from_reference = np.eye(4)
            current_from_reference[:3, :3] = cv2.Rodrigues(rotation)[0]
            current_from_reference[:3, 3] = translation.ravel()
            if not np.all(np.isfinite(current_from_reference)):
                raise ValueError("jaw PnP returned a non-finite transform")
            current_xyz = xyz[indices] @ current_from_reference[:3, :3].T + current_from_reference[:3, 3]
            if np.any(current_xyz[:, 2] <= 0):
                raise ValueError("jaw PnP placed reference fingers behind the current camera")
            projected, _ = cv2.projectPoints(xyz[indices], rotation, translation, self.K, None)
            residuals = np.linalg.norm(projected.reshape(-1, 2) - current_pixels[indices], axis=1)
            if not np.all(np.isfinite(residuals)) or float(np.percentile(residuals, 90)) > 2.0:
                raise ValueError("jaw PnP reprojection residual is too large")
            return WristMountMeasurement(
                True, np.linalg.inv(current_from_reference), mask, len(pa), len(indices), left, right,
                float(np.median(residuals)), float(np.percentile(residuals, 90)), None,
            )
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            try:
                mask = current_jaw_mask(depth_mm, self.arm)
            except (TypeError, ValueError):
                mask = empty_mask
            return WristMountMeasurement(False, None, mask, match_count, inlier_count,
                                         left_count, right_count, None, None, str(exc))


def refine_jaw_motion(ref_color: np.ndarray, ref_depth: np.ndarray, color: np.ndarray,
                      depth_mm: np.ndarray, K: np.ndarray, seed_current_to_reference: np.ndarray,
                      *, arm: Arm = "left") -> np.ndarray:
    """Refine camera motion on both open jaws using bidirectional Lucas–Kanade tracks.

    Returns reference-camera <- current-camera in mm. A failed registration raises ValueError.
    Only the reference depth sets the metric scale. Current depth gates near-field support.
    """
    from chem_demo.wrist_geometry import project, transform

    arm = _validate_arm(arm)
    _validate_observation(ref_color, ref_depth, K)
    _validate_observation(color, depth_mm, K)
    if ref_depth.shape != depth_mm.shape:
        raise ValueError("jaw images must have identical resolution")
    seed = np.asarray(seed_current_to_reference, dtype=float)
    if seed.shape != (4, 4) or not np.all(np.isfinite(seed)):
        raise ValueError("invalid jaw pose seed")
    if arm == "right":
        for frame in (ref_color, color):
            landmarks = _open_jaws(frame, arm)
            if not landmarks.ok or landmarks.state != "open":
                raise ValueError(f"jaw optical flow requires two visible open fingers: {landmarks.reason or landmarks.state}")
    gray_ref = cv2.cvtColor(ref_color, cv2.COLOR_BGR2GRAY)
    gray_now = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    height, width = ref_depth.shape
    if arm == "left":
        # Preserve the established left-camera feature support exactly.
        allowed = ((ref_depth > 55) & (ref_depth < 173)).astype(np.uint8)
        allowed[:, int(width * 0.38):int(width * 0.61)] = 0
        allowed[round(height * 7 / 12):] = 0
    else:
        allowed = (current_jaw_mask(ref_depth, arm) > 0).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(gray_ref, 1500, 0.002, 3, mask=allowed, blockSize=5)
    if pts is None:
        raise ValueError("no trackable jaw corners")
    uv_ref = pts.reshape(-1, 2)
    ref_z = sample_depth(ref_depth, uv_ref, window=1)
    valid = (ref_z > 50) & (ref_z < 175)
    uv_ref = uv_ref[valid]
    if len(uv_ref) < 10:
        raise ValueError("insufficient jaw depth for optical flow")
    xyz = backproject(K, uv_ref, ref_z[valid])
    predicted_xyz = transform(np.linalg.inv(seed), xyz)
    if np.any(predicted_xyz[:, 2] <= 0):
        raise ValueError("jaw pose seed puts corners behind camera")
    initial_pixels = project(K, predicted_xyz).astype(np.float32)
    flow = dict(winSize=(17, 17), maxLevel=2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    uv_now, forward_ok, patch_error = cv2.calcOpticalFlowPyrLK(gray_ref, gray_now, uv_ref, initial_pixels, **flow)
    uv_back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(gray_now, gray_ref, uv_now, uv_ref.copy(), **flow)
    inside = ((uv_now[:, 0] >= 0) & (uv_now[:, 0] < width - 1) &
              (uv_now[:, 1] >= 0) & (uv_now[:, 1] < height - 1))
    current_z = np.zeros(len(uv_now))
    current_z[inside] = sample_depth(depth_mm, uv_now[inside], window=1)
    reliable = (forward_ok.ravel().astype(bool) & reverse_ok.ravel().astype(bool) &
                (np.linalg.norm(uv_back - uv_ref, axis=1) < 0.6) & (patch_error.ravel() < 15) &
                (current_z > 50) & (current_z < 175))
    xyz, pixels = xyz[reliable], uv_now[reliable]
    if len(pixels) < 10:
        raise ValueError("insufficient bidirectional jaw tracks")
    ok, rotation, translation, indices = cv2.solvePnPRansac(
        xyz, pixels, K, None, iterationsCount=300, reprojectionError=1.3, flags=cv2.SOLVEPNP_EPNP)
    if not ok or indices is None or len(indices) < 10:
        raise ValueError("insufficient optical-flow PnP consensus")
    idx = indices.ravel()
    left = np.count_nonzero(pixels[idx, 0] < width / 2)
    if min(left, len(idx) - left) < 3:
        raise ValueError("optical-flow pose needs support from both jaws")
    spread = np.linalg.svd(xyz[idx] - xyz[idx].mean(axis=0), compute_uv=False)
    if spread[1] < 1.0 or spread[1] / spread[0] < 0.01:
        raise ValueError("optical-flow jaw points are nearly collinear")
    rotation, translation = cv2.solvePnPRefineLM(xyz[idx], pixels[idx], K, None, rotation, translation)
    current_from_reference = np.eye(4)
    current_from_reference[:3, :3] = cv2.Rodrigues(rotation)[0]
    current_from_reference[:3, 3] = translation.ravel()
    current_xyz = transform(current_from_reference, xyz[idx])
    if not np.all(np.isfinite(current_xyz)) or np.any(current_xyz[:, 2] <= 0):
        raise ValueError("invalid refined jaw pose")
    residuals = np.linalg.norm(project(K, current_xyz) - pixels[idx], axis=1)
    if np.percentile(residuals, 90) > 1.5:
        raise ValueError("refined jaw pose has excessive image error")
    return np.linalg.inv(current_from_reference)


__all__ = ["WristMountMeasurement", "WristMountReference", "current_jaw_mask", "refine_jaw_motion"]
