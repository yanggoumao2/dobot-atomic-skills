"""Fast OpenCV neck-ring localization for the known clear bottle.

This module is intentionally model-free and read-only. Top-camera pixels are intersected with a
fixed *left-base ring plane* in the corrected camera model, then transformed to the right base.
The image detector scans a calibrated portion of the left arm's white-table work area; no bottle
pixel or XY target is supplied to it. Wrist detection is independent of jaw state so it can still
report a visible blue rim with a closed gripper. A separate hover-pose gate checks the rim and open
jaw landmarks against the captured accepted hover view.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from chem_demo.camera_drift import load_corrected_camera
from dataclasses import replace

from chem_demo.wrist_landmarks import DEFAULT_TEMPLATE, GripperLandmarks, measure_open_pad_tips
from vision_pick.cup_locate import Camera, DEFAULT_SESSION


RING_PLANE_Z_MM = 377.49  # left-base apparent ring plane from the paired neck-ray check
# Calibrated left-base XY bounds of the white-table work patch. The rectangle is intentionally
# broad enough for normal placement variation; it excludes the left cap/rack and right-side clutter.
LEFT_WHITE_WORK_AREA_MM = (100.0, 430.0, -600.0, -180.0)  # xmin, xmax, ymin, ymax

# Captured at the accepted left-arm hover (Z=485 mm). These are image evidence for *that pose*,
# not a bottle target. Axes from the capture record are normalised below to (major, minor).
# Left wrist camera as finally fixed on 2026-11-22 21:3x: it sits tilted up, the pad tips are near the
# top edge and only partly visible, so the silhouette-shape confidence settles at 0.59 (two runs gave
# the same tips to 0.1 px). The hover gate accepts 0.55 for this camera; the rim is still compared to
# the jaws measured in the same image.
LEFT_HOVER_TEMPLATE = replace(DEFAULT_TEMPLATE, min_confidence=0.55)
# Re-baselined 2026-11-22 19:3x after Kunyu re-tightened the left wrist camera bracket: the whole
# view moved ~28 px up (jaw gap unchanged, 261.7 px) while the rim relative to the jaw midpoint stayed
# within 0.94 px of the original commissioning reference, which was
#   centre (422.33, 220.28), axes (66.82, 62.72), jaws (297.10, 128.0) / (557.87, 147.0).
#   19:3x values: centre (422.88, 192.01), axes (69.17, 63.80), jaws (298.44, 99.0) / (559.43, 119.0).
# Re-baselined again 21:3x from run-20261122_213106-c406a96331/03_wrist.png after the bracket was
# fixed for good; rim relative to the jaw midpoint (-4.97, 84.77) px vs (-5.15, 82.78) originally.
HOVER_REFERENCE_CENTER_PX = (429.3338623046875, 172.27313232421875)
HOVER_REFERENCE_AXES_PX = (67.30310821533203, 63.61422348022461)
HOVER_REFERENCE_JAWS_PX = (
    (303.4583333333333, 81.0),
    (565.1515151515151, 94.0),
)


@dataclass(frozen=True)
class RimEllipse:
    """One fitted ring ellipse. ``axes_px`` are ordered (major, minor)."""

    center_px: tuple[float, float]
    axes_px: tuple[float, float]
    angle_deg: float
    score: float


@dataclass(frozen=True)
class TopRimResult:
    ok: bool
    reason: str | None
    rim: RimEllipse | None
    right_base_xyz_mm: tuple[float, float, float] | None
    left_base_xyz_mm: tuple[float, float, float] | None
    candidate_count: int
    seconds: float

    @property
    def x(self) -> float | None:
        """Detected left-base X in mm, or ``None`` when localization was rejected."""
        return None if self.left_base_xyz_mm is None else self.left_base_xyz_mm[0]

    @property
    def y(self) -> float | None:
        """Detected left-base Y in mm, or ``None`` when localization was rejected."""
        return None if self.left_base_xyz_mm is None else self.left_base_xyz_mm[1]

    @property
    def z(self) -> float | None:
        """Apparent ring-plane Z in left-base mm, or ``None`` when localization was rejected."""
        return None if self.left_base_xyz_mm is None else self.left_base_xyz_mm[2]


@dataclass(frozen=True)
class WristRimResult:
    ok: bool
    reason: str | None
    rim: RimEllipse | None
    candidate_count: int
    seconds: float


@dataclass(frozen=True)
class WristHoverGate:
    ok: bool
    reason: str | None
    rim: RimEllipse | None
    jaw_landmarks: GripperLandmarks
    center_error_px: float | None
    axes_error_px: tuple[float, float] | None
    jaw_errors_px: tuple[float, float] | None
    # Rim minus where it should be relative to the jaws seen in this image (u right, v down), and how
    # the jaws moved in the frame since the reference (= how far the camera bracket crept).
    offset_px: tuple[float, float] | None = None
    camera_shift_px: tuple[float, float] | None = None
    jaws_used: str | None = None
    notes: tuple[str, ...] = ()


def _validate_bgr(image: np.ndarray) -> None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or
            image.shape[2] != 3 or min(image.shape[:2]) < 64):
        raise ValueError("image must be a uint8 BGR image at least 64x64 pixels")


def _ellipse(contour: np.ndarray, score: float) -> RimEllipse | None:
    if len(contour) < 5:
        return None
    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    if axis_a <= 0 or axis_b <= 0:
        return None
    if axis_a >= axis_b:
        major, minor, major_angle = axis_a, axis_b, angle
    else:
        major, minor, major_angle = axis_b, axis_a, angle + 90.0
    return RimEllipse((float(cx), float(cy)), (float(major), float(minor)),
                      float(major_angle % 180.0), float(np.clip(score, 0.0, 1.0)))


def _ellipse_coordinates(shape: tuple[int, int], rim: RimEllipse) -> np.ndarray:
    height, width = shape
    cx, cy = rim.center_px
    major, minor = rim.axes_px
    angle = math.radians(rim.angle_deg)
    yy, xx = np.ogrid[:height, :width]
    dx, dy = xx - cx, yy - cy
    xr = dx * math.cos(angle) + dy * math.sin(angle)
    yr = -dx * math.sin(angle) + dy * math.cos(angle)
    return np.sqrt((xr / (major / 2.0)) ** 2 + (yr / (minor / 2.0)) ** 2)


def _zone_profile(gray: np.ndarray, hsv: np.ndarray, rim: RimEllipse) -> tuple[float, float, float]:
    """Return dark-core, dark-ring, and white-table-surround fractions."""
    radial = _ellipse_coordinates(gray.shape, rim)
    core = radial < 0.52
    ring = (radial >= 0.72) & (radial <= 1.16)
    surround = (radial >= 1.22) & (radial <= 1.72)
    if not core.any() or not ring.any() or not surround.any():
        return 1.0, 0.0, 0.0
    dark = gray < 125
    white = (hsv[..., 1] < 100) & (hsv[..., 2] > 100)
    return (float(np.mean(dark[core])), float(np.mean(dark[ring])),
            float(np.mean(white[surround])))


def _left_roi_polygon(left_camera: Camera, plane_z_mm: float,
                      work_area_mm: Sequence[float]) -> np.ndarray:
    if len(work_area_mm) != 4:
        raise ValueError("work_area_mm must be (xmin, xmax, ymin, ymax)")
    xmin, xmax, ymin, ymax = map(float, work_area_mm)
    if not np.isfinite([xmin, xmax, ymin, ymax, plane_z_mm]).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError("work area and ring plane must be finite with increasing XY bounds")
    corners_left = np.array([
        [xmin, ymin, plane_z_mm], [xmax, ymin, plane_z_mm],
        [xmax, ymax, plane_z_mm], [xmin, ymax, plane_z_mm],
    ], dtype=float)
    # Plane height and ROI bounds are left-base quantities; use that frame's camera directly.
    return left_camera.project(corners_left).astype(np.float32)


def _inside(poly: np.ndarray, point: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(poly.reshape(-1, 1, 2), point, False) >= 0


def locate_top_rim(
    image: np.ndarray,
    *,
    right_camera: Camera | None = None,
    left_camera: Camera | None = None,
    plane_z_mm: float = RING_PLANE_Z_MM,
    work_area_mm: Sequence[float] = LEFT_WHITE_WORK_AREA_MM,
) -> TopRimResult:
    """Detect one dark bottle neck ring and map its center to both arm base frames.

    The top D405 image is expected in the corrected 1280x720 orientation. The pixel ellipse is
    localized by dark annular contrast plus a white-table surround; the calibrated left work area
    restricts candidates before accepting exactly one plausible ring. The returned Z is the known
    apparent left-base ring plane, not a measured bottle height or contact surface.
    """
    _validate_bgr(image)
    started = time.perf_counter()
    if right_camera is None or left_camera is None:
        default_right = load_corrected_camera(DEFAULT_SESSION, "right")
        default_left = load_corrected_camera(DEFAULT_SESSION, "left")
        right_camera = right_camera or default_right
        left_camera = left_camera or default_left
    height, width = image.shape[:2]
    if (width, height) != (1280, 720):
        raise ValueError(f"corrected top camera expects 1280x720 pixels, got {width}x{height}")
    roi = _left_roi_polygon(left_camera, plane_z_mm, work_area_mm)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates: list[RimEllipse] = []
    for threshold in (80, 95, 110, 125):
        mask = cv2.inRange(gray, 0, threshold)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < 55 or len(contour) < 12:
                continue
            rim = _ellipse(contour, 0.0)
            if rim is None:
                continue
            major, minor = rim.axes_px
            circularity = minor / major
            if not (20.0 <= minor <= 62.0 and major <= 70.0 and circularity >= 0.70):
                continue
            if not _inside(roi, rim.center_px):
                continue
            left_xyz = left_camera.ray_to_plane(rim.center_px, float(plane_z_mm))
            xmin, xmax, ymin, ymax = map(float, work_area_mm)
            if not (xmin <= left_xyz[0] <= xmax and ymin <= left_xyz[1] <= ymax):
                continue
            dark_core, dark_ring, white_surround = _zone_profile(gray, hsv, rim)
            # The cap and orange rack holes are dark blobs, but lack a clean bright core and white
            # table around an annular dark rim.
            if dark_core > 0.30 or dark_ring < 0.42 or white_surround < 0.40:
                continue
            score = (0.45 * dark_ring + 0.30 * (1.0 - dark_core) +
                     0.15 * white_surround + 0.10 * circularity)
            candidates.append(RimEllipse(rim.center_px, rim.axes_px, rim.angle_deg, score))

    # Multiple thresholds see the same ring. Collapse near-identical fits before the ambiguity gate.
    candidates.sort(key=lambda item: item.score, reverse=True)
    unique: list[RimEllipse] = []
    for candidate in candidates:
        if all(math.dist(candidate.center_px, kept.center_px) > 5.0 for kept in unique):
            unique.append(candidate)
    elapsed = time.perf_counter() - started
    if not unique:
        return TopRimResult(False, "no plausible dark annular rim in the calibrated left-table work area",
                            None, None, None, 0, elapsed)
    if len(unique) > 1 and unique[1].score >= unique[0].score - 0.08:
        return TopRimResult(False, "multiple plausible bottle rims in the calibrated work area",
                            None, None, None, len(unique), elapsed)
    best = unique[0]
    T_right_left = right_camera.T_base_camera @ np.linalg.inv(left_camera.T_base_camera)
    left_xyz = left_camera.ray_to_plane(best.center_px, float(plane_z_mm))
    right_xyz = T_right_left @ np.r_[left_xyz, 1.0]
    return TopRimResult(True, None, best,
                        tuple(float(v) for v in right_xyz[:3]), tuple(float(v) for v in left_xyz),
                        len(unique), elapsed)


def _nested_wrist_candidates(image: np.ndarray, *, saturated_only: bool = False) -> list[RimEllipse]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Broad blue/cyan hue range handles the blue ring under exposure changes. Geometric nesting
    # and annular color coverage remove the blue robot links and unrelated scene patches.
    # At contact the dark rubber pads can join a broad blue mask. A second pass
    # isolates the saturated rim while retaining the same annular geometry tests.
    lower, upper = ((85, 120, 20), (135, 255, 255)) if saturated_only else ((70, 15, 20), (145, 255, 250))
    blue = cv2.inRange(hsv, lower, upper)
    closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    h, w = blue.shape
    radial_blue = blue != 0
    results: list[RimEllipse] = []
    for index, contour in enumerate(contours):
        next_index, prev_index, child_index, parent_index = map(int, hierarchy[0][index])
        if parent_index != -1 or child_index == -1 or cv2.contourArea(contour) < 300:
            continue
        outer = _ellipse(contour, 0.0)
        # Small specular holes may precede the actual opening in OpenCV's sibling list.
        child_contour = contours[child_index]
        sibling = hierarchy[0][child_index][0]
        while sibling >= 0:
            if cv2.contourArea(contours[sibling]) > cv2.contourArea(child_contour):
                child_contour = contours[sibling]
            sibling = hierarchy[0][sibling][0]
        inner = _ellipse(child_contour, 0.0)
        if outer is None or inner is None:
            continue
        major, minor = outer.axes_px
        inner_major, inner_minor = inner.axes_px
        if not (26.0 <= minor <= 125.0 and major <= 145.0 and minor / major >= 0.50):
            continue
        if cv2.contourArea(child_contour) / max(cv2.contourArea(contour), 1.0) < 0.35:
            continue
        if not (0.55 <= inner_major / major <= 0.98 and 0.55 <= inner_minor / minor <= 0.98):
            continue
        concentric_error = math.dist(outer.center_px, inner.center_px) / max(minor, 1.0)
        if concentric_error > 0.18:
            continue
        cx, cy = outer.center_px
        if not (0.04 * w <= cx <= 0.96 * w and 0.04 * h <= cy <= 0.96 * h):
            continue
        radial = _ellipse_coordinates((h, w), outer)
        annulus = (radial >= 0.72) & (radial <= 1.18)
        blue_share = float(np.mean(radial_blue[annulus])) if annulus.any() else 0.0
        if blue_share < 0.12:
            continue
        ring_ratio = min(minor / major, inner_minor / inner_major)
        score = (0.45 * min(blue_share / 0.45, 1.0) +
                 0.25 * min(cv2.contourArea(child_contour) / cv2.contourArea(contour), 1.0) +
                 0.20 * max(0.0, 1.0 - concentric_error / 0.18) + 0.10 * ring_ratio)
        results.append(RimEllipse(outer.center_px, outer.axes_px, outer.angle_deg, score))
    results.sort(key=lambda candidate: candidate.score, reverse=True)
    return results


def detect_wrist_rim(image: np.ndarray) -> WristRimResult:
    """Find a unique blue bottle rim anywhere in the wrist image, without jaw-state assumptions.

    This also works when the jaw detector would report closed/ambiguous. It only reports a visible
    ring ellipse; callers must apply pose/occlusion checks before treating it as tracking evidence.
    """
    _validate_bgr(image)
    started = time.perf_counter()
    candidates = _nested_wrist_candidates(image)
    if not candidates:
        candidates = _nested_wrist_candidates(image, saturated_only=True)
    elapsed = time.perf_counter() - started
    if not candidates:
        return WristRimResult(False, "no plausible blue annular rim", None, 0, elapsed)
    if len(candidates) > 1 and candidates[1].score >= candidates[0].score * 0.90:
        return WristRimResult(False, "multiple plausible blue annular rims", None, len(candidates), elapsed)
    return WristRimResult(True, None, candidates[0], len(candidates), elapsed)


MAX_CAMERA_SHIFT_PX = 60.0   # beyond this the bracket moved so much that the reference relation is doubtful
LONE_PAD_WINDOW_PX = 45.0    # a single pad is trusted only near where the pads have been seen


def compare_wrist_to_hover(
    image: np.ndarray,
    *,
    center_tolerance_px: float = 12.0,
    axes_tolerance_px: float = 12.0,
    jaw_tolerance_px: float = 25.0,
) -> WristHoverGate:
    """Where the rim sits relative to the open jaws, against the accepted hover relation.

    Relative, not absolute: the rim is compared with the jaws seen in the SAME image, so a camera
    that crept on its bracket (19:3x, 21:3x and 22:47 on 2026-11-22, 7-15 px each time) changes
    nothing as long as the jaws are still seen. One pad is enough: when the other is cut off by the
    image edge (22:47), the visible one carries the camera shift, provided it lies within
    ``LONE_PAD_WINDOW_PX`` of the accepted camera-shift estimate from the other pad's reference.
    A camera shift is a note, not a refusal (up to ``MAX_CAMERA_SHIFT_PX``). ``ok`` means the rim is
    within ``center_tolerance_px`` of the jaw-centred target; ``offset_px`` lets the caller correct
    the arm and look again instead of giving up.
    """
    rim_result = detect_wrist_rim(image)
    landmarks = measure_open_pad_tips(image, template=LEFT_HOVER_TEMPLATE)
    if not rim_result.ok or rim_result.rim is None:
        return WristHoverGate(False, rim_result.reason, None, landmarks, None, None, None)
    rim = rim_result.rim
    notes: list[str] = []
    ref_left, ref_right = HOVER_REFERENCE_JAWS_PX
    if landmarks.ok and landmarks.left is not None and landmarks.right is not None:
        shifts = [np.subtract(landmarks.left, ref_left), np.subtract(landmarks.right, ref_right)]
        shift = (shifts[0] + shifts[1]) / 2.0
        used = "both"
        jaw_errors = (float(np.linalg.norm(shifts[0])), float(np.linalg.norm(shifts[1])))
    else:
        # The detector still reports a pad it found on one side; accept it only if it moved like a
        # camera shift would move it (the rim moved by about the same amount).
        rim_shift = np.subtract(rim.center_px, HOVER_REFERENCE_CENTER_PX)
        lone = []
        for point, ref, name in ((landmarks.left, ref_left, "left"), (landmarks.right, ref_right, "right")):
            if point is not None:
                pad_shift = np.subtract(point, ref)
                if np.linalg.norm(pad_shift - rim_shift) <= LONE_PAD_WINDOW_PX:
                    lone.append((name, pad_shift))
        if len(lone) != 1:
            return WristHoverGate(False, "open-jaw landmarks are not confidently visible (no usable pad)", rim,
                                  landmarks, None, None, None)
        used, shift = lone[0][0], lone[0][1]
        jaw_errors = None
        notes.append(f"only the {used} pad tip is visible; the other is outside the image")
    if float(np.linalg.norm(shift)) > MAX_CAMERA_SHIFT_PX:
        return WristHoverGate(False, f"wrist camera moved {np.linalg.norm(shift):.0f}px on its bracket; fix it",
                              rim, landmarks, None, None, jaw_errors, camera_shift_px=tuple(map(float, shift)),
                              jaws_used=used)
    if float(np.linalg.norm(shift)) > jaw_tolerance_px:
        notes.append(f"wrist camera crept {np.linalg.norm(shift):.0f}px on its bracket (tolerated, relative check)")
    target = np.add(HOVER_REFERENCE_CENTER_PX, shift)
    offset = np.subtract(rim.center_px, target)
    center_error = float(np.linalg.norm(offset))
    axes_error = tuple(abs(a - b) for a, b in zip(rim.axes_px, HOVER_REFERENCE_AXES_PX))
    if center_error > center_tolerance_px:
        reason = f"rim center relative to the jaws differs from accepted hover by {center_error:.1f}px"
    elif max(axes_error) > axes_tolerance_px:
        reason = f"rim size differs from accepted hover by {max(axes_error):.1f}px"
    else:
        reason = None
    return WristHoverGate(reason is None, reason, rim, landmarks, center_error, axes_error, jaw_errors,
                          offset_px=(float(offset[0]), float(offset[1])),
                          camera_shift_px=(float(shift[0]), float(shift[1])), jaws_used=used, notes=tuple(notes))


__all__ = [
    "HOVER_REFERENCE_AXES_PX",
    "HOVER_REFERENCE_CENTER_PX",
    "HOVER_REFERENCE_JAWS_PX",
    "LEFT_WHITE_WORK_AREA_MM",
    "RING_PLANE_Z_MM",
    "RimEllipse",
    "TopRimResult",
    "WristHoverGate",
    "WristRimResult",
    "compare_wrist_to_hover",
    "detect_wrist_rim",
    "locate_top_rim",
]
