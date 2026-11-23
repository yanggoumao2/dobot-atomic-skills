"""Measure the visible pad tips of an open wrist gripper in a camera frame.

The current camera view has two dark gripper pads entering from the top edge. This module locates
their silhouettes independently in each frame and returns the center of each pad's deepest visible
edge. It accepts open grippers only: a confidently close pair is reported as ``closed`` and rejected,
while missing, overlapping, or weak silhouettes are reported as ``ambiguous`` and rejected.

The default template is calibrated for the left wrist view used by ``chem_demo``. Supply another
``CameraTemplate`` for a different camera, resolution-independent ROI, or gripper appearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


Point2D = tuple[float, float]
JawState = Literal["open", "closed", "ambiguous"]


@dataclass(frozen=True)
class CameraTemplate:
    """Per-camera silhouette and location priors; ROIs are normalized to image width/height."""

    dark_threshold: int = 55
    left_center_x: tuple[float, float] = (0.18, 0.45)
    right_center_x: tuple[float, float] = (0.55, 0.84)
    max_component_top_y: float = 0.01
    max_tip_y: float = 0.68
    min_component_area: float = 0.005  # fraction of full image area
    min_component_height: float = 0.15  # fraction of image height
    min_component_width: float = 0.05  # fraction of image width
    min_aspect_ratio: float = 0.30
    max_aspect_ratio: float = 1.80
    max_tip_band_width: float = 0.45  # fraction of component width
    closed_gap_width: float = 0.04
    open_gap_width: float = 0.10
    bottom_band_height: float = 0.008  # fraction of image height
    min_candidate_ratio: float = 1.8  # winner area / second candidate area
    min_confidence: float = 0.65


DEFAULT_TEMPLATE = CameraTemplate()


@dataclass(frozen=True)
class GripperLandmarks:
    """Result of one open-gripper pad-tip measurement.

    ``ok`` is true only when both pad tips are confidently visible and the jaws are open. Point
    values are image pixels in (x, y) order. Failed results may retain measured points for
    diagnostics, but callers must gate their use on ``ok``.
    """

    ok: bool
    state: JawState
    left: Point2D | None
    right: Point2D | None
    gap_px: float | None
    confidence: float
    reason: str | None = None

    @property
    def points(self) -> dict[str, Point2D] | None:
        """Convenience mapping for callers that need both accepted landmarks."""
        if self.left is None or self.right is None:
            return None
        return {"left": self.left, "right": self.right}


@dataclass(frozen=True)
class _PadCandidate:
    point: Point2D
    area: int
    component_width: int
    component_height: int
    bottom_band_width: int


def _gray_u8(image: np.ndarray) -> np.ndarray:
    """Accept a uint8 gray, BGR, or BGRA image and return a grayscale view/copy."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 NumPy array")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("image must have shape (height, width), (height, width, 3), or (..., 4)")
    if min(gray.shape) < 16:
        raise ValueError("image dimensions are too small for gripper landmark detection")
    return gray


def _find_candidates(gray: np.ndarray, template: CameraTemplate) -> dict[str, list[_PadCandidate]]:
    height, width = gray.shape
    binary = (gray <= template.dark_threshold).astype(np.uint8)
    max_y = max(1, int(round(template.max_tip_y * height)))
    binary[max_y + 1 :, :] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    area_limit = max(1, int(round(template.min_component_area * height * width)))
    min_height = max(1, int(round(template.min_component_height * height)))
    min_width = max(1, int(round(template.min_component_width * width)))
    top_limit = max(0, int(round(template.max_component_top_y * height)))
    band_rows = max(2, int(round(template.bottom_band_height * height)))

    by_side: dict[str, list[_PadCandidate]] = {"left": [], "right": []}
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < area_limit or y > top_limit or w < min_width or h < min_height:
            continue
        aspect = w / h
        if not template.min_aspect_ratio <= aspect <= template.max_aspect_ratio:
            continue

        x_center = float(centroids[label, 0]) / width
        side = None
        if template.left_center_x[0] <= x_center <= template.left_center_x[1]:
            side = "left"
        elif template.right_center_x[0] <= x_center <= template.right_center_x[1]:
            side = "right"
        if side is None:
            continue

        # Work only in the component's bounding box. The center of the final few silhouette rows
        # is more stable than selecting a single potentially noisy extreme pixel.
        roi = labels[y : y + h, x : x + w] == label
        ys, xs = np.nonzero(roi)
        if not len(ys):
            continue
        bottom = int(ys.max())
        bottom_rows = ys >= max(0, bottom - band_rows + 1)
        bottom_xs = xs[bottom_rows]
        band_width = int(bottom_xs.max() - bottom_xs.min() + 1)
        if band_width / w > template.max_tip_band_width:
            continue
        point = (float(x + np.mean(bottom_xs)), float(y + bottom))
        if point[1] / height > template.max_tip_y:
            continue
        by_side[side].append(
            _PadCandidate(point, area, w, h, band_width)
        )
    return by_side


def _choose(candidates: list[_PadCandidate], template: CameraTemplate) -> _PadCandidate | None:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: item.area, reverse=True)
    if len(ranked) > 1 and ranked[0].area < template.min_candidate_ratio * ranked[1].area:
        return None
    return ranked[0]


def _candidate_confidence(candidate: _PadCandidate, image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    image_area = height * width
    area_score = min(1.0, candidate.area / max(1.0, image_area * 0.018))
    height_score = min(1.0, candidate.component_height / max(1.0, 0.3 * height))
    tip_score = max(0.0, 1.0 - candidate.bottom_band_width / candidate.component_width)
    return float(np.clip(0.40 * area_score + 0.20 * height_score + 0.40 * tip_score, 0.0, 1.0))


def measure_open_pad_tips(
    image: np.ndarray,
    *,
    template: CameraTemplate = DEFAULT_TEMPLATE,
) -> GripperLandmarks:
    """Measure both visible pad tips in an open gripper image.

    The selected landmark is the center of the pad silhouette's deepest few rows. The detector
    rejects missing or competing components, malformed pad shapes, and jaw gaps in the uncertain
    range between the calibrated closed and open thresholds.
    """
    gray = _gray_u8(image)
    _, width = gray.shape
    candidates = _find_candidates(gray, template)
    left = _choose(candidates["left"], template)
    right = _choose(candidates["right"], template)

    if left is None or right is None:
        return GripperLandmarks(
            ok=False,
            state="ambiguous",
            left=left.point if left else None,
            right=right.point if right else None,
            gap_px=None,
            confidence=0.0,
            reason="could not identify one unambiguous pad silhouette on each side",
        )

    left_point, right_point = left.point, right.point
    gap = float(np.linalg.norm(np.subtract(left_point, right_point)))
    closed_limit = template.closed_gap_width * width
    open_limit = template.open_gap_width * width
    confidence = min(_candidate_confidence(left, gray.shape),
                     _candidate_confidence(right, gray.shape))

    if left_point[0] >= right_point[0]:
        return GripperLandmarks(False, "ambiguous", left_point, right_point, gap, confidence,
                                "pad tip ordering is inconsistent with the camera template")
    if gap <= closed_limit:
        return GripperLandmarks(False, "closed", left_point, right_point, gap, confidence,
                                "gripper is closed or nearly closed; open-jaw landmarks are unsupported")
    if gap < open_limit:
        return GripperLandmarks(False, "ambiguous", left_point, right_point, gap, confidence,
                                "jaw gap is between the closed and open acceptance thresholds")
    if confidence < template.min_confidence:
        return GripperLandmarks(False, "ambiguous", left_point, right_point, gap, confidence,
                                f"pad-tip confidence {confidence:.3f} is below {template.min_confidence:.3f}")
    return GripperLandmarks(True, "open", left_point, right_point, gap, confidence)


__all__ = [
    "CameraTemplate",
    "DEFAULT_TEMPLATE",
    "GripperLandmarks",
    "measure_open_pad_tips",
]
