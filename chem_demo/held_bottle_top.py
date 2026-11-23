"""Measure the bottle hanging from the left jaws in one top-camera frame (read-only, offline-testable).

Why: the handoff plan hangs the bottle below the left flange from the commissioned neck grasp. That
is a prediction; the top camera can check it before the right arm moves (slip, tilt, a neck taken
off-centre). Kunyu 2026-11-22: measure, do not assume.

Method. The bottle is an upright cylinder of known radius and height (measured at the pick:
body ~68 mm, rim plane 134.3 mm above the bottom). With both sizes fixed, one oblique view fixes
the axis XY and the bottom height: the silhouette's size gives the distance along the ray. The
mask comes from Grounding DINO + SAM 2 on an enlarged crop around the predicted silhouette
(chem_demo.locate.detect_in_crop); the left jaws hide the neck, so contour points the model cannot
explain are trimmed and the fit is repeated (as in chem_demo.locate.fit_cylinder).

Coordinates: the left arm's base frame in the fingertip convention (cw) of the top-camera
calibration, i.e. the numbers of chem_demo.gripper_geometry.TABLE_Z_CW. The prediction for a left
flange at Z F is: axis at the flange XY, bottom at F - (NECK_GRASP_FLANGE_Z_CW - TABLE_Z_CW["left"]).
Both are cw numbers, so the flange->tips length does not enter the comparison.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from chem_demo.gripper_geometry import NECK_GRASP_FLANGE_Z_CW, RING_PLANE_Z_CW, TABLE_Z_CW
from vision_pick.cup_locate import Camera, frustum_silhouette

# Effective radius for the SAM mask outline: 32 mm, not the 34 mm of the 68 mm body. With 34 all six
# 2026-11-22 lift frames fitted ~50 mm too far along the ray (mask edges sit inside the glass edge);
# with 32 the axis is within 5 mm of the flange axis and the bottom within 7 mm of the known hang
# (04 frames on the table, 05 frames lifted 30 mm, measured lift 24-34 mm). Calibrated on those same frames.
BODY_RADIUS_MM = 32.0
BOTTLE_HEIGHT_MM = RING_PLANE_Z_CW - TABLE_Z_CW["left"]          # 134.3, measured at the pick
TIPS_ABOVE_BOTTOM_MM = NECK_GRASP_FLANGE_Z_CW - TABLE_Z_CW["left"]  # 115.7
VOCABULARY = {"glass bottle": "bottle", "glass jar": "bottle", "bottle": "bottle"}
# Only a grossly wrong hang (slipped bottle, wrong object) stops the handoff; anything smaller is left
# to the right wrist's close-range fine alignment, which may shift the hand up to 40 mm.
MAX_XY_OFFSET_MM = 40.0
MAX_BOTTOM_OFFSET_MM = 40.0
MAX_RMS_PX = 4.0


@dataclass
class HeldBottle:
    ok: bool
    reason: str | None = None
    predicted_axis_xy: tuple[float, float] | None = None
    predicted_bottom_z: float | None = None
    axis_xy: tuple[float, float] | None = None
    bottom_z: float | None = None
    xy_offset_mm: tuple[float, float] | None = None
    bottom_offset_mm: float | None = None
    rms_px: float | None = None
    kept_share: float | None = None
    mask_iou_with_prediction: float | None = None
    mask_score: float | None = None
    frame: str = "left base, fingertip convention (cw), mm"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predicted_hang(flange_pose: Sequence[float]) -> tuple[np.ndarray, float]:
    x, y, z = (float(v) for v in flange_pose[:3])
    return np.array([x, y]), z - TIPS_ABOVE_BOTTOM_MM


def _silhouette(camera: Camera, x: float, y: float, bottom_z: float) -> np.ndarray:
    return frustum_silhouette(camera, [x, y, BOTTLE_HEIGHT_MM, BODY_RADIUS_MM, BODY_RADIUS_MM], bottom_z)


def _fill(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    return mask


def _contour(mask: np.ndarray, count: int = 300) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    return points[np.linspace(0, len(points) - 1, min(count, len(points))).astype(int)]


def fit_hanging_cylinder(camera: Camera, mask: np.ndarray, start: Sequence[float], *, trim: float = 0.15,
                         rounds: int = 3, lower_share: float = 0.6) -> tuple[np.ndarray, float, float]:
    """(x, y, bottom_z) of a fixed-size upright cylinder whose outline matches the mask edge.

    Only the lower ``lower_share`` of the mask outline (along the image direction of world-down) is
    used: the upper edge is where the left jaws hide the neck, an occlusion boundary that pulled a
    fixed-size model ~200 mm away along the ray in the first try (all six 2026-11-22 lift frames).
    The sides give the scale (known body width), the bottom gives the height. Levenberg-Marquardt on
    signed contour-to-outline pixel distances; after each round the worst ``trim`` share is dropped.
    Returns params, RMS px over the kept points, and the kept share.
    """
    points = _contour(mask)
    a = camera.project(np.array([[start[0], start[1], start[2]], [start[0], start[1], start[2] - 50.0]]))
    down = (a[1] - a[0]) / max(np.linalg.norm(a[1] - a[0]), 1e-9)
    along = points @ down
    points = points[along >= along.max() - lower_share * (along.max() - along.min())]
    params = np.asarray(start, float).copy()
    keep = np.ones(len(points), bool)

    def residuals(p: np.ndarray, pts: np.ndarray) -> np.ndarray:
        outline = _silhouette(camera, *p).astype(np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(outline, (float(a), float(b)), True) for a, b in pts])

    for round_index in range(rounds):
        pts = points[keep]
        damping, current = 1e-2, residuals(params, pts)
        for _ in range(60):
            jacobian = np.empty((len(pts), 3))
            for k in range(3):
                delta = np.zeros(3)
                delta[k] = 0.2
                jacobian[:, k] = (residuals(params + delta, pts) - residuals(params - delta, pts)) / 0.4
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


def choose_mask(detections: Sequence[Any], predicted: np.ndarray) -> tuple[Any | None, float]:
    """The bottle mask overlapping the predicted silhouette most (IoU), or (None, 0)."""
    best, best_iou = None, 0.0
    for d in detections:
        if getattr(d, "role", "bottle") != "bottle":
            continue
        m = np.asarray(d.mask).astype(bool)
        union = (m | predicted).sum()
        iou = float((m & predicted).sum() / union) if union else 0.0
        if iou > best_iou:
            best, best_iou = d, iou
    return best, best_iou


def measure_held_bottle(image: np.ndarray, left_camera: Camera, flange_pose: Sequence[float], segmenter,
                        *, margin_px: int = 40) -> tuple[HeldBottle, np.ndarray | None]:
    """Detect and fit the hanging bottle; returns the result and the chosen mask (for drawing)."""
    from chem_demo.locate import detect_in_crop

    xy, bottom = predicted_hang(flange_pose)
    base = {"predicted_axis_xy": (float(xy[0]), float(xy[1])), "predicted_bottom_z": float(bottom)}
    outline = _silhouette(left_camera, xy[0], xy[1], bottom)
    h, w = image.shape[:2]
    if not ((outline[:, 0] > 0) & (outline[:, 0] < w) & (outline[:, 1] > 0) & (outline[:, 1] < h)).all():
        return HeldBottle(ok=False, reason="predicted bottle is not fully inside the top image", **base), None
    predicted = _fill(outline, (h, w)).astype(bool)
    x0, y0 = outline.min(axis=0)
    x1, y1 = outline.max(axis=0)
    box = [x0 - margin_px, y0 - margin_px, x1 + margin_px, y1 + margin_px]
    found = detect_in_crop(segmenter, image, box, VOCABULARY, margin=0, threshold=0.2)
    chosen, iou = choose_mask(found, predicted)
    if chosen is None or iou < 0.2:
        return HeldBottle(ok=False, reason=f"no bottle mask near the predicted hang (best IoU {iou:.2f})",
                          mask_iou_with_prediction=iou, **base), None
    mask = cv2.morphologyEx(np.asarray(chosen.mask, np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    params, rms, kept = fit_hanging_cylinder(left_camera, mask, [xy[0], xy[1], bottom])
    offset = params[:2] - xy
    result = HeldBottle(ok=True, axis_xy=(float(params[0]), float(params[1])), bottom_z=float(params[2]),
                        xy_offset_mm=(float(offset[0]), float(offset[1])),
                        bottom_offset_mm=float(params[2] - bottom), rms_px=rms, kept_share=kept,
                        mask_iou_with_prediction=iou, mask_score=float(chosen.score), **base)
    failed = []
    if rms > MAX_RMS_PX:
        failed.append(f"fit rms {rms:.1f} px > {MAX_RMS_PX}")
    if math.hypot(*offset) > MAX_XY_OFFSET_MM:
        failed.append(f"axis {math.hypot(*offset):.0f} mm from the flange axis")
    if abs(params[2] - bottom) > MAX_BOTTOM_OFFSET_MM:
        failed.append(f"bottom {params[2] - bottom:+.0f} mm from the predicted hang")
    if failed:
        result.ok, result.reason = False, "; ".join(failed)
    return result, mask


def draw(image: np.ndarray, left_camera: Camera, result: HeldBottle, mask: np.ndarray | None) -> np.ndarray:
    view = image.copy()
    if mask is not None:
        overlay = view.copy()
        overlay[mask.astype(bool)] = (0, 200, 0)
        view = cv2.addWeighted(overlay, 0.35, view, 0.65, 0)
    if result.predicted_axis_xy is not None:
        cv2.polylines(view, [np.round(_silhouette(left_camera, *result.predicted_axis_xy,
                                                  result.predicted_bottom_z)).astype(np.int32)], True,
                      (255, 0, 255), 1, cv2.LINE_AA)
    if result.axis_xy is not None:
        cv2.polylines(view, [np.round(_silhouette(left_camera, *result.axis_xy, result.bottom_z)).astype(np.int32)],
                      True, (0, 0, 255), 2, cv2.LINE_AA)
        text = (f"dxy ({result.xy_offset_mm[0]:+.0f},{result.xy_offset_mm[1]:+.0f}) "
                f"dz {result.bottom_offset_mm:+.0f} mm rms {result.rms_px:.1f}px")
        anchor = left_camera.project(np.array([[*result.axis_xy, result.bottom_z]]))[0]
        cv2.putText(view, text, (int(anchor[0]) - 140, int(anchor[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 255), 2)
    return view
