"""Closed-loop wrist-camera vision for the left-to-right bottle handoff (offline, read-only).

The left arm holds a blue-rimmed glass bottle by the neck. The right arm approaches with its tool
axis horizontal and must clamp the body from the side, guided only by its own wrist camera
(RealSense D405, serial 409122273461). This module never touches a robot, camera or network;
heavy models are injected as callables.

Frames. Camera: OpenCV (x right, y down, z forward), mm. Tool: the right Tool0 frame used by the
hand-eye calibration, ``tool_from_camera`` = ``T_flange_reference_camera`` (tool <- camera). In that
calibration the Tool0 origin sits at the fingertips, the jaw axis is tool Y, the tool axis is tool Z
and tool X is the third axis (``vertical`` below, along a vertical bottle's axis when the jaws open
horizontally).

Validity. The right-wrist calibration was validated only for downward open-jaw postures and the
checkerboard plane. Every result therefore carries ``calibration_validity="unverified_pose"``. The
alignment error is built from RELATIVE measurements (bottle against the jaw pads in the same
image); the calibration enters only through the camera-to-tool rotation and the pad projection,
never through an absolute base-frame position.

Estimate versus measurement is stated per field in each docstring. Values that are estimates:
the 68 mm body diameter (two-view mask measurement, 67.6 to 68.3 mm), the 91.6 +/- 3 mm right jaw
opening (repo estimate), the nominal 120 mm body height, all depth-free ranges (cylinder model
from apparent width), and every sigma reported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Sequence, Mapping

import cv2
import numpy as np

from chem_demo.neck_vision import RimEllipse, detect_wrist_rim

CALIBRATION_VALIDITY = "unverified_pose"
BODY_DIAMETER_MM = 68.0            # measured ~68 mm from two mask views (assumes an upright axis)
NOMINAL_BODY_HEIGHT_MM = 120.0     # estimate: wrist cylinder fit 120 mm, top fit 100 mm
DEFAULT_GRASP_HEIGHT_RATIO = 0.5   # fraction of the rim-to-bottom body length, 0 = rim
JAW_OPEN_MM = 91.6                 # estimate: repo's current right-gripper opening
JAW_OPEN_SIGMA_MM = 3.0
RANGE_MIN_MM, RANGE_MAX_MM = 60.0, 400.0
MIN_MASK_AREA_PX = 1500
MIN_BAND_AREA_PX = 150
EDGE_MIN_GRADIENT = 3.0            # grey levels per pixel, classical silhouette fallback
BORDER_TRUNCATION_PX = 2.0         # cross-section edge this close to the image border = cut off
BORDER_GATE_MARGIN_PX = 12.0
CLEARANCE_MM = 2.0                 # required free gap between bottle and each pad
_HEIGHT_TO_DIAMETER = NOMINAL_BODY_HEIGHT_MM / BODY_DIAMETER_MM


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


class _Result:
    """Mixin giving every result dataclass a JSON-serialisable ``to_dict``."""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}  # type: ignore[arg-type]


@dataclass
class BottleView(_Result):
    """Held-bottle measurement in one wrist image.

    ``ok`` is False whenever evidence is weak; ``reason`` then says why (partial fields stay filled
    for diagnostics only). Measured: rim ellipse, silhouette edges, width in pixels. Estimated:
    ``range_mm`` (depth where valid and consistent, else a cylinder model from the known body
    diameter), ``axis_point_cam_mm`` and every sigma. ``grasp_point_px`` lies on the bottle axis at
    ``grasp_height_ratio`` of the body length measured from the rim.
    """

    ok: bool = False
    reason: str | None = None
    confidence: float = 0.0
    calibration_validity: str = CALIBRATION_VALIDITY
    rim: dict[str, Any] | None = None
    rim_method: str | None = None
    silhouette_source: str | None = None   # "segmenter_mask", "segmenter_box" or "edges"
    body_diameter_mm: float = BODY_DIAMETER_MM
    grasp_height_ratio: float = DEFAULT_GRASP_HEIGHT_RATIO
    axis_dir_px: tuple[float, float] | None = None   # unit vector, rim -> body, image pixels
    body_length_px: float | None = None
    grasp_point_px: tuple[float, float] | None = None
    body_left_px: tuple[float, float] | None = None
    body_right_px: tuple[float, float] | None = None
    width_px: float | None = None
    axis_col_px: float | None = None
    sigma_px: tuple[float, float] = (1.5, 3.0)       # (across axis, along axis) estimates
    focal_px: float | None = None                    # effective focal length across the axis
    range_mm: float | None = None
    range_source: str | None = None                  # "depth" or "width"
    range_sigma_mm: float | None = None
    depth_valid_fraction: float | None = None
    axis_point_cam_mm: tuple[float, float, float] | None = None
    border_distance_px: float | None = None
    truncated_by_border: bool = False
    mask_area_px: int | None = None
    image_size: tuple[int, int] | None = None        # (width, height)
    notes: list[str] = field(default_factory=list)


@dataclass
class JawGeometry(_Result):
    """Jaw pads in the camera. Projected from calibration (estimate) or from measured tip pixels."""

    ok: bool = False
    reason: str | None = None
    confidence: float = 0.0
    calibration_validity: str = CALIBRATION_VALIDITY
    source: str | None = None                  # "projected_from_calibration" or "measured_tips"
    jaw_open_mm: float = JAW_OPEN_MM
    jaw_open_sigma_mm: float = JAW_OPEN_SIGMA_MM
    jaw_axis_tool: tuple[float, float, float] = (0.0, 1.0, 0.0)
    pad_midpoint_tool_mm: tuple[float, float, float] | None = None
    pad_midpoint_cam_mm: tuple[float, float, float] | None = None
    pad_midpoint_px: tuple[float, float] | None = None
    pad_tips_cam_mm: list[list[float]] | None = None
    pad_tips_px: list[list[float]] | None = None
    jaw_axis_cam: tuple[float, float, float] | None = None
    position_sigma_tool_mm: tuple[float, float, float] = (3.0, 3.0, 5.0)  # assumed
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass
class AlignmentError(_Result):
    """Bottle-axis point at the grasp row minus pad midpoint, in the right Tool0 frame.

    ``ok`` means the error is a valid measurement; ``gates_pass`` means it is also safe to close
    (``reason`` lists failed gates). ``lateral_mm`` is along the jaw axis (tool Y), ``vertical_mm``
    along tool X (bottle axis direction), ``range_mm`` along the tool axis (tool Z). Positive range
    means the bottle is beyond the pad midpoint. Sigmas are estimates, combined in quadrature.
    """

    ok: bool = False
    reason: str | None = None
    confidence: float = 0.0
    calibration_validity: str = CALIBRATION_VALIDITY
    lateral_mm: float | None = None
    vertical_mm: float | None = None
    range_mm: float | None = None
    sigma_mm: tuple[float, float, float] | None = None
    error_cam_mm: tuple[float, float, float] | None = None
    margin_mm: float | None = None             # (jaw_open - body diameter) / 2
    gates: dict[str, bool] = field(default_factory=dict)
    gates_pass: bool = False


@dataclass
class TransferVerdict(_Result):
    """Did the bottle move with the right arm during a small retreat? ``held`` is None if unsure."""

    ok: bool = False
    reason: str | None = None
    confidence: float = 0.0
    calibration_validity: str = CALIBRATION_VALIDITY
    held: bool | None = None
    bottle_shift_px: float | None = None
    bg_shift_px: float | None = None
    bg_shift_vector_px: tuple[float, float] | None = None
    bottle_shift_vector_px: tuple[float, float] | None = None
    bg_matches: int = 0
    bg_inliers: int = 0
    max_shift_px: float = 6.0
    min_bg_shift_px: float = 8.0


# --------------------------------------------------------------------------- shared helpers

def _check_K(K: Any) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("K must be a finite 3x3 intrinsic matrix")
    return K


def _check_T(T: Any) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]) or
            not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4)):
        raise ValueError("transform must be a finite rigid 4x4 matrix")
    return T


def _color_of(frame: Any) -> np.ndarray:
    return frame["color"] if isinstance(frame, dict) else frame


def _ray(K: np.ndarray, uv: Sequence[float]) -> np.ndarray:
    return np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))


def _sample(gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``gray`` (float32) at (N, 2) pixel points, border replicated."""
    mx = pts[:, 0].astype(np.float32).reshape(1, -1)
    my = pts[:, 1].astype(np.float32).reshape(1, -1)
    return cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def _ellipse_residual(contour: np.ndarray, center: tuple[float, float], axes: tuple[float, float],
                      angle_deg: float) -> float:
    pts = contour.reshape(-1, 2).astype(float) - center
    th = math.radians(angle_deg)
    xr = pts[:, 0] * math.cos(th) + pts[:, 1] * math.sin(th)
    yr = -pts[:, 0] * math.sin(th) + pts[:, 1] * math.cos(th)
    return float(np.mean(np.abs(np.sqrt((xr / (axes[0] / 2)) ** 2 + (yr / (axes[1] / 2)) ** 2) - 1.0)))


def _blue_band_rim(color: np.ndarray) -> tuple[RimEllipse | None, str | None]:
    """Fallback for rims the nested-ring detector rejects (large, or flat from a grazing view).

    Takes the largest saturated blue component and accepts it only if its outline is elliptical.
    Lower score than the nested-ring detector because it has no hole or concentricity evidence.
    """
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(cv2.inRange(hsv, (85, 120, 20), (135, 255, 255)), cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = sorted(((int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)), reverse=True)
    comps = [c for c in comps if c[0] >= MIN_BAND_AREA_PX]
    if not comps:
        return None, "no blue rim found"
    if len(comps) > 1 and comps[1][0] >= 0.5 * comps[0][0]:
        return None, "ambiguous: several comparable blue rims (two bottles?)"
    contours, _ = cv2.findContours((labels == comps[0][1]).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 12:
        return None, "blue rim outline too short to fit"
    (cx, cy), (a, b), angle = cv2.fitEllipse(contour)
    major, minor, angle = (a, b, angle) if a >= b else (b, a, angle + 90.0)
    height, width = color.shape[:2]
    residual = _ellipse_residual(contour, (cx, cy), (major, minor), angle)
    if minor / major < 0.12 or major > 0.9 * width or residual > 0.12:
        return None, "blue region is not an elliptical rim"
    score = 0.7 * max(0.0, 1.0 - residual / 0.12)
    return RimEllipse((float(cx), float(cy)), (float(major), float(minor)), float(angle % 180.0),
                      float(score)), None


def _find_rim(color: np.ndarray) -> tuple[RimEllipse | None, str | None, str | None]:
    """Classical rim first (nested blue ring), blue-band ellipse as a labelled fallback."""
    try:
        found = detect_wrist_rim(color)
    except ValueError as exc:
        return None, None, str(exc)
    if found.ok and found.rim is not None:
        return found.rim, "nested_ring", None
    if found.reason and "multiple" in found.reason:
        return None, None, "ambiguous: multiple blue rims (two bottles?)"
    rim, reason = _blue_band_rim(color)
    return rim, ("blue_band" if rim else None), reason


def _rim_dict(rim: RimEllipse) -> dict[str, Any]:
    return {"center_px": rim.center_px, "axes_px": rim.axes_px, "angle_deg": rim.angle_deg,
            "score": rim.score}


def _axes(rim: RimEllipse, sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Unit bottle-axis direction d (rim -> body, along the rim's minor axis) and lateral n."""
    th = math.radians(rim.angle_deg)
    d = sign * np.array([-math.sin(th), math.cos(th)])
    n = np.array([-d[1], d[0]])
    return d, (n if n[0] >= 0 else -n)


def _coerce_candidates(raw: Any, shape: tuple[int, int]) -> list[tuple[np.ndarray, float, bool]]:
    """Normalise segmenter output (arrays, dicts or objects with mask/box/score) to masks."""
    out: list[tuple[np.ndarray, float, bool]] = []
    for item in raw or []:
        if isinstance(item, np.ndarray):
            mask, box, score = item, None, 1.0
        elif isinstance(item, dict):
            mask, box, score = item.get("mask"), item.get("box"), item.get("score", 1.0)
        else:
            mask, box, score = (getattr(item, "mask", None), getattr(item, "box", None),
                                getattr(item, "score", 1.0))
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != shape:
                continue
            out.append(((mask > 0).astype(np.uint8), float(score), True))
        elif box is not None and len(box) == 4:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            rect = np.zeros(shape, np.uint8)
            rect[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 1
            out.append((rect, float(score), False))
    return out


# --------------------------------------------------------------------------- 1. bottle location

def locate_held_bottle(color_bgr: np.ndarray, depth_mm: np.ndarray | None, K: np.ndarray, *,
                       segmenter: Callable[[np.ndarray, str], Any] | None = None,
                       body_diameter_mm: float = BODY_DIAMETER_MM,
                       grasp_height_ratio: float = DEFAULT_GRASP_HEIGHT_RATIO,
                       prompt: str = "glass bottle",
                       down_px: Sequence[float] | None = None,
                       expected_px: Sequence[float] | None = None,
                       grasp_from_bottom_mm: float = 40.0) -> BottleView:
    """Find the blue-rimmed bottle in a right-wrist frame and measure it at the grasp cross-section.

    Steps: blue rim (classical) -> bottle axis from the rim's minor axis -> body silhouette from the
    injected ``segmenter(color_bgr, prompt)`` (masks, dicts with mask/box/score, or Detection-like
    objects) or, without one, a gradient-edge fallback (lower confidence; real clear glass often
    lacks the contrast, then ``ok=False``) -> width at the grasp cross-section -> range. Range uses
    ``depth_mm`` (aligned, 0 = invalid) only where valid and consistent with the width-based range
    (glass returns background depth); otherwise the cylinder model ``L = sqrt(R^2 + (f D / w)^2)``.

    Side view (no rim): with the jaws horizontal the wrist camera sits near the rim height and sees
    the blue rim edge-on, so the rim detector fails. When ``down_px`` (image direction of world-down,
    from the arm pose) and a segmenter are given, the bottle axis is taken along ``down_px``, the
    mask nearest ``expected_px`` is used, and the grasp row is ``grasp_from_bottom_mm`` above the
    lowest mask point (scaled by the measured body width). The bottom must be inside the image.
    """
    base = {"body_diameter_mm": float(body_diameter_mm), "grasp_height_ratio": float(grasp_height_ratio)}

    def fail(reason: str, **extra: Any) -> BottleView:
        return BottleView(ok=False, reason=reason, **base, **extra)

    if (not isinstance(color_bgr, np.ndarray) or color_bgr.dtype != np.uint8 or color_bgr.ndim != 3 or
            color_bgr.shape[2] != 3):
        return fail("color_bgr must be a uint8 BGR image")
    try:
        K = _check_K(K)
    except ValueError as exc:
        return fail(str(exc))
    if not (0.05 <= grasp_height_ratio <= 0.95) or not (20.0 <= body_diameter_mm <= 200.0):
        return fail("grasp_height_ratio must be in [0.05, 0.95] and body diameter in [20, 200] mm")
    height, width = color_bgr.shape[:2]
    size = (width, height)
    notes: list[str] = []
    if depth_mm is not None and np.shape(depth_mm) != (height, width):
        notes.append("depth ignored: shape differs from color")
        depth_mm = None

    rim, method, reason = _find_rim(color_bgr)
    sigma_px = (1.5, 3.0)
    if rim is None:
        if down_px is None or segmenter is None:
            return fail(reason or "no blue rim", image_size=size, notes=notes)
        side = _side_view_body(color_bgr, segmenter, prompt, down_px, expected_px, body_diameter_mm,
                               grasp_from_bottom_mm, size, notes)
        if isinstance(side, str):
            return fail(side, image_size=size, rim_method="side_view", notes=notes)
        c, d, n, s, length, t_min, t_max, area = side
        extra = {"rim": None, "rim_method": "side_view", "image_size": size}
        return _finish_view(color_bgr, depth_mm, K, base, extra, notes, "segmenter_mask", area, sigma_px,
                            c, d, n, s, length, t_min, t_max, rim_quality=0.6)
    c = np.array(rim.center_px)
    major, minor = rim.axes_px
    extra = {"rim": _rim_dict(rim), "rim_method": method, "image_size": size}

    # --- silhouette: segmenter first, gradient edges as a labelled fallback
    source: str | None = None
    area: int | None = None
    candidates = _coerce_candidates(segmenter(color_bgr, prompt), (height, width)) if segmenter else []
    if segmenter is not None and not candidates:
        notes.append("segmenter returned no usable mask; fell back to image edges")
    if candidates:
        probe = np.zeros((height, width), np.uint8)
        cv2.circle(probe, (int(round(c[0])), int(round(c[1]))), 2, 1, -1)
        covering = [cand for cand in candidates
                    if (cv2.dilate(cand[0], np.ones((31, 31), np.uint8)) & probe).any()]
        if not covering:
            return fail("segmenter mask does not cover the blue rim", **extra, notes=notes)
        covering.sort(key=lambda cand: -cand[1])
        if len(covering) > 1 and covering[1][1] >= 0.85 * covering[0][1]:
            return fail("ambiguous: several segmenter masks cover the rim", **extra, notes=notes)
        mask = cv2.morphologyEx(covering[0][0], cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count < 2:
            return fail("segmenter mask is empty after cleaning", **extra, notes=notes)
        mask = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        area = int(mask.sum())
        if area < MIN_MASK_AREA_PX:
            return fail(f"mask too small ({area} px)", **extra, mask_area_px=area, notes=notes)
        source = "segmenter_mask" if covering[0][2] else "segmenter_box"
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], axis=1).astype(float) - c
        q, _ = _axes(rim)
        side = float(np.mean(pts @ q))
        if abs(side) < 2.0:
            return fail("cannot tell which side of the rim the body is on", **extra, notes=notes)
        d, n = _axes(rim, 1.0 if side > 0 else -1.0)
        along = pts @ d
        length = float(along.max())
        if length < 0.5 * minor:
            return fail("mask does not extend beyond the rim", **extra, mask_area_px=area, notes=notes)
        s = grasp_height_ratio * length
        band = np.abs(along - s) <= 3.0
        if band.sum() < 10:
            return fail("no mask pixels at the grasp row", **extra, mask_area_px=area, notes=notes)
        lateral = pts[band] @ n
        t_min, t_max = float(lateral.min()), float(lateral.max())
    else:
        # Rim width is only a coarse scale here (assumes rim outer diameter ~ body diameter); it
        # places the cross-section and the search window, never the reported width.
        gray = cv2.GaussianBlur(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)
        mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        q, n0 = _axes(rim)
        energy = []
        for sign in (1.0, -1.0):
            ss, tt = np.meshgrid(np.arange(0.5 * major, 1.8 * major, 3.0),
                                 np.arange(-0.6 * major, 0.6 * major, 3.0), indexing="ij")
            pts = c + sign * ss.reshape(-1, 1) * q + tt.reshape(-1, 1) * n0
            energy.append(float(_sample(mag, pts).mean()))
        if max(energy) < 1.15 * min(energy) + 1e-6:
            return fail("cannot tell which side of the rim the body is on (no edge evidence)",
                        **extra, notes=notes)
        d, n = _axes(rim, 1.0 if energy[0] >= energy[1] else -1.0)
        length = _HEIGHT_TO_DIAMETER * major
        s = grasp_height_ratio * length
        half = int(1.8 * major)
        ts = np.arange(-half, half + 1, 1.0)
        prof = np.mean([_sample(gray, c + (s + off) * d + np.outer(ts, n)) for off in range(-4, 5, 2)],
                       axis=0)
        grad = np.abs(np.gradient(cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (0, 0),
                                                   1.5).ravel()))
        edges: list[float] = []
        # Search each side in 0.3 to 1.1 rim widths (body 0.6 to 2.2 rim widths wide) so that
        # background clutter outside that band cannot win; clean backgrounds work best.
        for ix in (np.where((ts < -0.3 * major) & (ts > -1.1 * major))[0],
                   np.where((ts > 0.3 * major) & (ts < 1.1 * major))[0]):
            peak = int(ix[np.argmax(grad[ix])])
            if grad[peak] < EDGE_MIN_GRADIENT:
                return fail("no silhouette edge found at the grasp row (supply a segmenter for glass)",
                            **extra, notes=notes)
            near = ix[(grad[ix] >= 0.5 * grad[peak]) & (np.abs(ix - peak) <= 4)]
            edges.append(float(ts[near.min()] if ts[peak] < 0 else ts[near.max()]))
        t_min, t_max = edges
        source, sigma_px = "edges", (2.0, 0.1 * length)
        if not (0.4 * major <= t_max - t_min <= 2.2 * major):
            return fail("edge pair width is implausible for this rim", **extra, notes=notes)
    return _finish_view(color_bgr, depth_mm, K, base, extra, notes, source, area, sigma_px,
                        c, d, n, s, length, t_min, t_max, rim_quality=min(1.0, max(rim.score, 0.0)))


TAIL_NARROW_RATIO = 0.6     # a tail narrower than this share of the body width is not bottle
TAIL_MIN_LENGTH_PX = 12.0   # shorter narrow ends are the rounded bottom itself
TAIL_MIN_SHARE_OF_WIDTH = 0.25


def _trim_narrow_tail(mask: np.ndarray, d: np.ndarray, n: np.ndarray, notes: list[str]) -> np.ndarray:
    """Cut off a narrow object the segmenter merged onto the bottle bottom.

    2026-11-23 16:07: from one approach step the white cylinder standing behind the bottle lined up
    under its bottom and SAM returned one mask for both (238 px long instead of 180). The bottom,
    hence the grasp row and the width range, moved ~80 mm and the approach was refused. The body
    width is taken over the middle of the mask; a tail that stays narrower than
    TAIL_NARROW_RATIO of it to the end, for at least TAIL_MIN_LENGTH_PX, is removed.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 50:
        return mask
    pts = np.stack([xs, ys], axis=1).astype(float)
    along, lateral = pts @ d, pts @ n
    top, bottom = float(along.min()), float(along.max())
    length = bottom - top
    if length < 4 * TAIL_MIN_LENGTH_PX:
        return mask
    bins = np.floor((along - top) / 2.0).astype(int)
    count = int(bins.max()) + 1
    lo = np.full(count, np.inf)
    hi = np.full(count, -np.inf)
    np.minimum.at(lo, bins, lateral)
    np.maximum.at(hi, bins, lateral)
    widths = np.where(np.isfinite(lo), hi - lo, 0.0)
    middle = widths[int(0.25 * count):max(int(0.6 * count), int(0.25 * count) + 1)]
    reference = float(np.median(middle[middle > 0])) if (middle > 0).any() else 0.0
    if reference <= 0:
        return mask
    narrow = widths < TAIL_NARROW_RATIO * reference
    start = None
    for index in range(count - 1, int(0.5 * count) - 1, -1):  # walk up from the bottom
        if not narrow[index]:
            break
        start = index
    # The rounded bottom tapers over a length that grows with the apparent size (11 px at 142 px wide,
    # 2026-11-23 16:48); a merged object is much longer (56 px at 117 px wide).
    if start is None or (count - start) * 2.0 < max(TAIL_MIN_LENGTH_PX, TAIL_MIN_SHARE_OF_WIDTH * reference):
        return mask
    cut = top + start * 2.0
    keep = along < cut
    trimmed = np.zeros_like(mask)
    trimmed[ys[keep], xs[keep]] = 1
    notes.append(f"trimmed a narrow {bottom - cut:.0f} px tail below the bottle (another object merged into the mask)")
    return trimmed


def _side_view_body(color_bgr, segmenter, prompt, down_px, expected_px, body_diameter_mm, grasp_from_bottom_mm,
                    size, notes):
    """Bottle body from a segmenter mask with the axis along ``down_px``; a reason string on failure."""
    width, height = size
    d = np.asarray(down_px, float)
    if d.shape != (2,) or np.linalg.norm(d) < 1e-6:
        return "down_px must be a non-zero 2-vector"
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    n = n if n[0] >= 0 else -n
    candidates = [cand for cand in _coerce_candidates(segmenter(color_bgr, prompt), (height, width))
                  if cand[2] and int(cand[0].sum()) >= MIN_MASK_AREA_PX]
    if not candidates:
        return "no bottle in the right wrist view (side view, no usable segmenter mask)"

    def cost(cand):
        ys, xs = np.nonzero(cand[0])
        value = -cand[1]
        if expected_px is not None:
            value += 0.8 * math.hypot(xs.mean() - expected_px[0], ys.mean() - expected_px[1]) / max(width, height)
        return value

    candidates.sort(key=cost)
    if len(candidates) > 1 and abs(cost(candidates[1]) - cost(candidates[0])) < 0.05:
        return "ambiguous: two similar bottle masks in the side view"
    mask = cv2.morphologyEx(candidates[0][0], cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return "segmenter mask is empty after cleaning"
    mask = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    mask = _trim_narrow_tail(mask, d, n, notes)
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    c = np.array([xs.mean(), ys.mean()])
    pts = np.stack([xs, ys], axis=1).astype(float) - c
    along, lateral_all = pts @ d, pts @ n
    bottom = float(along.max())
    low = along >= bottom - 2.0
    edge = np.stack([xs[low], ys[low]], axis=1)
    if ((edge[:, 0] <= BORDER_TRUNCATION_PX) | (edge[:, 0] >= width - 1 - BORDER_TRUNCATION_PX) |
            (edge[:, 1] <= BORDER_TRUNCATION_PX) | (edge[:, 1] >= height - 1 - BORDER_TRUNCATION_PX)).any():
        return "bottle bottom cut off by the image border (bottle lower than the view)"
    length = bottom - float(along.min())

    def width_at(s_row):
        band = np.abs(along - s_row) <= 3.0
        if band.sum() < 10:
            return None
        lat = lateral_all[band]
        return float(lat.min()), float(lat.max())

    s = bottom - 0.3 * length
    span = width_at(s)
    for _ in range(2):
        if span is None:
            return "no mask pixels at the grasp row"
        px_per_mm = (span[1] - span[0]) / body_diameter_mm
        s = bottom - grasp_from_bottom_mm * px_per_mm
        span = width_at(s)
    if span is None:
        return "no mask pixels at the grasp row"
    if along.min() > s - 3.0:
        notes.append("mask top at the grasp row: upper body hidden")
    return c, d, n, s, length, span[0], span[1], area


def _finish_view(color_bgr, depth_mm, K, base, extra, notes, source, area, sigma_px, c, d, n, s, length,
                 t_min, t_max, *, rim_quality):
    """Grasp cross-section -> pixels, width, range and the camera-frame axis point."""
    height, width = color_bgr.shape[:2]
    body_diameter_mm = base["body_diameter_mm"]

    def fail(reason: str, **more: Any) -> BottleView:
        return BottleView(ok=False, reason=reason, **base, **more)

    f_eff = math.hypot(K[0, 0] * n[0], K[1, 1] * n[1])
    g = c + s * d
    left, right = g + t_min * n, g + t_max * n
    center = g + 0.5 * (t_min + t_max) * n
    width_px = t_max - t_min
    border = float(min(min(p[0], p[1], width - 1 - p[0], height - 1 - p[1]) for p in (left, right)))
    geometry = {
        **extra, "silhouette_source": source, "mask_area_px": area, "notes": notes,
        "axis_dir_px": (float(d[0]), float(d[1])), "body_length_px": float(length),
        "grasp_point_px": (float(center[0]), float(center[1])), "body_left_px": (float(left[0]), float(left[1])),
        "body_right_px": (float(right[0]), float(right[1])), "width_px": float(width_px),
        "axis_col_px": float(center[0]), "sigma_px": sigma_px, "border_distance_px": border,
        "focal_px": float(f_eff),
    }
    if border <= BORDER_TRUNCATION_PX:
        return fail("silhouette truncated by the image border at the grasp row",
                    **geometry, truncated_by_border=True)
    if width_px < 8.0:
        return fail("silhouette too narrow to measure", **geometry)

    # --- range: cylinder model from the apparent width, depth only if valid and consistent
    radius = body_diameter_mm / 2.0
    ray = _ray(K, center)
    L = math.sqrt(radius ** 2 + (f_eff * body_diameter_mm / width_px) ** 2)
    z_width = L * float(ray[2] / np.linalg.norm(ray))
    z, range_source = z_width, "width"
    z_sigma = z_width * math.hypot(sigma_px[0] * 2 / width_px, 1.0 / body_diameter_mm)
    valid_fraction = None
    if depth_mm is not None:
        ss, tt = np.meshgrid(np.arange(-8, 9, 2.0), np.arange(-width_px / 6, width_px / 6 + 1, 2.0),
                             indexing="ij")
        pts = np.round(center + ss.reshape(-1, 1) * d + tt.reshape(-1, 1) * n).astype(int)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
        vals = np.asarray(depth_mm, float)[pts[inside, 1], pts[inside, 0]]
        valid = np.isfinite(vals) & (vals > 30) & (vals < 1000)
        valid_fraction = float(valid.mean()) if vals.size else 0.0
        if valid.sum() >= 15 and valid_fraction >= 0.3:
            z_depth = float(np.median(vals[valid])) + radius   # front surface + radius ~ axis
            if abs(z_depth - z_width) <= 0.2 * z_width:
                z, range_source, z_sigma = z_depth, "depth", max(4.0, 0.03 * z_depth)
            else:
                notes.append(f"depth {z_depth:.0f} mm disagrees with width range {z_width:.0f} mm "
                             "(glass shows background); using width")
    if not (RANGE_MIN_MM <= z <= RANGE_MAX_MM):
        return fail(f"range {z:.0f} mm outside [{RANGE_MIN_MM:.0f}, {RANGE_MAX_MM:.0f}] mm",
                    **geometry, range_mm=float(z), range_source=range_source)
    point = ray * (z / ray[2])
    confidence = min(0.85, 0.4 * rim_quality + 0.3 * (1.0 if source != "edges" else 0.6) +
                     0.3 * (1.0 if range_source == "depth" else 0.7))
    return BottleView(ok=True, reason=None, confidence=float(confidence), **base, **geometry,
                      range_mm=float(z), range_source=range_source, range_sigma_mm=float(z_sigma),
                      depth_valid_fraction=valid_fraction,
                      axis_point_cam_mm=(float(point[0]), float(point[1]), float(point[2])),
                      truncated_by_border=False)


# --------------------------------------------------------------------------- 2. jaw geometry

def jaw_geometry(K: np.ndarray, T_flange_camera: np.ndarray,
                 pad_offset_tool_mm: Sequence[float] = (0.0, 0.0, 0.0),
                 jaw_open_mm: float = JAW_OPEN_MM, *,
                 jaw_axis_tool: Sequence[float] = (0.0, 1.0, 0.0),
                 jaw_open_sigma_mm: float = JAW_OPEN_SIGMA_MM,
                 position_sigma_tool_mm: Sequence[float] = (3.0, 3.0, 5.0),
                 measured_tips_px: Sequence[Sequence[float]] | None = None) -> JawGeometry:
    """Where the pad midpoint and the jaw-opening segment sit in the image and the camera frame.

    Pure geometry. ``T_flange_camera`` is tool <- camera (the calibration's
    ``T_flange_reference_camera``); the pad contact midpoint is ``pad_offset_tool_mm`` from the
    Tool0 origin (the hand-eye origin sits at the fingertips, so the default offset is zero) and
    the pads open symmetrically along ``jaw_axis_tool`` (tool Y).

    Measured: nothing, unless ``measured_tips_px`` (two pad-tip pixels, e.g. from
    ``wrist_landmarks.measure_open_pad_tips``) is given; then pixels are measured and camera-frame
    positions follow from the known opening (``Z = fx * open / |tip gap|``). Estimated: jaw opening
    (91.6 +/- 3 mm), pad offset, the camera mount translation and rotation (validated only for
    downward postures), and ``position_sigma_tool_mm`` (assumed). Cross-check on the right
    reference frame: projected tips fall within ~10 px of the measured ones (gap 261 vs 275 px).
    """
    try:
        K, T = _check_K(K), _check_T(T_flange_camera)
        offset = np.asarray(pad_offset_tool_mm, float)
        axis = np.asarray(jaw_axis_tool, float)
        if offset.shape != (3,) or axis.shape != (3,) or np.linalg.norm(axis) < 1e-9 or jaw_open_mm <= 0:
            raise ValueError("pad offset, jaw axis and opening must be finite and non-degenerate")
    except (ValueError, TypeError) as exc:
        return JawGeometry(ok=False, reason=str(exc))
    axis = axis / np.linalg.norm(axis)
    sigma = tuple(float(v) for v in position_sigma_tool_mm)
    common = {"jaw_open_mm": float(jaw_open_mm), "jaw_open_sigma_mm": float(jaw_open_sigma_mm),
              "jaw_axis_tool": tuple(float(v) for v in axis), "position_sigma_tool_mm": sigma}
    provenance = {"jaw_open_mm": "estimate (repo: 91.6 +/- 3 mm)", "pad_offset_tool_mm": "estimate",
                  "camera_mount": "calibrated for downward open jaws only; unverified at this pose"}
    T_inv = np.linalg.inv(T)
    if measured_tips_px is not None:
        tips = np.asarray(measured_tips_px, float)
        gap = float(np.linalg.norm(tips[1] - tips[0])) if tips.shape == (2, 2) else 0.0
        if gap < 20.0:
            return JawGeometry(ok=False, reason="measured pad tips invalid or too close", **common)
        z = K[0, 0] * jaw_open_mm / gap
        cam = np.array([_ray(K, t) * z for t in tips])
        mid_cam = cam.mean(axis=0)
        mid_px = tips.mean(axis=0)
        source = "measured_tips"
        provenance["pad_tips_px"] = "measured"
        provenance["pad_depth"] = "estimate from the known opening"
        confidence = 0.6
    else:
        tips_tool = np.array([offset - axis * jaw_open_mm / 2, offset + axis * jaw_open_mm / 2])
        cam = (T_inv[:3, :3] @ tips_tool.T).T + T_inv[:3, 3]
        if (cam[:, 2] < 1.0).any():
            return JawGeometry(ok=False, reason="pad points fall behind the camera", **common)
        mid_cam = cam.mean(axis=0)
        proj = (K @ cam.T).T
        tips = proj[:, :2] / proj[:, 2:3]
        mid_px = (K @ mid_cam)[:2] / mid_cam[2]
        source = "projected_from_calibration"
        provenance["pad_tips_px"] = "estimate (projected from the mount calibration)"
        confidence = 0.45
    mid_tool = T[:3, :3] @ mid_cam + T[:3, 3]
    jaw_axis_cam = (cam[1] - cam[0]) / np.linalg.norm(cam[1] - cam[0])
    return JawGeometry(
        ok=True, reason=None, confidence=confidence, source=source,
        pad_midpoint_tool_mm=tuple(float(v) for v in mid_tool),
        pad_midpoint_cam_mm=tuple(float(v) for v in mid_cam),
        pad_midpoint_px=(float(mid_px[0]), float(mid_px[1])),
        pad_tips_cam_mm=cam.tolist(), pad_tips_px=np.asarray(tips).tolist(),
        jaw_axis_cam=tuple(float(v) for v in jaw_axis_cam), provenance=provenance, **common)


# --------------------------------------------------------------------------- 3. alignment

def alignment_error(view: BottleView, jaw: JawGeometry, *, tool_from_camera: np.ndarray,
                    clearance_mm: float = CLEARANCE_MM,
                    border_margin_px: float = BORDER_GATE_MARGIN_PX) -> AlignmentError:
    """Bottle body axis at the grasp row minus the pad midpoint, in the right Tool0 frame.

    Both points come from the same image and camera frame, so only the camera-to-tool rotation of
    ``tool_from_camera`` is applied to the difference. Gates (all must be True for ``gates_pass``):
    ``view_confident``; ``inside_image`` (bottle edges at least ``border_margin_px`` from the
    border); ``opening_wide_enough`` ((open - sigma) - D >= 2 * clearance); ``inside_pad_span``
    (|lateral| + D/2 <= (open - sigma)/2 - clearance, with margin (open - D)/2); ``not_touching_pads``
    (image-plane gap between bottle edges and pad tips >= clearance at the bottle's scale, a
    conservative proxy because pads and bottle may lie at different depths). Sigmas are estimates.
    """
    if not view.ok or view.axis_point_cam_mm is None or view.grasp_point_px is None:
        return AlignmentError(ok=False, reason=f"bottle view invalid: {view.reason}")
    if not jaw.ok or jaw.pad_midpoint_cam_mm is None or jaw.pad_tips_px is None:
        return AlignmentError(ok=False, reason=f"jaw geometry invalid: {jaw.reason}")
    try:
        T = _check_T(tool_from_camera)
    except ValueError as exc:
        return AlignmentError(ok=False, reason=str(exc))
    R = T[:3, :3]
    delta_cam = np.array(view.axis_point_cam_mm) - np.array(jaw.pad_midpoint_cam_mm)
    err_tool = R @ delta_cam
    lateral_axis = np.array(jaw.jaw_axis_tool)
    vertical_axis = np.cross(lateral_axis, [0.0, 0.0, 1.0])
    vertical_axis = vertical_axis / max(np.linalg.norm(vertical_axis), 1e-9)
    components = (float(err_tool @ lateral_axis), float(err_tool @ vertical_axis), float(err_tool[2]))

    # Per-component uncertainty: pixel noise and range noise of the bottle point, rotated into the
    # tool frame, plus the assumed pad-position sigma, all in quadrature.
    z = view.range_mm or 1.0
    focal = view.focal_px or 434.0
    sig_cam = np.array([view.sigma_px[0] * z / focal, view.sigma_px[1] * z / focal,
                        view.range_sigma_mm or 0.0])
    sig_cam[0] = math.hypot(sig_cam[0], view.axis_point_cam_mm[0] / z * sig_cam[2])
    sig_cam[1] = math.hypot(sig_cam[1], view.axis_point_cam_mm[1] / z * sig_cam[2])
    sig_view = np.sqrt((R ** 2) @ (sig_cam ** 2))
    sig_tool = np.sqrt(sig_view ** 2 + np.array(jaw.position_sigma_tool_mm) ** 2)
    sigma = (float(np.sqrt(np.sum((lateral_axis * sig_tool) ** 2))),
             float(np.sqrt(np.sum((vertical_axis * sig_tool) ** 2))), float(sig_tool[2]))

    D, opening, o_sigma = view.body_diameter_mm, jaw.jaw_open_mm, jaw.jaw_open_sigma_mm
    margin = (opening - D) / 2.0
    usable = (opening - o_sigma) / 2.0 - clearance_mm
    gates = {
        "view_confident": view.confidence >= 0.4,
        "inside_image": (view.border_distance_px or 0.0) >= border_margin_px,
        "opening_wide_enough": (opening - o_sigma) - D >= 2.0 * clearance_mm,
        "inside_pad_span": abs(components[0]) + D / 2.0 <= usable,
    }
    clear_px = clearance_mm * focal / z
    tips_u = sorted(float(p[0]) for p in jaw.pad_tips_px)
    edges_u = sorted([view.body_left_px[0], view.body_right_px[0]]) if view.body_left_px and view.body_right_px else None
    gates["not_touching_pads"] = bool(edges_u and edges_u[0] - tips_u[0] >= clear_px and
                                      tips_u[-1] - edges_u[1] >= clear_px)
    failed = [name for name, passed in gates.items() if not passed]
    return AlignmentError(
        ok=True, reason=("gates failed: " + ", ".join(failed)) if failed else None,
        confidence=float(min(view.confidence, jaw.confidence + 0.2)),
        lateral_mm=components[0], vertical_mm=components[1], range_mm=components[2], sigma_mm=sigma,
        error_cam_mm=tuple(float(v) for v in delta_cam), margin_mm=float(margin), gates=gates,
        gates_pass=not failed)


# --------------------------------------------------------------------------- 4. transfer check

def _exclusion_mask(shape: tuple[int, int], view: BottleView) -> np.ndarray:
    """Mask (255 = usable background) that removes the bottle and its surroundings."""
    keep = np.full(shape, 255, np.uint8)
    rim = view.rim
    if rim is None:
        return keep
    c = np.array(rim["center_px"])
    major, minor = rim["axes_px"]
    if view.axis_dir_px is not None and view.body_length_px and view.width_px:
        d = np.array(view.axis_dir_px)
        n = np.array([-d[1], d[0]])
        hw = max(view.width_px, major) / 2.0 + 15.0
        corners = [c - 0.7 * minor * d - hw * n, c - 0.7 * minor * d + hw * n,
                   c + (view.body_length_px + 10.0) * d + hw * n, c + (view.body_length_px + 10.0) * d - hw * n]
        cv2.fillPoly(keep, [np.round(np.array(corners)).astype(np.int32)], 0)
    else:
        cv2.circle(keep, (int(c[0]), int(c[1])), int(2.5 * major), 0, -1)
    return keep


def transfer_check(before: Any, after: Any, K: np.ndarray, *, background_shift_expected: bool = True,
                   segmenter: Callable[[np.ndarray, str], Any] | None = None,
                   max_shift_px: float = 6.0, min_bg_shift_px: float = 8.0,
                   exclude_mask: np.ndarray | None = None,
                   view_kwargs: Mapping[str, Any] | None = None) -> TransferVerdict:
    """Decide whether the bottle moved with the right arm during a small retreat.

    ``before``/``after`` are BGR wrist frames (or dicts with ``color``), taken with the left gripper
    already open. The right wrist camera moves with the arm, so a clamped bottle stays put in the
    image while the background (ORB features outside the bottle and ``exclude_mask``, e.g. the jaw
    pads, which are also static in this camera) shifts. ``held=True`` requires the background to
    shift by >= ``min_bg_shift_px`` and the bottle (rim centre, grasp axis) by <= ``max_shift_px``
    and well below the background shift; it is never returned when the background did not move.
    ``held=False`` means the bottle moved with the background. With
    ``background_shift_expected=False`` a static background no longer blocks ``held=False`` (the
    bottle visibly moved relative to the camera) but still blocks ``held=True``. All thresholds
    are estimates, not tuned on robot data.
    """
    common = {"max_shift_px": float(max_shift_px), "min_bg_shift_px": float(min_bg_shift_px)}
    frames = [_color_of(before), _color_of(after)]
    # ``view_kwargs``: the side-view hints (down_px, expected_px) of the approach. With the jaws
    # horizontal the rim is seen edge-on and never found (2026-11-23 16:58: the bottle was in the right
    # jaws but the check gave up on "rim not found"); then the body's grasp point is tracked instead.
    views = [locate_held_bottle(f, None, K, segmenter=segmenter, **dict(view_kwargs or {})) for f in frames]
    if all(v.rim is not None for v in views):
        shift_vec = np.array(views[1].rim["center_px"]) - np.array(views[0].rim["center_px"])
    elif all(v.ok and v.grasp_point_px for v in views):
        shift_vec = np.zeros(2)
    else:
        return TransferVerdict(ok=False, reason="bottle not found in both frames (no rim, no side-view body)",
                               **common)
    if all(v.ok and v.grasp_point_px for v in views):
        axis_shift = np.array(views[1].grasp_point_px) - np.array(views[0].grasp_point_px)
        if np.linalg.norm(axis_shift) > np.linalg.norm(shift_vec):
            shift_vec = axis_shift
    bottle_shift = float(np.linalg.norm(shift_vec))

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    masks = [_exclusion_mask(g.shape, v) for g, v in zip(grays, views)]
    if exclude_mask is not None:
        masks = [cv2.bitwise_and(m, np.where(exclude_mask > 0, 0, 255).astype(np.uint8)) for m in masks]
    orb = cv2.ORB_create(1500)
    kp_a, des_a = orb.detectAndCompute(grays[0], masks[0])
    kp_b, des_b = orb.detectAndCompute(grays[1], masks[1])
    result = {"bottle_shift_px": bottle_shift,
              "bottle_shift_vector_px": (float(shift_vec[0]), float(shift_vec[1])), **common}
    if des_a is None or des_b is None or len(kp_a) < 15 or len(kp_b) < 15:
        return TransferVerdict(ok=False, reason="too few background features outside the bottle", **result)
    matches = [m for m in cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_a, des_b)
               if m.distance <= 64]
    if len(matches) < 12:
        return TransferVerdict(ok=False, reason="too few background matches", bg_matches=len(matches),
                               **result)
    pa = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pb = np.float32([kp_b[m.trainIdx].pt for m in matches])
    _, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    keep = (inliers.ravel() > 0) if inliers is not None else np.zeros(len(matches), bool)
    if keep.sum() < 12:
        return TransferVerdict(ok=False, reason="background motion model has too few inliers",
                               bg_matches=len(matches), bg_inliers=int(keep.sum()), **result)
    bg_vec = np.median(pb[keep] - pa[keep], axis=0)
    bg_shift = float(np.linalg.norm(bg_vec))
    result.update(bg_shift_px=bg_shift, bg_shift_vector_px=(float(bg_vec[0]), float(bg_vec[1])),
                  bg_matches=len(matches), bg_inliers=int(keep.sum()))
    bg_moved = bg_shift >= min_bg_shift_px
    with_background = float(np.linalg.norm(shift_vec - bg_vec)) <= max(max_shift_px, 0.3 * bg_shift)
    if bg_moved and bottle_shift <= max_shift_px and bottle_shift <= 0.35 * bg_shift:
        return TransferVerdict(ok=True, reason=None, held=True,
                               confidence=float(min(0.85, 0.5 + 0.35 * min(1.0, bg_shift / (4 * min_bg_shift_px)))),
                               **result)
    if bottle_shift > max_shift_px and with_background and (bg_moved or not background_shift_expected):
        return TransferVerdict(ok=True, reason="bottle moved with the background, not with the arm",
                               held=False, confidence=0.7, **result)
    if not bg_moved:
        return TransferVerdict(ok=False, held=None, confidence=0.0,
                               reason=f"background shifted only {bg_shift:.1f} px: the arm did not "
                                      "move enough to judge", **result)
    return TransferVerdict(ok=False, held=None, confidence=0.0,
                           reason="bottle and background motion are inconsistent", **result)


__all__ = [
    "AlignmentError", "BottleView", "CALIBRATION_VALIDITY", "JawGeometry", "TransferVerdict",
    "alignment_error", "jaw_geometry", "locate_held_bottle", "transfer_check",
]
