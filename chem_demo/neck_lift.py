"""Compare a bottle rim against the tool and the stationary bench during a short lift."""
from __future__ import annotations

import cv2
import numpy as np


def _track_region(before, after, mask):
    first = cv2.goodFeaturesToTrack(before, 220, .008, 6, mask=mask.astype(np.uint8))
    if first is None or len(first) < 8:
        return None
    second, ok, _ = cv2.calcOpticalFlowPyrLK(before, after, first, None)
    if second is None:
        return None
    back, reverse_ok, _ = cv2.calcOpticalFlowPyrLK(after, before, second, None)
    if back is None:
        return None
    valid = (ok.ravel() != 0) & (reverse_ok.ravel() != 0)
    valid &= np.linalg.norm(back[:, 0] - first[:, 0], axis=1) < 0.8
    a, b = first[valid, 0], second[valid, 0]
    if len(a) < 8:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.8)
    if matrix is None or inliers.sum() < max(8, .65 * len(a)):
        return None
    points = a[inliers.ravel() != 0]
    if np.ptp(points[:, 0]) < .15 * before.shape[1]:
        return None
    return matrix


def _median_depth(depth, mask):
    values = depth[mask]
    values = values[(values > 50) & (values < 1500)]
    return (float(np.median(values)), int(values.size)) if values.size else (float("nan"), 0)


def _depth_check(depth_before, depth_after, rim, lift_mm):
    """Lift seen in depth: the table drops away by the lift, a held rim keeps its distance.

    A vertical lift moves the wrist camera along its own viewing direction, so the image barely
    shifts and the optical-flow test below cannot tell a held bottle from a static one; the depth
    camera measures that motion directly. Replays of every lift so far (2026-11-21/22, 15 and 30 mm):
    table +13.5..+31 mm, rim +0.7..+6.7 mm. Returns None when the depth is unusable (then the
    optical-flow test decides).
    """
    if depth_before is None or depth_after is None or not depth_before.any() or not depth_after.any():
        return None
    h, w = depth_before.shape
    y, x = np.mgrid[:h, :w]
    (cx, cy), (major, _minor) = rim.center_px, rim.axes_px
    r = np.hypot(x - cx, y - cy)
    ring = (r < major / 2 + 3) & (r > major / 2 - 8)
    table = (r > major * 1.5) & (y > cy + 60) & (np.abs(x - cx) < 250)
    rim0, n0 = _median_depth(depth_before, ring)
    rim1, n1 = _median_depth(depth_after, ring)
    tab0, m0 = _median_depth(depth_before, table)
    tab1, m1 = _median_depth(depth_after, table)
    if min(n0, n1) < 300 or min(m0, m1) < 2000:
        return None
    rim_delta, table_delta = rim1 - rim0, tab1 - tab0
    record = {"method": "depth", "lift_mm": round(lift_mm, 2), "rim_depth_delta_mm": round(rim_delta, 1),
              "table_depth_delta_mm": round(table_delta, 1)}
    if not .6 * lift_mm <= table_delta <= 1.4 * lift_mm:
        return None  # the background does not show the lift cleanly: let the image test decide
    if abs(rim_delta) <= .35 * lift_mm:
        return {"held": True, **record}
    if rim_delta >= .65 * lift_mm:
        return {"held": False, "reason": "瓶口距离随台面一起变远：瓶子留在了台面上", **record}
    return None


def verify_neck_lift(before, after):
    """Require a visible rim moving with the jaws, plus an independently moving bench image."""
    return {**_compare(before, after), "before": before.get("evidence"), "after": after.get("evidence")}


def _compare(before, after):
    from chem_demo.neck_vision import detect_wrist_rim

    rims = [detect_wrist_rim(f["color"]) for f in (before, after)]
    if any(not r.ok or r.rim is None for r in rims):
        return {"held": None, "reason": "试提前后未能连续看清同一瓶口"}
    a, b = (d.rim for d in rims)
    z_delta = after["pose"][2] - before["pose"][2]
    if z_delta < 8 or np.linalg.norm(np.subtract(before["pose"][:2], after["pose"][:2])) > 1:
        return {"held": None, "reason": "缺少独立的竖直试提位移"}
    by_depth = _depth_check(before["depth"], after["depth"], a, z_delta)
    if by_depth is not None:
        return by_depth
    gray = [cv2.cvtColor(f["color"], cv2.COLOR_BGR2GRAY) for f in (before, after)]
    depth = before["depth"]
    h, w = depth.shape
    y, x = np.ogrid[:h, :w]
    u, v = a.center_px
    # The bottle and its shoulder may fill the center even below the colored rim.
    clear = (np.abs(x - u) > max(1.35 * max(a.axes_px), .13 * w))
    near = (depth > 40) & (depth < 175) & (y < h * .55) & clear
    far = (depth > 195) & clear & ((y > h * .58) | (x < .16 * w) | (x > .84 * w))
    jaw = _track_region(*gray, near)
    bench = _track_region(*gray, far)
    if jaw is None or bench is None:
        return {"held": None, "reason": "无法分别跟踪夹爪与静止台面"}
    point = np.array([u, v, 1.])
    jaw_error = float(np.linalg.norm(jaw @ point - b.center_px))
    bench_error = float(np.linalg.norm(bench @ point - b.center_px))
    # Near the optical axis a stationary object may barely shift in XY; in that case
    # require a visible difference in scale between jaw motion and bench motion.
    jaw_scale = float(np.linalg.norm(jaw[:, 0]))
    bench_scale = float(np.linalg.norm(bench[:, 0]))
    rim_scale = float(max(b.axes_px) / max(a.axes_px))
    separation = np.linalg.norm((jaw - bench) @ point)
    consistent = jaw_error <= 3.5 and abs(rim_scale - jaw_scale) <= .035
    distinguishable = ((separation >= 8 and bench_error >= jaw_error + 5) or
                       (abs(jaw_scale - bench_scale) >= .065 and
                        abs(rim_scale - bench_scale) >= .055))
    return {"held": True if consistent and distinguishable else None,
            "jaw_error_px": round(jaw_error, 2), "bench_error_px": round(bench_error, 2),
            "rim_scale": round(rim_scale, 4), "jaw_scale": round(jaw_scale, 4),
            "bench_scale": round(bench_scale, 4), "lift_mm": round(z_delta, 2)}
