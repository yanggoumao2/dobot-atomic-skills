"""Right arm picks the dropper (bulb wrapped in blue paper) out of its tube in the rack and lifts it,
tool straight down the whole time. No teaching.

Sequence from the photo pose (both arms there, disabled):
  1. coarse: top-camera frame -> blue blob near the rack -> bulb XY (right base, ~1 cm).
  2. hover: fingertips ``--hover`` mm above the table over that point.
  3. refine (closed loop): right-wrist colour + depth -> blue blob -> 3D points of the bulb ->
     bulb top Z and axis XY -> move over it; repeat, then the same ``--pre-clear`` mm above the bulb.
  4. grasp: jaws pre-closed to bulb + 2 x ``--jaw-margin``, straight down until the fingertips are
     ``--grip-depth`` mm below the bulb top, close to bulb - ``--squeeze``.
  5. lift ``--lift`` mm straight up (tool stays vertical), check with the wrist camera that the bulb
     came along, take wrist / top / front frames, disable with the dropper held (brakes hold it).
``--put-back`` lowers it back into its tube, opens, rises and returns to the photo pose.

Coordinates: controller poses (Tool0 = flange, right base). The top-camera calibration and the
right-wrist hand-eye use the fingertip convention (see gripper_geometry): with the tool straight
down, the flange Z read from the controller is the fingertip height in those frames, so a vision
point at Z is reached by commanding the flange to Z (table: 240.18).

    C:/software/anaconda3/envs/dobot-xtrainer/python.exe -X utf8 -m chem_demo.run_dropper --preflight
    ... --live --until look      (move over the bulb, look, back to the photo pose; no grasp)
    ... --live                   (grasp and lift; ends holding the dropper, arm disabled)
    ... --put-back               (put the held dropper back into its tube, photo pose)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from chem_demo.wrist_geometry import backproject, pose_matrix, transform
from chem_demo.wrist_runtime import ROOT, TABLE_Z, camera_serial, platform_height

DATA = ROOT / "calibration-data" / "chem-demo" / "dropper"
RIGHT_MODEL = (ROOT / "calibration-data" / "chem-demo" / "wrist-board" /
               "right-20261121_155150-recovery-20261121_161320_332390" / "checker_multiview_validation.json")
TABLE = TABLE_Z["right"]
RACK_HINT_XY = (-299.0, -255.0)   # bulb, top frame 11-25 15:50 (rack moved toward the base; was -345,-365)
RACK_HINT_RADIUS = 110.0
ASSUMED_BULB_TOP = 163.0          # above the table, until the wrist measures it (wrist depth 11-25 00:58)
BLUE_LO, BLUE_HI = (85, 45, 40), (125, 255, 255)   # OpenCV HSV; blue paper (pale in the top view)
BLUE_PALE_LO = (88, 20, 60)       # fallback near the expected bulb only (paper over the orange rack)
PALE_WINDOW_PX = 70
BULB_Z_GATE = 40.0                # a bulb candidate must be within this of the expected bulb-top height
DRIP_STEP_CODE = 2                # each drip squeezes 2 more codes (0.7 mm); no release in between, so
DRIP_MIN_CODE = 13                # no air is drawn back in at the tip (Kunyu 11-25 17:17)
PRIOR_TOL_MM = 6.0                # tube centre may shift this much between runs (seen: ~3 mm). The prior
                                  # stays the run-start value: chaining measurements let it drift 10 mm (17:03)
SUCK_CODE = 26                    # squeezed bulb stalls the jaws at ~14 (5.0 mm). 20 (7.2 mm) drew too little
                                  # water (Kunyu 11-25 17:09); 26 = 9.3 mm
TUBE_RIM_RADIUS = 14.3            # 50 ml tube mouth seen by the wrist (free fits 11-25: 14.0-15.0)
JAW_OPEN_MM = 91.6                # 255 = fully open; 180 held the 64 mm bottle -> linear, 0.359 mm/code
MM_PER_CODE = JAW_OPEN_MM / 255.0
GRIPPER_OPEN = 255
TOOL_DOWN_TOL_DEG = 0.6


def log_line(text: str) -> None:
    print(text, flush=True)


# ---------------------------------------------------------------------------------------- vision
def blue_mask(color: np.ndarray, min_area: int = 12, lo: Sequence[int] = BLUE_LO) -> np.ndarray:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, tuple(lo), BLUE_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def top_bulb(image: np.ndarray, camera: Any, hint_xy: Sequence[float] = RACK_HINT_XY,
             radius: float = RACK_HINT_RADIUS) -> dict[str, Any]:
    """Blue blob whose ground point (at the assumed bulb top) lies near the rack; nearest to the hint wins."""
    mask = blue_mask(image, min_area=6)
    count, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > 900:  # bottle caps and the wash-bottle are bigger than the bulb seen from the top
            continue
        px = [float(v) for v in cents[i]]
        xy = camera.ray_to_plane(px, TABLE + ASSUMED_BULB_TOP)[:2]
        d = math.dist(xy, hint_xy)
        if d <= radius and (best is None or d < best["dist_to_hint"]):
            best = {"pixel": px, "area": area, "xy": [float(xy[0]), float(xy[1])], "dist_to_hint": round(d, 1)}
    if best is None:
        return {"ok": False, "reason": f"顶部图里试管架附近（{radius:.0f} mm 内）没有蓝色块"}
    return {"ok": True, **best}


def wrist_bulb(frame: dict, K: np.ndarray, T_base_camera: np.ndarray, expected: Sequence[float],
               search_mm: float = 70.0) -> dict[str, Any]:
    """Bulb from one wrist RGB-D frame: blue blob with valid depth nearest to ``expected`` (x, y, top z).
    If nothing qualifies, retry once with a paler threshold inside a window around ``expected``: the
    translucent paper over the orange rack reads grey-blue, S~30 (11-25 16:35)."""
    out = _wrist_bulb(frame, K, T_base_camera, expected, search_mm, blue_mask(frame["color"], min_area=40))
    if out.get("ok") or frame.get("depth") is None:
        return out
    u, v = _project(K, T_base_camera, np.array([list(expected)], float))[0]
    h, w = frame["color"].shape[:2]
    window = np.zeros((h, w), np.uint8)
    cv2.circle(window, (int(round(u)), int(round(v))), PALE_WINDOW_PX, 255, -1)
    pale = blue_mask(frame["color"], min_area=40, lo=BLUE_PALE_LO) & window
    second = _wrist_bulb(frame, K, T_base_camera, expected, search_mm, pale)
    if second.get("ok"):
        second["pale_threshold"] = True
        return second
    return out


def _wrist_bulb(frame: dict, K: np.ndarray, T_base_camera: np.ndarray, expected: Sequence[float],
                search_mm: float, mask: np.ndarray) -> dict[str, Any]:
    color, depth = frame["color"], frame.get("depth")
    if depth is None:
        return {"ok": False, "reason": "没有深度图"}
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, count):
        vs, us = np.nonzero(labels == i)
        d = depth[vs, us].astype(float)
        ok = d > 0
        if ok.sum() < 25:
            continue
        pts = transform(T_base_camera, backproject(K, np.column_stack([us[ok], vs[ok]]).astype(float), d[ok]))
        z_top = float(np.percentile(pts[:, 2], 95))
        if abs(z_top - expected[2]) > BULB_Z_GATE:
            # the finger pads' felt reads bluish and sits right under the camera (11-25 16:58)
            continue
        lo, hi = np.percentile(pts[:, :2], 5, axis=0), np.percentile(pts[:, :2], 95, axis=0)
        centre = (lo + hi) / 2.0
        dist = math.dist([*centre, z_top], expected)
        candidates.append({"label": i, "xy": [float(centre[0]), float(centre[1])], "top_z": z_top,
                           "bottom_z": float(np.percentile(pts[:, 2], 5)), "extent_xy": (hi - lo).tolist(),
                           "pixels": int(len(vs)), "depth_points": int(ok.sum()), "dist": dist,
                           "pixel": [float(us.mean()), float(vs.mean())]})
    if not candidates:
        return {"ok": False, "reason": "腕部画面里没有带深度的蓝色块", "mask": mask}
    best = min(candidates, key=lambda c: c["dist"])
    if best["dist"] <= search_mm:
        # The paper is pale at the top: blue pixels miss part of the bulb. Use every depth point
        # near the blob axis for the top, and the ring around it for the tube rims.
        h, w = depth.shape
        vs, us = np.mgrid[0:h:2, 0:w:2]
        us, vs = us.reshape(-1), vs.reshape(-1)
        d = depth[vs, us].astype(float)
        ok = d > 0
        pts = transform(T_base_camera, backproject(K, np.column_stack([us[ok], vs[ok]]).astype(float), d[ok]))
        r = np.hypot(pts[:, 0] - best["xy"][0], pts[:, 1] - best["xy"][1])
        low = pts[:, 2] < expected[2] + BULB_Z_GATE       # nothing from the gripper above the bulb
        core = pts[(r < 9.0) & low]
        ring = pts[(r >= 14.0) & (r < 32.0) & low]
        if len(core) >= 20:
            best["top_z_blue"] = best["top_z"]
            best["xy_blue"] = list(best["xy"])
            best["top_z"] = float(np.percentile(core[:, 2], 97))
            upper = core[core[:, 2] > best["top_z"] - 15.0]
            if len(upper) >= 10:
                best["xy"] = [float(upper[:, 0].mean()), float(upper[:, 1].mean())]
        if len(ring) >= 50:
            best["rim_z"] = float(np.percentile(ring[:, 2], 97))
            best["ring_points"] = int(len(ring))
    if best["dist"] > search_mm:
        return {"ok": False, "reason": f"最近的蓝色块离预期 {best['dist']:.0f} mm > {search_mm:.0f}", "mask": mask,
                "candidates": [{k: v for k, v in c.items() if k != 'label'} for c in candidates]}
    best["mask_label"] = best.pop("label")
    best["mask"] = (labels == best["mask_label"]).astype(np.uint8)
    return {"ok": True, **best}


def tube_rim(depth: np.ndarray, K: np.ndarray, T_base_camera: np.ndarray, bulb_xy: Sequence[float],
             rim_z: float, search_mm: float = 45.0, seed: int = 0,
             prior: Sequence[float] | None = None, prior_tol: float = PRIOR_TOL_MM) -> dict[str, Any]:
    """Circle of the tube mouth around the dropper: depth points at the rim height near the bulb, RANSAC
    circle of the known radius (50 ml tube; free fits on 11-25 gave 14.0-15.0 mm). The radius is fixed
    because a leaning dropper hides part of the rim and a free fit on that arc shrinks (16:41: 12.0 mm).
    The dropper tip rests in the conical bottom, so the centre is where the bulb must go for the
    dropper to stand vertical."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w]
    us, vs, z = us.reshape(-1), vs.reshape(-1), depth.reshape(-1).astype(float)
    ok = z > 0
    pts = transform(T_base_camera, backproject(K, np.column_stack([us[ok], vs[ok]]).astype(float), z[ok]))
    sel = (np.abs(pts[:, 2] - rim_z) < 4.0) & (np.hypot(pts[:, 0] - bulb_xy[0], pts[:, 1] - bulb_xy[1]) < search_mm)
    xy = pts[sel, :2]
    if len(xy) < 60:
        return {"ok": False, "reason": f"管口高度附近深度点太少（{len(xy)}）"}
    R = TUBE_RIM_RADIUS
    rng = np.random.default_rng(seed)
    best, best_c = None, None
    for _ in range(600):
        a, b = xy[rng.choice(len(xy), 2, replace=False)]
        d = math.dist(a, b)
        if not 4.0 < d < 2 * R:
            continue
        mid, half = (a + b) / 2, math.sqrt(R * R - (d / 2) ** 2)
        n = np.array([-(b - a)[1], (b - a)[0]]) / d
        for c in (mid + half * n, mid - half * n):
            if prior is not None and math.dist(c, prior) > prior_tol:
                continue   # the dropper's own tube does not move; neighbours/lying tubes also give circles
            inl = np.abs(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]) - R) < 1.2
            if best is None or inl.sum() > best.sum():
                best, best_c = inl, c
    if best is None or best.sum() < max(60, 0.3 * len(xy)):
        return {"ok": False, "reason": f"管口圆拟合内点不足（{0 if best is None else int(best.sum())}/{len(xy)}）"}
    c = np.array(best_c, float)
    for _ in range(10):   # fixed-radius least squares on the inliers
        q = xy[best] - c
        r = np.hypot(q[:, 0], q[:, 1])
        J = -q / r[:, None]
        c = c - np.linalg.lstsq(J, r - R, rcond=None)[0]
    q = xy[best] - c
    residual = float(np.abs(np.hypot(q[:, 0], q[:, 1]) - R).mean())
    ang = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
    coverage = int(len(np.unique(np.floor((ang + 180) / 30))) * 30)   # degrees of rim seen, 30 deg bins
    if coverage < 120:
        return {"ok": False, "reason": f"管口只看到 {coverage}° 圆弧，圆心不可靠"}
    if prior is not None and math.dist(c, prior) > prior_tol + 2.0:
        return {"ok": False, "reason": f"管口圆心离上次 {math.dist(c, prior):.1f} mm（>{prior_tol}）"}
    return {"ok": True, "centre": [float(c[0]), float(c[1])], "radius": R, "inliers": int(best.sum()),
            "points": int(len(xy)), "residual_mm": residual, "arc_deg": coverage}


def left_in_tube(frame: dict, K: np.ndarray, T_base_camera: np.ndarray, centre_xy: Sequence[float],
                 rim_z: float, bulb_top_z: float) -> dict[str, Any]:
    """After the lift, look back down at the tube: a dropper left behind shows its blue bulb near the
    tube centre at the old bulb height. The rim fit proves the tube was in view. Used because a
    straightened grasp hides the bulb completely between the pads (11-25 16:21), so the finger-gap
    sliver test reads "not held" even when it is held."""
    rim = tube_rim(frame["depth"], K, T_base_camera, centre_xy, rim_z, search_mm=30.0)
    seen = wrist_bulb(frame, K, T_base_camera, [*centre_xy, bulb_top_z], search_mm=35.0)
    out = {"tube_visible": bool(rim.get("ok")), "rim": {k: v for k, v in rim.items() if k != "ok"},
           "bulb_in_tube": bool(seen.get("ok"))}
    if seen.get("ok"):
        out["bulb_seen_xy"] = [round(v, 1) for v in seen["xy"]]
        out["bulb_seen_top_above_table"] = round(seen["top_z"] - TABLE, 1)
    return out


def jaw_axis_xy(rpy: Sequence[float], T_tool_camera: np.ndarray) -> np.ndarray:
    """Unit XY direction (base) along which the jaws close: the wrist image's u axis (the fingers come in
    from the left and right of the wrist frame)."""
    R = pose_matrix([0, 0, 0, *rpy])[:3, :3] @ T_tool_camera[:3, :3]
    a = R[:2, 0]
    return a / np.linalg.norm(a)


def jaw_rotation_deg(rpy: Sequence[float], T_tool_camera: np.ndarray, lean_xy: Sequence[float]) -> float:
    """Smallest rotation about vertical (deg, -90..90) that lines the jaw axis up with the lean."""
    a = jaw_axis_xy(rpy, T_tool_camera)
    delta = math.degrees(math.atan2(lean_xy[1], lean_xy[0]) - math.atan2(a[1], a[0]))
    return (delta + 90.0) % 180.0 - 90.0


def pad_offset_xy(rpy: Sequence[float]) -> np.ndarray:
    """Base XY of the pad midpoint relative to the tool axis (hand-eye: pads at Tool0 [+3, 0])."""
    from chem_demo.gripper_geometry import PAD_CENTRE_TOOL_X_MM
    R = pose_matrix([0, 0, 0, *rpy])[:3, :3]
    return (R @ np.array([PAD_CENTRE_TOOL_X_MM, 0.0, 0.0]))[:2]


def grab_top() -> np.ndarray:
    """Top D405 colour as 1280x720 (calibrated size). 1280x720 no longer streams on the shared USB hub
    (11-25: fails every time, 848x480 every time works); 848x480 has the same field of view, so
    scaling it up gives the calibrated geometry to well under the ~1 cm the coarse step needs."""
    from chem_demo.cameras import TOP_ROTATE_180, grab_realsense
    color, _, _ = grab_realsense(camera_serial("top"), width=848, height=480)
    color = cv2.resize(color, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return cv2.rotate(color, cv2.ROTATE_180) if TOP_ROTATE_180 else color


def draw_wrist(color: np.ndarray, seen: dict, K: np.ndarray, T_base_camera: np.ndarray,
               tips_xyz: Sequence[float], path: Path) -> None:
    view = color.copy()
    if seen.get("ok"):
        view[seen["mask"] > 0] = (0.5 * view[seen["mask"] > 0] + [127, 0, 127]).astype(np.uint8)
        top = np.array([[*seen["xy"], seen["top_z"]]])
        px = _project(K, T_base_camera, top)[0]
        cv2.drawMarker(view, (int(px[0]), int(px[1])), (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
    # where the fingertips' vertical line would pass at the bulb top
    if seen.get("ok"):
        under = _project(K, T_base_camera, np.array([[tips_xyz[0], tips_xyz[1], seen["top_z"]]]))[0]
        cv2.drawMarker(view, (int(under[0]), int(under[1])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.putText(view, "red: bulb top  green: tool axis at that height", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    cv2.imwrite(str(path), view)


def _project(K: np.ndarray, T_base_camera: np.ndarray, pts: np.ndarray) -> np.ndarray:
    cam = transform(np.linalg.inv(T_base_camera), pts)
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3]


# ---------------------------------------------------------------------------------------- run
class DropperPick:
    def __init__(self, arm: Any, args: argparse.Namespace, out_dir: Path, *, stop_pressed=lambda: False):
        from task_skills.run_cartesian_jog import Jog, Limits
        self.arm, self.args, self.out_dir, self.stop_pressed = arm, args, out_dir, stop_pressed
        state = arm.read()
        self.start_pose = [float(v) for v in state["pose"]]
        xyz = self.start_pose[:3]
        limits = Limits(xyz, args.radius, TABLE + args.min_tip, xyz[2] + 5.0)
        self.jog = Jog(arm, limits, record_path=out_dir / "unused.csv", ask=lambda _p: "", out=log_line,
                       stop_pressed=stop_pressed)
        model = json.loads(RIGHT_MODEL.read_text(encoding="utf-8"))
        self.T_tool_camera = np.asarray(model["T_flange_reference_camera"], float)
        self.camera = None
        self.K = None
        self.log: dict[str, Any] = {"start": datetime.now().isoformat(timespec="seconds"),
                                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                    "start_pose": self.start_pose, "steps": []}
        self.frames = 0
        last = last_tube_centre()
        self.tube_prior = None if last is None else last["xy"]
        self.log["tube_prior"] = last

    # -- helpers
    def step(self, name: str, **fields: Any) -> None:
        self.log["steps"].append({"step": name, "t": round(time.time(), 2), **fields})

    def save(self) -> None:
        (self.out_dir / "dropper_run.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=2, default=str),
                                                       encoding="utf-8")

    def straight(self, target: Sequence[float], label: str) -> None:
        target = [round(float(v), 3) for v in target]
        finals = self.jog.check_route([target])
        if self.stop_pressed():
            raise RuntimeError("操作员停止")
        self.jog.run_route([target], finals)
        state = self.arm.settle()
        self.step("move", label=label, target=target, measured=[round(v, 2) for v in state["pose"][:3]])
        log_line(f"{label}：到达 {[round(v, 1) for v in target]}")

    def travel(self, target: Sequence[float], label: str) -> None:
        legs, finals, travel_z = self.jog._plan_travel([round(float(v), 3) for v in target], label, "-")
        if self.stop_pressed():
            raise RuntimeError("操作员停止")
        self.jog.run_route(legs, finals)
        state = self.arm.settle()
        self.step("travel", label=label, legs=legs, travel_z=travel_z,
                  measured=[round(v, 2) for v in state["pose"][:3]])
        log_line(f"{label}：到达 {[round(v, 1) for v in target]}（平移高度 {travel_z:.0f}）")

    def open_camera(self) -> None:
        from chem_demo.wrist_camera import LiveWristCamera
        self.camera = LiveWristCamera(camera_serial("right_wrist"), fps=15, warmup=6)
        self.K = np.asarray(self.camera.K, float)
        self.log["wrist_K"] = self.K.tolist()

    def look(self, label: str, expected: Sequence[float]) -> dict[str, Any]:
        state = self.arm.settle()
        pose = [float(v) for v in state["pose"]]
        frame = self.camera.grab(flush=3, require_fresh_pair=True)
        self.frames += 1
        name = f"{self.frames:02d}_{label}"
        folder = self.out_dir / "frames"
        folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / f"{name}.png"), frame["color"])
        np.save(folder / f"{name}_depth.npy", frame["depth"].astype(np.float32))
        T = pose_matrix(pose) @ self.T_tool_camera
        self.last_view = {"depth": frame["depth"], "T": T, "name": name}
        seen = wrist_bulb(frame, self.K, T, expected, search_mm=self.args.search_mm)
        plane = platform_height(T, self.K, frame["depth"],
                                cv2.dilate((seen.get("mask") if seen.get("mask") is not None
                                            else np.zeros(frame["depth"].shape, np.uint8)).astype(np.uint8),
                                           np.ones((25, 25), np.uint8)), TABLE, window=40.0)
        draw_wrist(frame["color"], seen, self.K, T, pose[:3], folder / f"{name}_seen.jpg")
        record = {k: v for k, v in seen.items() if k != "mask"}
        record.update(frame=name, pose=[round(v, 3) for v in pose], table_plane=plane)
        self.step("look", **record)
        if not seen.get("ok"):
            raise RuntimeError(f"右腕画面 {name} 里没找到捏头：{seen.get('reason')}")
        rim_txt = f"，管口离台面 {seen['rim_z'] - TABLE:.0f} mm" if seen.get("rim_z") else ""
        log_line(f"  {name}：捏头顶 ({seen['xy'][0]:.1f}, {seen['xy'][1]:.1f}, {seen['top_z']:.1f})，"
                 f"顶离台面 {seen['top_z'] - TABLE:.0f} mm{rim_txt}；"
                 f"台面 {plane.get('z', '?')}（应 {TABLE}）")
        return record

    def refine(self, tip_above_table: float, guess: list[float], label: str) -> list[float]:
        """Look and move over the bulb at this height until the correction is below tol."""
        args = self.args
        z = TABLE + tip_above_table
        tag = {"高位": "high", "低位": "low"}.get(label, "look")
        for it in range(args.max_iter):
            seen = self.look(f"{tag}{it}", guess)
            guess = [seen["xy"][0], seen["xy"][1], seen["top_z"]]
            error = math.dist(guess[:2], self.jog.xyz[:2])
            shift = math.dist(guess[:2], self.coarse_xy)
            log_line(f"  {label} 第 {it + 1} 次：指尖轴离捏头 {error:.1f} mm（离粗位置 {shift:.0f} mm）")
            if shift > args.max_shift:
                raise RuntimeError(f"腕部结果离粗位置 {shift:.0f} mm > {args.max_shift:.0f}：可能认错，停止")
            if error <= args.tol and it > 0:
                return guess
            target = list(guess[:2])
            if error > args.max_step:
                k = args.max_step / error
                target = [self.jog.xyz[i] + k * (guess[i] - self.jog.xyz[i]) for i in range(2)]
            if error > args.tol * 0.5:
                self.straight([target[0], target[1], z], f"{label}对准")
        raise RuntimeError(f"{args.max_iter} 次后仍未对准到 {args.tol} mm 以内")

    def grip(self, code: int, speed: int, label: str) -> dict:
        feedback = self.arm.grip_to(int(code), speed)
        self.step("gripper", label=label, **feedback)
        log_line(f"夹爪{label}：{feedback.get('start')} -> {feedback.get('final')}（目标 {code}）")
        return feedback

    def rotate_tool(self, rz: float, label: str) -> None:
        """Turn the tool about vertical to yaw ``rz`` in place. Checked every few degrees (the IK branch
        check refuses >15 deg joint jumps per sample), then sent as one MovL."""
        start = list(self.jog.rpy)
        delta = math.remainder(rz - start[2], 360.0)
        steps = max(1, math.ceil(abs(delta) / 5.0))
        seed = list(self.jog.joints)
        for k in range(1, steps + 1):
            rpy = [start[0], start[1], start[2] + delta * k / steps]
            seed = self.jog.check(list(self.jog.xyz), rpy, seed)
        if self.stop_pressed():
            raise RuntimeError("操作员停止")
        final = [start[0], start[1], start[2] + delta]
        self.arm.move(list(self.jog.xyz), final, seed)
        self.arm.wait(list(self.jog.xyz), final, seed, self.stop_pressed)
        self.jog.rpy, self.jog.joints = final, list(seed)
        state = self.arm.settle()
        self.step("rotate", label=label, rz_from=round(start[2], 2), rz_to=round(final[2], 2),
                  measured=[round(v, 2) for v in state["pose"]])
        log_line(f"{label}：工具绕竖直轴转 {delta:+.1f}°（rz {start[2]:.1f} -> {final[2]:.1f}）")

    def measure_lean(self, bulb: Sequence[float], rim_z: float | None, label: str) -> dict[str, Any]:
        """Tube-mouth centre from the last wrist frame; lean = bulb minus centre (horizontal)."""
        if rim_z is None:
            raise RuntimeError("没测到管口高度，无法求管口圆心")
        view = self.last_view
        rim = tube_rim(view["depth"], self.K, view["T"], bulb[:2], rim_z, prior=self.tube_prior)
        if not rim.get("ok"):
            if self.tube_prior is None:
                raise RuntimeError(f"{label}：{rim['reason']}（没有合爪）")
            # A dropper leaning over its own rim hides most of it (17:00: 90 deg arc). The tube does not
            # move (centre within ~3 mm across all 11-25 runs), so use the last measured centre.
            log_line(f"  {label}：{rim['reason']}，用上次测得的管口圆心")
            rim = {"ok": True, "centre": list(self.tube_prior), "radius": TUBE_RIM_RADIUS, "inliers": 0,
                   "from_prior": True, "fit_failure": rim["reason"]}
        lean = [bulb[0] - rim["centre"][0], bulb[1] - rim["centre"][1]]
        out = {**rim, "frame": view["name"], "bulb_xy": [round(v, 2) for v in bulb[:2]],
               "lean_xy": [round(v, 2) for v in lean], "lean_mm": round(math.hypot(*lean), 2)}
        self.step("rim", label=label, **out)
        log_line(f"  {label}：管口圆心 ({rim['centre'][0]:.1f}, {rim['centre'][1]:.1f}) 半径 {rim['radius']:.1f} mm"
                 f"（内点 {rim['inliers']}）；捏头偏 ({lean[0]:+.1f}, {lean[1]:+.1f}) = {out['lean_mm']:.1f} mm")
        if out["lean_mm"] > self.args.max_lean:
            raise RuntimeError(f"捏头偏离管口圆心 {out['lean_mm']:.0f} mm > {self.args.max_lean:.0f}：可能认错管口，停止")
        return out

    def snap_down(self) -> None:
        from task_skills.run_cartesian_jog import down_tilt_deg
        tilt = down_tilt_deg(self.jog.rpy)
        if tilt > 0.05:
            if tilt > TOOL_DOWN_TOL_DEG:
                raise RuntimeError(f"工具偏离竖直 {tilt:.2f}° > {TOOL_DOWN_TOL_DEG}°，不自动摆正")
            self.jog.snap_down()
            self.step("snap_down", tilt_before=round(tilt, 3), rpy=self.jog.rpy)

    # -- sequence
    def run(self, coarse: dict) -> str:
        args = self.args
        self.coarse_xy = coarse["xy"]
        self.log["coarse"] = coarse
        self.snap_down()
        home = list(self.jog.xyz)
        home_rpy = list(self.jog.rpy)
        if self.arm.gripper_position() < GRIPPER_OPEN - 5:
            self.grip(GRIPPER_OPEN, 10, "张开")
        self.open_camera()
        guess = [*self.coarse_xy, TABLE + ASSUMED_BULB_TOP]
        self.travel([*self.coarse_xy, TABLE + args.hover], "到捏头粗位置上方")
        for n in range(1, args.attempts + 1):
            self.attempt_no = n
            if n > 1:
                log_line(f"—— 第 {n} 次抓取 ——")
            result = self.attempt(guess, home, home_rpy)
            if result != "retry":
                return result
            plan = self.log["plan"]
            guess = [*plan["bulb_xy"], TABLE + plan["bulb_top_above_table"]]
        return "提起后没确认夹住（重试次数用完，夹爪已张开，在悬停高度）"

    def attempt(self, guess: list[float], home: list[float], home_rpy: list[float]) -> str:
        """Align, (straighten,) grasp and lift once. Returns the result, or "retry" when the lift came
        up empty with the dropper back in its tube (jaws opened, arm at the hover height)."""
        args = self.args
        tag = "" if self.attempt_no == 1 else f"_try{self.attempt_no}"
        guess = self.refine(args.hover, guess, "高位")
        if args.straighten:
            # The dropper leans in its 50 ml tube (tip in the conical bottom, stem on the rim). Closing
            # the jaws across the lean squeezes it as it leans (11-25 15:55: ~13 mm off at the bulb,
            # jaws perpendicular to it). Turn the jaws onto the lean, then close around the tube axis:
            # the closing pad pushes the bulb over the tip and the dropper pivots upright.
            lean = self.measure_lean(guess, self._last_look().get("rim_z"), "高位管口")
            if lean["lean_mm"] >= args.align_min:
                turn = jaw_rotation_deg(self.jog.rpy, self.T_tool_camera, lean["lean_xy"])
                self.log["jaw_turn_deg"] = round(turn, 2)
                if abs(turn) > 1.0:
                    self.rotate_tool(self.jog.rpy[2] + turn, "夹爪合拢方向对准倾斜方向")
                    guess = self.refine(args.hover, guess, "高位")
        bulb_top = guess[2] - TABLE
        pre_tip = bulb_top + args.pre_clear
        if pre_tip < args.hover - 5:
            self.straight([self.jog.xyz[0], self.jog.xyz[1], TABLE + pre_tip], "降到捏头上方")
            guess = self.refine(pre_tip, guess, "低位")
        bulb_top = guess[2] - TABLE
        seen = self._last_look()
        width = args.bulb_width   # blue pixels see only part of the bulb (11-24 22:28 probe: ~10-12 mm)
        rim = seen.get("rim_z")
        rim_above = None if rim is None else rim - TABLE
        centre_xy, lean_mm = list(guess[:2]), 0.0
        if args.straighten:
            lean = self.measure_lean(guess, rim, "低位管口")
            self.log["lean"] = lean
            if lean["lean_mm"] >= args.align_min:
                jaw = jaw_axis_xy(self.jog.rpy, self.T_tool_camera)
                along = abs(float(np.dot(jaw, lean["lean_xy"])))
                across = math.sqrt(max(0.0, lean["lean_mm"] ** 2 - along ** 2))
                if across > args.max_across:
                    raise RuntimeError(f"倾斜方向与夹爪合拢方向还差 {across:.1f} mm（>{args.max_across}），没有合爪")
                pad = pad_offset_xy(self.jog.rpy)
                centre_xy = [lean["centre"][0] - pad[0], lean["centre"][1] - pad[1]]
                lean_mm = lean["lean_mm"]
        tips = bulb_top - args.grip_depth
        if rim_above is not None:
            tips = max(tips, rim_above + args.rim_clear)
        if tips > bulb_top - 12.0:
            raise RuntimeError(f"管口 {rim_above:.0f} mm 离捏头顶 {bulb_top:.0f} mm 太近，夹不住捏头")
        plan = {"bulb_xy": [round(v, 2) for v in guess[:2]], "bulb_top_above_table": round(bulb_top, 1),
                "rim_above_table": None if rim_above is None else round(rim_above, 1),
                "bulb_width_mm": round(width, 1), "grip_tips_above_table": round(tips, 1),
                "grasp_flange_xy": [round(v, 2) for v in centre_xy], "lean_mm": round(lean_mm, 2),
                "pre_close_code": int(min(255, round((width + 2 * args.jaw_margin + 2 * lean_mm) / MM_PER_CODE))),
                "close_code": int(max(0, round((width - args.squeeze) / MM_PER_CODE))),
                "lift_mm": args.lift}
        self.log["plan"] = plan
        log_line(f"计划：{plan}")
        if args.until == "look":
            self.grip(GRIPPER_OPEN, 10, "保持张开")
            if math.dist(centre_xy, self.jog.xyz[:2]) > 0.5:
                self.straight([*centre_xy, self.jog.xyz[2]], "移到管口圆心正上方（只看）")
                self.look("centre", guess)
            self.straight([self.jog.xyz[0], self.jog.xyz[1], TABLE + args.hover], "回到悬停高度")
            self.restore_yaw(home_rpy)
            self.straight(home, "直线回拍照位")
            return "只看：已对准并回拍照位"

        grasp_z = TABLE + plan["grip_tips_above_table"]
        if grasp_z < TABLE + args.min_tip:
            raise RuntimeError(f"抓取高度 {plan['grip_tips_above_table']} mm 低于下限 {args.min_tip}")
        # Check the whole descent + lift before the jaws move.
        lift_z = grasp_z + args.lift
        out_xyz = self.retreat_point([*centre_xy, lift_z])
        # The whole descent + lift (+ retreat) must be reachable before the jaws move: at the first
        # rack position the arm topped out 82 mm up with the dropper tip still in its tube (00:59).
        saved = (list(self.jog.xyz), list(self.jog.joints))
        try:
            route = ([[*centre_xy, self.jog.xyz[2]], [*centre_xy, grasp_z], [*centre_xy, lift_z]]
                     + ([out_xyz] if out_xyz else []))
            self.jog.check_route(route)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"提起 {args.lift:.0f} mm 够不到（{error}）：没有合爪。把试管架往右臂底座方向挪近") from error
        finally:
            self.jog.xyz, self.jog.joints = saved
        self.grip(plan["pre_close_code"], 5, "预合到捏头宽度+余量" + ("+倾斜" if lean_mm else ""))
        if math.dist(centre_xy, self.jog.xyz[:2]) > 0.5:
            self.straight([*centre_xy, self.jog.xyz[2]], f"移到管口圆心正上方（捏头偏 {lean_mm:.1f} mm）")
        self.straight([self.jog.xyz[0], self.jog.xyz[1], grasp_z], "下降到抓取高度")
        before = self.camera.grab(flush=3, require_fresh_pair=True)
        cv2.imwrite(str(self.out_dir / "frames" / f"grasp_before_close{tag}.png"), before["color"])
        fb = self.grip(plan["close_code"], 3, "合爪")
        held_code = fb.get("final")
        if args.suck:
            # The closing jaws squeeze the bulb with the tip under water; opening them a little lets
            # the bulb spring back and draw water up. Fixed code from Kunyu (11-25): no camera check.
            sucked = self.grip(args.suck_code, 2, "松一点吸液")
            time.sleep(args.suck_wait)
            held_code = sucked.get("final")
            self.log["suck"] = {"squeezed_code": fb.get("final"), "released_code": held_code,
                                "wait_s": args.suck_wait}
        closed = self.camera.grab(flush=3, require_fresh_pair=True)
        cv2.imwrite(str(self.out_dir / "frames" / f"grasp_closed{tag}.png"), closed["color"])
        self.log["grasp"] = {"pose": [*self.jog.xyz, *self.jog.rpy], "held_code": held_code}
        self.save()
        self.straight([self.jog.xyz[0], self.jog.xyz[1], lift_z], f"竖直提起 {lift_z - grasp_z:.0f} mm")
        if out_xyz is not None:
            self.straight(out_xyz, f"斜向内上方移出 {args.retreat_in:.0f} mm")
        lifted = self.camera.grab(flush=3, require_fresh_pair=True)
        cv2.imwrite(str(self.out_dir / "frames" / f"lifted{tag}.png"), lifted["color"])
        np.save(self.out_dir / "frames" / f"lifted_depth{tag}.npy", lifted["depth"].astype(np.float32))
        self.vertical_overlay(lifted)
        held = self.held_check(closed, lifted)
        if lean_mm or args.straighten:
            pose = [*self.jog.xyz, *self.jog.rpy]
            tube = left_in_tube(lifted, self.K, pose_matrix(pose) @ self.T_tool_camera,
                                self.log.get("lean", {}).get("centre", plan["bulb_xy"]),
                                TABLE + plan["rim_above_table"], TABLE + plan["bulb_top_above_table"])
            held["tube_check"] = tube
            held["sliver_held"] = held["held"]
            # Either sign of the dropper coming along is enough, unless its bulb is seen back in the tube.
            # 11-25 runs: 15:55 sliver+tube, 16:21 tube only (bulb hidden), 16:37 sliver only (rim fit
            # failed), 16:27 slipped = bulb in tube, sliver at another depth.
            held["held"] = (not tube["bulb_in_tube"]) and (
                held["sliver_held"] or tube["tube_visible"])
        self.log["held_check"] = held
        self.log.setdefault("attempts", []).append({"attempt": self.attempt_no, "plan": plan, "held_check": held,
                                                    "grasp": self.log["grasp"]})
        log_line(f"提起后腕部判断：{held}")
        tube = held.get("tube_check") or {}
        if not held.get("held") and tube.get("bulb_in_tube") and self.attempt_no < args.attempts:
            self.save()
            log_line("滴管还在管里（提起时滑脱）：张开夹爪，回悬停高度重新对准")
            self.grip(GRIPPER_OPEN, 10, "张开（重试）")
            self.straight([*plan["bulb_xy"], TABLE + args.hover], "回到捏头上方悬停高度")
            return "retry"
        self.log["lifted_pose"] = [*self.jog.xyz, *self.jog.rpy]
        self.log["vertical_top_z"] = lift_z
        self.save()
        self.camera.pause()
        self.side_frames("lifted")
        return "提起" if held.get("held") else "提起后没确认夹住"

    def _last_look(self) -> dict[str, Any]:
        return [s for s in self.log["steps"] if s["step"] == "look"][-1]

    def restore_yaw(self, rpy: Sequence[float]) -> None:
        if abs(math.remainder(self.jog.rpy[2] - rpy[2], 360.0)) > 0.5:
            self.rotate_tool(rpy[2], "工具转回原朝向")

    def vertical_overlay(self, frame: dict) -> None:
        """Lifted wrist frame with the vertical line through the pad midpoint drawn on it: a vertical
        dropper hangs along it (visual check only; the stem is transparent, no usable depth)."""
        pose = [*self.jog.xyz, *self.jog.rpy]
        T = pose_matrix(pose) @ self.T_tool_camera
        pad = pad_offset_xy(self.jog.rpy)
        top = np.array([self.jog.xyz[0] + pad[0], self.jog.xyz[1] + pad[1], self.jog.xyz[2] - 16.0])
        pts = np.array([top - [0, 0, d] for d in range(0, 181, 10)])
        uv = _project(self.K, T, pts)
        view = frame["color"].copy()
        for a, b in zip(uv[:-1], uv[1:]):
            cv2.line(view, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 0), 1)
        cv2.imwrite(str(self.out_dir / "frames" / "lifted_vertical.jpg"), view)

    def retreat_point(self, at: Sequence[float]) -> list[float] | None:
        """Second leg after the vertical lift: toward the right base and up (reach grows ~1 mm per mm)."""
        if self.args.retreat_in <= 0:
            return None
        r = math.hypot(at[0], at[1])
        k = (r - self.args.retreat_in) / r
        return [at[0] * k, at[1] * k, at[2] + self.args.retreat_rise * self.args.retreat_in]

    def held_check(self, closed: dict, lifted: dict) -> dict[str, Any]:
        """The camera rides with the gripper. Closed jaws hide most of the bulb; what stays visible is a
        blue sliver in the finger gap at the pads. Held = that sliver is still there after the lift,
        at about the same depth (a bulb left behind would be ~lift mm farther or gone).
        (The first version compared whole-image blue masks and picked up a blue cap on the table, 00:58.)"""
        pad_tool = np.array([3.0, 0.0, -16.0, 1.0])            # pad midpoint, hand-eye (fingertip) frame
        cam = np.linalg.inv(self.T_tool_camera) @ pad_tool
        uv = self.K @ cam[:3]
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        h, w = closed["color"].shape[:2]
        roi = np.zeros((h, w), bool)
        y0, y1 = max(0, int(v - 70)), min(h, int(v + 40))
        x0, x1 = max(0, int(u - 35)), min(w, int(u + 35))
        roi[y0:y1, x0:x1] = True
        out = {"pad_px": [round(float(u), 1), round(float(v), 1)], "roi": [x0, y0, x1, y1]}
        for tag, frame in (("closed", closed), ("lifted", lifted)):
            blue = (blue_mask(frame["color"], 5) > 0) & roi
            depth = frame["depth"][blue & (frame["depth"] > 0)]
            out[f"{tag}_blue_px"] = int(blue.sum())
            out[f"{tag}_depth_mm"] = None if depth.size == 0 else round(float(np.median(depth)), 1)
            view = frame["color"].copy()
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
            view[blue] = (255, 0, 255)
            cv2.imwrite(str(self.out_dir / "frames" / f"held_{tag}.jpg"), view)
        same_depth = (out["closed_depth_mm"] is not None and out["lifted_depth_mm"] is not None
                      and abs(out["closed_depth_mm"] - out["lifted_depth_mm"]) < 25.0)
        out["held"] = out["closed_blue_px"] >= 10 and out["lifted_blue_px"] >= 10 and same_depth
        return out

    def side_frames(self, tag: str) -> None:
        from chem_demo.cameras import grab_c920
        try:
            cv2.imwrite(str(self.out_dir / f"{tag}_top.png"), grab_top())
        except Exception as error:  # noqa: BLE001 - evidence only
            log_line(f"顶部相机取图失败：{error}")
        try:
            cv2.imwrite(str(self.out_dir / f"{tag}_front.png"), grab_c920())
        except Exception as error:  # noqa: BLE001
            log_line(f"正面 C920 取图失败：{error}")

    def put_back(self, saved: dict) -> str:
        grasp = saved["grasp"]
        lifted = saved["lifted_pose"]
        here = list(self.jog.xyz)
        if math.dist(here, lifted[:3]) > 3.0:
            raise RuntimeError(f"右臂不在上次提起的位置（差 {math.dist(here, lifted[:3]):.1f} mm），不自动放回")
        gx, gy, gz = grasp["pose"][:3]
        top = saved.get("vertical_top_z", gz + saved["plan"]["lift_mm"])
        if math.dist(here[:2], [gx, gy]) > 1.0:
            self.straight([gx, gy, top], "放回：斜向回到抓取点正上方")
        self.straight([gx, gy, gz], "放回：竖直下降到抓取高度")
        here = [gx, gy, gz]
        self.grip(saved["plan"]["pre_close_code"], 5, "松开")
        self.straight([here[0], here[1], grasp["pose"][2] + self.args.pre_clear + self.args.grip_depth], "竖直退出")
        self.grip(GRIPPER_OPEN, 10, "张开")
        self.straight([gx, gy, TABLE + self.args.hover], "升到悬停高度")
        self.restore_yaw(saved["start_pose"][3:6])
        self.straight(saved["start_pose"][:3], "直线回拍照位")
        return "已放回"


def last_tube_centre(max_age_h: float = 12.0, runs: int = 5) -> dict[str, Any] | None:
    """Tube-mouth centre: median of the newest ``runs`` live runs that measured it (recent, near the
    rack hint). A median so one bad fit (17:03: 10 mm off) does not carry into the next run."""
    found = []
    for path in sorted(DATA.glob("live-*/dropper_run.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        if time.time() - path.stat().st_mtime > max_age_h * 3600 or len(found) >= runs:
            break
        try:
            lean = json.loads(path.read_text(encoding="utf-8")).get("lean") or {}
        except (OSError, ValueError):
            continue
        centre = lean.get("centre")
        if centre and not lean.get("from_prior") and math.dist(centre, RACK_HINT_XY) <= RACK_HINT_RADIUS:
            found.append((path.parent.name, centre))
    if not found:
        return None
    xy = np.median(np.array([c for _, c in found], float), axis=0)
    return {"xy": [float(xy[0]), float(xy[1])], "run": f"median of {len(found)}: " + ", ".join(n for n, _ in found)}


def coarse_from_top(args, out_dir: Path) -> dict[str, Any]:
    if args.bulb_xy:
        return {"xy": list(args.bulb_xy), "source": "command line"}
    from chem_demo.camera_drift import load_corrected_camera
    from vision_pick.cup_locate import DEFAULT_SESSION
    image = grab_top()
    cv2.imwrite(str(out_dir / "top.png"), image)
    found = top_bulb(image, load_corrected_camera(DEFAULT_SESSION, "right"))
    view = image.copy()
    if found.get("ok"):
        cv2.circle(view, (int(found["pixel"][0]), int(found["pixel"][1])), 18, (255, 0, 255), 2)
    cv2.imwrite(str(out_dir / "top_bulb.jpg"), view)
    if not found.get("ok"):
        last = last_tube_centre()
        if last is None:
            raise RuntimeError(found["reason"])
        # From the top the paper over the orange rack can read grey (11-25 16:40); the tube itself
        # does not move, and the wrist loop searches +-search_mm around this anyway.
        log_line(f"顶部相机没认出捏头（{found['reason']}）：用上次右腕测得的管口圆心 "
                 f"({last['xy'][0]:.1f}, {last['xy'][1]:.1f})（{last['run']}）作粗位置")
        return {"xy": last["xy"], "source": f"last wrist tube centre ({last['run']})", "top": found}
    log_line(f"顶部相机：捏头约在 ({found['xy'][0]:.1f}, {found['xy'][1]:.1f})（像素 {found['pixel']}，面积 {found['area']}）")
    return {**found, "source": "top camera blue blob"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="右臂从试管架夹起滴管（蓝色捏头），竖直提起")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="只读：顶部定位 + 控制器逆解查整条路线，不动")
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--put-back", action="store_true", help="把手里的滴管放回原管，回拍照位")
    mode.add_argument("--drip", action="store_true",
                      help="原地滴加：从当前夹爪值起每滴再捏紧 --drip-step 码，共 --drips 滴，中间不松（手里须有吸了水的滴管）")
    mode.add_argument("--go-home", action="store_true",
                      help="手里没有滴管时用：张开夹爪、转回原朝向、直线回拍照位（右臂须在悬停高度以上）")
    p.add_argument("--until", choices=("look", "lift"), default="lift")
    p.add_argument("--speed", type=int, default=10, help="直线速度 %%（1-10）")
    p.add_argument("--bulb-xy", type=float, nargs=2, help="跳过顶部相机，直接给粗位置（右臂基座 mm）")
    p.add_argument("--hover", type=float, default=200.0, help="高位对准时指尖离台面 mm")
    p.add_argument("--pre-clear", type=float, default=25.0, help="低位对准时指尖在捏头顶上方 mm")
    p.add_argument("--grip-depth", type=float, default=40.0,
                   help="合爪时指尖在捏头顶下方 mm（受管口限制）。11-25 16:27 用 30 时捏头落在三角夹垫窄尖处，提起滑脱")
    p.add_argument("--rim-clear", type=float, default=8.0, help="合爪时指尖至少高出管口 mm")
    p.add_argument("--bulb-width", type=float, default=11.0, help="捏头直径 mm（蓝色只看到一部分，先用估计值）")
    p.add_argument("--jaw-margin", type=float, default=12.0, help="预合时每侧余量 mm")
    p.add_argument("--squeeze", type=float, default=7.0,
                   help="合爪目标比捏头宽度小这么多 mm（11 mm 捏头夹到约 4 mm；4 时夹到 6.8 mm，提起滑脱过一次）")
    p.add_argument("--lift", type=float, default=160.0,
                   help="竖直提起 mm（11-25 实测 130 时管尖只高出管口几 mm，留 3 cm 余量）")
    p.add_argument("--attempts", type=int, default=2, help="提起后滴管仍在管里时，张开重新对准再抓，总次数")
    p.add_argument("--suck", action="store_true", help="合爪捏紧后松到 --suck-code 吸液，再提起（管尖须在水里）")
    p.add_argument("--suck-code", type=int, default=SUCK_CODE,
                   help=f"吸液时夹爪松到的码值（默认 {SUCK_CODE} ≈ {SUCK_CODE * 91.6 / 255:.1f} mm；合紧时停在约 14）")
    p.add_argument("--suck-wait", type=float, default=2.0, help="松开后等水上来的秒数")
    p.add_argument("--drip-step", type=int, default=DRIP_STEP_CODE, help="每滴多捏紧的码值（1 码 ≈ 0.36 mm）")
    p.add_argument("--drip-min", type=int, default=DRIP_MIN_CODE, help="最紧捏到的码值（捏头捏扁时约 13）")
    p.add_argument("--drips", type=int, default=5, help="滴数")
    p.add_argument("--drip-dwell", type=float, default=1.5, help="每滴后停留秒数")
    p.add_argument("--drip-speed", type=int, default=2, help="夹爪速度（1-10，慢一点容易出单滴）")
    p.add_argument("--no-straighten", dest="straighten", action="store_false",
                   help="不做扶正（夹爪不转向、对准捏头而不是管口圆心）")
    p.add_argument("--align-min", type=float, default=2.0, help="捏头偏离管口圆心小于此值就不转向/不平移 mm")
    p.add_argument("--max-lean", type=float, default=25.0, help="捏头偏离管口圆心超过此值视为认错 mm")
    p.add_argument("--max-across", type=float, default=3.0, help="转向后倾斜在夹爪不合拢方向上的残余上限 mm")
    p.add_argument("--retreat-in", type=float, default=0.0, help="竖直提起后再朝底座方向收回 mm（0 = 不收）")
    p.add_argument("--retreat-rise", type=float, default=0.9, help="收回时每 mm 升高多少 mm")
    p.add_argument("--tol", type=float, default=1.5)
    p.add_argument("--max-step", type=float, default=40.0)
    p.add_argument("--max-shift", type=float, default=70.0)
    p.add_argument("--max-iter", type=int, default=5)
    p.add_argument("--search-mm", type=float, default=70.0)
    p.add_argument("--min-tip", type=float, default=80.0, help="指尖离台面的最低高度 mm")
    p.add_argument("--radius", type=float, default=320.0, help="以拍照位为圆心的水平软限位 mm")
    return p


def _stop():
    try:
        from chem_demo.run_neck_pick import _stop_key
        return _stop_key()
    except Exception:  # noqa: BLE001 - no console when started in the background
        return False


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from chem_demo.bottle_observation import read_passive_robot_state
    from chem_demo.run_neck_pick import NeckArm
    from chem_demo.stations import near_photo
    from task_skills.run_cartesian_jog import DEFAULT_PROFILE, start_refusal, snap_down_rpy

    profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
    states = {a: read_passive_robot_state(profile["arms"][a]["ip"]) for a in ("right", "left")}
    tag = ("preflight" if args.preflight else "putback" if args.put_back else "gohome" if args.go_home
           else "drip" if args.drip
           else f"live-{args.until}")
    out_dir = DATA / f"{tag}-{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_line(f"记录目录 {out_dir}")
    for a, s in states.items():
        log_line(f"{a}: mode {s['mode']} pose {[round(v, 1) for v in s['pose']]}")
    if not near_photo("left", states["left"]["pose"]):
        log_line("左臂不在拍照位，拒绝启动（没有发送任何指令）")
        return 5
    if not (args.put_back or args.go_home or args.drip) and not near_photo("right", states["right"]["pose"]):
        log_line("右臂不在拍照位，拒绝启动（没有发送任何指令）")
        return 5
    problem = start_refusal(states["right"]["mode"])
    if problem:
        log_line(f"拒绝启动：{problem}")
        return 5

    arm = NeckArm(profile, "right", args.speed, log=log_line)
    pick = None
    result, code = "未开始", 1
    try:
        if args.preflight:
            coarse = coarse_from_top(args, out_dir)
            pick = DropperPick(arm, args, out_dir)
            pick.jog.rpy = snap_down_rpy(pick.jog.rpy)
            hover = [*coarse["xy"], TABLE + args.hover]
            legs, finals, tz = pick.jog._plan_travel(hover, "悬停点", "-")
            log_line(f"悬停点可达：{len(legs)} 段，平移高度 {tz:.0f}，到达关节 {[round(v, 1) for v in finals[-1]]}")
            hover_joints = list(finals[-1])
            pick.jog.xyz, pick.jog.joints = list(hover), list(finals[-1])
            grasp_z = TABLE + ASSUMED_BULB_TOP - args.grip_depth   # 11-25 00:58 measured: tips 133 mm
            finals = pick.jog.check_route([[*coarse["xy"], grasp_z]])
            log_line(f"下降到假设抓取高度 Z={grasp_z:.0f} 可达，关节 {[round(v, 1) for v in finals[-1]]}")
            pick.jog.xyz, pick.jog.joints = [*coarse["xy"], grasp_z], list(finals[-1])
            base_rpy = list(pick.jog.rpy)
            # The straightening grasp turns the jaws onto the lean (any yaw within +-90 deg): check the
            # turn at the hover and the whole descent + lift at the turned yaws as well.
            for turn in ((0.0, 45.0, 90.0, -45.0, -90.0) if args.straighten else (0.0,)):
                rpy = [base_rpy[0], base_rpy[1], base_rpy[2] + turn]
                try:
                    seed = list(hover_joints)
                    for k in range(1, max(1, math.ceil(abs(turn) / 5.0)) + 1):
                        part = [base_rpy[0], base_rpy[1], base_rpy[2] + turn * k / max(1, math.ceil(abs(turn) / 5.0))]
                        seed = pick.jog.check(list(hover), part, seed)
                    pick.jog.rpy, pick.jog.xyz, pick.jog.joints = rpy, list(hover), seed
                    finals = pick.jog.check_route([[*coarse["xy"], grasp_z]])
                    pick.jog.xyz, pick.jog.joints = [*coarse["xy"], grasp_z], list(finals[-1])
                    lo, hi = grasp_z, 560.0     # highest reachable flange Z straight above the bulb
                    while hi - lo > 2.0:
                        mid = (lo + hi) / 2
                        try:
                            pick.arm.plan([*coarse["xy"], mid], rpy, pick.jog.joints)
                            lo = mid
                        except RuntimeError:
                            hi = mid
                    up = [*coarse["xy"], grasp_z + args.lift]
                    legs = [up] + ([pick.retreat_point(up)] if args.retreat_in > 0 else [])
                    pick.jog.check_route(legs)
                    log_line(f"夹爪转 {turn:+.0f}°：转向可达，下降+竖直提起 {args.lift:.0f} mm 可达"
                             f"（最高法兰 Z≈{lo:.0f}，最多提 {lo - grasp_z:.0f} mm）")
                except Exception as error:  # noqa: BLE001 - report every yaw
                    log_line(f"夹爪转 {turn:+.0f}°：不可达 —— {error}")
                    if turn == 0.0:
                        raise
            pick.jog.rpy = base_rpy
            pick.jog.xyz, pick.jog.joints = list(hover), list(hover_joints)
            pick.jog.check_route([list(pick.start_pose[:3])])
            log_line("从悬停点直线回拍照位可达")
            result, code = "预检完成（未动）", 0
            pick.log["coarse"] = coarse
            return code
        if args.drip:
            runs = sorted(DATA.glob("live-lift-*/dropper_run.json"), key=lambda q: q.stat().st_mtime)
            saved = json.loads(runs[-1].read_text(encoding="utf-8")) if runs else {}
            here = [float(v) for v in states["right"]["pose"]]
            if "lifted_pose" not in saved or math.dist(here[:3], saved["lifted_pose"][:3]) > 3.0:
                raise RuntimeError("右臂不在上次提起滴管的位置，不滴加")
            arm.open_session(home=False)
            pick = DropperPick(arm, args, out_dir, stop_pressed=_stop)
            start = int(arm.gripper_position())
            target, done = start, []
            for i in range(1, args.drips + 1):
                target = max(args.drip_min, target - args.drip_step)
                fb = pick.grip(target, args.drip_speed, f"第 {i} 滴：捏紧到 {target}")
                done.append(int(fb.get("final", target)))
                time.sleep(args.drip_dwell)
                if target <= args.drip_min:
                    log_line(f"已捏到最紧 {args.drip_min}，停止")
                    break
            pick.log["drip"] = {"start_code": start, "reached": done, **{k: getattr(args, k) for k in
                                ("drip_step", "drip_min", "drips", "drip_dwell", "drip_speed")}}
            result, code = f"滴加 {len(done)} 滴完成（夹爪 {start} -> {done[-1] if done else start}，中间不松）", 0
            return code
        if args.go_home:
            from chem_demo.stations import photo_pose
            here = [float(v) for v in states["right"]["pose"]]
            photo_xyz, photo_rpy = photo_pose("right")
            low = here[2] < TABLE + args.hover - 2.0
            if low and here[2] < TABLE + args.min_tip:
                raise RuntimeError(f"右臂指尖离台面 {here[2] - TABLE:.0f} mm，太低，不自动回位")
            arm.open_session(home=False)
            if low and arm.gripper_position() < GRIPPER_OPEN - 60:
                raise RuntimeError("右臂在悬停高度以下且夹爪未张开（可能夹着或卡着东西），不自动回位")
            pick = DropperPick(arm, args, out_dir, stop_pressed=_stop)
            pick.jog.limits.z_max = max(pick.jog.limits.z_max, photo_xyz[2] + 5.0, here[2] + 5.0)
            pick.jog.limits.center = tuple(photo_xyz[:2])
            if low:   # jaws already open: straight up is the way it came down
                pick.straight([here[0], here[1], TABLE + args.hover], "竖直升到悬停高度")
            pick.grip(GRIPPER_OPEN, 10, "张开")
            pick.restore_yaw(photo_rpy)
            pick.straight(list(photo_xyz), "直线回拍照位")
            result, code = "已回拍照位", 0
            return code
        if args.put_back:
            runs = sorted(DATA.glob("live-lift-*/dropper_run.json"))
            if not runs:
                raise RuntimeError("找不到上次提起的记录")
            saved = json.loads(runs[-1].read_text(encoding="utf-8"))
            if "lifted_pose" not in saved:
                raise RuntimeError(f"{runs[-1]} 没有提起记录")
            arm.open_session(home=False)
            pick = DropperPick(arm, args, out_dir, stop_pressed=_stop)
            pick.jog.limits.z_max = max(pick.jog.limits.z_max, saved["start_pose"][2] + 5)
            pick.jog.limits.center = tuple(saved["start_pose"][:2])
            result = pick.put_back(saved)
            code = 0
            return code
        coarse = coarse_from_top(args, out_dir)
        arm.open_session(home=False)
        arm.set_linear_speed(args.speed)
        pick = DropperPick(arm, args, out_dir, stop_pressed=_stop)
        result = pick.run(coarse)
        code = 0
    except Exception as error:  # noqa: BLE001 - report, keep the arm where it stopped, then disable
        result, code = f"中止：{type(error).__name__}: {error}", 1
        log_line(f"已中止，机械臂停在当前位置：{error}")
        try:
            if arm.session_open:
                arm.stop()
        except Exception:  # noqa: BLE001
            pass
    finally:
        if pick is not None and pick.camera is not None:
            try:
                pick.camera.close()
            except Exception:  # noqa: BLE001
                pass
        arm.close()
        if pick is not None:
            pick.log.update(end=datetime.now().isoformat(timespec="seconds"), result=result)
            pick.save()
        log_line(f"结果：{result}；记录 {out_dir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
