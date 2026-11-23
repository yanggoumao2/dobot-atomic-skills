"""Scene localisation from one top-camera frame taken at the photo pose.

Pipeline: Grounding DINO + SAM 2 masks (chem_demo.segment), with a second, 3x-enlarged pass
around each balance (beakers on the reflective pans are missed at full scale) and around the
tube rack (the dropper). Each vessel mask is fitted with an upright cylinder standing on a known
support plane: the table (right-base Z 240.18) or a balance pan (rim touched by the right arm on
2026-11-20: Z 339.73, i.e. 99.55 mm above the table). The cylinder's axis XY, height and radius
are free; the base height is not, which removes the depth ambiguity of a single view. Contour
points the model cannot explain (the glass rod standing in a beaker) are trimmed and refitted.

Everything is computed in the right arm's base frame and reported for both arms.

    .venv-perception/Scripts/python.exe -X utf8 -m chem_demo.locate --image <top.png> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from chem_demo.camera_drift import load_corrected_camera as load_camera  # 11-04 calibration + bump correction
from vision_pick.cup_locate import DEFAULT_SESSION, Camera, frustum_silhouette

TABLE_Z = 240.18          # right base, fingertip touch on the white table (2026-11-04)
PAN_Z = 339.73            # right base, fingertip touch on the right balance pan rim (2026-11-20)
# Platform region on the table plane, right base (mm). Detections whose ground point falls outside
# are dropped: the grippers at the photo pose sit at y > -230 (right) and y < -800 (left).
PLATFORM_X = (-520.0, 160.0)
PLATFORM_Y = (-820.0, -220.0)
CROP_SCALE = 3
PAN_HEIGHT_TOLERANCE = 12.0  # free fits of the real pan beakers 2026-11-21..23: 77-82 good, 108/115 wrong
PAN_BEAKER_HEIGHT = 81.0  # right pan beaker, free fit at 21:28 (rms 1.1 px); provisional until measured
MID_Y = -531.0  # half way between the two bases (left base at y ≈ -1062 in the right-base frame)


@dataclass
class Vessel:
    name: str
    role: str
    support: str          # table | right_pan | left_pan
    x: float
    y: float
    base_z: float
    height: float
    radius: float
    rms_px: float
    kept_share: float
    score: float

    @property
    def rim_z(self) -> float:
        return self.base_z + self.height


def _contour(mask: np.ndarray, count: int = 300) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    return points[np.linspace(0, len(points) - 1, min(count, len(points))).astype(int)]


def fit_cylinder(camera: Camera, mask: np.ndarray, base_z: float, trim: float = 0.15,
                 rounds: int = 3, fixed_height: float | None = None) -> tuple[np.ndarray, float, float]:
    """Upright cylinder (x, y, h, r) on z = base_z whose projected silhouette matches the mask edge.

    Levenberg-Marquardt on signed pixel distances of the mask contour to the model outline. After
    each round the worst-explained points (largest |distance|) are dropped, so a rod or a reflection
    sticking out of the outline does not stretch the cylinder. Returns params, RMS px, kept share.
    """
    points = _contour(mask)
    ys, xs = np.nonzero(mask)
    centre = camera.ray_to_plane((xs.mean(), ys.mean()), base_z + 40.0)
    params = np.array([centre[0], centre[1], fixed_height or 80.0, 35.0])
    keep = np.ones(len(points), bool)

    def residuals(p: np.ndarray, pts: np.ndarray) -> np.ndarray:
        if fixed_height is not None:
            p = np.array([p[0], p[1], fixed_height, p[3]])
        if p[2] < 5 or p[3] < 5:
            return np.full(len(pts), 1e3)
        outline = frustum_silhouette(camera, [p[0], p[1], p[2], p[3], p[3]], base_z)
        outline = outline.astype(np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(outline, (float(a), float(b)), True) for a, b in pts])

    for round_index in range(rounds):
        pts = points[keep]
        damping, current = 1e-2, residuals(params, pts)
        for _ in range(60):
            jacobian = np.empty((len(pts), 4))
            for k in range(4):
                delta = np.zeros(4)
                delta[k] = 0.1
                jacobian[:, k] = (residuals(params + delta, pts) - residuals(params - delta, pts)) / 0.2
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
    if fixed_height is not None:
        params[2] = fixed_height
    final = residuals(params, points[keep])
    return params, float(np.sqrt(np.mean(final ** 2))), float(keep.mean())


def ground_point(camera: Camera, box: Sequence[float], z: float) -> np.ndarray:
    """Bottom-centre of a box, back-projected to height z: a rough position for filtering."""
    return camera.ray_to_plane(((box[0] + box[2]) / 2, box[3]), z)


def on_platform(point: np.ndarray) -> bool:
    return PLATFORM_X[0] <= point[0] <= PLATFORM_X[1] and PLATFORM_Y[0] <= point[1] <= PLATFORM_Y[1]


def detect_in_crop(segmenter, image: np.ndarray, box: Sequence[float], vocabulary: dict[str, str],
                   margin: int = 25, threshold: float = 0.25) -> list:
    """Run the detector on an enlarged crop; boxes and masks are mapped back to the full image."""
    h, w = image.shape[:2]
    x0, y0 = max(int(box[0]) - margin, 0), max(int(box[1]) - margin, 0)
    x1, y1 = min(int(box[2]) + margin, w), min(int(box[3]) + margin, h)
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=CROP_SCALE, fy=CROP_SCALE, interpolation=cv2.INTER_CUBIC)
    found = segmenter.detect(crop, vocabulary=vocabulary, box_threshold=threshold, size_filter=False)
    for d in found:
        d.box = [d.box[0] / CROP_SCALE + x0, d.box[1] / CROP_SCALE + y0,
                 d.box[2] / CROP_SCALE + x0, d.box[3] / CROP_SCALE + y0]
        small = cv2.resize(d.mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        d.mask = np.zeros((h, w), np.uint8)
        d.mask[y0:y1, x0:x1] = small
    return found


def locate_scene(image: np.ndarray, camera: Camera, segmenter) -> dict:
    """Detect, assign identities and fit vessels; returns a JSON-ready report."""
    detections = [d for d in segmenter.detect(image)
                  if on_platform(ground_point(camera, d.box, TABLE_Z if d.role != "balance" else PAN_Z))]
    balances = [d for d in detections if d.role == "balance"]
    report: dict = {"vessels": [], "balances": [], "rack": None, "dropper": None, "notes": []}

    # Balances are named by where they stand, not by order: with one balance hidden behind an arm,
    # ordering once called the left balance "right_pan" (2026-11-20 21:59). The right arm's balance
    # stands nearer the right base (y > MID_Y in the right-base frame).
    pans = {}
    for d in sorted(balances, key=lambda d: -d.score):
        y = ground_point(camera, d.box, PAN_Z)[1]
        name = "right_pan" if y > MID_Y else "left_pan"
        if name not in pans:
            pans[name] = d
            report["balances"].append({"name": name, "box": [round(v, 1) for v in d.box], "score": round(d.score, 3)})
    for name in ("right_pan", "left_pan"):
        if name not in pans:
            report["notes"].append(f"没看到 {name}（被手臂挡住？先回拍照位）")

    candidates = []  # (name, detection, support)
    for name, d in pans.items():
        found = [b for b in detect_in_crop(segmenter, image, d.box, {"glass beaker": "beaker", "glass cup": "beaker"},
                                           margin=5, threshold=0.2) if b.role == "beaker"]
        # Keep only vessel-shaped hits inside the balance box: the crop margin once let the glass
        # rod of the centre beaker (a thin vertical strip) outscore the real beaker (2026-11-20).
        found = [b for b in found if _overlap(b.box, d.box) > 0.8
                 and (b.box[2] - b.box[0]) > 0.4 * (b.box[3] - b.box[1])]
        if found:
            best = max(found, key=lambda b: b.score)
            candidates.append(({"right_pan": "right_beaker", "left_pan": "left_beaker"}[name], best, name))
        else:
            report["notes"].append(f"{name} 上没找到烧杯")
    table_beakers = [d for d in detections if d.role == "beaker"
                     and all(_overlap(d.box, p.box) < 0.5 for p in pans.values())]
    if table_beakers:
        candidates.append(("centre_beaker", max(table_beakers, key=lambda d: d.score), "table"))
    bottles = [d for d in detections if d.role == "bottle" and all(_overlap(d.box, p.box) < 0.5 for p in pans.values())]
    if bottles:
        candidates.append(("bottle", max(bottles, key=lambda d: d.score), "table"))

    for name, d, support in candidates:
        base_z = TABLE_Z if support == "table" else PAN_Z
        # Opening removes thin parts (the glass rod standing in the centre beaker, ~8 px wide)
        # that SAM merges into the vessel mask; the vessel walls survive.
        mask = cv2.morphologyEx(d.mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        mask = _largest_component(mask) if mask.any() else d.mask
        params, rms, kept = fit_cylinder(camera, mask, base_z)
        if support != "table" and (abs(params[2] - PAN_BEAKER_HEIGHT) > PAN_HEIGHT_TOLERANCE or rms > 2.5):
            # SAM sometimes keeps only the upper part of a beaker on the mirror-like pan, and the
            # free fit then collapses (h 5 mm, 21:59). It also overshoots: h 108/115 mm on 2026-11-23
            # (real 82), and a wrong height drags the axis XY along the view ray (27 mm), which a pour
            # cannot afford. Refit with the pan-beaker height fixed.
            params, rms, kept = fit_cylinder(camera, mask, base_z, fixed_height=PAN_BEAKER_HEIGHT)
            report["notes"].append(f"{name}: 自由拟合失败，按固定杯高 {PAN_BEAKER_HEIGHT:.0f} mm 重拟合，只信 XY")
        vessel = Vessel(name, d.role, support, *(float(v) for v in params[:2]), base_z,
                        float(params[2]), float(params[3]), rms, kept, d.score)
        report["vessels"].append({**asdict(vessel), "rim_z": vessel.rim_z,
                                  "box": [round(v, 1) for v in d.box], "_mask": d.mask})

    racks = [d for d in detections if d.role == "rack"]
    if racks:
        rack = max(racks, key=lambda d: d.score)
        centre = camera.ray_to_plane(((rack.box[0] + rack.box[2]) / 2, (rack.box[1] + rack.box[3]) / 2), TABLE_Z + 50)
        report["rack"] = {"box": [round(v, 1) for v in rack.box], "approx_xy": [round(float(v), 1) for v in centre[:2]],
                          "score": round(rack.score, 3)}
        droppers = detect_in_crop(segmenter, image, rack.box,
                                  {"plastic dropper pipette": "dropper", "plastic pipette": "dropper"})
        if droppers:
            boxes = np.array([d.box for d in droppers])
            union = [boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()]
            report["dropper"] = {"box": [round(float(v), 1) for v in union],
                                 "parts": [{"box": [round(v, 1) for v in d.box], "score": round(d.score, 3)} for d in droppers],
                                 "note": "斜靠在试管架上，位置仅为像素框；三维姿态待腕部相机确定"}
        else:
            report["notes"].append("试管架附近没找到滴管")
    return report


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of box a that lies inside box b."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area > 0 else 0.0


def to_left(point_right: Sequence[float], T_right_left: np.ndarray) -> list[float]:
    p = np.linalg.inv(T_right_left) @ np.array([*point_right, 1.0])
    return [round(float(v), 1) for v in p[:3]]


def draw(image: np.ndarray, camera: Camera, report: dict) -> np.ndarray:
    view = image.copy()
    angles = np.linspace(0, 2 * math.pi, 90)
    for v in report["vessels"]:
        overlay = view.copy()
        overlay[v["_mask"].astype(bool)] = (255, 160, 0)
        view = cv2.addWeighted(overlay, 0.3, view, 0.7, 0)
        for z, color in ((v["base_z"], (255, 0, 0)), (v["rim_z"], (0, 0, 255))):
            ring = np.column_stack([v["x"] + v["radius"] * np.cos(angles), v["y"] + v["radius"] * np.sin(angles),
                                    np.full(90, z)])
            cv2.polylines(view, [np.round(camera.project(ring)).astype(np.int32)], True, color, 2, cv2.LINE_AA)
        top = camera.project(np.array([[v["x"], v["y"], v["rim_z"]]]))[0]
        cv2.putText(view, f"{v['name']} h{v['height']:.0f} d{2 * v['radius']:.0f}", (int(top[0]) - 60, int(top[1]) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for key, color in (("rack", (0, 120, 255)), ("dropper", (255, 0, 255))):
        if report.get(key):
            x0, y0, x1, y1 = map(int, report[key]["box"])
            cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)
            cv2.putText(view, key, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return view


def main(argv: Sequence[str] | None = None) -> int:
    from chem_demo.segment import Segmenter

    parser = argparse.ArgumentParser(description="拍照位顶部图：识别 + 器皿三维定位（离线，不动机械臂）")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    right, left = load_camera(args.session, "right"), load_camera(args.session, "left")
    T_right_left = right.T_base_camera @ np.linalg.inv(left.T_base_camera)
    image = cv2.imread(str(args.image))
    segmenter = Segmenter()
    t0 = time.time()
    report = locate_scene(image, right, segmenter)
    report["seconds"] = round(time.time() - t0, 2)
    cv2.imwrite(str(args.out / "locate.jpg"), draw(image, right, report))
    for v in report["vessels"]:
        v.pop("_mask")
        v["right_base"] = {"axis_xy": [round(v["x"], 1), round(v["y"], 1)], "rim_z": round(v["rim_z"], 1)}
        v["left_base"] = {"axis_xyz_at_rim": to_left([v["x"], v["y"], v["rim_z"]], T_right_left)}
    report.update({"image": str(args.image), "frame": "right base unless stated, mm",
                   "table_z": TABLE_Z, "pan_z": PAN_Z})
    (args.out / "locate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for v in report["vessels"]:
        print(f"{v['name']:14s} {v['support']:9s} xy=({v['x']:7.1f},{v['y']:7.1f}) h={v['height']:5.1f} "
              f"d={2 * v['radius']:5.1f} rim_z={v['rim_z']:6.1f} rms={v['rms_px']:.2f}px score={v['score']:.2f} "
              f"left={v['left_base']['axis_xyz_at_rim']}")
    print(json.dumps({k: report[k] for k in ("rack", "dropper", "notes", "seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
