"""Locate an upright cup on a known plane from one top-camera image.

Geometry, not a box centre: the cup is modelled as a frustum standing on a horizontal plane of
known height (measured once per surface, not per object). Its axis position, height and both
radii are fitted so that the projected silhouette (convex hull of the projected top and bottom
circles, with the calibrated intrinsics, distortion and camera pose) matches the segmented mask.
No object dimensions are entered. The segmentation is a plug-in: this first version picks the
most saturated cyan blob; an open-vocabulary segmenter can replace it without touching the fit.

    .venv-calibration/Scripts/python.exe -X utf8 -m vision_pick.cup_locate \
        --image calibration-data/cup-images/pos01/board_001.png --table-z 240.18 --arm right
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = ROOT / "calibration-data" / "cal-20261103-153131-49df4d"
CIRCLE_SAMPLES = 90
CROP_MARGIN_PX = 30
WORKSPACE_RADIUS_MM = (250.0, 700.0)
OPENING_SHARE = (0.3, 0.85)  # the opening's share of the cup outline in the five real photos: 0.50-0.65  # horizontal distance from the arm base that can hold a cup


@dataclass
class Camera:
    K: np.ndarray
    D: np.ndarray
    T_base_camera: np.ndarray  # mm

    @property
    def center(self) -> np.ndarray:
        return self.T_base_camera[:3, 3]

    def project(self, points_base: np.ndarray) -> np.ndarray:
        R, C = self.T_base_camera[:3, :3], self.T_base_camera[:3, 3]
        rvec, _ = cv2.Rodrigues(R.T)
        tvec = -R.T @ C
        pixels, _ = cv2.projectPoints(points_base.reshape(-1, 1, 3).astype(np.float64), rvec, tvec,
                                      self.K, self.D)
        return pixels.reshape(-1, 2)

    def ray_to_plane(self, pixel: Sequence[float], z: float) -> np.ndarray:
        normalized = cv2.undistortPoints(np.array([[pixel]], np.float64), self.K, self.D).reshape(2)
        ray = self.T_base_camera[:3, :3] @ np.array([normalized[0], normalized[1], 1.0])
        scale = (z - self.center[2]) / ray[2]
        return self.center + scale * ray


def load_camera(session: Path, arm: str) -> Camera:
    state = json.loads((session / "session.json").read_text(encoding="utf-8"))
    intrinsics = json.loads((session / state["intrinsics"]["result"]["artifact_path"]).read_text(encoding="utf-8"))
    fit = state["arms"][arm]["fit"]
    if not fit.get("usable") or fit.get("stale"):
        raise RuntimeError(f"{arm} 臂拟合不可用")
    extrinsics = json.loads((session / fit["artifact_path"]).read_text(encoding="utf-8"))
    T = np.array(extrinsics["T_base_camera"], dtype=float)
    T[:3, 3] *= 1000.0  # metres -> mm
    return Camera(np.array(intrinsics["K"], float), np.array(intrinsics["distortion"], float).reshape(-1), T)


def _largest_filled(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not candidates:
        raise RuntimeError("没有找到杯子颜色的区域")
    blob = (labels == max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(blob)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def _best_seed(seed: np.ndarray, hsv: np.ndarray, min_area: int,
               accept: Callable[[float, float], bool] | None) -> np.ndarray:
    """The most saturated candidate inside the workspace, not the largest blob.

    On 2026-11-04 the camera's exposure drifted and the blue arm covers rose from S 65-120 to
    S 150-186, passing the seed threshold and out-sizing the cup (median S 255): the largest
    blob was an arm at the image edge. Geometry rejects such blobs, saturation ranks the rest.
    """
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(seed)
    ranked = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        u, v = centroids[i]
        if accept is not None and not accept(float(u), float(v)):
            continue
        ranked.append((float(np.median(hsv[..., 1][labels == i])), i))
    if not ranked:
        raise RuntimeError("工作区内没有找到杯子颜色的区域（画面边缘或够不到的地方的色块已排除）")
    return convex_fill((labels == max(ranked)[1]).astype(np.uint8))


def convex_fill(blob: np.ndarray) -> np.ndarray:
    """Both the rim opening and the cup outline are convex; filling the hull closes bites that a
    shadowed or glaring patch leaves in the colour mask. With the cup near the arm, the dark
    inside bottom (V 22-26) fell below the seed's V>=25 and cut a notch into the opening, which
    shrank the fitted rim (2026-11-04)."""
    points = cv2.findNonZero(blob)
    filled = np.zeros_like(blob)
    if points is not None:
        cv2.fillConvexPoly(filled, cv2.convexHull(points), 1)
    return filled


def segment_saturated_cyan(image: np.ndarray, min_area: int = 800,
                           accept: Callable[[float, float], bool] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """First-version segmenter -> (opening, whole cup).

    The inside seen through the rim is deeply saturated (median S ~255), so it is the seed and,
    being bounded by the rim, it is the opening ellipse. Arm covers can also pass the seed
    threshold when the exposure drifts, so `accept(u, v)` (a workspace test on the blob centre)
    and the saturation ranking pick the seed. The lit outer wall is paler; it is the part of a
    looser cyan mask connected to the seed.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    seed = cv2.inRange(hsv, (80, 150, 25), (104, 255, 255))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    seed_blob = _best_seed(seed, hsv, min_area, accept)
    loose = cv2.inRange(hsv, (78, 45, 0), (106, 255, 255))
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels = cv2.connectedComponents(loose | seed_blob)
    touching = np.unique(labels[seed_blob > 0])
    touching = touching[touching > 0]
    cup = convex_fill(_largest_filled(np.isin(labels, touching).astype(np.uint8), min_area))
    return dark_opening(hsv, cup, seed_blob), cup


def dark_opening(hsv: np.ndarray, cup: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """The rim opening = the darker part of the cup (the inside is shaded, the outer wall is lit).

    Saturation alone failed near the arm: there the lit wall was as saturated as the inside,
    so the "opening" covered the whole cup (2026-11-04, third position). Brightness separated
    the opening cleanly in all five real photos. The threshold is Otsu on the cup's own V
    values, so it follows the exposure; without a clear split it falls back to the seed.
    """
    values = hsv[..., 2][cup > 0].reshape(-1, 1).astype(np.uint8)
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((hsv[..., 2] <= threshold) & (cup > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if count < 2:
        return fallback
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    opening = convex_fill((labels == largest).astype(np.uint8))
    share = opening.sum() / max(int(cup.sum()), 1)
    return opening if OPENING_SHARE[0] <= share <= OPENING_SHARE[1] else fallback


@dataclass
class CupFit:
    x: float
    y: float
    table_z: float
    height: float
    top_radius: float
    bottom_radius: float
    iou: float

    @property
    def rim_z(self) -> float:
        return self.table_z + self.height


def frustum_silhouette(camera: Camera, params: Sequence[float], table_z: float) -> np.ndarray:
    x, y, height, top_r, bottom_r = params
    angles = np.linspace(0.0, 2.0 * math.pi, CIRCLE_SAMPLES, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    top = np.column_stack([x + top_r * ring[:, 0], y + top_r * ring[:, 1], np.full(len(ring), table_z + height)])
    bottom = np.column_stack([x + bottom_r * ring[:, 0], y + bottom_r * ring[:, 1], np.full(len(ring), table_z)])
    pixels = camera.project(np.vstack([top, bottom]))
    return cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)


def top_ellipse(camera: Camera, params: Sequence[float], table_z: float) -> np.ndarray:
    x, y, height, top_r, _ = params
    angles = np.linspace(0.0, 2.0 * math.pi, CIRCLE_SAMPLES, endpoint=False)
    top = np.column_stack([x + top_r * np.cos(angles), y + top_r * np.sin(angles),
                           np.full(CIRCLE_SAMPLES, table_z + height)])
    return camera.project(top)


def model_scorer(camera: Camera, opening: np.ndarray, cup: np.ndarray, table_z: float):
    """Overlap of a frustum model with the segmentation: mean IoU of silhouette and rim opening."""
    ys, xs = np.nonzero(cup)
    x0, x1 = max(xs.min() - CROP_MARGIN_PX, 0), min(xs.max() + CROP_MARGIN_PX, cup.shape[1])
    y0, y1 = max(ys.min() - CROP_MARGIN_PX, 0), min(ys.max() + CROP_MARGIN_PX, cup.shape[0])
    observed_cup = cup[y0:y1, x0:x1].astype(bool)
    observed_open = opening[y0:y1, x0:x1].astype(bool)
    scale = 4  # sub-pixel rasterisation of the model polygons

    def raster(polygon: np.ndarray) -> np.ndarray:
        canvas = np.zeros(((y1 - y0) * scale, (x1 - x0) * scale), np.uint8)
        cv2.fillPoly(canvas, [np.round((polygon - [x0, y0]) * scale).astype(np.int32)], 1)
        return cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA) > 0.5

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / union) if union else 0.0

    def score(params: Sequence[float]) -> float:
        params = np.asarray(params, float)
        if params[2] <= 5 or params[3] <= 5 or params[4] <= 5:
            return 0.0
        return 0.5 * (iou(raster(frustum_silhouette(camera, params, table_z)), observed_cup)
                      + iou(raster(top_ellipse(camera, params, table_z)), observed_open))

    return score


def fit_cup(camera: Camera, opening: np.ndarray, cup: np.ndarray, table_z: float,
            prior: dict | None = None) -> CupFit:
    """Fit x, y, height and both radii to the whole silhouette and the rim opening together.

    The silhouette alone is ambiguous (a short cup with a tiny base explains the same outline);
    the opening pins the rim circle, and the bottom contact on the known plane pins the rest.
    Starts are the rim ellipse back-projected with the learned size (when known) and a generic
    guess; each is searched and refined, the best-scoring result wins. A generic start alone got
    stuck at a wrong local optimum for a cup closer to the arm (2026-11-04, score 0.74).
    """
    score = model_scorer(camera, opening, cup, table_z)
    ys, xs = np.nonzero(cup)
    centre = camera.ray_to_plane((xs.mean(), ys.mean()), table_z + 40.0)
    starts = [np.array([centre[0], centre[1], 80.0, 40.0, 35.0])]
    for size in ([prior] if prior else []) + [{"height": 90.0, "top_radius": 40.0, "bottom_radius": 37.0}]:
        x, y = rim_position(camera, opening, table_z + size["height"])
        starts.insert(0, np.array([x, y, size["height"], size["top_radius"], size["bottom_radius"]]))
    results = []
    for start in starts:
        best, best_score = start, score(start)
        step = np.array([8.0, 8.0, 15.0, 6.0, 6.0])
        rng = np.random.default_rng(0)
        for _ in range(4):  # coarse search on the overlap score, only to get into the basin
            for _ in range(80):
                trial = best + rng.normal(0.0, 1.0, 5) * step
                trial_score = score(trial)
                if trial_score > best_score:
                    best, best_score = trial, trial_score
            step *= 0.5
        best = refine_edges(camera, opening, cup, table_z, best)
        results.append((score(best), best))
    best_score, best = max(results, key=lambda item: item[0])
    return CupFit(float(best[0]), float(best[1]), table_z, float(best[2]), float(best[3]),
                  float(best[4]), best_score)


def _contour_points(mask: np.ndarray, count: int = 240) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    return points[np.linspace(0, len(points) - 1, min(count, len(points))).astype(int)]


def refine_edges(camera: Camera, opening: np.ndarray, cup: np.ndarray, table_z: float,
                 start: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Levenberg-Marquardt on signed pixel distances of the observed edges to the model outlines.

    Unlike the overlap score this is smooth in the parameters, so the optimum is well defined.
    """
    cup_edge, open_edge = _contour_points(cup), _contour_points(opening)

    def residuals(params: np.ndarray) -> np.ndarray:
        silhouette = frustum_silhouette(camera, params, table_z).astype(np.float32).reshape(-1, 1, 2)
        rim = top_ellipse(camera, params, table_z).astype(np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(silhouette, tuple(p), True) for p in cup_edge]
                        + [cv2.pointPolygonTest(rim, tuple(p), True) for p in open_edge])

    params, damping = np.array(start, float), 1e-2
    current = residuals(params)
    for _ in range(iterations):
        jacobian = np.empty((len(current), len(params)))
        for k in range(len(params)):
            delta = np.zeros_like(params)
            delta[k] = 0.05
            jacobian[:, k] = (residuals(params + delta) - residuals(params - delta)) / 0.1
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        while True:
            step = -np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), gradient)
            candidate = params + step
            trial = residuals(candidate)
            if trial @ trial < current @ current:
                params, current, damping = candidate, trial, max(damping / 3, 1e-6)
                break
            damping *= 4
            if damping > 1e6:
                return params
        if np.abs(step).max() < 1e-3:
            break
    return params


HOME_DOWN_RPY = (180.0, 0.0, 89.972)  # right arm home after v; Tool0 Y = base +X there
WALL_HALF_MM = 1.0       # TCP goes on the wall centre line, just inside the outer rim
HOVER_ABOVE_RIM_MM = 30.0


def jaw_axis_xy(rpy: Sequence[float]) -> np.ndarray:
    """Horizontal jaw opening direction: Tool0 Y (chemistry/geometry.py convention)."""
    from task_skills.chemistry import geometry

    axis = np.array(geometry.tool_axis(list(rpy), "Y"), float)[:2]
    return axis / np.linalg.norm(axis)


def rim_grasp_points(fit: CupFit, rpy: Sequence[float] = HOME_DOWN_RPY,
                     base_xy: Sequence[float] = (0.0, 0.0)) -> dict:
    """Rim pinch with one finger inside, one outside: the jaws must close along the cup radius.

    Keep the locked tool orientation and pick the rim point where the radius is parallel to
    the jaw axis, so the wrist never turns (turning would also swing any fingertip offset
    that the flange-based calibration absorbed). Of the two such points take the one nearer
    the arm base (more reach margin).
    """
    jaw = jaw_axis_xy(rpy)
    radius = fit.top_radius - WALL_HALF_MM
    options = [np.array([fit.x, fit.y]) + sign * radius * jaw for sign in (1.0, -1.0)]
    chosen = min(options, key=lambda p: np.linalg.norm(p - np.asarray(base_xy, float)))
    rim = [float(chosen[0]), float(chosen[1]), fit.rim_z]
    hover = [rim[0], rim[1], fit.rim_z + HOVER_ABOVE_RIM_MM]
    return {"rim_point": [round(v, 2) for v in rim],
            "hover_point": [round(v, 2) for v in hover],
            "tool_rpy": [float(v) for v in rpy],
            "jaw_axis_xy": [round(float(v), 4) for v in jaw],
            "note": "TCP = Tool0 flange coordinates, same as the calibration touches"}


def draw(image: np.ndarray, camera: Camera, fit: CupFit, mask: np.ndarray, grasp: dict) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (0, 255, 255), 1)
    angles = np.linspace(0, 2 * math.pi, 120)
    for radius, z, colour in ((fit.top_radius, fit.rim_z, (0, 0, 255)), (fit.bottom_radius, fit.table_z, (255, 0, 0))):
        circle = np.column_stack([fit.x + radius * np.cos(angles), fit.y + radius * np.sin(angles), np.full(120, z)])
        cv2.polylines(out, [np.round(camera.project(circle)).astype(np.int32)], True, colour, 1, cv2.LINE_AA)
    axis = camera.project(np.array([[fit.x, fit.y, fit.table_z], [fit.x, fit.y, fit.rim_z]]))
    cv2.line(out, tuple(np.round(axis[0]).astype(int)), tuple(np.round(axis[1]).astype(int)), (0, 200, 0), 1)
    rim = camera.project(np.array([grasp["rim_point"]]))[0]
    cv2.circle(out, tuple(np.round(rim).astype(int)), 4, (0, 255, 0), -1)
    return out


OBJECT_MEMORY = ROOT / "calibration-data" / "vision-pick" / "objects" / "cyan_cup.json"
MEMORY_MIN_IOU = 0.9   # only clean fits teach the object's size
MEMORY_KEEP = 30


def read_size(memory_path: Path | None) -> dict | None:
    if memory_path is None or not memory_path.exists():
        return None
    entries = json.loads(memory_path.read_text(encoding="utf-8")).get("fits", [])
    if not entries:
        return None
    return {key: float(np.median([e[key] for e in entries])) for key in ("height", "top_radius", "bottom_radius")}


def learn_size(memory_path: Path | None, fit: CupFit, image_path: Path) -> dict:
    """Object size learned from its own good fits; the median is what every trial uses.

    One photo's height is not stable (a pale wall that blends into the table shifts the fitted
    base; two photos of the same unmoved cup gave 88.7 and 97.7 mm). The cup's size does not
    change between trials, so pooling fits makes it steadier with every trial, with nothing
    measured by hand.
    """
    entries = []
    if memory_path is not None and memory_path.exists():
        entries = json.loads(memory_path.read_text(encoding="utf-8")).get("fits", [])
    if fit.iou >= MEMORY_MIN_IOU:
        entries = [e for e in entries if e.get("image") != str(image_path)]  # one vote per photo
        entries.append({"height": round(fit.height, 2), "top_radius": round(fit.top_radius, 2),
                         "bottom_radius": round(fit.bottom_radius, 2), "iou": round(fit.iou, 4),
                         "image": str(image_path), "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        entries = entries[-MEMORY_KEEP:]
        if memory_path is not None:
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory_path.write_text(json.dumps({"object": "cyan cup", "fits": entries}, ensure_ascii=False,
                                              indent=2), encoding="utf-8")
    source = entries or [{"height": fit.height, "top_radius": fit.top_radius, "bottom_radius": fit.bottom_radius}]
    size = {key: float(np.median([e[key] for e in source])) for key in ("height", "top_radius", "bottom_radius")}
    size["n"] = len(entries)
    size["height_spread"] = float(np.ptp([e["height"] for e in source])) if source else 0.0
    return size


def rim_pixel(opening: np.ndarray) -> tuple[float, float]:
    contours, _ = cv2.findContours(opening.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (u, v), _, _ = cv2.fitEllipse(max(contours, key=len))
    return float(u), float(v)


def rim_position(camera: Camera, opening: np.ndarray, rim_z: float) -> tuple[float, float]:
    """Cup axis XY from the rim opening: the crisp, saturated ellipse, back-projected at rim height.

    The ellipse centre is not exactly the projected circle centre under perspective, but at
    ~0.8 m with a ~40 mm rim the difference is well under 0.5 mm.
    """
    point = camera.ray_to_plane(rim_pixel(opening), rim_z)
    return float(point[0]), float(point[1])


def locate_and_save(image: np.ndarray, image_path: Path, camera: Camera, table_z: float, arm: str,
                    rpy: Sequence[float], out_dir: Path, stem: str,
                    memory_path: Path | None = OBJECT_MEMORY) -> dict:
    """Segment, fit, take XY from the rim with the learned size, choose the rim point; save evidence."""
    def in_workspace(u: float, v: float) -> bool:
        point = camera.ray_to_plane((u, v), table_z + 50.0)  # roughly mid-cup
        return WORKSPACE_RADIUS_MM[0] <= math.hypot(point[0], point[1]) <= WORKSPACE_RADIUS_MM[1]

    opening, mask = segment_saturated_cyan(image, accept=in_workspace)
    frame_fit = fit_cup(camera, opening, mask, table_z, prior=read_size(memory_path))
    size = learn_size(memory_path, frame_fit, image_path)
    x, y = rim_position(camera, opening, table_z + size["height"])
    # The safety check must judge the model that is actually used: rim position + learned size.
    used = [x, y, size["height"], size["top_radius"], size["bottom_radius"]]
    fit = CupFit(x, y, table_z, size["height"], size["top_radius"], size["bottom_radius"],
                 model_scorer(camera, opening, mask, table_z)(used))
    grasp = rim_grasp_points(fit, rpy=rpy)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw(image, camera, fit, mask, grasp)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 255), 1)
    overlay_path = out_dir / f"{stem}_cup_{arm}.png"
    cv2.imwrite(str(overlay_path), overlay)
    report = {"image": str(image_path), "overlay": str(overlay_path), "arm": arm, "fit": asdict(fit),
              "frame_fit": asdict(frame_fit), "size": size, "rim_z": fit.rim_z, "grasp": grasp,
              "rim_pixel": [round(v, 3) for v in rim_pixel(opening)],
              "camera_xy": [round(float(v), 2) for v in camera.center[:2]],  # the camera's foot point
              "method": "xy from rim ellipse at learned rim height; size = median of good fits",
              "units": "mm, arm base frame"}
    (out_dir / f"{stem}_cup_{arm}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    return report


def grab_frame(index: int, rotate_180: bool = True, warmup: int = 20, max_failures: int = 60) -> np.ndarray:
    """One frame from the top camera after auto-exposure settles; rotated like the calibration images."""
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开相机 {index}：请关闭占用它的程序（标定向导预览、拍照程序等）")
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        good, failures, frame = 0, 0, None
        while good < warmup:
            ok, raw = capture.read()
            if not ok or raw is None:
                failures += 1  # MSMF drops the first frames while the stream starts
                if failures >= max_failures:
                    raise RuntimeError(f"连续 {failures} 次读不到相机帧")
                cv2.waitKey(30)
                continue
            good, frame = good + 1, raw
    finally:
        capture.release()
    if frame.shape[:2] != (720, 1280):
        raise RuntimeError(f"相机分辨率 {frame.shape[1]}x{frame.shape[0]}，不是标定时的 1280x720")
    return cv2.rotate(frame, cv2.ROTATE_180) if rotate_180 else frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="顶部相机单张图定位立在已知平面上的杯子")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="已旋转 180° 的顶部相机图像")
    source.add_argument("--camera", type=int, help="直接从这个相机取一帧（顶部相机是 0），旋转 180° 后保存")
    parser.add_argument("--table-z", type=float, required=True, help="放置平面在该臂基座中的 Z（mm）")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--rpy", type=float, nargs=3, default=list(HOME_DOWN_RPY),
                        help="抓取时锁定的工具姿态，决定夹爪合拢方向（默认右臂初始位摆正）")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--out", type=Path, help="结果目录；--image 时默认图像同目录的 evidence/")
    parser.add_argument("--show", action="store_true", help="弹窗显示定位结果，按任意键关闭")
    args = parser.parse_args(argv)

    try:
        camera = load_camera(args.session, args.arm)
        if args.image is not None:
            image = cv2.imread(str(args.image))
            if image is None:
                raise RuntimeError(f"读不到图像 {args.image}")
            image_path, out_dir, stem = args.image, args.out or args.image.parent / "evidence", args.image.stem
        else:
            if args.out is None:
                raise RuntimeError("--camera 需要同时给 --out")
            image = grab_frame(args.camera)
            args.out.mkdir(parents=True, exist_ok=True)
            image_path, out_dir, stem = args.out / "frame.png", args.out, "frame"
            cv2.imwrite(str(image_path), image)
        report = locate_and_save(image, image_path, camera, args.table_z, args.arm, args.rpy, out_dir, stem)
    except RuntimeError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.show:
        view = cv2.imread(report["overlay"])
        fit = report["fit"]
        cv2.putText(view, f"h={fit['height']:.1f} rim_d={2 * fit['top_radius']:.1f} mm  any key = close",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("cup locate", view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
