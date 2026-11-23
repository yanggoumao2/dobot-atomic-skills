"""Side-by-side demo video: the C920 recording (left) and what the vision saw, when it saw it (right).

Right panel = every recognition the run made, drawn on the frame it was made on, shown from the
moment it was computed until the next one (time-synchronised with the recording by wall clock):

* top camera, pour scene: Grounding DINO + SAM 2 vessels, fitted beaker/bottle cylinders (locate.jpg)
* top camera, neck pick: blue rim found by the classical ring detector (re-run here on the same frame)
* left wrist, hover check: blue rim relative to the jaws (re-run here on the same frame)
* top camera at the handoff station: SAM 2 mask of the hanging bottle + fitted cylinder (hang_top_fit)
* right wrist, approach: SAM 2 bottle mask (seg_NNN_mask) + servo error from right_obs_NNN.json
* top camera while pouring: planned lip and the located beaker rim projected (run_pour.lip_overlay)

These are keyframes, not 30 fps tracking; each shows "x s ago". Run with the perception venv (PIL).

  python -m chem_demo.make_demo_video <handoff run dir> --video c920.mp4 --start-ms <epoch ms>
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/msyh.ttc"
PANEL = (960, 540)


@dataclass
class Event:
    t: float            # epoch seconds when the result existed
    title: str
    image: np.ndarray
    source: str = "top_neck"


# One tile per camera / recognition, all on screen at once (canvas 1920x1080: front video 1280x720
# top-left, three 640x360 tiles down the right, two along the bottom).
TILES = [
    ("top_neck", "顶视 · 瓶口定位", (1280, 0)),
    ("left_wrist", "左腕 · 悬停/抬起核验", (1280, 360)),
    ("scene", "顶视 · 烧杯识别 DINO+SAM2", (1280, 720)),
    ("top_hold", "顶视 · 悬挂拟合 / 倾倒对准", (0, 720)),
    ("right_wrist", "右腕 · SAM2 瓶身分割", (640, 720)),
]
TILE = (640, 360)
HIGHLIGHT_S = 2.0


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def caption(image: np.ndarray, lines: list[str], size: int = 26, top: bool = True) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _font(size)
    h = (size + 8) * len(lines) + 10
    y0 = 0 if top else pil.height - h
    draw.rectangle([0, y0, pil.width, y0 + h], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((12, y0 + 6 + i * (size + 8)), line, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def fit(image: np.ndarray, size=PANEL) -> np.ndarray:
    w, h = size
    scale = min(w / image.shape[1], h / image.shape[0])
    resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return out


def _mtime(path: Path) -> float:
    return os.path.getmtime(path)


def _ellipse(image, rim, colour=(0, 255, 255)):
    out = image.copy()
    (cx, cy), (a, b), ang = rim.center_px, rim.axes_px, rim.angle_deg
    cv2.ellipse(out, ((float(cx), float(cy)), (float(a), float(b)), float(ang)), colour, 3)
    cv2.drawMarker(out, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    return out


def neck_events(run: Path) -> list[Event]:
    from chem_demo.camera_drift import load_corrected_camera
    from chem_demo.neck_vision import detect_wrist_rim, locate_top_rim
    from vision_pick.cup_locate import DEFAULT_SESSION

    left_cam = load_corrected_camera(DEFAULT_SESSION, "left")
    right_cam = load_corrected_camera(DEFAULT_SESSION, "right")
    events = []
    for path in sorted(run.glob("[0-9][0-9]_top.png")):
        image = cv2.imread(str(path))
        m = locate_top_rim(image, right_camera=right_cam, left_camera=left_cam)
        if m.ok:
            image = _ellipse(image, m.rim)
            xyz = m.left_base_xyz_mm
            text = f"瓶口 ({xyz[0]:.0f}, {xyz[1]:.0f}) mm"
        else:
            text = f"未找到：{m.reason}"
        events.append(Event(_mtime(path), f"顶视相机：定位蓝色瓶口  {text}", image))
    for path in sorted(run.glob("[0-9][0-9]_wrist.png")):
        image = cv2.imread(str(path))
        r = detect_wrist_rim(image)
        rim = getattr(r, "rim", None)
        if rim is not None:
            image = _ellipse(image, rim)
        events.append(Event(_mtime(path), "左腕相机：悬停核对瓶口与两指的相对位置", image, "left_wrist"))
    for path in sorted(run.glob("[0-9][0-9]_lift.png")):
        events.append(Event(_mtime(path), "左腕相机：抬起核验（深度判断是否夹住）", cv2.imread(str(path)), "left_wrist"))
    return events


def handoff_events(run: Path) -> list[Event]:
    events = []
    fit_path = run / "hang_top_fit.png"
    if fit_path.exists():
        events.append(Event(_mtime(fit_path), "顶视相机：SAM2 分割被举瓶子 + 圆柱拟合悬挂位置", cv2.imread(str(fit_path)),
                            "top_hold"))
    for mask_path in sorted(run.glob("seg_*_mask.png")):
        frame_path = mask_path.with_name(mask_path.name.replace("_mask", ""))
        frame, mask = cv2.imread(str(frame_path)), cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue
        over = frame.copy()
        over[mask > 0] = (0.45 * over[mask > 0] + 0.55 * np.array([0, 220, 0])).astype(np.uint8)
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, contours, -1, (0, 255, 0), 2)
        n = mask_path.name.split("_")[1]
        obs = run / f"right_obs_{n}.json"
        text = "右腕相机：SAM2 分割瓶身，伺服对准夹取点"
        if obs.exists():
            err = json.loads(obs.read_text(encoding="utf-8")).get("error")
            if isinstance(err, (list, tuple)) and len(err) >= 3:
                text += f"  误差 ({err[0]:+.0f},{err[1]:+.0f}) 距离 {err[2]:.0f} mm"
        events.append(Event(_mtime(mask_path), text, over, "right_wrist"))
    return events


def pour_events(pour: Path) -> list[Event]:
    events = []
    loc = pour / "locate" / "locate.jpg"
    if loc.exists():
        events.append(Event(_mtime(loc), "顶视相机：DINO+SAM2 识别烧杯/瓶子，圆柱拟合三维位置（倒液目标）", cv2.imread(str(loc)),
                            "scene"))
    top = pour / "top_pour.png"
    plan_path, result = pour / "pour_plan.json", pour.parent / "result.json"
    if top.exists() and plan_path.exists():
        from chem_demo.pour_plan import Vessel
        from chem_demo.run_pour import lip_overlay
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        beaker = Vessel(**plan["beaker"])
        image = lip_overlay(cv2.imread(str(top)), plan["lip_xyz"], beaker)
        events.append(Event(_mtime(top), "顶视相机：倾倒中，红点=规划瓶口唇边，绿圈=识别出的杯口", image, "top_hold"))
    return events


def _tile_image(event: Event | None, label: str, age: float | None, hot: bool) -> np.ndarray:
    if event is None:
        tile = caption(np.zeros((TILE[1], TILE[0], 3), np.uint8), [label, "等待识别"], 22)
    else:
        lines = [label, event.title.split("：", 1)[-1][:34]]
        tile = caption(fit(event.image, TILE), lines, 20)
        tile = caption(tile, [f"{int(age)} s 前"], 18, top=False)
    colour = (0, 230, 255) if hot else (70, 70, 70)
    cv2.rectangle(tile, (1, 1), (TILE[0] - 2, TILE[1] - 2), colour, 5 if hot else 2)
    return tile


def build(run: Path, video: Path, start_ms: float, out: Path) -> Path:
    events = sorted(neck_events(run) + handoff_events(run) + pour_events(run / "pour"), key=lambda e: e.t)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    raw = out.with_suffix(".raw.mp4")
    index = 0
    cache: dict[tuple, np.ndarray] = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = start_ms / 1000.0 + index / fps
        index += 1
        canvas = np.zeros((1080, 1920, 3), np.uint8)
        canvas[0:720, 0:1280] = caption(fit(frame, (1280, 720)), [f"正面相机（原始画面）  t = {index / fps:5.1f} s"], 26)
        for source, label, (x, y) in TILES:
            latest = None
            for e in events:
                if e.source == source and e.t <= t:
                    latest = e
            age = None if latest is None else t - latest.t
            hot = age is not None and age < HIGHLIGHT_S
            key = (source, id(latest), hot, None if age is None else int(age))
            if key not in cache:
                cache[key] = _tile_image(latest, label, age, hot)
            canvas[y:y + TILE[1], x:x + TILE[0]] = cache[key]
        if writer is None:
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
        writer.write(canvas)
    cap.release()
    if writer is None:
        raise RuntimeError("录像里没有读到画面")
    writer.release()
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", str(out)], check=True)
        raw.unlink()
        return out
    return raw


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="左：正面录像；右：同一时刻的视觉识别结果")
    parser.add_argument("run", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-ms", type=float, required=True, help="录像开始的 Unix 毫秒时间")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or args.run / "demo_side_by_side.mp4"
    print(build(args.run, args.video, args.start_ms, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
