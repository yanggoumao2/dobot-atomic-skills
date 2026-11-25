"""Dropper demo video: the C920 recording plus what the vision and the gripper did, when they did it.

Same layout as make_demo_video (front camera 1280x720 top-left, five 640x360 tiles):

* top camera: blue bulb coarse location (top_bulb.jpg), the scene after the lift (lifted_top.png)
* right wrist: bulb seen in each look (NN_*_seen.jpg: red = bulb top, green = tool axis at that height)
* right wrist: tube-mouth fit + lean of the bulb, which sets the jaw turn (rim steps)
* right wrist: held check (pad window closed / lifted) and the vertical line after the lift
* gripper: code over time with the step label (grasp, suction, every drop, release)

Run with the perception venv (PIL):

  python -m chem_demo.make_dropper_video <session dir> --video c920.mkv --start-ms <epoch ms>

<session dir>/session.json lists the run_dropper folders (lift, drip, put-back) recorded in this video.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from chem_demo.make_demo_video import TILE, Event, HIGHLIGHT_S, _tile_image, caption, fit

TILES = [
    ("top", "顶视 · 捏头定位", (1280, 0)),
    ("wrist", "右腕 · 捏头识别与对准", (1280, 360)),
    ("rim", "右腕 · 管口拟合与扶正", (1280, 720)),
    ("held", "右腕 · 夹持/竖直核验", (0, 720)),
    ("grip", "夹爪 · 夹取/吸液/滴加", (640, 720)),
]
GRIP_OPEN = 1000   # gripper codes: 0 closed .. 1000 open


def _mtime(path: Path) -> float:
    return os.path.getmtime(path)


def _grip_panel(code: int, history: list[int]) -> np.ndarray:
    """Drawn into the middle band of the tile (the tile adds its own title on top and age at the bottom)."""
    panel = np.zeros((360, 640, 3), np.uint8)
    band = caption(np.zeros((150, 640, 3), np.uint8),
                   [f"夹爪码值 {code}", "（0 合拢 / 1000 全开）", "历次: " + " → ".join(str(v) for v in history[-7:])], 26)
    panel[80:230] = band
    # small codes matter (suction 26, drops 24..16): log scale so they stay visible
    x0, x1, y = 40, 600, 275
    cv2.rectangle(panel, (x0, y - 16), (x1, y + 16), (90, 90, 90), 2)
    frac = np.log1p(max(code, 0)) / np.log1p(GRIP_OPEN)
    cv2.rectangle(panel, (x0, y - 16), (x0 + int((x1 - x0) * frac), y + 16), (0, 200, 255), -1)
    return panel


def _rim_view(run: Path, log: dict, rim: dict) -> np.ndarray | None:
    """Raw wrist frame with the fitted tube mouth (cyan circle), its centre (cyan) and the bulb (red)."""
    try:
        from chem_demo.run_dropper import RIGHT_MODEL, _project
        from chem_demo.wrist_geometry import pose_matrix
        look = next(s for s in log["steps"] if s["step"] == "look" and s.get("frame") == rim.get("frame"))
        image = cv2.imread(str(run / "frames" / f"{rim['frame']}.png"))
        model = json.loads(RIGHT_MODEL.read_text(encoding="utf-8"))
        T = pose_matrix(look["pose"]) @ np.asarray(model["T_flange_reference_camera"], float)
        K = np.asarray(log["wrist_K"], float)
        z = float(look["rim_z"])
        (cx, cy), r = rim["centre"], float(rim["radius"])
        ring = np.array([[cx + r * np.cos(a), cy + r * np.sin(a), z] for a in np.linspace(0, 2 * np.pi, 72)])
        uv = _project(K, T, ring).astype(np.int32)
        cv2.polylines(image, [uv.reshape(-1, 1, 2)], True, (255, 255, 0), 2)
        c = _project(K, T, np.array([[cx, cy, z]]))[0]
        b = _project(K, T, np.array([[*rim["bulb_xy"], z]]))[0]
        cv2.drawMarker(image, (int(c[0]), int(c[1])), (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.drawMarker(image, (int(b[0]), int(b[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        cv2.arrowedLine(image, (int(c[0]), int(c[1])), (int(b[0]), int(b[1])), (0, 0, 255), 2, tipLength=0.3)
        return image
    except Exception as error:  # noqa: BLE001 - fall back to the detection frame
        print(f"管口叠加失败 {rim.get('frame')}: {error}")
        return None


def run_events(run: Path) -> list[Event]:
    log_path = run / "dropper_run.json"
    if not log_path.exists():
        return []
    log = json.loads(log_path.read_text(encoding="utf-8"))
    events: list[Event] = []
    top = run / "top_bulb.jpg"
    if top.exists():
        coarse = log.get("coarse") or {}
        xy = coarse.get("xy")
        if (coarse.get("top") or {}).get("ok") is False:
            text = "未认出捏头，用上次腕部测得的管口位置"
        else:
            text = f"蓝色捏头粗定位 ({xy[0]:.0f}, {xy[1]:.0f}) mm" if xy else "蓝色捏头粗定位"
        events.append(Event(_mtime(top), f"顶视相机：{text}", cv2.imread(str(top)), "top"))
    lifted_top = run / "lifted_top.png"
    if lifted_top.exists():
        events.append(Event(_mtime(lifted_top), "顶视相机：滴管已竖直提出试管", cv2.imread(str(lifted_top)), "top"))
    frames = run / "frames"
    for path in sorted(frames.glob("[0-9][0-9]_*_seen.jpg")):
        name = path.name[:-len("_seen.jpg")]
        stage = "低位" if "_low" in name else "高位"
        events.append(Event(_mtime(path), f"右腕：{stage}识别捏头（红=捏头顶 绿=工具轴）", cv2.imread(str(path)), "wrist"))
    turn = log.get("jaw_turn_deg")
    for s in log.get("steps", []):
        if s["step"] != "rim":
            continue
        image = _rim_view(run, log, s)
        if image is None:
            path = frames / f"{s.get('frame', '')}_seen.jpg"
            image = cv2.imread(str(path)) if path.exists() else np.zeros((480, 848, 3), np.uint8)
        text = f"青圈=管口 红=捏头 偏{s.get('lean_mm', 0):.0f}mm"
        if turn is not None:
            text += f" 爪转{float(turn):+.0f}°扶正"
        events.append(Event(float(s["t"]), "右腕：" + text, image, "rim"))
    for name, text in (("held_closed.jpg", "合爪后：指间蓝色区域与深度"),
                       ("held_lifted.jpg", "提起后：蓝色仍在两指间 → 夹住"),
                       ("lifted_vertical.jpg", "提起后：绿线=竖直线，滴管沿线下垂")):
        path = frames / name
        if path.exists():
            events.append(Event(_mtime(path), f"右腕：{text}", cv2.imread(str(path)), "held"))
    history: list[int] = []
    for s in log.get("steps", []):
        if s["step"] != "gripper":
            continue
        code = int(s.get("final", s.get("target", 0)))
        history.append(code)
        events.append(Event(float(s["t"]), f"夹爪：{s.get('label', '')}", _grip_panel(code, list(history)), "grip"))
    return events


def build(session: Path, video: Path, start_ms: float, out: Path) -> Path:
    runs = json.loads((session / "session.json").read_text(encoding="utf-8"))["runs"]
    events = sorted((e for r in runs for e in run_events(Path(r["dir"]))), key=lambda e: e.t)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps > 120:   # mkv may report a timebase instead of a frame rate
        fps = 30.0
    raw = out.with_suffix(".raw.mp4")
    writer, index, cache = None, 0, {}
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
    parser = argparse.ArgumentParser(description="滴管演示：左上正面录像，周围是同一时刻的视觉识别与夹爪码值")
    parser.add_argument("session", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-ms", type=float, required=True, help="录像开始的 Unix 毫秒时间")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    print(build(args.session, args.video, args.start_ms, args.out or args.session / "dropper_side_by_side.mp4"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
