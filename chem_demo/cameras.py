"""Grab one colour frame from each camera on the platform.

RealSense D405s are opened by serial number through the RealSense SDK, so their roles do not
depend on the Windows enumeration order (OpenCV indices moved between sessions). The roles are
kept in ``chem_demo/camera_roles.json``; ``--list`` prints the serials to fill it in.
The front Logitech C920 is opened by its DirectShow name with ffmpeg; it is not calibrated and
only serves as an observer.

    .venv-perception/Scripts/python.exe -X utf8 -m chem_demo.cameras --list
    .venv-perception/Scripts/python.exe -X utf8 -m chem_demo.cameras --out <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROLES_PATH = Path(__file__).with_name("camera_roles.json")
C920_NAME = "HD Pro Webcam C920"
TOP_ROTATE_180 = True  # the top camera is mounted upside down; calibration images are rotated


def list_realsense() -> list[dict[str, str]]:
    import pyrealsense2 as rs
    devices = []
    for device in rs.context().query_devices():
        devices.append({"serial": device.get_info(rs.camera_info.serial_number),
                        "name": device.get_info(rs.camera_info.name),
                        "usb": device.get_info(rs.camera_info.usb_type_descriptor)})
    return devices


def grab_realsense(serial: str, width: int = 1280, height: int = 720, warmup: int = 20, fps: int = 15,
                   depth: bool = False) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Colour frame (BGR) after auto exposure settles; optionally the aligned depth (mm, uint16)."""
    import pyrealsense2 as rs
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)  # D405 colour 1280x720 runs at 5 or 15 fps only
    if depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    try:
        frames = None
        for _ in range(warmup):
            frames = pipeline.wait_for_frames(5000)
        if depth:
            frames = rs.align(rs.stream.color).process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth_mm, scale = None, None
        if depth:
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            raw = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)
            depth_mm = (raw * scale * 1000.0).astype(np.uint16)
        return color, depth_mm, scale
    finally:
        pipeline.stop()


def grab_c920(width: int = 1920, height: int = 1080, settle_s: float = 2.0) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("找不到 ffmpeg，无法按名称打开 C920")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "c920.png"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "dshow",
                   "-video_size", f"{width}x{height}", "-i", f"video={C920_NAME}",
                   "-ss", str(settle_s), "-frames:v", "1", "-y", str(path)]
        subprocess.run(command, check=True, timeout=60)
        image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError("C920 没有返回图像")
    return image


def load_roles() -> dict[str, str]:
    if not ROLES_PATH.exists():
        raise RuntimeError(f"缺少 {ROLES_PATH.name}：先运行 --list 并按画面填写 top/left_wrist/right_wrist 的序列号")
    return json.loads(ROLES_PATH.read_text(encoding="utf-8"))


def snapshot(out: Path, roles: dict[str, str], with_c920: bool = True, depth: bool = True) -> dict[str, str]:
    """Save one frame per camera into ``out``; returns {role: path}. Top frame is rotated 180°."""
    out.mkdir(parents=True, exist_ok=True)
    saved = {}
    for role, serial in roles.items():
        color, depth_mm, _ = grab_realsense(serial, depth=depth and role != "top")
        if role == "top" and TOP_ROTATE_180:
            color = cv2.rotate(color, cv2.ROTATE_180)
        path = out / f"{role}.png"
        cv2.imwrite(str(path), color)
        saved[role] = str(path)
        if depth_mm is not None:
            depth_path = out / f"{role}_depth.png"
            cv2.imwrite(str(depth_path), depth_mm)
            saved[f"{role}_depth"] = str(depth_path)
    if with_c920:
        path = out / "front.png"
        cv2.imwrite(str(path), grab_c920())
        saved["front"] = str(path)
    (out / "snapshot.json").write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                   "roles": roles, "files": saved},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="平台相机取帧（只读，不动机械臂）")
    parser.add_argument("--list", action="store_true", help="列出 RealSense 序列号，并各存一张图以便认出角色")
    parser.add_argument("--out", type=Path, help="输出目录")
    parser.add_argument("--no-front", action="store_true", help="不拍正面 C920")
    args = parser.parse_args(argv)
    if args.list:
        out = args.out or ROOT / "calibration-data" / "chem-demo" / "camera-list"
        out.mkdir(parents=True, exist_ok=True)
        for device in list_realsense():
            color, _, _ = grab_realsense(device["serial"])
            path = out / f"{device['serial']}.png"
            cv2.imwrite(str(path), color)
            print(json.dumps({**device, "image": str(path)}, ensure_ascii=False))
        return 0
    if args.out is None:
        parser.error("需要 --out")
    print(json.dumps(snapshot(args.out, load_roles(), with_c920=not args.no_front), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
