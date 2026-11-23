"""Shared pieces of the wrist-camera programs: table heights, calibration files, the vision worker.

Runs in the robot env (dobot-xtrainer). Nothing here moves the arm.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from chem_demo.wrist_geometry import fit_plane, backproject, transform

ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_PYTHON = ROOT / ".venv-perception" / "Scripts" / "python.exe"
CALIBRATION_DIR = ROOT / "calibration-data" / "chem-demo" / "wrist-calibration"
ROLES_PATH = Path(__file__).with_name("camera_roles.json")

# Flange Z with the closed fingertips touching the white platform, each arm's own base frame.
# Right: touched 2026-11-04 (chem_demo.locate.TABLE_Z). Left: the same plane carried into the left
# base through the 11-04 top-camera calibration (243.18-243.25 over the platform), not touched.
TABLE_Z = {"right": 240.18, "left": 243.2}


def camera_serial(role: str) -> str:
    roles = json.loads(ROLES_PATH.read_text(encoding="utf-8"))
    if role not in roles:
        raise RuntimeError(f"camera_roles.json 里没有 {role}")
    return roles[role]


def calibration_path(arm: str, directory: Path = CALIBRATION_DIR) -> Path:
    return directory / f"{arm}_wrist.json"


def save_calibration(arm: str, result: dict[str, Any], gripper: np.ndarray,
                     directory: Path = CALIBRATION_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    mask_path = directory / f"{arm}_gripper_mask.png"
    cv2.imwrite(str(mask_path), gripper.astype(np.uint8) * 255)
    record = {**result, "gripper_mask": mask_path.name}
    record["T_flange_camera"] = np.asarray(result["T_flange_camera"]).round(6).tolist()
    record["K"] = np.asarray(result["K"]).round(4).tolist()
    path = calibration_path(arm, directory)
    if path.exists():  # keep earlier calibrations; never overwrite one silently
        path.rename(path.with_name(f"{path.stem}.{time.strftime('%Y%m%d_%H%M%S')}.json"))
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_calibration(arm: str, path: Path | None = None) -> dict[str, Any]:
    path = path or calibration_path(arm)
    if not path.exists():
        raise RuntimeError(f"没有 {arm} 腕部相机标定（{path}）：先运行 chem_demo.run_wrist_calib")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["T_flange_camera"] = np.array(record["T_flange_camera"], float)
    record["K"] = np.array(record["K"], float)
    mask = cv2.imread(str(path.parent / record["gripper_mask"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"缺少夹爪掩码 {record['gripper_mask']}")
    record["gripper"] = (mask > 0).astype(np.uint8)
    record["path"] = str(path)
    return record


def platform_height(T_base_camera: np.ndarray, K: np.ndarray, depth: np.ndarray, exclude: np.ndarray,
                    expected_z: float, window: float = 40.0, stride: int = 6) -> dict[str, Any]:
    """Height (base Z, mm) of the largest flat surface near the expected table height in this frame.

    The calibration cannot see the camera height along the tool axis by itself; the white platform
    around the object provides it at run time (see run_bottle_pick). Glass has no depth, so the
    bottle does not disturb this.
    """
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    us, vs = us.reshape(-1), vs.reshape(-1)
    d = depth[vs, us].astype(float)
    ok = (d > 0) & (exclude[vs, us] == 0)
    if ok.sum() < 80:
        return {"ok": False, "reason": f"有效深度点只有 {int(ok.sum())} 个"}
    base = transform(T_base_camera, backproject(K, np.column_stack([us[ok], vs[ok]]).astype(float), d[ok]))
    near = base[np.abs(base[:, 2] - expected_z) < window]
    if len(near) < 80:
        return {"ok": False, "reason": f"台面附近（±{window:.0f} mm）只有 {len(near)} 个点；"
                                       f"画面中位高度 {np.median(base[:, 2]):.1f}"}
    normal, offset, inliers = fit_plane(near, tolerance=2.0)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    z = near[inliers][:, 2]
    return {"ok": True, "z": round(float(np.median(z)), 2), "points": int(inliers.sum()),
            "share": round(float(inliers.mean()), 3), "tilt_deg": round(tilt, 2)}


class VisionClient:
    """The GPU worker (chem_demo.vision_worker) as a child process; one request at a time."""

    def __init__(self, python: Path = PERCEPTION_PYTHON, out: Callable[[str], None] = print,
                 start_timeout: float = 180.0):
        self.out = out
        self.process = subprocess.Popen([str(python), "-X", "utf8", "-m", "chem_demo.vision_worker"], cwd=ROOT,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        self.replies: "queue.Queue[dict]" = queue.Queue()
        self.stderr_tail: list[str] = []
        self.next_id = 0
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain, daemon=True).start()
        self.ready = self._wait(lambda m: m.get("type") == "ready", start_timeout, "加载识别模型")

    def _read(self) -> None:
        for line in self.process.stdout:
            try:
                self.replies.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.replies.put({"type": "exit"})

    def _drain(self) -> None:
        for line in self.process.stderr:
            self.stderr_tail = (self.stderr_tail + [line.rstrip()])[-20:]

    def _wait(self, accept: Callable[[dict], bool], timeout: float, what: str) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.replies.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty as error:
                raise RuntimeError(f"识别进程 {timeout:.0f} s 内没有完成：{what}") from error
            if message.get("type") == "exit":
                raise RuntimeError(f"识别进程已退出（{what}）：" + " | ".join(self.stderr_tail[-5:]))
            if accept(message):
                return message

    def request(self, payload: dict, timeout: float = 120.0) -> dict:
        self.next_id += 1
        ident = self.next_id
        self.process.stdin.write(json.dumps({**payload, "id": ident}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        return self._wait(lambda m: m.get("type") == "reply" and m.get("id") == ident, timeout, payload["cmd"])

    def locate_top(self, image_path: str, out_dir: str) -> dict:
        return self.request({"cmd": "locate", "image": image_path, "out": out_dir})

    def detect_wrist(self, frame: dict, image_path: str, mask_path: str,
                     near: Sequence[float] | None = None, exclude_path: str | None = None) -> dict:
        return self.request({"cmd": "detect", "image": image_path, "mask": mask_path,
                             "near": None if near is None else [float(v) for v in near],
                             "exclude": exclude_path})

    def close(self) -> None:
        try:
            self.process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.process.kill()


def save_frame(frame: dict, folder: Path, name: str) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    color_path, depth_path = folder / f"{name}.png", folder / f"{name}_depth.png"
    cv2.imwrite(str(color_path), frame["color"])
    cv2.imwrite(str(depth_path), np.clip(frame["depth"], 0, 65535).astype(np.uint16))
    return {"color": str(color_path), "depth": str(depth_path)}
