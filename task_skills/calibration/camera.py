"""Explicit, operator-triggered local OpenCV camera preview."""

from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

from .state import CalibrationError, utc_now


class CameraManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._capture: cv2.VideoCapture | None = None
        self._meta: dict[str, Any] = {"open": False, "device_opened_at_startup": False}

    def start(self, device: int, width: int, height: int) -> dict[str, Any]:
        with self._lock:
            if self._capture is not None:
                return dict(self._meta)
            if isinstance(device, bool) or not isinstance(device, int) or not 0 <= device <= 32:
                raise CalibrationError("INVALID_CAMERA_DEVICE", "相机编号必须在 0–32")
            if isinstance(width, bool) or not isinstance(width, int) or not 64 <= width <= 8192:
                raise CalibrationError("INVALID_CAMERA_WIDTH", "预览宽度必须在 64–8192")
            if isinstance(height, bool) or not isinstance(height, int) or not 64 <= height <= 8192:
                raise CalibrationError("INVALID_CAMERA_HEIGHT", "预览高度必须在 64–8192")
            capture = cv2.VideoCapture(device)
            self._capture = capture
            try:
                if not capture.isOpened():
                    raise CalibrationError("CAMERA_OPEN_FAILED", "无法打开相机；请检查设备占用与 OpenCV 后端")
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise CalibrationError("CAMERA_FRAME_FAILED", "相机已打开但无法读取首帧")
                actual_height, actual_width = frame.shape[:2]
                self._meta = {
                    "open": True,
                    "device_opened_at_startup": False,
                    "opened_utc": utc_now(),
                    "device": device,
                    "requested_wh": [width, height],
                    "actual_wh": [actual_width, actual_height],
                    "backend": capture.getBackendName() if hasattr(capture, "getBackendName") else "opencv",
                    "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                }
                return dict(self._meta)
            except Exception:
                capture.release()
                self._capture = None
                self._meta = {"open": False, "device_opened_at_startup": False}
                raise

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._meta)

    def _read_locked(self) -> np.ndarray:
        if self._capture is None:
            raise CalibrationError("CAMERA_NOT_RUNNING", "相机预览未启动", status=409)
        ok, frame = self._capture.read()
        if not ok or frame is None or frame.ndim != 3:
            self.stop()
            raise CalibrationError("CAMERA_FRAME_FAILED", "读取相机帧失败，预览已关闭")
        self._meta["last_frame_utc"] = utc_now()
        return frame

    def frame_jpeg(self) -> bytes:
        with self._lock:
            frame = self._read_locked()
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if not ok:
                raise CalibrationError("CAMERA_ENCODE_FAILED", "相机帧编码失败")
            return encoded.tobytes()

    def capture_png(self) -> tuple[bytes, tuple[int, int]]:
        with self._lock:
            frame = self._read_locked()
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise CalibrationError("CAMERA_ENCODE_FAILED", "相机帧编码失败")
            height, width = frame.shape[:2]
            return encoded.tobytes(), (width, height)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
            self._capture = None
            previous = dict(self._meta)
            self._meta = {"open": False, "device_opened_at_startup": False,
                          "closed_utc": utc_now()}
            return {**previous, **self._meta}

    def close(self) -> None:
        self.stop()
