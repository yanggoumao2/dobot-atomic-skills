"""Wrist D405 frames for the robot process: a live pipeline kept open, and a synthetic stand-in.

LiveWristCamera opens one RealSense by serial (chem_demo/camera_roles.json) with colour and depth
aligned to colour, and keeps it running so each grab after a move only drops a few stale frames.

SimWristCamera renders what a wrist camera rigidly fixed to the flange would see, from the arm's
current pose: a textured table, an upright transparent bottle (no depth, like real glass) and two
gripper fingers fixed in the image. It is used by the offline tests and --sim runs, with a hidden
camera mount that the calibration has to recover.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import cv2
import numpy as np

from chem_demo.wrist_geometry import intrinsics_matrix, pose_matrix
from vision_pick.cup_locate import Camera, frustum_silhouette

FRESH_FRAME_RETRY_LIMIT = 20


MAX_RECONNECTS = 3
RECONNECT_WAIT_S = 20.0
_LOST_MARKERS = ("no device connected", "device disconnected", "failed to reconnect")


def device_lost(error: BaseException) -> bool:
    """librealsense errors that mean the camera dropped off USB (not a slow frame)."""
    text = str(error).lower()
    return any(marker in text for marker in _LOST_MARKERS)


class LiveWristCamera:
    # 848x480 @ 30: same field of view as 1280x720 (fx 435 vs 652.5). On this rig 1280x720 colour +
    # depth delivered < 1 frame/s from the left wrist D405 (2026-11-21, USB 3.2), 848x480@30 ~11/s.
    def __init__(self, serial: str, width: int = 848, height: int = 480, fps: int = 30, warmup: int = 30):
        import pyrealsense2 as rs

        self.rs = rs
        self.serial = serial
        self.width, self.height, self.fps = width, height, fps
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._config = config
        self.reconnects = 0
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as error:
            if not device_lost(error):
                raise
            # 2026-11-23: all three D405s share one bus-powered USB 3 hub; the left wrist drops off and
            # re-enumerates a few seconds later. Wait for it instead of failing the run.
            self._wait_for_device()
            self.pipeline = rs.pipeline()
            profile = self.pipeline.start(config)
            self.reconnects += 1
        self.restarts = 0
        self.streaming = True
        self.align = rs.align(rs.stream.color)
        self.depth_scale_m_per_raw_unit = profile.get_device().first_depth_sensor().get_depth_scale()
        self.scale_mm = self.depth_scale_m_per_raw_unit * 1000.0
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = intrinsics_matrix(intr.fx, intr.fy, intr.ppx, intr.ppy)
        self.distortion = list(intr.coeffs)
        # The D405 colour lens is not distortion-free (k1 ≈ -0.05, k2 ≈ +0.06 on the left wrist,
        # tens of pixels at the corners). Frames are undistorted here, so everything downstream
        # (features, masks, fits) is a pinhole camera with matrix K.
        D = np.array(self.distortion, float)
        self.maps = cv2.initUndistortRectifyMap(self.K, D, None, self.K, (width, height), cv2.CV_32FC1)
        self._last_sdk_sample: dict[str, Any] | None = None
        for _ in range(warmup):  # auto exposure
            self.pipeline.wait_for_frames(5000)

    def describe(self) -> dict[str, Any]:
        """Camera calibration and units needed to reproduce a captured observation."""
        return {
            "source": "realsense",
            "serial": self.serial,
            "stream": {"width_px": self.width, "height_px": self.height, "fps": self.fps},
            "intrinsics": {
                "matrix_px": self.K.tolist(),
                "fx_px": float(self.K[0, 0]), "fy_px": float(self.K[1, 1]),
                "cx_px": float(self.K[0, 2]), "cy_px": float(self.K[1, 2]),
                "distortion_coefficients": self.distortion,
                "distortion_removed_from_output": True,
            },
            "depth_scale_m_per_raw_unit": float(self.depth_scale_m_per_raw_unit),
            "depth_scale_mm_per_raw_unit": float(self.scale_mm),
            "depth_output_unit": "mm",
            "depth_alignment": "aligned to color",
        }

    def grab(self, flush: int = 4, *, require_fresh_pair: bool = False) -> dict[str, Any]:
        """Undistorted colour (BGR) and depth (float32 mm, 0 = invalid) taken after the arm settled.

        A stalled stream ("Frame didn't arrive within 5000") is restarted once: with three D405s on
        this PC a wrist stream left idle for minutes stalled during the 2026-11-22 handoff run.
        """
        if not self.streaming:
            self.resume()
        try:
            return self._grab_once(flush, require_fresh_pair=require_fresh_pair)
        except RuntimeError as error:
            if device_lost(error) and self.reconnects < MAX_RECONNECTS:
                self._reopen_after_loss()
                return self._grab_once(flush, require_fresh_pair=require_fresh_pair)
            if "didn't arrive" not in str(error) or self.restarts >= 3:
                raise
            self.restarts += 1
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline.start(self._config)
            self._last_sdk_sample = None
            for _ in range(10):
                self.pipeline.wait_for_frames(5000)
            return self._grab_once(flush, require_fresh_pair=require_fresh_pair)

    def _wait_for_device(self, timeout_s: float = RECONNECT_WAIT_S) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if any(d.get_info(self.rs.camera_info.serial_number) == self.serial
                   for d in self.rs.context().query_devices()):
                time.sleep(1.0)  # let Windows finish binding the interfaces
                return
            time.sleep(0.5)
        raise RuntimeError(f"RealSense {self.serial} did not come back within {timeout_s:.0f} s after dropping off USB")

    def _reopen_after_loss(self, warmup: int = 10) -> None:
        """New pipeline on the same config after the device re-enumerated."""
        self.reconnects += 1
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001 - the old pipeline is already dead
            pass
        self.streaming = False
        self._wait_for_device()
        self.pipeline = self.rs.pipeline()
        self.pipeline.start(self._config)
        self.streaming = True
        self._last_sdk_sample = None
        for _ in range(warmup):
            self.pipeline.wait_for_frames(5000)

    def pause(self) -> None:
        """Stop streaming (USB bandwidth: three D405s on one PC stalled when all streamed at once)."""
        if self.streaming:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.streaming = False

    def resume(self, warmup: int = 10) -> None:
        """Restart a paused stream; one retry if the first frames do not arrive."""
        for attempt in range(2):
            try:
                self.pipeline.start(self._config)
            except RuntimeError as error:
                if not device_lost(error) or self.reconnects >= MAX_RECONNECTS:
                    raise
                self._reopen_after_loss(warmup)
                return
            self.streaming = True
            self._last_sdk_sample = None
            try:
                for _ in range(warmup):
                    self.pipeline.wait_for_frames(5000)
                return
            except RuntimeError:
                self.pause()
                if attempt == 1:
                    raise

    def _grab_once(self, flush: int = 4, *, require_fresh_pair: bool = False) -> dict[str, Any]:
        retries = 0
        if require_fresh_pair:
            for _ in range(max(0, flush)):  # discard queued frames, including frames during motion
                self.pipeline.wait_for_frames(5000)
            for _ in range(FRESH_FRAME_RETRY_LIMIT):
                raw_frames = self.pipeline.wait_for_frames(5000)
                frames = self.align.process(raw_frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise RuntimeError("RealSense frameset is missing an aligned color or depth frame")
                sdk = self._sdk_metadata(color_frame, depth_frame)
                previous = self._last_sdk_sample
                fresh = previous is None or all(
                    sdk[f"{stream}_frame_counter"] != previous[f"{stream}_frame_counter"] and
                    (sdk[f"{stream}_timestamp_domain"] != previous[f"{stream}_timestamp_domain"] or
                     sdk[f"{stream}_timestamp_ms"] > previous[f"{stream}_timestamp_ms"])
                    for stream in ("color", "depth")
                )
                if fresh:
                    break
                retries += 1
            else:
                raise RuntimeError(
                    f"RealSense did not provide a new color/depth frame pair after {FRESH_FRAME_RETRY_LIMIT} attempts")
        else:
            raw_frames = None
            for _ in range(max(0, flush) + 1):  # preserve the shared camera's existing normal-pick behavior
                raw_frames = self.pipeline.wait_for_frames(5000)
            frames = self.align.process(raw_frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("RealSense frameset is missing an aligned color or depth frame")
            sdk = self._sdk_metadata(color_frame, depth_frame)
        sdk["freshness_retries"] = retries
        if sdk["color_timestamp_domain"] == sdk["depth_timestamp_domain"]:
            sdk["color_depth_timestamp_delta_ms"] = abs(
                sdk["color_timestamp_ms"] - sdk["depth_timestamp_ms"])
        else:
            sdk["color_depth_timestamp_delta_ms"] = None
        self._last_sdk_sample = sdk.copy()
        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.scale_mm
        color = cv2.remap(color, *self.maps, cv2.INTER_LINEAR)
        depth = cv2.remap(depth, *self.maps, cv2.INTER_NEAREST)  # never blend depths across an edge
        return {"color": color, "depth": depth, "t": time.time(),
                "capture_metadata": {"source": "realsense", "sdk": sdk}}

    @staticmethod
    def _sdk_metadata(color_frame: Any, depth_frame: Any) -> dict[str, Any]:
        return {
            "color_timestamp_ms": float(color_frame.get_timestamp()),
            "color_frame_counter": int(color_frame.get_frame_number()),
            "color_timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
            "depth_timestamp_ms": float(depth_frame.get_timestamp()),
            "depth_frame_counter": int(depth_frame.get_frame_number()),
            "depth_timestamp_domain": str(depth_frame.get_frame_timestamp_domain()),
            "timestamp_units": "milliseconds",
            "frame_counter_units": "frames",
            "synchronized_frameset": True,
        }

    def close(self) -> None:
        if not getattr(self, "streaming", True):
            return
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001 - closing must not mask the real error
            pass
        self.streaming = False


def look_at(position: Sequence[float], target: Sequence[float], up_hint: Sequence[float]) -> np.ndarray:
    """4x4 frame <- camera with the optical axis (camera +Z) from position towards target."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up_hint, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = position
    return T


# Flange frame of a tool pointing down: +Z points at the table. The fingertip convention of
# wrist_geometry puts the origin at the fingertips. This mount resembles the real one: the camera
# sits behind and above the fingers and looks forward-down past them.
SIM_T_FLANGE_CAMERA = look_at([4.0, 95.0, -230.0], [0.0, -60.0, 120.0], [1.0, 0.0, 0.0])


class SimWristCamera:
    def __init__(self, arm: Any, *, table_z: float, bottle: dict | None = None,
                 T_flange_camera: np.ndarray = SIM_T_FLANGE_CAMERA, width: int = 640, height: int = 360,
                 seed: int = 3):
        self.arm, self.table_z, self.bottle = arm, float(table_z), bottle
        self._grasp_z: float | None = None
        self._frame_counter = 0
        self.T_flange_camera = T_flange_camera
        self.width, self.height = width, height
        f = 0.5 * width / math.tan(math.radians(87.0 / 2))  # D405 colour HFOV ≈ 87°
        self.K = intrinsics_matrix(f, f, width / 2 - 0.5, height / 2 - 0.5)
        rng = np.random.default_rng(seed)
        noise = rng.random((1024, 1024)).astype(np.float32)
        texture = cv2.GaussianBlur(noise, (0, 0), 3.0)
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        self.texture = (60 + 160 * texture).astype(np.uint8)  # 1 px = 1 mm on the table
        self.gripper = np.zeros((height, width), np.uint8)  # two fingers, fixed in the image
        cv2.fillPoly(self.gripper, [np.array([[0, height], [int(0.38 * width), height],
                                              [int(0.30 * width), int(0.62 * height)], [0, int(0.70 * height)]])], 1)
        cv2.fillPoly(self.gripper, [np.array([[width, height], [int(0.62 * width), height],
                                              [int(0.70 * width), int(0.62 * height)], [width, int(0.70 * height)]])], 1)

    def describe(self) -> dict[str, Any]:
        return {
            "source": "simulation",
            "serial": None,
            "stream": {"width_px": self.width, "height_px": self.height, "fps": None},
            "intrinsics": {"matrix_px": self.K.tolist(), "fx_px": float(self.K[0, 0]),
                           "fy_px": float(self.K[1, 1]), "cx_px": float(self.K[0, 2]),
                           "cy_px": float(self.K[1, 2]), "distortion_coefficients": [0.0] * 5,
                           "distortion_removed_from_output": True},
            "depth_scale_m_per_raw_unit": None,
            "depth_scale_mm_per_raw_unit": None,
            "depth_output_unit": "mm",
            "depth_alignment": "synthetic color and depth share a pixel grid",
        }

    def camera(self) -> Camera:
        pose = self.arm.read()["pose"]
        return Camera(self.K, np.zeros(5), pose_matrix(pose) @ self.T_flange_camera)

    def bottle_base_z(self) -> float:
        """A bottle between closed jaws rides with the flange (grasp = first frame seen closed)."""
        pose = self.arm.read()["pose"]
        closed = self.arm.gripper_position() < 128
        if closed and self.bottle is not None and self._grasp_z is None:
            if math.hypot(pose[0] - self.bottle["x"], pose[1] - self.bottle["y"]) < 0.8 * self.bottle["r"]:
                self._grasp_z = pose[2]
        if not closed:
            self._grasp_z = None
        return self.table_z + (0.0 if self._grasp_z is None else pose[2] - self._grasp_z)

    def bottle_mask(self, camera: Camera | None = None) -> np.ndarray:
        camera = camera or self.camera()
        mask = np.zeros((self.height, self.width), np.uint8)
        if self.bottle is None:
            return mask
        b = self.bottle
        outline = frustum_silhouette(camera, [b["x"], b["y"], b["h"], b["r"], b["r"]], self.bottle_base_z())
        cv2.fillPoly(mask, [np.round(outline).astype(np.int32)], 1)
        return mask

    def grab(self, flush: int = 0) -> dict[str, Any]:
        camera = self.camera()
        R, C = camera.T_base_camera[:3, :3], camera.T_base_camera[:3, 3]
        vs, us = np.mgrid[0:self.height, 0:self.width].astype(np.float64)
        rays = np.stack([(us - self.K[0, 2]) / self.K[0, 0], (vs - self.K[1, 2]) / self.K[1, 1],
                         np.ones_like(us)], axis=-1) @ R.T
        scale = (self.table_z - C[2]) / rays[..., 2]            # camera-Z depth = scale (ray z_cam = 1)
        hit = C + rays * scale[..., None]
        tu = np.mod(np.round(hit[..., 0]).astype(np.int64), 1024)
        tv = np.mod(np.round(hit[..., 1]).astype(np.int64), 1024)
        gray = self.texture[tv, tu].copy()
        depth = np.where(scale > 0, scale, 0.0).astype(np.float32)
        bottle = self.bottle_mask(camera).astype(bool) & (self.gripper == 0)
        gray[bottle] = (0.55 * gray[bottle] + 90).astype(np.uint8)  # glass: see-through, a bit lighter
        depth[bottle] = 0.0                                          # and no depth, like the D405 on glass
        gray[self.gripper == 1] = 25
        depth[self.gripper == 1] = 0.0
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        captured_at = time.time()
        self._frame_counter += 1
        return {"color": color, "depth": depth, "t": captured_at, "truth_bottle": bottle.astype(np.uint8),
                "capture_metadata": {"source": "simulation", "timestamp_unix_s": captured_at,
                                     "frame_counter": self._frame_counter,
                                     "timestamp_units": "seconds since Unix epoch",
                                     "frame_counter_units": "frames"}}

    def close(self) -> None:
        pass


class SimVision:
    """Stand-in for the GPU worker in --sim and tests: returns the rendered bottle mask."""

    def __init__(self, camera: SimWristCamera, coarse_error: Sequence[float] = (0.0, 0.0)):
        self.camera = camera
        self.coarse_error = coarse_error

    def locate_top(self, image_path: str, out_dir: str) -> dict[str, Any]:
        """Top-camera result with a chosen error (the real one is off by about 1 cm)."""
        b = self.camera.bottle
        xy = [b["x"] + self.coarse_error[0], b["y"] + self.coarse_error[1]]
        bottle = {"x": xy[0], "y": xy[1], "height": b["h"] * 0.7, "score": 0.4,
                  "left_base": {"axis_xyz_at_base": [*xy, self.camera.table_z]}}
        return {"ok": True, "vessels": {"bottle": bottle}, "notes": [], "overlay": None}

    def detect_wrist(self, frame: dict[str, Any], image_path: str, mask_path: str,
                     near: Sequence[float] | None = None, exclude_path: str | None = None) -> dict[str, Any]:
        mask = frame.get("truth_bottle")
        if mask is None or not mask.any():
            return {"ok": False, "error": "画面里没有瓶子"}
        cv2.imwrite(mask_path, mask * 255)
        ys, xs = np.nonzero(mask)
        return {"ok": True, "mask": mask_path, "score": 0.9,
                "box": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]}

    def close(self) -> None:
        pass
