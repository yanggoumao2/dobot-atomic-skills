"""Use a verified open-jaw wrist optical model on settled observations without commanding a robot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from chem_demo.wrist_geometry import pose_matrix
from chem_demo.wrist_mount import WristMountReference, refine_jaw_motion


class OpenWristCameraModel:
    """Recover base <- color camera in mm at a recorded calibration posture."""

    def __init__(self, model_path: Path, *, serial: str, K: np.ndarray):
        model_path = Path(model_path)
        data = json.loads(model_path.read_text(encoding="utf-8"))
        model_types = {f"{arm}_open_wrist_checker_plane_v1": arm for arm in ("left", "right")}
        model_types["right_open_wrist_checker_multiview_v1"] = "right"
        self.arm = model_types.get(data.get("type"))
        if not self.arm or data.get("arm", self.arm) != self.arm or data.get("checker_plane_passed") is not True:
            raise ValueError("a verified open wrist checker calibration is required")
        if data["serial"] != serial or not np.allclose(data["K"], K, rtol=1e-6, atol=1e-5):
            raise ValueError("camera serial or intrinsics differ from calibration")
        self.K = np.array(data["K"], dtype=float)
        self.mount = np.asarray(data["T_flange_reference_camera"], dtype=float)
        if (self.mount.shape != (4, 4) or not np.isfinite(self.mount).all() or
                not np.allclose(self.mount[3], [0, 0, 0, 1]) or
                not np.allclose(self.mount[:3, :3].T @ self.mount[:3, :3], np.eye(3)) or
                not np.isclose(np.linalg.det(self.mount[:3, :3]), 1.0)):
            raise ValueError("invalid calibrated rigid transform")
        paths = {}
        for role in ("color", "depth_mm"):
            resource = data["reference_files"][role]
            filename = resource["name"]
            if Path(filename).name != filename:
                raise ValueError("reference images must be beside the model")
            path = model_path.parent / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != resource["sha256"]:
                raise ValueError(f"reference image checksum mismatch: {role}")
            paths[role] = path
        self.ref_color = cv2.imread(str(paths["color"]))
        self.ref_depth = np.load(paths["depth_mm"], allow_pickle=False)
        self.reference = WristMountReference.from_frame(self.ref_color, self.ref_depth, self.K, arm=self.arm)
        self.poses = np.array(list(data["tcp_poses_mm_deg"].values()))

    def at_recorded_pose(self, pose_mm_deg, color_bgr, depth_mm, *, gripper_open: bool) -> np.ndarray:
        """Use aligned, undistorted frames. Refuse unseen poses and unobservable/closed fingers."""
        pose = np.asarray(pose_mm_deg, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all() or not gripper_open:
            raise ValueError("a finite six-axis pose and an open gripper are required")
        close_xyz = np.linalg.norm(self.poses[:, :3] - pose[:3], axis=1) <= 0.5
        pose_rotation = pose_matrix(pose)[:3, :3]
        angle = [np.degrees(np.arccos(np.clip((np.trace(pose_matrix(p)[:3, :3].T @ pose_rotation) - 1) / 2, -1, 1)))
                 for p in self.poses]
        if not np.any(close_xyz & (np.array(angle) <= 0.5)):
            raise ValueError("outside verified calibration postures")
        seed = self.reference.measure(color_bgr, depth_mm)
        if not seed.ok:
            raise ValueError(seed.reason)
        jaw = refine_jaw_motion(self.ref_color, self.ref_depth, color_bgr, depth_mm, self.K,
                                seed.T_reference_current, arm=self.arm)
        return pose_matrix(pose) @ self.mount @ jaw
