"""Read-only project integration API for validated camera-to-base calibration.

This module never activates scene transformations, writes taught poses or opens
hardware. Callers receive an immutable view bound to session hashes and must
revalidate whenever they use it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import calibrate as c
from . import table_calibration as t
from .state import (
    CalibrationError,
    SessionStore,
    board_status,
    fit_status,
    load_json,
    resolve_session_path,
    verification_status,
)


class CalibrationTransform:
    """Validated, read-only camera-to-base transform for one arm."""

    def __init__(self, store: SessionStore, session_id: str, arm: str, *,
                 expected_camera_id: str, expected_robot_id: str,
                 expected_base_frame: str, expected_tcp_name: str,
                 expected_unit: str, allow_synthetic: bool = False):
        self.store = store
        self.session_id = session_id
        self.arm = arm
        self.expected_camera_id = expected_camera_id
        self.expected_robot_id = expected_robot_id
        self.expected_base_frame = expected_base_frame
        self.expected_tcp_name = expected_tcp_name
        self.expected_unit = expected_unit
        self.allow_synthetic = allow_synthetic
        self._validated = self._validate()

    def _validate(self) -> dict[str, Any]:
        if self.arm not in ("left", "right"):
            raise CalibrationError("INVALID_ARM", "只支持 left/right")
        state = self.store.read(self.session_id)
        session_dir = self.store.session_dir(self.session_id)
        if state.get("synthetic_only") and not self.allow_synthetic:
            raise CalibrationError("SYNTHETIC_REJECTED", "synthetic_only 标定不能用于真实控制")
        if state["settings"]["camera_id"] != self.expected_camera_id:
            raise CalibrationError("CAMERA_ID_MISMATCH", "标定相机 ID 与调用方要求不一致")
        metadata = state["arms"][self.arm]["metadata"]
        for key, expected in (("robot_id", self.expected_robot_id),
                              ("base_frame", self.expected_base_frame),
                              ("tcp_name", self.expected_tcp_name),
                              ("unit", self.expected_unit)):
            if metadata[key] != expected:
                raise CalibrationError("METADATA_MISMATCH", f"{key} 与调用方要求不一致")
        fit_ok, fit_reason = fit_status(state, session_dir, self.arm)
        if not fit_ok:
            raise CalibrationError("FIT_NOT_VALID", fit_reason)
        verify_ok, verify_reason = verification_status(state, session_dir, self.arm)
        if not verify_ok:
            raise CalibrationError("VERIFICATION_NOT_VALID", verify_reason)
        board_ok, board_reason = board_status(state, session_dir)
        if not board_ok:
            raise CalibrationError("BOARD_NOT_VALID", board_reason)
        fit_record = state["arms"][self.arm]["fit"]
        fit_artifact = load_json(resolve_session_path(session_dir, fit_record["artifact_path"]))
        verification = load_json(resolve_session_path(
            session_dir, state["arms"][self.arm]["verification"]["result"]["artifact_path"]))
        if fit_artifact.get("synthetic_only") != bool(state.get("synthetic_only")):
            raise CalibrationError("SYNTHETIC_MARKER_MISMATCH", "标定 synthetic 标记不一致")
        if verification.get("synthetic_only") != bool(state.get("synthetic_only")):
            raise CalibrationError("SYNTHETIC_MARKER_MISMATCH", "验收 synthetic 标记不一致")
        return {
            "state": state,
            "fit_record": fit_record,
            "fit_artifact": fit_artifact,
            "verification": verification,
        }

    def _reload(self) -> dict[str, Any]:
        self._validated = self._validate()
        return self._validated

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._reload()["state"]["arms"][self.arm]["metadata"])

    @property
    def camera_id(self) -> str:
        return self._reload()["state"]["settings"]["camera_id"]

    @property
    def synthetic_only(self) -> bool:
        return bool(self._reload()["state"].get("synthetic_only"))

    @property
    def session_hashes(self) -> dict[str, str]:
        data = self._reload()
        return {
            "fit_input_hash": data["fit_record"]["input_hash"],
            "fit_artifact_sha256": data["fit_record"]["artifact_sha256"],
            "verification_input_hash": data["state"]["arms"][self.arm]["verification"]["result"]["input_hash"],
            "verification_artifact_sha256": data["state"]["arms"][self.arm]["verification"]["result"]["artifact_sha256"],
        }

    def point_base_m(self, u_px: float, v_px: float, board_z_m: float = 0.0,
                     *, allow_extrapolation: bool = False) -> np.ndarray:
        """Map one original-image pixel to the arm base frame, in metres."""
        data = self._reload()
        if isinstance(u_px, bool) or isinstance(v_px, bool) or isinstance(board_z_m, bool):
            raise CalibrationError("INVALID_PIXEL_QUERY", "像素与板坐标必须是有限数字")
        values = np.asarray([u_px, v_px, board_z_m], dtype=float)
        if not np.all(np.isfinite(values)):
            raise CalibrationError("NONFINITE_PIXEL_QUERY", "像素与板坐标不允许 NaN/Inf")
        state = data["state"]
        fit = data["fit_artifact"]
        session_dir = self.store.session_dir(self.session_id)
        intr_record = state["intrinsics"]["result"]
        intrinsics = load_json(resolve_session_path(session_dir, intr_record["artifact_path"]))
        K = np.asarray(intrinsics["K"], dtype=float)
        D = np.asarray(intrinsics["distortion"], dtype=float).reshape(-1)
        X = c.validate_transform(fit["T_base_camera"])
        B = c.validate_transform(fit["T_base_board"])
        predicted, board_point = t.pixel_to_plane(
            (float(u_px), float(v_px)), intrinsics, K, D, X, B, float(board_z_m),
        )
        board = fit["board"]
        inside = bool(0 <= board_point[0] <= (board["cols"] - 1) * board["square_m"]
                      and 0 <= board_point[1] <= (board["rows"] - 1) * board["square_m"])
        if not inside and not allow_extrapolation:
            raise CalibrationError("OUTSIDE_BOARD", "像素位于棋盘内角点矩形外；需要显式外推")
        return predicted

    def point_base_mm(self, u_px: float, v_px: float, board_z_m: float = 0.0,
                      *, allow_extrapolation: bool = False) -> np.ndarray:
        return self.point_base_m(u_px, v_px, board_z_m,
                                 allow_extrapolation=allow_extrapolation) * 1000.0

    def snapshot(self) -> dict[str, Any]:
        data = self._reload()
        state = data["state"]
        return {
            "session_id": self.session_id,
            "arm": self.arm,
            "camera_id": state["settings"]["camera_id"],
            "metadata": state["arms"][self.arm]["metadata"],
            "board": state["settings"]["board"],
            "synthetic_only": bool(state.get("synthetic_only")),
            "T_base_camera": data["fit_artifact"]["T_base_camera"],
            "T_base_board": data["fit_artifact"]["T_base_board"],
            "fit_metrics": state["arms"][self.arm]["fit"]["summary"],
            "verification_metrics": state["arms"][self.arm]["verification"]["result"]["summary"],
            "hashes": self.session_hashes,
            "scope": data["fit_artifact"]["validation_scope"],
            "verification_scope": data["verification"]["validation_scope"],
        }


def load_validated_calibration(data_dir: str | Path, session_id: str, arm: str, *,
                               expected_camera_id: str, expected_robot_id: str,
                               expected_base_frame: str, expected_tcp_name: str,
                               expected_unit: str,
                               allow_synthetic: bool = False) -> CalibrationTransform:
    """Load a validated calibration without mutating scene or pose state."""
    store = SessionStore(data_dir)
    return CalibrationTransform(
        store, session_id, arm,
        expected_camera_id=expected_camera_id,
        expected_robot_id=expected_robot_id,
        expected_base_frame=expected_base_frame,
        expected_tcp_name=expected_tcp_name,
        expected_unit=expected_unit,
        allow_synthetic=allow_synthetic,
    )
