"""Right-wrist closed-loop observer for the handoff cycle.

Wraps the offline vision in ``handoff_vision`` around the live right D405: grabs a fresh
undistorted RGB-D pair, finds the bottle held by the left hand, measures where the pad
midpoint is relative to the bottle body axis, and returns the correction in the right Tool0
axes (X = along the bottle axis, Y = jaw axis, Z = tool axis towards the bottle).

The right wrist calibration was verified only for downward open-jaw postures, so every
measurement is relative (bottle against pads in the same image) and carries that caveat.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Any, Callable

import cv2
import numpy as np

from chem_demo.handoff_vision import alignment_error, jaw_geometry, locate_held_bottle, transfer_check
from chem_demo.wrist_camera import LiveWristCamera
from chem_demo.gripper_geometry import pad_offset_tip_frame
from chem_demo.wrist_runtime import ROOT

RIGHT_MODEL = (ROOT / "calibration-data" / "chem-demo" / "wrist-board" /
               "right-20261121_155150-recovery-20261121_161320_332390" / "checker_multiview_validation.json")
# Pad midpoint in the HAND-EYE frame, whose origin is at the fingertips (not the flange): [+3, 0, -16]
# from the right-wrist board model. Until 2026-11-22 the flange offset (then 215 mm) was used here,
# which projected the pads ~150 px above where they are in the image.
PAD_OFFSET_TOOL_MM = pad_offset_tip_frame()


def servo_error(err_tool: Sequence[float], rpy: Sequence[float]) -> list[float]:
    """Bottle grasp-point error -> what the cycle commands, for a tool that may be pitched down.

    ``err_tool`` = bottle grasp point (on the axis, ``GRASP_ABOVE_BOTTOM_MM`` above the bottom) minus
    the pad midpoint in Tool0 (X, Y, Z). Returned as [vertical, lateral, s] in tool axes: the world-
    vertical part of the error (expressed along tool X, which the cycle moves as "vertical"), the
    jaw-axis (tool Y) part, and ``s``, how far to advance along tool Z to bring the pads onto the
    axis (s = horizontal distance / cos(pitch)). The vertical part was dropped until 2026-11-22 ("any
    height on the body is fine"); with a 120 mm height error the pads then missed the bottle.
    """
    from chem_demo.handoff_plan import _rot
    e = np.asarray(err_tool, float)
    up = _rot([float(v) for v in rpy]).T @ np.array([0.0, 0.0, 1.0])   # world up in tool axes
    e_perp = e - (e @ up) * up
    z = np.array([0.0, 0.0, 1.0])
    h = z - (z @ up) * up
    if float(h @ h) < 0.25:  # tool axis within 60 deg of vertical: not a side grasp
        raise ValueError("tool axis too steep for a side grasp")
    s = float(e_perp @ h / (h @ h))
    # Vertical error as a tool-frame vector; advancing s along tool Z also moves s*(z.up) vertically.
    vertical = (float(e @ up) - s * float(z @ up)) * up
    return [float(vertical[0]), float(e_perp[1] + vertical[1]), s + float(vertical[2])]


def side_view_hints(K: np.ndarray, T_tool_camera: np.ndarray, pose: Sequence[float] | None, jaw: Any,
                    expected_range_mm: float | None) -> dict[str, Any]:
    """Image direction of world-down at the pad midpoint, and where the bottle should appear.

    ``expected_range_mm`` = planned distance from the pads to the bottle axis along tool Z (the
    standoff on the first look); the expected pixel is that point projected, a soft prior only.
    """
    from chem_demo.handoff_plan import _rot
    if pose is None or not getattr(jaw, "ok", False):
        return {}
    R_cam_tool = np.asarray(T_tool_camera, float)[:3, :3].T
    down_cam = R_cam_tool @ (_rot([float(v) for v in pose[3:6]]).T @ np.array([0.0, 0.0, -1.0]))
    mid = np.array(jaw.pad_midpoint_cam_mm, float)
    hints: dict[str, Any] = {}
    target = mid + (R_cam_tool @ np.array([0.0, 0.0, float(expected_range_mm)]) if expected_range_mm else 0.0)
    if target[2] > 1.0:
        a = np.asarray(K, float) @ target
        b = np.asarray(K, float) @ (target + 20.0 * down_cam)
        if b[2] > 1.0:
            pa, pb = a[:2] / a[2], b[:2] / b[2]
            if np.linalg.norm(pb - pa) > 1e-3:
                hints["down_px"] = (pb - pa).tolist()
            hints["expected_px"] = pa.tolist()
    return hints


class RightWristObserver:
    def __init__(self, stack: Any, serial: str, directory: Path, *, camera: Any = None,
                 model_path: Path = RIGHT_MODEL, segment: bool = True, out: Callable[[str], None] = print):
        data = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.T_tool_camera = np.asarray(data["T_flange_reference_camera"], float)
        self.directory = Path(directory)
        self.out = out
        self.count = 0
        self._near = None
        self.expected_range_mm: float | None = None   # set by the cycle from the plan (standoff)
        if camera is None:
            camera = LiveWristCamera(serial, fps=15, warmup=6)  # shared USB hub: 15 fps
            if stack is not None:
                stack.callback(camera.close)
        self.camera = camera
        self.K = np.asarray(camera.K, float)
        if hasattr(camera, "pause"):
            camera.pause()  # streams only when the right arm needs it (USB bandwidth, see wrist_camera)
        self.client = None
        if segment:
            from chem_demo.wrist_runtime import VisionClient
            self.client = VisionClient(out=out)
            if stack is not None:
                stack.callback(self.client.close)

    # The injected segmenter writes the frame, asks the GPU worker for the bottle mask and reads it back.
    def _segmenter(self, color: np.ndarray, prompt: str):
        if self.client is None:
            return []
        image_path = self.directory / f"seg_{self.count:03d}.png"
        mask_path = self.directory / f"seg_{self.count:03d}_mask.png"
        cv2.imwrite(str(image_path), color)
        reply = self.client.detect_wrist({}, str(image_path), str(mask_path), near=self._near)
        if not reply.get("ok"):
            return []
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return [] if mask is None else [{"mask": mask, "score": float(reply.get("score", 1.0))}]

    def _grab(self) -> dict[str, Any]:
        frame = self.camera.grab(flush=2, require_fresh_pair=True)
        self.count += 1
        return frame

    def capture(self) -> dict[str, Any]:
        """A fresh colour + depth pair, used before/after the retreat by the transfer check."""
        frame = self._grab()
        cv2.imwrite(str(self.directory / f"right_capture_{self.count:03d}.png"), frame["color"])
        return {"color": frame["color"].copy(), "depth": None if frame.get("depth") is None else frame["depth"].copy()}

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        frame = self._grab()
        color, depth = frame["color"], frame.get("depth")
        jaw = jaw_geometry(self.K, self.T_tool_camera, PAD_OFFSET_TOOL_MM)
        hints = side_view_hints(self.K, self.T_tool_camera, state.get("pose"), jaw, self.expected_range_mm)
        self._near = hints.get("expected_px")
        self._hints = hints
        view = locate_held_bottle(color, depth, self.K, segmenter=self._segmenter, **hints)
        error = alignment_error(view, jaw, tool_from_camera=self.T_tool_camera)
        cv2.imwrite(str(self.directory / f"right_obs_{self.count:03d}.png"), color)
        record = {"view": view.to_dict(), "jaw": jaw.to_dict(), "error": error.to_dict(),
                  "right_pose": state.get("pose")}
        (self.directory / f"right_obs_{self.count:03d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        # ``ok`` = a valid measurement (usable for servoing); ``safe_to_close`` = the close gates also pass.
        result = {"ok": bool(error.ok), "safe_to_close": bool(error.ok and error.gates_pass),
                  "reason": error.reason, "confidence": error.confidence,
                  "calibration_validity": error.calibration_validity,
                  "evidence": f"right_obs_{self.count:03d}.json"}
        if error.ok:
            pose = state.get("pose")
            if pose is None:
                raise ValueError("observe needs the right pose to express the error for a pitched tool")
            result["error_tool_mm"] = servo_error([error.vertical_mm, error.lateral_mm, error.range_mm], pose[3:])
            result["raw_error_tool_mm"] = [error.vertical_mm, error.lateral_mm, error.range_mm]
        return result

    def transfer(self, before: Any, after: Any) -> dict[str, Any]:
        """Bottle static in the image while the background shifts => it moved with the right hand.

        The pads are static in this camera as well, so they are masked out of the background.
        """
        exclude = None
        depth = after.get("depth") if isinstance(after, dict) else None
        if depth is not None:
            from chem_demo.wrist_mount import current_jaw_mask
            try:
                exclude = current_jaw_mask(depth, "right")
            except Exception:  # noqa: BLE001 - fall back to no mask; the verdict then needs more background shift
                exclude = None
        hints = getattr(self, "_hints", None) or {}
        side = {k: hints[k] for k in ("down_px", "expected_px") if hints.get(k) is not None}
        verdict = transfer_check(before, after, self.K, segmenter=self._segmenter, exclude_mask=exclude,
                                 view_kwargs=side)
        return verdict.to_dict()
