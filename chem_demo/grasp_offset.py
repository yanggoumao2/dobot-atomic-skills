"""Where the bottle really sits in the right jaws, from the blue mouth ring in two top frames.

Why (2026-11-23): after a handoff the bottle sat ~20 mm further along the tool axis, ~26 mm higher
and ~12 mm further along the pour than the pad-offset model said, which put the lip 20-30 mm off the
beaker axis at the pre-tilt. One top frame cannot tell "further along the pour" from "higher" (both
move the ring up the image), but the upright side hover and the 60 deg pre-tilt see the same in-hand
offset through different rotations, so the two frames together fix all three components.

Inputs: a run_pour --until pretilt directory (pour_plan.json, result.json, top_side.png,
top_pretilt.png). Output: the correction to ADD to the offsets that run used
(--grasp-shift-mm / --grasp-up-mm / --grasp-pour-mm).

Observations: upright, the whole ring is seen -> its blue centroid; tilted, only the far edge of the
ring shows above the glass -> the blue band's top row and mean column.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from chem_demo.gripper_geometry import FINGERTIP_FROM_FLANGE_MM
from chem_demo.pour_plan import NECK_RADIUS_MM

BLUE_LO, BLUE_HI = (95, 80, 60), (130, 255, 255)
ROI_MARGIN_PX = 50
MIN_BLUE_PX = 30


def _rotm(axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def ring_points(plan: Mapping[str, Any], tool_axis: np.ndarray, delta: np.ndarray,
                fingertip_mm: float = FINGERTIP_FROM_FLANGE_MM) -> tuple[np.ndarray, np.ndarray]:
    """Mouth ring (cw coordinates) at the side hover (upright) and at the pre-tilt, with the bottle
    moved by ``delta`` = (along tool, up, along pour) in the hold frame."""
    d = np.asarray(plan["pour_dir"], float)
    up = np.array([0.0, 0.0, 1.0])
    k = np.cross(up, d)
    shift = delta[0] * tool_axis + delta[1] * up + delta[2] * d
    a = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    cw = np.array([0.0, 0.0, fingertip_mm])

    side = [s for s in plan["steps"] if s["phase"] == "transit"][-1]
    centre = np.asarray(side["lip"], float) - NECK_RADIUS_MM * d + shift
    upright = centre + NECK_RADIUS_MM * (np.outer(np.cos(a), d) + np.outer(np.sin(a), k)) + cw

    theta = float(plan["theta_pre_deg"])
    R = _rotm(k, theta)
    lip = np.asarray(plan["lip_xyz"], float) + R @ shift
    u = R @ d
    centre = lip - NECK_RADIUS_MM * u
    tilted = centre + NECK_RADIUS_MM * (np.outer(np.cos(a), u) + np.outer(np.sin(a), k)) + cw
    return upright, tilted


def blue_pixels(image: np.ndarray, near: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    x0, y0 = (near.min(axis=0) - ROI_MARGIN_PX).astype(int)
    x1, y1 = (near.max(axis=0) + ROI_MARGIN_PX).astype(int)
    roi = np.zeros_like(mask)
    roi[max(y0, 0):y1, max(x0, 0):x1] = 1
    ys, xs = np.nonzero(mask * roi)
    return xs.astype(float), ys.astype(float)


def _features(upright_px: np.ndarray, tilted_px: np.ndarray) -> np.ndarray:
    return np.array([upright_px[:, 0].mean(), upright_px[:, 1].mean(),
                     tilted_px[:, 0].mean(), tilted_px[:, 1].min()])


def solve(plan, tool_axis, camera, side_image, pretilt_image, iterations: int = 6) -> dict[str, Any]:
    """Correction (along tool, up, along pour) in mm that makes the predicted rings match the blue."""
    up0, tilt0 = ring_points(plan, tool_axis, np.zeros(3))
    sx, sy = blue_pixels(side_image, camera.project(up0))
    tx, ty = blue_pixels(pretilt_image, camera.project(tilt0))
    if len(sx) < MIN_BLUE_PX or len(tx) < MIN_BLUE_PX:
        raise RuntimeError(f"蓝色瓶口像素太少（竖直 {len(sx)}，预倾 {len(tx)}）")
    observed = np.array([sx.mean(), sy.mean(), tx.mean(), ty.min()])

    def predict(delta: np.ndarray) -> np.ndarray:
        u, t = ring_points(plan, tool_axis, delta)
        return _features(camera.project(u), camera.project(t))

    delta = np.zeros(3)
    for _ in range(iterations):
        f0 = predict(delta)
        J = np.column_stack([(predict(delta + e) - f0) / 1.0 for e in np.eye(3)])
        step, *_ = np.linalg.lstsq(J, observed - f0, rcond=None)
        delta = delta + step
        if np.linalg.norm(step) < 0.1:
            break
    residual = observed - predict(delta)
    return {"correction_mm": [round(float(v), 1) for v in delta],
            "residual_px": [round(float(v), 1) for v in residual],
            "before_px": [round(float(v), 1) for v in observed - predict(np.zeros(3))],
            "blue_px": [int(len(sx)), int(len(tx))]}


def solve_run(directory: Path) -> dict[str, Any]:
    import cv2

    from chem_demo.camera_drift import load_corrected_camera
    from chem_demo.handoff_plan import _rot
    from vision_pick.cup_locate import DEFAULT_SESSION

    plan = json.loads((directory / "pour_plan.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    tool = _rot(result["start"]["right"][3:])[:, 2].copy()
    tool[2] = 0.0
    tool /= np.linalg.norm(tool)
    camera = load_corrected_camera(DEFAULT_SESSION, "right")
    out = solve(plan, tool, camera, cv2.imread(str(directory / "top_side.png")),
                cv2.imread(str(directory / "top_pretilt.png")))
    used = plan.get("config", {})
    before = [used.get("grasp_along_tool_mm", 0.0), used.get("grasp_up_mm", 0.0), used.get("grasp_along_pour_mm", 0.0)]
    out["used_mm"] = before
    out["new_offsets_mm"] = [round(b + c, 1) for b, c in zip(before, out["correction_mm"])]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="由 --until pretilt 运行目录的两张顶视图求瓶子在右爪中的偏移")
    parser.add_argument("run", type=Path)
    out = solve_run(parser.parse_args(argv).run)
    print(json.dumps(out, ensure_ascii=False))
    t, u, p = out["new_offsets_mm"]
    print(f"建议参数：--grasp-shift-mm {t} --grasp-up-mm {u} --grasp-pour-mm {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
