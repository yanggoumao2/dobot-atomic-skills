"""Re-estimate the top camera's pose after it was bumped, without a calibration board.

Model: the camera rotated on its mount and did not translate. Under pure rotation the old and new
images are related by rays_new = M @ rays_old (undistorted, normalised) for every scene point at
any depth, so SIFT matches on fixed rig parts (base plates, panel holes, rails) between a frame
from the calibrated period and a current frame determine M. RANSAC rejects matches on things that
moved (arms, objects, people). The residual tells whether the no-translation assumption holds.

New pose: T_base_camera_new = T_base_camera_old with R_new = R_old @ M.T (camera centre kept).

    .venv-perception/Scripts/python.exe -X utf8 -m chem_demo.camera_drift --old <png>... --new <png>...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from vision_pick.cup_locate import DEFAULT_SESSION, Camera, load_camera

CORRECTION_PATH = Path(__file__).with_name("camera_correction.json")


def rays(camera: Camera, pixels: np.ndarray) -> np.ndarray:
    normalised = cv2.undistortPoints(pixels.reshape(-1, 1, 2).astype(np.float64), camera.K, camera.D).reshape(-1, 2)
    r = np.column_stack([normalised, np.ones(len(normalised))])
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation M minimising |b - M a| over unit vectors."""
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def reprojection_px(camera: Camera, M: np.ndarray, old_px: np.ndarray, new_px: np.ndarray) -> np.ndarray:
    predicted = (M @ rays(camera, old_px).T).T
    pixels, _ = cv2.projectPoints(predicted.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.K, camera.D)
    return np.linalg.norm(pixels.reshape(-1, 2) - new_px, axis=1)


def match(old: np.ndarray, new: np.ndarray, sift) -> tuple[np.ndarray, np.ndarray]:
    ko, do = sift.detectAndCompute(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), None)
    kn, dn = sift.detectAndCompute(cv2.cvtColor(new, cv2.COLOR_BGR2GRAY), None)
    pairs = [a for a, b in cv2.BFMatcher().knnMatch(do, dn, k=2) if a.distance < 0.75 * b.distance]
    return (np.float64([ko[p.queryIdx].pt for p in pairs]), np.float64([kn[p.trainIdx].pt for p in pairs]))


def estimate(camera: Camera, old_px: np.ndarray, new_px: np.ndarray, threshold_px: float = 2.0,
             iterations: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ro, rn = rays(camera, old_px), rays(camera, new_px)
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(ro), bool)
    for _ in range(iterations):
        pick = rng.choice(len(ro), 2, replace=False)
        M = kabsch(ro[pick], rn[pick])
        inliers = reprojection_px(camera, M, old_px, new_px) < threshold_px
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    M = kabsch(ro[best_inliers], rn[best_inliers])
    for _ in range(3):  # refit on the consensus set
        best_inliers = reprojection_px(camera, M, old_px, new_px) < threshold_px
        M = kabsch(ro[best_inliers], rn[best_inliers])
    return M, best_inliers


def corrected(camera: Camera, M: np.ndarray) -> Camera:
    T = camera.T_base_camera.copy()
    T[:3, :3] = T[:3, :3] @ M.T
    return Camera(camera.K, camera.D, T)


def load_corrected_camera(session: Path, arm: str) -> Camera:
    """Calibrated camera with the stored drift correction applied (if any)."""
    camera = load_camera(session, arm)
    if CORRECTION_PATH.exists():
        data = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
        camera = corrected(camera, np.array(data["M"], float))
    return camera


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="无标定板估计顶部相机碰歪后的旋转")
    parser.add_argument("--old", type=Path, nargs="+", required=True, help="标定有效期内拍的顶部图（已旋转 180°）")
    parser.add_argument("--new", type=Path, nargs="+", required=True, help="当前顶部图")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--save", action="store_true", help=f"写入 {CORRECTION_PATH.name}")
    parser.add_argument("--vis", type=Path, help="内点可视化输出路径")
    args = parser.parse_args(argv)
    camera = load_camera(args.session, "right")
    sift = cv2.SIFT_create(10000)
    old_all, new_all, sources = [], [], []
    for o in args.old:
        for n in args.new:
            a, b = match(cv2.imread(str(o)), cv2.imread(str(n)), sift)
            old_all.append(a); new_all.append(b); sources += [(o.name, n.name)] * len(a)
    old_px, new_px = np.vstack(old_all), np.vstack(new_all)
    M, inliers = estimate(camera, old_px, new_px)
    errors = reprojection_px(camera, M, old_px, new_px)[inliers]
    angle = float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))
    shift = old_px[inliers] - new_px[inliers]
    result = {"M": M.tolist(), "angle_deg": round(angle, 4), "matches": int(len(old_px)), "inliers": int(inliers.sum()),
              "rms_px": round(float(np.sqrt(np.mean(errors ** 2))), 3), "max_px": round(float(errors.max()), 3),
              "median_image_shift_px": np.median(new_px[inliers] - old_px[inliers], 0).round(2).tolist(),
              "old": [str(p) for p in args.old], "new": [str(p) for p in args.new], "session": str(args.session),
              "model": "pure rotation about the camera centre; R_new = R_old @ M.T",
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps({k: v for k, v in result.items() if k not in ("old", "new")}, ensure_ascii=False, indent=2))
    if args.vis:
        image = cv2.imread(str(args.new[0]))
        for a, b in zip(old_px[inliers], new_px[inliers]):
            cv2.arrowedLine(image, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 255), 1, tipLength=0.3)
            cv2.circle(image, tuple(map(int, b)), 2, (0, 255, 0), -1)
        cv2.imwrite(str(args.vis), image)
    if args.save:
        CORRECTION_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {CORRECTION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
