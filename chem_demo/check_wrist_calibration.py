"""Replay saved open-jaw left D405 observations, without opening any devices.

Test a changing camera-to-gripper transform. Register the near jaw features to
the reference jaws (3D to 2D), then evaluate *different*, distant static scene
features in robot coordinates. Each evaluated pose is excluded from the mount
fit. Output is a diagnostic report, never an active robot calibration.

    python -m chem_demo.check_wrist_calibration <left-capture-folder>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chem_demo.run_wrist_calib import ACCEPT, MOVES
from chem_demo.wrist_landmarks import measure_open_pad_tips
from chem_demo.wrist_geometry import (
    PairObservation, backproject, fit_plane, pose_matrix, sample_depth, solve_hand_eye, transform,
)


def features(color, depth, jaws):
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    mask = None
    if jaws:
        # Open jaws occupy the upper left/right parts of this camera's image.
        h, w = depth.shape
        mask = cv2.dilate(((depth > 0) & (depth < 175)).astype(np.uint8), np.ones((13, 13), np.uint8))
        mask[:, round(0.38 * w):round(0.61 * w)] = 0
        mask[round(0.6 * h):] = 0
    keys, descriptors = cv2.SIFT_create(
        nfeatures=5000, contrastThreshold=0.005, edgeThreshold=15,
    ).detectAndCompute(gray, mask)
    if descriptors is None or len(descriptors) < 2:
        raise ValueError("insufficient image features")
    return np.float32([key.pt for key in keys]), descriptors


def matches(first, second):
    a, da = first
    b, db = second
    matcher = cv2.BFMatcher()
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
    kept = [m for m, n in matcher.knnMatch(da, db, k=2)
            if m.distance < 0.8 * n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    pa = a[[m.queryIdx for m in kept]]
    pb = b[[m.trainIdx for m in kept]]
    _, distinct = np.unique(np.round(pa), axis=0, return_index=True)
    return pa[distinct], pb[distinct]


def current_to_reference_jaws(first, current, depth, K):
    pa, pb = matches(first, current)
    z = sample_depth(depth, pa, window=1)
    keep = (z > 45) & (z < 175)
    xyz = backproject(K, pa[keep], z[keep])
    pixels = pb[keep].astype(float)
    if len(xyz) < 8:
        raise ValueError("insufficient reference jaw depths")
    ok, rot, t, inliers = cv2.solvePnPRansac(
        xyz, pixels, K, None, iterationsCount=500, reprojectionError=1.8, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 8:
        raise ValueError("jaw registration is ambiguous")
    idx = inliers.ravel()
    left = int((pa[keep][idx, 0] < depth.shape[1] / 2).sum())
    if min(left, len(idx) - left) < 2:
        raise ValueError("jaw registration lacks support on both sides")
    rot, t = cv2.solvePnPRefineLM(xyz[idx], pixels[idx], K, None, rot, t)
    current_from_ref = np.eye(4)
    current_from_ref[:3, :3] = cv2.Rodrigues(rot)[0]
    current_from_ref[:3, 3] = t.ravel()
    return np.linalg.inv(current_from_ref), len(idx)


def scene_plane(K, depth):
    vs, us = np.mgrid[:depth.shape[0]:8, :depth.shape[1]:8]
    z = depth[vs, us]
    keep = z > 200
    if keep.sum() < 50:
        raise ValueError("table plane is not visible")
    cloud = backproject(K, np.column_stack((us[keep], vs[keep])), z[keep])
    _, _, inliers = fit_plane(cloud, tolerance=3.0)
    if inliers.sum() < 50:
        raise ValueError("insufficient table plane support")
    return cloud[inliers][::max(1, int(inliers.sum()) // 600)]


def observation(folder, label, group=None):
    manifest = folder / "acquisition.json"
    if group is None and manifest.is_file():
        group = json.loads(manifest.read_text(encoding="utf-8")).get("pose_groups", {}).get(label)
    if group is not None:
        captures = group.get("captures") or []
        color_name = captures[0]["files"]["color_png"] if captures else None
        depth_name = group.get("representative_depth_file")
        if any(not isinstance(name, str) or Path(name).name != name for name in (color_name, depth_name)):
            raise ValueError(f"{label}: capture manifest lacks a complete local RGB-D pair")
        color_path, depth_path = folder / color_name, folder / depth_name
    else:
        color_path = folder / f"{label}.png"
        depth_path = folder / f"{label}_depth_temporal_median_mm.npy"
        if not depth_path.exists():
            depth_path = folder / f"{label}_depth.png"
    color = cv2.imread(str(color_path))
    depth = (np.load(depth_path, allow_pickle=False) if depth_path.suffix == ".npy" else
             cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED))
    if (color is None or depth is None or depth.ndim != 2 or color.shape[:2] != depth.shape
            or not np.all(np.isfinite(depth)) or np.any(depth < 0)):
        raise ValueError(f"{label}: missing or inconsistent RGB-D pair")
    return color, depth


def scene_pair(first_scene, scene, ref_depth, depth, K, correction, reference, pose, label):
    pa, pb = matches(first_scene, scene)
    da, db = sample_depth(ref_depth, pa), sample_depth(depth, pb)
    keep = (da > 210) & (db > 200)
    if keep.sum() < ACCEPT["min_points"]:
        raise ValueError(f"{label}: too few scene correspondences")
    return PairObservation(reference, pose, backproject(K, pa[keep], da[keep]),
                           transform(correction, backproject(K, pb[keep], db[keep])), label)


def pair_error(pair, mount):
    return np.linalg.norm(transform(pair.T0 @ mount, pair.x0) -
                          transform(pair.Ti @ mount, pair.xi), axis=1)


def replay(folder: Path, K: np.ndarray, table_z: float) -> dict:
    required = [move[0] for move in MOVES]
    poses = json.loads((folder / "poses.json").read_text(encoding="utf-8"))
    if any(label not in poses for label in required):
        raise ValueError("incomplete nine-pose sequence")
    reference = pose_matrix(poses["ref"])
    pairs, rows, planes = [], {}, {}
    tips = {}
    for label in required:
        color, depth = observation(folder, label)
        pads = measure_open_pad_tips(color)
        if not pads.ok:
            raise ValueError(f"{label}: open pads are not observable ({pads.reason})")
        tips[label] = {"left": pads.left, "right": pads.right}
        jaw, scene = features(color, depth, True), features(color, depth, False)
        plane = scene_plane(K, depth)
        if label == "ref":
            ref_depth, ref_jaw, ref_scene = depth, jaw, scene
            planes[label] = plane
            continue
        correction, support = current_to_reference_jaws(ref_jaw, jaw, ref_depth, K)
        pair = scene_pair(ref_scene, scene, ref_depth, depth, K, correction,
                          reference, pose_matrix(poses[label]), label)
        pairs.append(pair)
        planes[label] = transform(correction, plane)
        rows[label] = {"jaw_inliers": support, "scene_matches": len(pair.x0)}

    full = solve_hand_eye(pairs, reference, planes["ref"], table_z)
    for i, pair in enumerate(pairs):
        fit = solve_hand_eye(pairs[:i] + pairs[i+1:], reference, planes["ref"], table_z)
        T = fit["T_flange_camera"]
        error = pair_error(pair, T)
        rows[pair.label].update(held_out_median_mm=float(np.median(error)),
                                held_out_p90_mm=float(np.percentile(error, 90)))
    manifest_path = folder / "acquisition.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("capture_problems") or manifest["status"] not in ("acquired", "complete"):
            raise ValueError(f"incomplete or unstable acquisition: {manifest['status']}, {manifest.get('capture_problems')}")
        if manifest["capture_burst"] > 1:
            label = "ref_return"
            poses[label] = manifest["pose_groups"][label]["captures"][0]["robot_before"]["tcp_pose_mm_deg"]
            color, depth = observation(folder, label)
            pads = measure_open_pad_tips(color)
            if not pads.ok:
                raise ValueError(f"{label}: {pads.reason}")
            tips[label] = {"left": pads.left, "right": pads.right}
            corr, count = current_to_reference_jaws(ref_jaw, features(color, depth, True), ref_depth, K)
            pair = scene_pair(ref_scene, features(color, depth, False), ref_depth, depth,
                              K, corr, reference, pose_matrix(poses[label]), label)
            err = pair_error(pair, full["T_flange_camera"])
            rows[label] = {"jaw_inliers": count, "scene_matches": len(err), "held_out_median_mm": float(np.median(err)),
                           "held_out_p90_mm": float(np.percentile(err, 90))}
            planes[label] = transform(corr, scene_plane(K, depth))
    heights = {label: float(np.median(transform(pose_matrix(poses[label]) @ full["T_flange_camera"], plane)[:, 2]))
               for label, plane in planes.items()}
    worst = max(row["held_out_median_mm"] for row in rows.values())
    spread = max(heights.values()) - min(heights.values())
    return {"capture": folder.name, "model": "open jaws; per-view PnP and reference hand-eye",
            "purpose": "offline diagnostic only", "poses": rows, "open_pad_tips_px": tips,
            "full_fit_pairs": full["pairs"],
            "worst_held_out_median_mm": worst, "plane_heights_mm": heights,
            "plane_height_spread_mm": spread, "thresholds": dict(ACCEPT),
            "held_out_median_and_height_within_limits": bool(
                worst <= ACCEPT["median_mm"] and spread <= ACCEPT["height_spread_mm"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--metadata", type=Path, help="matching original run record with K, arm and table_z_reference")
    args = parser.parse_args()
    if not args.folder.is_dir():
        parser.error("capture folder does not exist")
    cv2.setRNGSeed(0)
    try:
        manifest = args.folder / "acquisition.json"
        if args.metadata is None and manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            K = metadata["camera"]["intrinsics"]["matrix_px"]
            table_z = metadata["table_z_reference_mm"]
        else:
            source = args.metadata or args.folder / "result_rejected.json"
            metadata = json.loads(source.read_text(encoding="utf-8"))
            if Path(metadata["frames"]).resolve() != args.folder.resolve():
                raise ValueError("metadata belongs to a different capture")
            K = metadata["K"]
            table_z = metadata["table_z_reference"]
        if metadata["arm"] != "left":
            raise ValueError("this open-pad geometry requires the left wrist")
        result = replay(args.folder, np.array(K, float), table_z)
    except (KeyError, OSError, ValueError, RuntimeError, cv2.error) as exc:
        result = {"capture": args.folder.name, "purpose": "offline diagnostic only", "error": str(exc)}
    path = args.folder / "open_jaw_replay.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(path)
    print(json.dumps({k: v for k, v in result.items() if k not in {"poses", "full_fit_pairs", "plane_heights_mm"}}))
    return 0 if result.get("held_out_median_and_height_within_limits") else 1


if __name__ == "__main__":
    raise SystemExit(main())
