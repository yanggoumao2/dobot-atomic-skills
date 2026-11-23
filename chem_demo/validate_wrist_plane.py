"""Replay a fixed metric checkerboard with independently measured, plane-constrained stereo.

Usage: python -m chem_demo.validate_wrist_plane CAPTURE --seed PRIOR_JSON --seed-reference RGBD_FOLDER
The prior camera mount only identifies repeated grid sites. It is not the fitted hand-eye transform.
The plane correction is specific to the measured checker corners and is never installed as a global depth correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from chem_demo.check_wrist_board import detect_checkerboard
from chem_demo.check_wrist_calibration import observation
from chem_demo.run_wrist_calib import ACCEPT, MOVES
from chem_demo.wrist_geometry import PairObservation, backproject, pose_matrix, sample_depth, solve_hand_eye, transform
from chem_demo.wrist_mount import WristMountReference, current_jaw_mask, refine_jaw_motion


@dataclass
class View:
    ids: np.ndarray
    rays: np.ndarray
    depth: np.ndarray
    metric_xyz: np.ndarray
    flange: np.ndarray
    jaw: np.ndarray


def robust_linear(design, values):
    weights = np.ones(len(values))
    for _ in range(8):
        coeff, _, rank, _ = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=1e-8)
        if rank < 3:
            raise ValueError("board points do not span a plane")
        residual = design @ coeff - values
        scale = max(0.2, 1.4826 * np.median(np.abs(residual)))
        weights = (1.0 + (residual / scale) ** 2) ** -0.25
    return coeff


def fit_disparity(views):
    # Inverse depth is affine in normalized image coordinates for a pinhole view of a plane.
    design = np.vstack([v.rays for v in views])
    observed_bias = np.concatenate([100000 / v.metric_xyz[:, 2] - 100000 / v.depth for v in views])
    return robust_linear(design, observed_bias)


def measured_board_points(view, coeff):
    # The plane fit excludes the evaluated corner. The reference is shared when training the disparity model.
    disparity = 100000 / view.depth + view.rays @ coeff
    indices = np.arange(len(disparity))
    depth = np.empty(len(indices))
    for held in np.array_split(np.random.default_rng(187).permutation(indices), 4):
        train = np.setdiff1d(indices, held)
        plane = robust_linear(view.rays[train], disparity[train])
        inverse_depth = view.rays[held] @ plane
        if np.any(inverse_depth <= 0):
            raise ValueError("fitted plane has nonpositive disparity")
        depth[held] = 100000 / inverse_depth
    return view.rays * depth[:, None]


def register(reference, ref_color, ref_depth, color, depth, K, arm="left"):
    seed = reference.measure(color, depth)
    if not seed.ok:
        raise ValueError(seed.reason)
    return refine_jaw_motion(ref_color, ref_depth, color, depth, K, seed.T_reference_current, arm=arm)


def collect(folder, seed, old_reference, cols, rows, square):
    data = json.loads((folder / "acquisition.json").read_text(encoding="utf-8"))
    labels = [move[0] for move in MOVES] + ["ref_return"]
    groups = data["pose_groups"]
    arm = data["arm"]
    if (arm not in ("left", "right") or data["status"] != "complete" or data["capture_problems"] or
            data["capture_burst"] < 2 or set(groups) != set(labels) or not data["return_home_drift"]["stable"]):
        raise ValueError("expected a complete, stable wrist burst and independent return")
    K = np.array(data["camera"]["intrinsics"]["matrix_px"])
    ref_color, ref_depth = observation(folder, "ref")
    if arm == "left":
        if old_reference is None or seed is None:
            raise ValueError("left capture requires its same-camera prior and reference")
        prior_data = json.loads((old_reference / "acquisition.json").read_text(encoding="utf-8"))
        if (data["camera"]["serial"] != prior_data["camera"]["serial"] or
                not np.allclose(K, prior_data["camera"]["intrinsics"]["matrix_px"])):
            raise ValueError("seed camera/intrinsics do not match")
        old_color, old_depth = observation(old_reference, "ref")
        prior = WristMountReference.from_frame(old_color, old_depth, K)
        seed = seed @ register(prior, old_color, old_depth, ref_color, ref_depth, K)
    else:
        from chem_demo.wrist_runtime import camera_serial
        if data["camera"]["serial"] != camera_serial("right_wrist"):
            raise ValueError("capture serial does not match the right wrist")
        if seed is not None or old_reference is not None:
            raise ValueError("right wrist uses its own observed board features")
    reference = WristMountReference.from_frame(ref_color, ref_depth, K, arm=arm)
    views = {}
    for label in labels:
        group = groups[label]
        if not group["pose_stability"]["stable"] or len(group["captures"]) != data["capture_burst"]:
            raise ValueError(f"{label}: unstable or incomplete burst")
        if arm == "right" and group.get("gripper_position", 0) < 250:
            raise ValueError(f"{label}: fully open right gripper unverified")
        color_file = group["captures"][0]["files"]["color_png"]
        stem = Path(color_file).stem
        median_file = group["representative_depth_file"]
        if (color_file != f"{stem}.png" or median_file != f"{stem}_depth_temporal_median_mm.npy"
                or not (folder / median_file).is_file()):
            raise ValueError(f"{label}: inconsistent burst file names or missing temporal median")
        color, depth = observation(folder, label, group)
        base_detection = detect_checkerboard(color, cols, rows, square, K=K)
        board = base_detection
        if arm == "right":
            from chem_demo.wrist_checker_sites import augment_right_capture
            jaw_mask = current_jaw_mask(depth, arm="right")
            board = augment_right_capture(color, depth, K, board, exclude_mask=jaw_mask,
                                          cols=cols, rows=rows, square_mm=square)
        if not board.get("handeye_support", {}).get("passed"):
            raise ValueError(f"{label}: insufficient observable checker corners")
        if board["metric_pnp"]["status"] != "PASS":
            raise ValueError(f"{label}: metric board reprojection check failed")
        pixels = np.asarray(board["used_corner_pixels"])
        grid = np.asarray(board["used_object_points_local_mm"])
        pose = board["metric_pnp"]
        local = np.eye(4)
        local[:3, :3] = cv2.Rodrigues(np.array(pose["rvec_rad_local_to_camera"]))[0]
        local[:3, 3] = pose["tvec_mm_local_origin_in_camera"]
        xyz = transform(local, grid)
        flange = pose_matrix(group["captures"][0]["robot_before"]["tcp_pose_mm_deg"])
        if label == "ref":
            ref_chart = local
            ref_detection = base_detection
            jaw = np.eye(4)
        else:
            jaw = register(reference, ref_color, ref_depth, color, depth, K, arm=arm)
        if label == "ref":
            ids = np.rint(grid[:, :2] / square).astype(int)
        elif arm == "right":
            from chem_demo.right_wrist_grid import register_augmented_checkerboard_sites
            match = register_augmented_checkerboard_sites(
                ref_color, ref_depth, color, depth, K, reference_base_detection=ref_detection,
                current_base_detection=base_detection, current_augmented_detection=board,
                reference_exclude_mask=current_jaw_mask(ref_depth, arm="right"),
                current_exclude_mask=jaw_mask, cols=cols, rows=rows, square_mm=square)
            ids = np.asarray(match.get("current_site_ids_in_reference", []))
            if (not match.get("accepted") or ids.shape != (len(grid), 2) or
                    not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids, axis=0)) != len(ids)):
                raise ValueError(f"{label}: unresolved right checker grid: {match.get('reason')}")
        else:
            ref_world = views["ref"].flange @ seed @ ref_chart
            coords = transform(np.linalg.inv(ref_world) @ flange @ seed @ jaw, xyz)
            ids = np.rint(coords[:, :2] / square).astype(int)
            if (len(np.unique(ids, axis=0)) != len(ids) or
                    np.max(np.linalg.norm(coords[:, :2] - square * ids, axis=1)) >= 0.3 * square):
                raise ValueError(f"{label}: prior cannot disambiguate physical grid sites")
        z = sample_depth(depth, pixels, window=2)
        ok = z > 185  # exclude missing depths and foreground fingers
        if ok.sum() < ACCEPT["min_points"]:
            raise ValueError(f"{label}: missing checker stereo support")
        views[label] = View(ids[ok], backproject(K, pixels[ok], np.ones(ok.sum())), z[ok],
                            xyz[ok], flange, jaw)
    return data, views, ref_chart


def evaluate(views, ref_chart, square, table_z):
    ref = views["ref"]
    moved = [label for label in views if label not in ("ref", "ref_return")]
    pairs = {}
    nominal = {}
    for label, v in views.items():
        grid = np.column_stack((v.ids * square, np.zeros(len(v.ids))))
        nominal[label] = transform(ref_chart, grid)
        pairs[label] = PairObservation(ref.flange, v.flange, nominal[label], transform(v.jaw, v.metric_xyz), label)
    scores = {}
    for label in [*moved, "ref_return"]:
        others = [key for key in moved if key != label]
        mount = solve_hand_eye([pairs[key] for key in others], ref.flange, ref.metric_xyz, table_z)["T_flange_camera"]
        coeff = fit_disparity([ref, *(views[key] for key in others)])
        v = views[label]
        world = v.flange @ mount @ v.jaw
        current = transform(world, measured_board_points(v, coeff))
        reference = transform(ref.flange @ mount, measured_board_points(ref, coeff))
        expected = transform(ref.flange @ mount, nominal[label])
        optical = transform(world, v.metric_xyz)
        raw = transform(world, v.rays * v.depth[:, None])
        common = {tuple(site): i for i, site in enumerate(ref.ids)}
        idx = [(common[tuple(site)], i) for i, site in enumerate(v.ids) if tuple(site) in common]
        if not idx:
            raise ValueError(f"{label}: no corresponding physical grid points")
        iref, icur = np.asarray(idx).T
        distance = np.linalg.norm(current[icur] - reference[iref], axis=1)
        scores[label] = {
            "shared_physical_corners": len(idx), "measured_corners": len(current),
            "stereo_pair_median_mm": float(np.median(distance)), "stereo_pair_p90_mm": float(np.percentile(distance, 90)),
            "board_3d_median_mm": float(np.median(np.linalg.norm(current - expected, axis=1))),
            "optical_3d_median_mm": float(np.median(np.linalg.norm(optical - expected, axis=1))),
            "raw_3d_median_mm": float(np.median(np.linalg.norm(raw - expected, axis=1))),
            "board_height_mm": float(np.median(current[:, 2])), "reference_board_height_mm": float(np.median(reference[:, 2])),
        }
    heights = [s[key] for s in scores.values() for key in ("board_height_mm", "reference_board_height_mm")]
    worst_pair = max(s["stereo_pair_median_mm"] for s in scores.values())
    worst_board = max(s["board_3d_median_mm"] for s in scores.values())
    worst_optical = max(s["optical_3d_median_mm"] for s in scores.values())
    height_spread = float(np.ptp(heights))
    fewest = min(s["shared_physical_corners"] for s in scores.values())
    passed = (max(worst_pair, worst_board, worst_optical) <= ACCEPT["median_mm"] and
              height_spread <= ACCEPT["height_spread_mm"] and fewest >= ACCEPT["min_points"])
    full_coeff = fit_disparity([ref, *(views[key] for key in moved)])
    return {"checker_plane_passed": bool(passed), "poses": scores, "fewest_shared_physical_corners": fewest,
            "worst_stereo_pair_median_mm": worst_pair, "worst_board_3d_median_mm": worst_board,
            "worst_optical_3d_median_mm": worst_optical,
            "held_out_plane_height_spread_mm": height_spread,
            "limits": {k: ACCEPT[k] for k in ("median_mm", "height_spread_mm", "min_points")},
            "disparity_coefficients_xy1": full_coeff.tolist(),
            "formula": "100000 / corrected_Z_mm = 100000 / measured_Z_mm + [normalized_x, normalized_y, 1] dot coefficients"}


def evaluate_multiview(views, ref_chart, square, table_z, original):
    """Cross-check every sufficiently overlapping pair. Select the anchor using visibility alone."""
    ref = views["ref"]
    moved = [key for key in views if key not in ("ref", "ref_return")]
    indices = {k: {tuple(site): i for i, site in enumerate(v.ids)} for k, v in views.items()}
    minimum = ACCEPT["min_points"]
    observations = {}
    for key in moved:
        view = views[key]
        grid = np.column_stack((view.ids * square, np.zeros(len(view.ids))))
        observations[key] = PairObservation(ref.flange, view.flange, transform(ref_chart, grid),
                                            transform(view.jaw, view.metric_xyz), key)
    coverage = {key: min(len(indices[key].keys() & indices[other].keys()) for other in views if other != key)
                for key in views if key != "ref_return"}
    anchor = max(coverage, key=coverage.get)
    records = []
    for first, second in combinations(views, 2):
        common = sorted(indices[first].keys() & indices[second].keys())
        row = {"first": first, "second": second, "common": len(common)}
        records.append(row)
        if len(common) < minimum:
            continue
        train = [key for key in moved if key not in (first, second)]
        mount = solve_hand_eye([observations[k] for k in train], ref.flange, ref.metric_xyz, table_z)["T_flange_camera"]
        coeff = fit_disparity([ref, *(views[k] for k in train)])
        endpoints = []
        for key in (first, second):
            view = views[key]
            points = measured_board_points(view, coeff)[[indices[key][site] for site in common]]
            endpoints.append(transform(view.flange @ mount @ view.jaw, points))
        errors = np.linalg.norm(endpoints[0] - endpoints[1], axis=1)
        row.update(median_mm=float(np.median(errors)), p90_mm=float(np.percentile(errors, 90)))
    worst = max((row["median_mm"] for row in records if "median_mm" in row), default=None)
    passed = (coverage[anchor] >= minimum and worst is not None and worst <= ACCEPT["median_mm"] and
              max(original[key] for key in ("worst_stereo_pair_median_mm", "worst_board_3d_median_mm",
                                             "worst_optical_3d_median_mm")) <= ACCEPT["median_mm"] and
              original["held_out_plane_height_spread_mm"] <= ACCEPT["height_spread_mm"])
    return {**original, "original_reference_check_passed": original["checker_plane_passed"],
            "original_reference_min_shared": original["fewest_shared_physical_corners"],
            "original_reference_worst_pair_median_mm": original["worst_stereo_pair_median_mm"],
            "fewest_shared_physical_corners": coverage[anchor], "worst_stereo_pair_median_mm": worst,
            "poses_scored_against": "ref",
            "checker_plane_passed": passed, "validation_anchor": anchor,
            "anchor_selection": "maximise minimum jointly measured sites over all other poses, before scoring errors",
            "minimum_shared_by_candidate_anchor": coverage,
            "minimum_shared_with_validation_anchor": coverage[anchor], "pairwise_scores": records,
            "worst_pairwise_holdout_median_mm": worst,
            "training_rule": "exclude both endpoints from moved training poses; fixed ref supplies chart and height datum; ref_return never trains"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--seed-reference", type=Path)
    parser.add_argument("--cols", type=int, default=11)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--square-mm", type=float, default=30)
    parser.add_argument("--multiview", action="store_true", help="also validate every sufficiently overlapping pair")
    args = parser.parse_args()
    seed = (np.array(json.loads(args.seed.read_text(encoding="utf-8"))["seed_T_flange_reference_camera"])
            if args.seed else None)
    data, views, ref_chart = collect(args.capture, seed, args.seed_reference, args.cols, args.rows, args.square_mm)
    result = evaluate(views, ref_chart, args.square_mm, data["table_z_reference_mm"])
    if args.multiview:
        if data["arm"] != "right":
            parser.error("multiview validation is currently supported for the right wrist")
        result = evaluate_multiview(views, ref_chart, args.square_mm, data["table_z_reference_mm"], result)
    result.update(capture=args.capture.name, serial=data["camera"]["serial"], square_mm=args.square_mm,
                  arm=data["arm"], scope=f"open {data['arm']} jaws; fixed flat checker in tested region; calibrated-board plane stereo only",
                  height_datum="board surface at nominal table Z; absolute fingertip contact height not surveyed",
                  seed_reference=args.seed_reference.name if args.seed_reference else None,
                  seed_path=str(args.seed) if args.seed else None)
    if result["checker_plane_passed"]:
        ref = views["ref"]
        training = []
        for label, v in views.items():
            if label not in ("ref", "ref_return"):
                grid = np.column_stack((v.ids * args.square_mm, np.zeros(len(v.ids))))
                training.append(PairObservation(ref.flange, v.flange, transform(ref_chart, grid),
                                                transform(v.jaw, v.metric_xyz), label))
        fitted = solve_hand_eye(training, ref.flange, ref.metric_xyz, data["table_z_reference_mm"])
        model_type = ("right_open_wrist_checker_multiview_v1" if args.multiview else
                      f"{data['arm']}_open_wrist_checker_plane_v1")
        result.update(type=model_type, K=data["camera"]["intrinsics"]["matrix_px"],
                      T_flange_reference_camera=fitted["T_flange_camera"].tolist(),
                      T_reference_camera_local_board=ref_chart.tolist(),
                      tcp_poses_mm_deg={label: group["captures"][0]["robot_before"]["tcp_pose_mm_deg"]
                                        for label, group in data["pose_groups"].items()})
        result["reference_files"] = {
            role: {"name": name, "sha256": hashlib.sha256((args.capture / name).read_bytes()).hexdigest()}
            for role, name in (("color", "ref.png"), ("depth_mm", "ref_depth_temporal_median_mm.npy"))}
    path = args.capture / ("checker_multiview_validation.json" if args.multiview else "checker_plane_validation.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(path, result["checker_plane_passed"], result["worst_stereo_pair_median_mm"],
          result["held_out_plane_height_spread_mm"], result["fewest_shared_physical_corners"])
    if args.multiview:
        print(f"Validation anchor {result['validation_anchor']}: minimum "
              f"{result['minimum_shared_with_validation_anchor']} shared; original ref minimum "
              f"{result['original_reference_min_shared']}.")
    return 0 if result["checker_plane_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
