"""Fixed camera + stationary 12x9-square board + independently taught TCP points.

T_A_B maps B to A. Internal translations are metres. No robot motion is sent.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
if __package__:
    from . import calibrate as c
else:  # preserve the original standalone CLI
    import calibrate as c

BOARD = {"cols": 11, "rows": 8, "square_m": 0.03}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply(T, points):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def rigid_fit(source, target):
    """Kabsch rigid fit: planar points allowed, collinear points rejected; no scale fit."""
    a, b = c.finite(source), c.finite(target)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("Need matching Nx3 point arrays")
    ac, bc = a - a.mean(0), b - b.mean(0)
    for cloud in (ac, bc):
        s = np.linalg.svd(cloud, compute_uv=False)
        if s[0] < 1e-9 or s[1] / s[0] < 0.05:
            raise ValueError("Points are collinear or too narrow; cover the board in both directions")
    U, _, Vt = np.linalg.svd(ac.T @ bc)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    return c.validate_transform(c.transform(R, b.mean(0) - R @ a.mean(0)))


def pixel_to_plane(pixel, cam, K, D, T_base_camera, T_base_board, board_z=0):
    """Intersect an undistorted ray with board-coordinate plane z=board_z."""
    u, v, board_z = c.finite([*pixel, board_z])
    w, h = cam["image_size_wh"]
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("Pixel outside original calibrated image")
    X, B = c.validate_transform(T_base_camera), c.validate_transform(T_base_board)
    xy = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, D).reshape(2)
    origin, ray = X[:3, 3], X[:3, :3] @ np.r_[xy, 1.]
    normal, p0 = B[:3, 2], apply(B, np.array([0., 0., board_z]))
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-8:
        raise ValueError("Viewing ray is parallel to the plane")
    distance = float(normal @ (p0 - origin) / denominator)
    if distance <= 0:
        raise ValueError("Plane intersection is behind the camera")
    point = origin + distance * ray
    return point, apply(c.inverse(B), point)


def detect_command(args):
    board = {"cols": args.cols, "rows": args.rows, "square_m": args.square_m}
    obj = c.board_points(**board)
    image, size, corners = c.detect(args.image, args.cols, args.rows)
    corners = c.ordered(corners, args.corner_order)
    out = c.output_dir(args.out)
    c.save_image(out / "source.png", c.read_image(args.image))
    for i, (u, v) in enumerate(corners.reshape(-1, 2)):
        pos = (round(float(u)), round(float(v)))
        cv2.circle(image, pos, 3, (0, 0, 255), -1)
        label_pos = (pos[0] + 3, pos[1] - 4)
        cv2.putText(image, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 3)
        cv2.putText(image, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 0, 0), 1)
    c.save_image(out / "numbered_corners.png", image)
    record = {**c.base_result("board_observation"), "board": board,
              "image_size_wh": list(size), "source_image": "source.png",
              "source_sha256": digest(out / "source.png"), "corner_order": args.corner_order,
              "corners_uv": corners.reshape(-1, 2).tolist()}
    c.write_json(out / "observation.json", record)
    with (out / "corners.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["corner_id", "row", "col", "board_x_m", "board_y_m", "u_px", "v_px"])
        for i, (xyz, uv) in enumerate(zip(obj, corners.reshape(-1, 2))):
            writer.writerow([i, i // args.cols, i % args.cols, *xyz[:2], *uv])
    # Empty coordinates intentionally: no fabricated robot measurements.
    with (out / "teach_points.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["corner_id", "split", "x", "y", "z"])
    print(f"Detected {len(obj)} corners. Use numbered_corners.png to identify physical TCP targets: {out}")
    return 0


def load_observation(path, cam):
    obs = c.load_json(path)
    if obs.get("kind") != "board_observation" or obs["board"] != cam["board"]:
        raise ValueError("Board definition differs from intrinsics")
    if obs["image_size_wh"] != cam["image_size_wh"]:
        raise ValueError("Observation size differs from intrinsics")
    source = Path(path).resolve().parent / obs["source_image"]
    if digest(source) != obs["source_sha256"]:
        raise ValueError("Reference image has changed; recapture and reteach points")
    obj = c.board_points(**obs["board"])
    uv = c.finite(obs["corners_uv"])
    if uv.shape != (len(obj), 2):
        raise ValueError("Unexpected corner array")
    return obs, obj, uv.astype(np.float32).reshape(-1, 1, 2), source


def taught_points(path, n, unit):
    rows = c.read_rows(path, ["corner_id", "split", "x", "y", "z"])
    ids = [int(row["corner_id"]) for row in rows]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= n for i in ids):
        raise ValueError("corner_id must be unique and within the detected board")
    if any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("split must be train or val")
    c.counts(rows, min_train=6, min_val=3)
    xyz = c.finite([[r[k] for k in ("x", "y", "z")] for r in rows])
    return rows, np.array(ids), xyz * (0.001 if unit == "mm" else 1.)


def fit_command(args):
    cam, K, D = c.camera_file(args.intrinsics)
    obs, obj, corners, source = load_observation(args.observation, cam)
    rows, ids, measured = taught_points(args.points, len(obj), args.unit)
    for row in rows:
        for key in ("unit", "robot_id", "base_frame", "tcp_name"):
            if key in row and row[key] != getattr(args, key):
                raise ValueError(f"CSV {key} differs from the requested calibration")
    train = np.array([r["split"] == "train" for r in rows])
    B = rigid_fit(obj[ids[train]], measured[train])
    C, rms = c.pose_from_board(obj, corners, K, D)
    X = c.validate_transform(B @ c.inverse(C))
    reports = []
    for row, i, reference in zip(rows, ids, measured):
        predicted, _ = pixel_to_plane(corners[i, 0], cam, K, D, X, B)
        reports.append({**row, "reference_base_m": reference.tolist(),
                        "predicted_base_m": predicted.tolist(),
                        "rigid_error_mm": float(np.linalg.norm(apply(B, obj[i]) - reference)*1000),
                        "pixel_plane_error_mm": float(np.linalg.norm(predicted-reference)*1000)})
    passed = rms <= args.max_rms_px and all(max(r["rigid_error_mm"], r["pixel_plane_error_mm"])
                                           <= args.max_error_mm for r in reports)
    result = {**c.base_result("fixed_board_extrinsics"), "robot_id": args.robot_id,
              "synthetic_only": bool(cam.get("synthetic_only") or obs.get("synthetic_only")),
              "robot_model": "DOBOT Nova2", "base_frame": args.base_frame, "tcp_name": args.tcp_name,
              "input_unit": args.unit, "camera_id": cam["camera_id"], "board": obs["board"],
              "intrinsics_sha256": digest(args.intrinsics), "observation_sha256": digest(args.observation),
              "teach_points_sha256": digest(args.points), "image_size_wh": cam["image_size_wh"],
              "T_base_board": B.tolist(), "T_camera_board": C.tolist(), "T_base_camera": X.tolist(),
              "board_normal_base": B[:3, 2].tolist(), "pnp_rms_px": rms,
              "thresholds": {"max_rms_px": args.max_rms_px, "max_error_mm": args.max_error_mm},
              "points": reports, "validation_passed": bool(passed),
              "validation_scope": "held-out taught points on this board plane; not independent robot metrology or off-plane accuracy"}
    out = c.output_dir(args.out)
    c.write_json(out / "extrinsics.json", result)
    image = c.read_image(source)
    cv2.drawFrameAxes(image, K, D, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], obs["board"]["square_m"]*2)
    c.save_image(out / "board_axes.png", image)
    val = [r["pixel_plane_error_mm"] for r in reports if r["split"] == "val"]
    print(f"Validation={passed}; PnP RMS={rms:.4f} px; held-out max={max(val):.3f} mm; {out}")
    return 0 if passed else 2


def load_solution(intrinsics_path, extrinsics_path, robot_id):
    cam, K, D = c.camera_file(intrinsics_path)
    ext = c.load_json(extrinsics_path)
    if ext.get("kind") != "fixed_board_extrinsics" or not ext.get("validation_passed"):
        raise ValueError("Extrinsics must pass the supplied validation thresholds")
    if digest(intrinsics_path) != ext["intrinsics_sha256"]:
        raise ValueError("Intrinsics hash mismatch")
    if ext["robot_id"] != robot_id:
        raise ValueError("robot_id mismatch: each Nova2 has a different base frame")
    return cam, K, D, ext


def point_command(args):
    cam, K, D, ext = load_solution(args.intrinsics, args.extrinsics, args.robot_id)
    X, B = c.validate_transform(ext["T_base_camera"]), c.validate_transform(ext["T_base_board"])
    if args.depth_z_m is not None:
        p = c.point_in_base(*args.pixel, args.depth_z_m, cam, K, D,
                            {"mode": "eye-to-hand", "T_base_camera": X})
        bp = apply(c.inverse(B), p)
        mode = "registered_optical_Z_depth"
    else:
        p, bp = pixel_to_plane(args.pixel, cam, K, D, X, B, args.board_z_m)
        mode = "intersection_with_board_coordinate_plane"
    board = ext["board"]
    inside = 0 <= bp[0] <= (board["cols"]-1)*board["square_m"] and 0 <= bp[1] <= (board["rows"]-1)*board["square_m"]
    if not inside and not args.allow_extrapolation:
        raise ValueError("Outside inner-corner rectangle; independently validate there before --allow-extrapolation")
    print(json.dumps({"robot_id": args.robot_id, "base_frame": ext["base_frame"], "mode": mode,
                      "synthetic_only": bool(cam.get("synthetic_only") or ext.get("synthetic_only")),
                      "point_base_m": p.tolist(), "point_base_mm": (p*1000).tolist(),
                      "point_board_m": bp.tolist(), "inside_board_inner_rectangle": bool(inside)}, indent=2))
    return 0


def verify_command(args):
    cam, K, D, ext = load_solution(args.intrinsics, args.extrinsics, args.robot_id)
    rows = c.read_rows(args.csv, ["u_px", "v_px", "board_z_m", "reference_x_m", "reference_y_m", "reference_z_m"])
    if len(rows) < 3:
        raise ValueError("Need at least 3 independent measured points")
    reports = []
    seen = set()
    for row in rows:
        pixel = tuple(c.finite([row["u_px"], row["v_px"]]))
        if pixel in seen:
            raise ValueError("Duplicate verification pixel")
        seen.add(pixel)
        p, _ = pixel_to_plane(pixel, cam, K, D, ext["T_base_camera"], ext["T_base_board"], float(row["board_z_m"]))
        ref = c.finite([row[k] for k in ("reference_x_m", "reference_y_m", "reference_z_m")])
        reports.append({**row, "predicted_base_m": p.tolist(), "error_mm": float(np.linalg.norm(p-ref)*1000)})
    errors = np.array([r["error_mm"] for r in reports])
    result = {**c.base_result("plane_point_validation"), "robot_id": args.robot_id,
              "points": reports, "rms_mm": float(np.sqrt(np.mean(errors**2))), "max_mm": float(max(errors)),
              "validation_passed": bool(max(errors) <= args.max_error_mm), "threshold_mm": args.max_error_mm,
              "scope": "provided independent reference points; reference independence must be ensured by operator"}
    out = c.output_dir(args.out)
    c.write_json(out / "validation.json", result)
    print(f"Independent plane points: {result['validation_passed']}; max={max(errors):.3f} mm")
    return 0 if result["validation_passed"] else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("detect", help="Number the 88 internal corners and freeze their image coordinates")
    a.add_argument("--image", required=True)
    a.add_argument("--cols", type=int, default=11)
    a.add_argument("--rows", type=int, default=8)
    a.add_argument("--square-m", type=c.positive, default=.03)
    a.add_argument("--corner-order", choices=["normal", "reverse"], default="normal")
    a.add_argument("--out", required=True)
    a.set_defaults(func=detect_command)
    a = sub.add_parser("fit", help="Rigid fit of taught board points followed by camera-to-base composition")
    for key in ("intrinsics", "observation", "points", "robot-id", "base-frame", "tcp-name", "out"):
        a.add_argument("--" + key, required=True)
    a.add_argument("--unit", choices=["mm", "m"], required=True)
    a.add_argument("--max-rms-px", type=c.positive, required=True)
    a.add_argument("--max-error-mm", type=c.positive, required=True)
    a.set_defaults(func=fit_command)
    for name, func in (("point", point_command), ("verify", verify_command)):
        a = sub.add_parser(name)
        for key in ("intrinsics", "extrinsics", "robot-id"):
            a.add_argument("--" + key, required=True)
        if name == "point":
            a.add_argument("--pixel", nargs=2, type=float, required=True)
            g = a.add_mutually_exclusive_group(required=True)
            g.add_argument("--board-z-m", type=float, help="Signed z in board frame; 0 is printed board surface")
            g.add_argument("--depth-z-m", type=c.positive, help="Registered RGB optical-axis depth, metres")
            a.add_argument("--allow-extrapolation", action="store_true")
        else:
            a.add_argument("--csv", required=True)
            a.add_argument("--max-error-mm", type=c.positive, required=True)
            a.add_argument("--out", required=True)
        a.set_defaults(func=func)
    return p


def main():
    args = parser().parse_args()
    try:
        return args.func(args)
    except (ValueError, OSError, KeyError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
