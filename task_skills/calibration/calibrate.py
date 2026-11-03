"""Offline robot vision calibration. All geometric translations use metres.

T_A_B transforms column coordinates from frame B to frame A.
See README.md for acquisition contracts and validation limitations.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def finite(value, label="value"):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{label}: NaN/Inf is not allowed")
    return a


def positive(value):
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return v


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def output_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def base_result(kind):
    return {"schema_version": 1, "kind": kind,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "length_unit": "metre", "transform_convention": "T_A_B maps B to A"}


def read_rows(path, required):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {', '.join(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def manifest(path, required):
    rows = read_rows(path, required)
    root = Path(path).resolve().parent
    seen = set()
    for row in rows:
        if row["split"] not in ("train", "val"):
            raise ValueError("split must be train or val")
        p = (root / row["image"]).resolve()
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f"Duplicate image content in manifest: {p}")
        seen.add(digest)
        row["image"] = str(p)
        row["sha256"] = digest
    return rows


def board_points(cols, rows, square_m):
    if cols < 3 or rows < 3 or square_m <= 0:
        raise ValueError("Board needs at least 3x3 inner corners and positive square size")
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return points


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_image(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    data.tofile(str(path))


def detect(path, cols, rows, expected_size=None):
    image = read_image(path)
    size = (image.shape[1], image.shape[0])
    if expected_size is not None and tuple(expected_size) != size:
        raise ValueError(f"Image size {size} differs from calibration {expected_size}: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # SB returns subpixel corners, including better support for tilted boards.
    ok, corners = cv2.findChessboardCornersSB(
        gray, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        raise ValueError(f"Complete board not detected: {path}")
    return image, size, corners.astype(np.float32)


def ordered(corners, order):
    if order not in ("normal", "reverse"):
        raise ValueError("corner_order must be normal or reverse")
    return corners.copy() if order == "normal" else corners[::-1].copy()


def transform(R, t):
    T = np.eye(4)
    T[:3, :3] = finite(R).reshape(3, 3)
    T[:3, 3] = finite(t).reshape(3)
    return T


def inverse(T):
    R = T[:3, :3]
    return transform(R.T, -R.T @ T[:3, 3])


def validate_transform(T):
    T = finite(T, "transform")
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("Expected homogeneous 4x4 transform")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError("Invalid rotation matrix")
    return T


def pose(values):
    """x,y,z (m), qx,qy,qz,qw; output T_base_tool."""
    values = finite(values, "pose").reshape(7)
    x, y, z, w = values[3:]
    norm = np.linalg.norm(values[3:])
    if not np.isclose(norm, 1, atol=1e-3):
        raise ValueError(f"Quaternion must have unit norm, got {norm}")
    x, y, z, w = values[3:] / norm
    R = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    return transform(R, values[:3])


def rotation_error(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


def mean_transform(Ts):
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return transform(R, np.mean([T[:3, 3] for T in Ts], axis=0))


def residual(T, ref):
    return {"translation_mm": float(np.linalg.norm(T[:3, 3]-ref[:3, 3])*1000),
            "rotation_deg": rotation_error(ref[:3, :3].T @ T[:3, :3])}


def pose_from_board(obj, corners, K, D):
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP failed")
    R = cv2.Rodrigues(rvec)[0]
    if np.any((R @ obj.T + tvec)[2] <= 0):
        raise ValueError("PnP board points behind camera")
    projected = cv2.projectPoints(obj, rvec, tvec, K, D)[0]
    rms = float(np.sqrt(np.mean(np.sum((projected-corners)**2, axis=2))))
    return transform(R, tvec), rms


def camera_file(path):
    data = load_json(path)
    if data.get("kind") != "intrinsics" or not data.get("validation_passed"):
        raise ValueError("Use an intrinsics file that passed validation")
    K = finite(data["K"])
    D = finite(data["distortion"])
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    return data, K, D


def counts(rows, min_train=8, min_val=3):
    a = sum(r["split"] == "train" for r in rows)
    b = sum(r["split"] == "val" for r in rows)
    if a < min_train or b < min_val:
        raise ValueError(f"Need >= {min_train} train and {min_val} val views; got {a}, {b}")


def intrinsics(args):
    rows = manifest(args.manifest, ["image", "split"])
    counts(rows, 10, 3)
    obj = board_points(args.cols, args.rows, args.square_m)
    out = output_dir(args.out)
    views, size = [], None
    for i, row in enumerate(rows):
        image, size, corners = detect(row["image"], args.cols, args.rows, size)
        views.append(corners)
        cv2.drawChessboardCorners(image, (args.cols, args.rows), corners, True)
        save_image(out / f"corners_{i:03d}.png", image)
    training = [v for r, v in zip(rows, views) if r["split"] == "train"]
    rms, K, D, _, _ = cv2.calibrateCamera([obj]*len(training), training, size, None, None)
    finite(K); finite(D); finite(rms)
    reports = []
    for row, corners in zip(rows, views):
        _, err = pose_from_board(obj, corners, K, D)
        reports.append({**row, "reprojection_rms_px": err})
    passed = all(r["reprojection_rms_px"] <= args.max_rms_px for r in reports)
    data = {**base_result("intrinsics"), "camera_id": args.camera_id,
            "image_size_wh": list(size), "model": "opencv_pinhole_5coeff",
            "K": K.tolist(), "distortion": D.reshape(-1).tolist(),
            "board": {"cols": args.cols, "rows": args.rows, "square_m": args.square_m},
            "train_rms_px": float(rms), "views": reports,
            "max_allowed_view_rms_px": args.max_rms_px, "validation_passed": passed,
            "validation_scope": "held-out image reprojection with per-image PnP; not metric robot accuracy"}
    write_json(out / "intrinsics.json", data)
    print(f"Intrinsic validation: {passed}; train RMS={rms:.4f} px; output={out}")
    return 0 if passed else 2


POSE_KEYS = ["x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"]


def excitation(Ts):
    # Relative rotations must excite at least two independent axes.
    vectors = [cv2.Rodrigues(Ts[i][:3, :3].T @ Ts[j][:3, :3])[0].reshape(3)
               for i in range(len(Ts)) for j in range(i+1, len(Ts))]
    singular = np.linalg.svd(np.asarray(vectors), compute_uv=False)
    if singular[1] < 0.1:
        raise ValueError("Insufficient rotation diversity: rotate around multiple nonparallel axes")
    return singular.tolist()


def solve_handeye(Bs, Cs, mode, method="park"):
    excitation(Bs)
    robot = Bs if mode == "eye-in-hand" else [inverse(B) for B in Bs]
    methods = {"park": cv2.CALIB_HAND_EYE_PARK, "tsai": cv2.CALIB_HAND_EYE_TSAI,
               "horaud": cv2.CALIB_HAND_EYE_HORAUD}
    R, t = cv2.calibrateHandEye([B[:3, :3] for B in robot], [B[:3, 3] for B in robot],
                               [C[:3, :3] for C in Cs], [C[:3, 3] for C in Cs],
                               method=methods[method])
    return validate_transform(transform(R, t))


def fixed_target(B, X, C, mode):
    return (B if mode == "eye-in-hand" else inverse(B)) @ X @ C


def handeye(args):
    cam, K, D = camera_file(args.intrinsics)
    rows = manifest(args.manifest, ["image", "split", "corner_order", "image_time_s", "pose_time_s", "stationary", *POSE_KEYS])
    counts(rows)
    board = cam["board"]
    if board["cols"] == board["rows"]:
        raise ValueError("Use a nonsquare inner-corner grid to reduce board-axis ambiguity")
    obj = board_points(**{"cols": board["cols"], "rows": board["rows"], "square_m": board["square_m"]})
    Bs, Cs, reports = [], [], []
    out = output_dir(args.out)
    for i, row in enumerate(rows):
        times = finite([row["image_time_s"], row["pose_time_s"]], "timestamps")
        dt = float(abs(times[0]-times[1])*1000)
        if row["stationary"] != "1" or dt > args.max_time_delta_ms:
            raise ValueError(f"View {i}: require stationary=1 and timestamp delta <= {args.max_time_delta_ms} ms")
        image, _, corners = detect(row["image"], board["cols"], board["rows"], cam["image_size_wh"])
        corners = ordered(corners, row["corner_order"])
        C, rms = pose_from_board(obj, corners, K, D)
        if rms > args.max_rms_px:
            raise ValueError(f"View {i}: PnP RMS {rms:.3f}px exceeds threshold")
        B = pose([row[k] for k in POSE_KEYS])
        Bs.append(B); Cs.append(C)
        reports.append({**row, "time_delta_ms": dt, "pnp_rms_px": rms})
        cv2.drawFrameAxes(image, K, D, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], board["square_m"]*2)
        save_image(out / f"axes_{i:03d}.png", image)
    ix = [i for i, r in enumerate(rows) if r["split"] == "train"]
    X = solve_handeye([Bs[i] for i in ix], [Cs[i] for i in ix], args.mode, args.method)
    constants = [fixed_target(B, X, C, args.mode) for B, C in zip(Bs, Cs)]
    reference = mean_transform([constants[i] for i in ix])
    for report, T in zip(reports, constants):
        report.update(residual(T, reference))
    passed = all(r["translation_mm"] <= args.max_translation_mm and
                 r["rotation_deg"] <= args.max_rotation_deg for r in reports)
    key = "T_tool_camera" if args.mode == "eye-in-hand" else "T_base_camera"
    data = {**base_result("handeye"), "mode": args.mode, "method": args.method,
            "camera_id": cam["camera_id"], "tool_frame": args.tool_frame,
            "intrinsics_sha256": hashlib.sha256(Path(args.intrinsics).read_bytes()).hexdigest(),
            key: X.tolist(), "fixed_target_reference": reference.tolist(),
            "fixed_target_frame": "base" if args.mode == "eye-in-hand" else "tool",
            "rotation_excitation_singular_values": excitation([Bs[i] for i in ix]),
            "views": reports, "validation_passed": passed,
            "thresholds": {"translation_mm": args.max_translation_mm, "rotation_deg": args.max_rotation_deg,
                           "time_delta_ms": args.max_time_delta_ms, "pnp_rms_px": args.max_rms_px},
            "validation_scope": "held-out rigid-target consistency; external metrology still required"}
    write_json(out / "handeye.json", data)
    print(f"Hand-eye validation: {passed}; {key}; output={out}")
    return 0 if passed else 2


def load_handeye(path, intrinsic_path):
    data = load_json(path)
    if data.get("kind") != "handeye" or not data.get("validation_passed"):
        raise ValueError("Use a handeye file that passed validation")
    if data["intrinsics_sha256"] != hashlib.sha256(Path(intrinsic_path).read_bytes()).hexdigest():
        raise ValueError("Intrinsics file differs from the one used for hand-eye calibration")
    return data


def point_in_base(u, v, z, cam, K, D, hand, robot_pose=None):
    finite([u, v, z])
    w, h = cam["image_size_wh"]
    if z <= 0 or not (0 <= u < w and 0 <= v < h):
        raise ValueError("Need positive optical-axis Z depth and pixel within calibrated image")
    ray = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, D).reshape(2)
    p = np.array([ray[0]*z, ray[1]*z, z, 1])
    if hand["mode"] == "eye-in-hand":
        if robot_pose is None:
            raise ValueError("Eye-in-hand requires synchronized T_base_tool pose")
        T = robot_pose @ validate_transform(hand["T_tool_camera"])
    else:
        T = validate_transform(hand["T_base_camera"])
    return (T @ p)[:3]


def point_command(args):
    cam, K, D = camera_file(args.intrinsics)
    hand = load_handeye(args.handeye, args.intrinsics)
    p = point_in_base(*args.pixel, args.depth_z_m, cam, K, D, hand,
                      pose(args.pose) if args.pose is not None else None)
    print(json.dumps({"point_base_m": p.tolist()}, indent=2))
    return 0


def verify_points(args):
    cam, K, D = camera_file(args.intrinsics)
    hand = load_handeye(args.handeye, args.intrinsics)
    required = ["u_px", "v_px", "depth_z_m", "reference_x_m", "reference_y_m", "reference_z_m"]
    if hand["mode"] == "eye-in-hand":
        required += POSE_KEYS
    rows = read_rows(args.csv, required)
    if len(rows) < 3:
        raise ValueError("Use at least 3 independent reference points; cover the whole workspace")
    reports = []
    for r in rows:
        B = pose([r[k] for k in POSE_KEYS]) if hand["mode"] == "eye-in-hand" else None
        vals = finite([r[k] for k in required[:6]])
        p = point_in_base(*vals[:3], cam, K, D, hand, B)
        error = float(np.linalg.norm(p-vals[3:6])*1000)
        reports.append({**r, "predicted_base_m": p.tolist(), "error_mm": error})
    errors = [r["error_mm"] for r in reports]
    result = {**base_result("external_point_validation"), "points": reports,
              "rms_mm": float(np.sqrt(np.mean(np.square(errors)))), "max_mm": max(errors),
              "threshold_mm": args.max_error_mm, "validation_passed": max(errors) <= args.max_error_mm}
    out = output_dir(args.out)
    write_json(out / "point_validation.json", result)
    print(f"External point validation: {result['validation_passed']}; max={max(errors):.3f} mm")
    return 0 if result["validation_passed"] else 2


def depth_check(args):
    rows = read_rows(args.csv, ["depth_raw", "reference_z_m"])
    if len(rows) < 3:
        raise ValueError("Need >=3 independently measured depth samples")
    raw = finite([r["depth_raw"] for r in rows])
    ref = finite([r["reference_z_m"] for r in rows])
    if np.any(raw <= 0) or np.any(ref <= 0):
        raise ValueError("Depth and reference Z must be positive")
    errors = (raw * args.scale_m_per_unit-ref)*1000
    report = {**base_result("depth_check"), "sample_count": len(rows),
              "scale_m_per_unit": args.scale_m_per_unit,
              "bias_mm": float(np.mean(errors)), "rms_mm": float(np.sqrt(np.mean(errors**2))),
              "max_abs_mm": float(np.max(np.abs(errors))),
              "validation_passed": bool(np.max(np.abs(errors)) <= args.max_error_mm),
              "max_allowed_mm": args.max_error_mm,
              "scope": "checks supplied optical-axis depth pairs; does not align RGB/depth or change device calibration"}
    out = output_dir(args.out)
    write_json(out / "depth_check.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["validation_passed"] else 2


def inspect_board(args):
    cam, K, D = camera_file(args.intrinsics)
    b = cam["board"]
    obj = board_points(b["cols"], b["rows"], b["square_m"])
    image, _, corners = detect(args.image, b["cols"], b["rows"], cam["image_size_wh"])
    corners = ordered(corners, args.corner_order)
    C, rms = pose_from_board(obj, corners, K, D)
    cv2.drawChessboardCorners(image, (b["cols"], b["rows"]), corners, True)
    cv2.drawFrameAxes(image, K, D, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], b["square_m"]*2)
    u, v = corners[0, 0]
    cv2.putText(image, "ORIGIN", (int(u)+5, int(v)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out = output_dir(args.out)
    save_image(out / "board_axes.png", image)
    write_json(out / "board_pose.json", {"T_camera_board": C.tolist(), "rms_px": rms,
                                        "corner_order": args.corner_order})
    print(f"Inspect physical origin and axis directions: {out / 'board_axes.png'}")
    return 0


def stereo(args):
    left, K1, D1 = camera_file(args.left_intrinsics)
    right, K2, D2 = camera_file(args.right_intrinsics)
    if left["board"] != right["board"] or left["image_size_wh"] != right["image_size_wh"]:
        raise ValueError("This stereo implementation requires the same board and image dimensions")
    b = left["board"]
    if b["cols"] == b["rows"]:
        raise ValueError("Use a nonsquare inner-corner grid")
    rows = read_rows(args.manifest, ["left_image", "right_image", "split", "left_corner_order",
                                   "right_corner_order", "left_time_s", "right_time_s", "stationary"])
    counts(rows)
    root = Path(args.manifest).resolve().parent
    obj = board_points(b["cols"], b["rows"], b["square_m"])
    out = output_dir(args.out)
    ls, rs, targets_left, targets_right = [], [], [], []
    seen_left, seen_right = set(), set()
    for row in rows:
        if row["split"] not in ("train", "val"):
            raise ValueError("split must be train or val")
        times = finite([row["left_time_s"], row["right_time_s"]])
        if row["stationary"] != "1" or abs(times[0]-times[1])*1000 > args.max_time_delta_ms:
            raise ValueError("Stereo pairs require stationary=1 and timestamps within tolerance")
        for prefix, cam, K, D, arrays, targets, seen in [
                ("left", left, K1, D1, ls, targets_left, seen_left),
                ("right", right, K2, D2, rs, targets_right, seen_right)]:
            p = (root / row[prefix+"_image"]).resolve()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate stereo image: {p}")
            seen.add(digest)
            _, _, corners = detect(p, b["cols"], b["rows"], cam["image_size_wh"])
            corners = ordered(corners, row[prefix+"_corner_order"])
            T, rms = pose_from_board(obj, corners, K, D)
            if rms > args.max_rms_px:
                raise ValueError("Stereo view PnP RMS exceeds threshold")
            arrays.append(corners); targets.append(T)
            row[prefix+"_sha256"] = digest
    ix = [i for i, r in enumerate(rows) if r["split"] == "train"]
    size = tuple(left["image_size_wh"])
    rms, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
        [obj]*len(ix), [ls[i] for i in ix], [rs[i] for i in ix], K1, D1, K2, D2, size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8))
    finite(rms)
    T = validate_transform(transform(R, t))
    reports = [{**row, **residual(T @ L, Rb)}
               for row, L, Rb in zip(rows, targets_left, targets_right)]
    passed = (rms <= args.max_rms_px and all(r["translation_mm"] <= args.max_translation_mm and
              r["rotation_deg"] <= args.max_rotation_deg for r in reports))
    # Rectification is saved only for a meaningful nonzero baseline.
    rectification = None
    if np.linalg.norm(t) > 1e-6:
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, t)
        rectification = {k: finite(v).tolist() for k, v in zip(["R1", "R2", "P1", "P2", "Q"], [R1, R2, P1, P2, Q])}
    result = {**base_result("stereo"), "left_camera_id": left["camera_id"],
              "right_camera_id": right["camera_id"], "T_right_left": T.tolist(),
              "left_intrinsics_sha256": hashlib.sha256(Path(args.left_intrinsics).read_bytes()).hexdigest(),
              "right_intrinsics_sha256": hashlib.sha256(Path(args.right_intrinsics).read_bytes()).hexdigest(),
              "train_rms_px": float(rms), "E": finite(E).tolist(), "F": finite(F).tolist(),
              "rectification": rectification, "views": reports, "validation_passed": bool(passed),
              "thresholds": {"rms_px": args.max_rms_px, "translation_mm": args.max_translation_mm,
                             "rotation_deg": args.max_rotation_deg, "time_delta_ms": args.max_time_delta_ms},
              "validation_scope": "held-out board consistency; does not implement RGB-D device registration"}
    write_json(out / "stereo.json", result)
    print(f"Stereo validation: {passed}; train RMS={rms:.4f}px")
    return 0 if passed else 2


def capture(args):
    out = output_dir(args.out)
    cap = cv2.VideoCapture(args.device)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise ValueError("Cannot open camera; this command supports OpenCV video devices only")
    print("SPACE: save raw frame; Q/ESC: quit. Timestamps are HOST RECEIPT times, not exposure timestamps.")
    count = 0
    try:
        with (out / "capture_log.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "host_receive_time_utc", "width", "height"])
            while True:
                ok, frame = cap.read()
                received = datetime.now(timezone.utc).isoformat()
                if not ok:
                    raise ValueError("Camera frame read failed")
                preview = frame.copy()
                cv2.putText(preview, f"SPACE save | Q quit | saved {count}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Calibration capture", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key == 32:
                    name = f"frame_{count:04d}.png"
                    save_image(out / name, frame)
                    writer.writerow([name, received, frame.shape[1], frame.shape[0]])
                    f.flush()
                    count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="Capture raw images from an OpenCV camera")
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--width", type=int)
    c.add_argument("--height", type=int)
    c.add_argument("--out", required=True)
    c.set_defaults(func=capture)
    c = sub.add_parser("intrinsics", help="Fit camera intrinsics and validate held-out views")
    c.add_argument("--manifest", required=True)
    c.add_argument("--cols", type=int, required=True)
    c.add_argument("--rows", type=int, required=True)
    c.add_argument("--square-m", type=positive, required=True)
    c.add_argument("--camera-id", required=True)
    c.add_argument("--max-rms-px", type=positive, required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=intrinsics)
    c = sub.add_parser("handeye", help="Fit eye-in-hand or eye-to-hand calibration")
    c.add_argument("--intrinsics", required=True)
    c.add_argument("--manifest", required=True)
    c.add_argument("--mode", choices=["eye-in-hand", "eye-to-hand"], required=True)
    c.add_argument("--tool-frame", required=True, help="Name of frame used by robot pose feedback")
    c.add_argument("--method", choices=["park", "tsai", "horaud"], default="park")
    c.add_argument("--max-rms-px", type=positive, required=True)
    c.add_argument("--max-time-delta-ms", type=positive, required=True)
    c.add_argument("--max-translation-mm", type=positive, required=True)
    c.add_argument("--max-rotation-deg", type=positive, required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=handeye)
    c = sub.add_parser("point", help="Raw RGB pixel + aligned optical Z depth -> base point")
    c.add_argument("--intrinsics", required=True)
    c.add_argument("--handeye", required=True)
    c.add_argument("--pixel", nargs=2, type=float, required=True, metavar=("U", "V"))
    c.add_argument("--depth-z-m", type=positive, required=True)
    c.add_argument("--pose", nargs=7, type=float, help="x y z qx qy qz qw, synchronized T_base_tool")
    c.set_defaults(func=point_command)
    c = sub.add_parser("verify-points", help="Compare against independent metric reference points")
    c.add_argument("--intrinsics", required=True)
    c.add_argument("--handeye", required=True)
    c.add_argument("--csv", required=True)
    c.add_argument("--max-error-mm", type=positive, required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=verify_points)
    c = sub.add_parser("depth-check", help="Validate supplied depth scale against reference Z")
    c.add_argument("--csv", required=True)
    c.add_argument("--scale-m-per-unit", type=positive, required=True)
    c.add_argument("--max-error-mm", type=positive, required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=depth_check)
    c = sub.add_parser("inspect-board", help="Inspect board origin before hand-eye or stereo calibration")
    c.add_argument("--image", required=True)
    c.add_argument("--intrinsics", required=True)
    c.add_argument("--corner-order", choices=["normal", "reverse"], default="normal")
    c.add_argument("--out", required=True)
    c.set_defaults(func=inspect_board)
    c = sub.add_parser("stereo", help="Fixed-intrinsics stereo extrinsics and rectification")
    c.add_argument("--left-intrinsics", required=True)
    c.add_argument("--right-intrinsics", required=True)
    c.add_argument("--manifest", required=True)
    c.add_argument("--max-time-delta-ms", type=positive, required=True)
    c.add_argument("--max-rms-px", type=positive, required=True)
    c.add_argument("--max-translation-mm", type=positive, required=True)
    c.add_argument("--max-rotation-deg", type=positive, required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=stereo)
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
