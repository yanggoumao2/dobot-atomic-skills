"""Generate simulated chessboard images/poses and exercise the complete offline pipeline.

python synthetic_demo.py --out demo_run
No robot or camera hardware is used. Output is explicitly synthetic.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from . import calibrate as c
else:  # preserve the original standalone demo
    import calibrate as c


def quaternion(R):
    rv = cv2.Rodrigues(R)[0].reshape(3)
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return [0., 0., 0., 1.]
    return [*(rv / angle * np.sin(angle/2)), float(np.cos(angle/2))]


def make_T(rot, translation):
    return c.transform(cv2.Rodrigues(np.array(rot, float))[0], translation)


def render(C, K, cols=8, rows=5, s=0.025, size=(960, 720)):
    scale = 2
    image = np.full((size[1]*scale, size[0]*scale, 3), 180, np.uint8)
    rvec = cv2.Rodrigues(C[:3, :3])[0]
    Ks = K.copy()
    Ks[:2] *= scale
    for y in range(-1, rows):
        for x in range(-1, cols):
            obj = np.array([[x*s, y*s, 0], [(x+1)*s, y*s, 0],
                            [(x+1)*s, (y+1)*s, 0], [x*s, (y+1)*s, 0]], float)
            pixels = cv2.projectPoints(obj, rvec, C[:3, 3], Ks, np.zeros(5))[0].reshape(-1, 2)
            if np.any(pixels[:, 0] < 2) or np.any(pixels[:, 0] >= size[0]*scale-2) or np.any(pixels[:, 1] < 2) or np.any(pixels[:, 1] >= size[1]*scale-2):
                raise ValueError("Synthetic board leaves image bounds")
            colour = 20 if (x+y) % 2 == 0 else 245
            cv2.fillConvexPoly(image, np.round(pixels).astype(np.int32), (colour,)*3)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def save_csv(path, keys, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def correct_order(path, C, K):
    _, _, corners = c.detect(path, 8, 5)
    expected = cv2.projectPoints(np.zeros((1, 3)), cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], K, np.zeros(5))[0].reshape(2)
    return "normal" if np.linalg.norm(corners[0, 0]-expected) < np.linalg.norm(corners[-1, 0]-expected) else "reverse"


def generate(root):
    root = c.output_dir(root)
    images = root / "images"
    images.mkdir()
    rng = np.random.default_rng(918)
    K = np.array([[800., 0, 480], [0, 790., 360], [0, 0, 1]])
    X_in = make_T([0.12, -0.18, 0.06], [0.04, 0.02, 0.08])
    Y_in = make_T([0.2, 0.1, -0.15], [0.4, 0.1, 0.3])
    X_out = make_T([0.2, -0.1, 0.3], [0.3, -0.2, 0.5])
    Y_out = make_T([0.1, 0.2, -0.1], [0.03, 0.01, 0.1])
    X_stereo = make_T([0.01, 0.02, -0.005], [-0.075, 0.002, 0.001])
    camera_rows, in_rows, out_rows, stereo_rows = [], [], [], []
    for i in range(24):
        R = cv2.Rodrigues(rng.uniform([-.5, -.5, -.35], [.5, .5, .35]))[0]
        centre = np.array([rng.uniform(-.055, .055), rng.uniform(-.045, .045), rng.uniform(.58, .85)])
        C = c.transform(R, centre-R @ np.array([.0875, .05, 0]))
        Cr = X_stereo @ C
        name, right = f"images/left_{i:03d}.png", f"images/right_{i:03d}.png"
        c.save_image(root / name, render(C, K))
        c.save_image(root / right, render(Cr, K))
        order = correct_order(root / name, C, K)
        rorder = correct_order(root / right, Cr, K)
        split = "train" if i < 18 else "val"
        camera_rows.append({"image": name, "split": split})
        common = {"image": name, "split": split, "corner_order": order,
                  "image_time_s": float(i), "pose_time_s": float(i), "stationary": 1}
        for B, rows in [(Y_in @ c.inverse(C) @ c.inverse(X_in), in_rows),
                        (X_out @ C @ c.inverse(Y_out), out_rows)]:
            rows.append({**common, **dict(zip(c.POSE_KEYS, [*B[:3, 3], *quaternion(B[:3, :3])]))})
        stereo_rows.append({"left_image": name, "right_image": right, "split": split,
                            "left_corner_order": order, "right_corner_order": rorder,
                            "left_time_s": float(i), "right_time_s": float(i), "stationary": 1})
    save_csv(root / "intrinsics.csv", ["image", "split"], camera_rows)
    keys = ["image", "split", "corner_order", "image_time_s", "pose_time_s", "stationary", *c.POSE_KEYS]
    save_csv(root / "handeye_in.csv", keys, in_rows)
    save_csv(root / "handeye_out.csv", keys, out_rows)
    save_csv(root / "stereo.csv", list(stereo_rows[0]), stereo_rows)
    truth = {"synthetic_only": True, "K": K.tolist(), "T_tool_camera": X_in.tolist(),
             "T_base_camera": X_out.tolist(), "T_right_left": X_stereo.tolist()}
    c.write_json(root / "ground_truth.json", truth)
    references = []
    for u, v, z in [(300, 250, .6), (600, 400, .8), (450, 500, .7), (700, 200, .9)]:
        pc = np.linalg.inv(K) @ np.array([u, v, 1.]) * z
        p = (X_out @ np.r_[pc, 1])[:3]
        references.append(dict(zip(["u_px", "v_px", "depth_z_m", "reference_x_m", "reference_y_m", "reference_z_m"], [u, v, z, *p])))
    save_csv(root / "reference_points.csv", list(references[0]), references)
    save_csv(root / "depth.csv", ["depth_raw", "reference_z_m"],
             [{"depth_raw": z*1000, "reference_z_m": z} for z in [.4, .6, .8]])
    return truth


def run(root):
    root = root.resolve()
    script = Path(c.__file__).resolve()
    def command(*args, expected_code=0):
        cmd = [sys.executable, str(script), *map(str, args)]
        print("RUN:", subprocess.list2cmdline(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode != expected_code:
            raise AssertionError(f"Expected exit {expected_code}, got {result.returncode}")
    camera = root / "camera/intrinsics.json"
    command("intrinsics", "--manifest", root / "intrinsics.csv", "--cols", 8, "--rows", 5,
            "--square-m", .025, "--camera-id", "SIMULATED_CAMERA", "--max-rms-px", 1, "--out", root / "camera")
    for mode, name in [("eye-in-hand", "in"), ("eye-to-hand", "out")]:
        command("handeye", "--intrinsics", camera, "--manifest", root / f"handeye_{name}.csv",
                "--mode", mode, "--tool-frame", "SIMULATED_TOOL", "--max-rms-px", 1,
                "--max-time-delta-ms", 20, "--max-translation-mm", 8, "--max-rotation-deg", 2,
                "--out", root / f"handeye_{name}")
    # Both simulated cameras deliberately use the same lens model.
    command("stereo", "--left-intrinsics", camera, "--right-intrinsics", camera,
            "--manifest", root / "stereo.csv", "--max-rms-px", 1, "--max-time-delta-ms", 20,
            "--max-translation-mm", 8, "--max-rotation-deg", 2, "--out", root / "stereo")
    command("verify-points", "--intrinsics", camera, "--handeye", root / "handeye_out/handeye.json",
            "--csv", root / "reference_points.csv", "--max-error-mm", 8, "--out", root / "point_check")
    command("depth-check", "--csv", root / "depth.csv", "--scale-m-per-unit", .001,
            "--max-error-mm", 1, "--out", root / "depth_check")
    first = c.read_rows(root / "handeye_in.csv", ["corner_order"])[0]
    command("inspect-board", "--image", root / first["image"], "--intrinsics", camera,
            "--corner-order", first["corner_order"], "--out", root / "inspect")
    command("point", "--intrinsics", camera, "--handeye", root / "handeye_out/handeye.json",
            "--pixel", 480, 360, "--depth-z-m", .7)
    truth = c.load_json(root / "ground_truth.json")
    summary = {"synthetic_only": True, "real_hardware_tested": False, "results": {}}
    for name, key in [("in", "T_tool_camera"), ("out", "T_base_camera")]:
        fitted = c.load_json(root / f"handeye_{name}/handeye.json")
        err = c.residual(np.array(fitted[key]), np.array(truth[key]))
        if err["translation_mm"] > 8 or err["rotation_deg"] > 2:
            raise AssertionError(f"Ground-truth recovery failed: {err}")
        summary["results"][key] = err
    stereo = c.load_json(root / "stereo/stereo.json")
    summary["results"]["T_right_left"] = c.residual(np.array(stereo["T_right_left"]), np.array(truth["T_right_left"]))
    stereo_error = summary["results"]["T_right_left"]
    if stereo_error["translation_mm"] > 8 or stereo_error["rotation_deg"] > 2:
        raise AssertionError(f"Synthetic stereo recovery failed: {stereo_error}")
    Kfit = np.array(c.load_json(camera)["K"])
    Ktrue = np.array(truth["K"])
    focal_error = float(np.max(np.abs(np.diag(Kfit)[:2]/np.diag(Ktrue)[:2]-1)))
    if focal_error > .03:
        raise AssertionError("Synthetic intrinsic recovery differs by >3%")
    summary["results"]["max_focal_relative_error"] = focal_error
    summary["results"]["external_points"] = c.load_json(root / "point_check/point_validation.json")
    # Corrupt only validation robot poses: training stays unchanged, rejection is expected.
    rows = c.read_rows(root / "handeye_in.csv", ["split", "x_m"])
    for row in rows:
        if row["split"] == "val":
            row["x_m"] = float(row["x_m"]) + .05
    save_csv(root / "bad_validation.csv", list(rows[0]), rows)
    command("handeye", "--intrinsics", camera, "--manifest", root / "bad_validation.csv",
            "--mode", "eye-in-hand", "--tool-frame", "SIMULATED_TOOL", "--max-rms-px", 1,
            "--max-time-delta-ms", 20, "--max-translation-mm", 8, "--max-rotation-deg", 2,
            "--out", root / "rejected_handeye", expected_code=2)
    rejected = c.load_json(root / "rejected_handeye/handeye.json")
    np.testing.assert_allclose(rejected["T_tool_camera"],
                               c.load_json(root / "handeye_in/handeye.json")["T_tool_camera"], atol=1e-10)
    rows = c.read_rows(root / "handeye_in.csv", ["pose_time_s"])
    rows[0]["pose_time_s"] = 99
    save_csv(root / "bad_timestamp.csv", list(rows[0]), rows)
    command("handeye", "--intrinsics", camera, "--manifest", root / "bad_timestamp.csv",
            "--mode", "eye-in-hand", "--tool-frame", "SIMULATED_TOOL", "--max-rms-px", 1,
            "--max-time-delta-ms", 20, "--max-translation-mm", 8, "--max-rotation-deg", 2,
            "--out", root / "rejected_timestamp", expected_code=1)
    summary["negative_checks"] = {"bad_validation_rejected": True,
                                  "held_out_data_did_not_change_fit": True,
                                  "bad_timestamp_rejected": True}
    c.write_json(root / "synthetic_report.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    generate(args.out)
    if not args.generate_only:
        run(args.out)
