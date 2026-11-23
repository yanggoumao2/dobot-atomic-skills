"""Independent geometric checks for the open-jaw optical and flat-checker depth models."""
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from chem_demo.validate_wrist_plane import View, evaluate_multiview, fit_disparity, measured_board_points
from chem_demo.wrist_mount import refine_jaw_motion
from chem_demo.wrist_optical import OpenWristCameraModel


class CheckerPlaneTests(unittest.TestCase):
    def test_multiview_keeps_original_failure_and_requires_a_single_anchor(self):
        ids = np.indices((7, 7)).reshape(2, -1).T
        def view(selected):
            points = np.column_stack((selected * 30., np.full(len(selected), 300.)))
            return View(selected, points / 300., points[:, 2], points, np.eye(4), np.eye(4))
        views = {"ref": view(ids[:35]), "x-": view(ids), "z-yaw+": view(ids[10:]), "ref_return": view(ids[:35])}
        original = {"checker_plane_passed": False, "fewest_shared_physical_corners": 25,
                    "worst_stereo_pair_median_mm": 0.8, "worst_board_3d_median_mm": 0.7,
                    "worst_optical_3d_median_mm": 0.6, "held_out_plane_height_spread_mm": 1.}
        with patch("chem_demo.validate_wrist_plane.solve_hand_eye", return_value={"T_flange_camera": np.eye(4)}) as fit:
            result = evaluate_multiview(views, np.eye(4), 30., 0., original)
            self.assertTrue(result["checker_plane_passed"])
            self.assertFalse(result["original_reference_check_passed"])
            self.assertEqual(result["validation_anchor"], "x-")
            self.assertEqual(result["minimum_shared_with_validation_anchor"], 35)
            tested = [pair for pair in result["pairwise_scores"] if "median_mm" in pair]
            for pair, call in zip(tested, fit.call_args_list, strict=True):
                trained = {p.label for p in call.args[0]}
                self.assertFalse(trained.intersection((pair["first"], pair["second"], "ref_return")))
            del views["x-"]
            self.assertFalse(evaluate_multiview(views, np.eye(4), 30., 0., original)["checker_plane_passed"])

    def test_metric_inverse_depth_model_and_disjoint_plane_interpolation(self):
        xy = np.indices((8, 8)).reshape(2, -1).T / 16 - 0.2
        rays = np.c_[xy, np.ones(len(xy))]
        bias = np.array([10, 2, -0.4])
        views = []
        for offset in (290, 310, 330):
            actual = 100000 / (rays @ [35, 17, offset])
            raw = 100000 / (100000 / actual - rays @ bias)
            views.append(View(xy, rays, raw, rays * actual[:, None], np.eye(4), np.eye(4)))
        correction = fit_disparity(views[:2])
        np.testing.assert_allclose(correction, bias, atol=1e-9)
        np.testing.assert_allclose(measured_board_points(views[2], correction), views[2].metric_xyz, atol=1e-8)

    def test_unobservable_fingers_rejected(self):
        image = np.zeros((480, 848, 3), np.uint8)
        depth = np.full((480, 848), 310, np.float32)
        with self.assertRaisesRegex(ValueError, "no trackable"):
            refine_jaw_motion(image, depth, image, depth, np.eye(3), np.eye(4))

    def test_real_independent_still_and_closed_gripper(self):
        parent = Path(__file__).resolve().parents[1] / "calibration-data/chem-demo/wrist-board"
        manifest = parent / "left-20261121_152610/checker_plane_validation.json"
        still = parent / "stationary-left-20261121_152346"
        if not manifest.exists() or not still.exists():
            self.skipTest("local checkerboard observations unavailable")
        md = json.loads((still / "capture.json").read_text(encoding="utf-8"))
        K = np.array(md["camera"]["intrinsics"]["matrix_px"])
        model = OpenWristCameraModel(manifest, serial=md["camera"]["serial"], K=K)
        color = cv2.imread(str(still / "ref.png"))
        depth = np.load(still / "ref_depth_temporal_median_mm.npy")
        pose = md["before"]["pose"]
        result = model.at_recorded_pose(pose, color, depth, gripper_open=True)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(np.linalg.det(result[:3, :3]), 1.0, places=6)
        with self.assertRaisesRegex(ValueError, "open gripper"):
            model.at_recorded_pose(pose, color, depth, gripper_open=False)
        with self.assertRaisesRegex(ValueError, "verified calibration postures"):
            model.at_recorded_pose(np.array(pose) + [120, 0, 0, 0, 0, 0], color, depth, gripper_open=True)
