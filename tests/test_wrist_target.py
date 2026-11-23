import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from chem_demo.wrist_target import (
    _camera_parameters,
    _local_disparity_depth,
    as_pair_observations,
    load_target_capture,
)


def pair(label):
    pixels = np.array([[20.0, 20.0], [21.0, 20.0], [22.0, 20.0], [23.0, 20.0], [24.0, 20.0]])
    points = np.column_stack((np.arange(5, dtype=float), np.zeros(5), np.full(5, 500.0)))
    identity = np.eye(4)
    return SimpleNamespace(
        label=label,
        unique_inliers=5,
        region_ids=np.array(["logo"] * 5, dtype=object),
        T_base_flange_ref=identity,
        T_base_flange_pose=identity,
        pixels_ref_px=pixels,
        pixels_pose_px=pixels,
        points_ref_camera_mm=points,
        points_pose_camera_mm=points,
    )


class WristTargetTests(unittest.TestCase):
    def test_camera_corrections_must_cover_every_pair(self):
        capture = SimpleNamespace(pairs={"x+": pair("x+"), "y+": pair("y+")})
        with self.assertRaisesRegex(ValueError, "missing for poses: y\\+"):
            as_pair_observations(capture, {"x+": np.eye(4)})
        with self.assertRaisesRegex(ValueError, "camera correction must be a finite 4x4"):
            as_pair_observations(capture, {"x+": np.eye(4), "y+": None})

    def test_camera_correction_maps_pose_points_to_reference_camera(self):
        capture = SimpleNamespace(pairs={"x+": pair("x+")})
        correction = np.eye(4)
        correction[0, 3] = 5.0
        observed = as_pair_observations(capture, {"x+": correction})
        np.testing.assert_allclose(observed[0].xi[:, 0], np.arange(5, dtype=float) + 5.0)
        np.testing.assert_allclose(observed[0].xi[:, 0] - observed[0].x0[:, 0], 5.0)

    def test_local_depth_fit_rejects_paper_edge_and_keeps_interior(self):
        h, w = 64, 64
        vv, uu = np.mgrid[:h, :w]
        depth = (1.0 / (0.002 + 1.0e-6 * uu + 2.0e-6 * vv)).astype(np.float32)
        paper = np.zeros((h, w), np.uint8)
        paper[10:55, 10:55] = 255
        support = paper.copy()
        edge_distance = cv2.distanceTransform((paper > 0).astype(np.uint8), cv2.DIST_L2, 3)
        interior = _local_disparity_depth(depth, support, edge_distance, 32.0, 32.0)
        self.assertIsNotNone(interior)
        self.assertAlmostEqual(interior[0], float(depth[32, 32]), delta=0.1)
        self.assertIsNone(_local_disparity_depth(depth, support, edge_distance, 12.0, 32.0))

    def test_metadata_rejects_invalid_focal_length_and_nonfinite_table_height(self):
        metadata = {"camera": {"intrinsics": {"matrix_px": np.eye(3).tolist()}},
                    "table_z_reference_mm": 243.2}
        K, height = _camera_parameters(metadata)
        np.testing.assert_allclose(K, np.eye(3))
        self.assertEqual(height, 243.2)
        invalid = {"camera": {"intrinsics": {"matrix_px": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
                   "table_z_reference_mm": 243.2}
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            _camera_parameters(invalid)
        metadata["table_z_reference_mm"] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _camera_parameters(metadata)

    def test_requested_inlier_threshold_cannot_be_lowered(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "cannot be lower"):
                load_target_capture(Path(tmp), minimum_inliers=29)


if __name__ == "__main__":
    unittest.main()
