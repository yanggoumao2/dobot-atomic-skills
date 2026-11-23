"""Focused checks for offline open-jaw mount measurement."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from chem_demo import check_wrist_calibration as legacy
from chem_demo.wrist_mount import WristMountReference, current_jaw_mask, refine_jaw_motion


ROOT = Path(__file__).resolve().parents[1]
BURST = ROOT / "calibration-data" / "chem-demo" / "wrist-burst-review" / "left-20261121_150323"
RIGHT_OPEN = ROOT / "calibration-data" / "chem-demo" / "wrist-board" / "stationary-right-20261121_154444"
RIGHT_HALF_OPEN = ROOT / "calibration-data" / "chem-demo" / "wrist-board" / "stationary-right-20261121_154144"


class WristMountTests(unittest.TestCase):
    def test_depth_mask_keeps_near_fingers_and_suppresses_printed_scene(self) -> None:
        depth = np.full((480, 848), 300.0, np.float32)
        depth[40:180, 160:280] = 100.0
        depth[45:190, 560:700] = 105.0
        depth[120:220, 390:470] = 110.0  # central printed logo-like region
        depth[350:440, 160:280] = 100.0  # lower scene region

        mask = current_jaw_mask(depth)

        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)), {0, 255})
        self.assertGreater(int(mask[100, 220]), 0)
        self.assertGreater(int(mask[100, 620]), 0)
        self.assertEqual(int(mask[165, 430]), 0)
        self.assertEqual(int(mask[390, 220]), 0)
        self.assertEqual(int(mask[250, 620]), 0)

    @unittest.skipUnless(BURST.is_dir(), "local wrist burst capture is unavailable")
    def test_reference_self_registration_and_existing_pnp_parity(self) -> None:
        manifest = json.loads((BURST / "acquisition.json").read_text(encoding="utf-8"))
        K = np.asarray(manifest["camera"]["intrinsics"]["matrix_px"], float)
        color = cv2.imread(str(BURST / "ref.png"))
        depth = np.load(BURST / "ref_depth_temporal_median_mm.npy", allow_pickle=False)
        reference = WristMountReference.from_frame(color, depth, K)

        cv2.setRNGSeed(0)
        measured = reference.measure(color, depth)

        self.assertTrue(measured.ok, measured.reason)
        self.assertIsNotNone(measured.T_reference_current)
        self.assertGreaterEqual(measured.inliers, 8)
        self.assertGreaterEqual(measured.left_inliers, 2)
        self.assertGreaterEqual(measured.right_inliers, 2)
        assert measured.T_reference_current is not None
        np.testing.assert_allclose(measured.T_reference_current, np.eye(4), atol=1e-5)
        self.assertGreater(int(measured.jaw_mask.sum()), 0)
        # The logo lies within the center exclusion band and must never enter the jaw mask.
        self.assertEqual(int(measured.jaw_mask[170, 450]), 0)

        current_color = cv2.imread(str(BURST / "yaw+.png"))
        current_depth = np.load(BURST / "yaw+_depth_temporal_median_mm.npy", allow_pickle=False)
        cv2.setRNGSeed(0)
        measured_yaw = reference.measure(current_color, current_depth)
        self.assertTrue(measured_yaw.ok, measured_yaw.reason)

        cv2.setRNGSeed(0)
        old_reference = legacy.features(color, depth, True)
        old_current = legacy.features(current_color, current_depth, True)
        expected, _ = legacy.current_to_reference_jaws(old_reference, old_current, depth, K)
        assert measured_yaw.T_reference_current is not None
        np.testing.assert_allclose(measured_yaw.T_reference_current, expected, atol=1e-5)

    @unittest.skipUnless(BURST.is_dir(), "local wrist burst capture is unavailable")
    def test_closed_or_invalid_frames_return_no_transform(self) -> None:
        manifest = json.loads((BURST / "acquisition.json").read_text(encoding="utf-8"))
        K = np.asarray(manifest["camera"]["intrinsics"]["matrix_px"], float)
        color = cv2.imread(str(BURST / "ref.png"))
        depth = np.load(BURST / "ref_depth_temporal_median_mm.npy", allow_pickle=False)
        reference = WristMountReference.from_frame(color, depth, K)

        closed_or_unobservable = np.full_like(color, 255)
        rejected = reference.measure(closed_or_unobservable, depth)
        self.assertFalse(rejected.ok)
        self.assertIsNone(rejected.T_reference_current)
        self.assertIn("open gripper", rejected.reason or "")

        invalid_depth = depth.copy()
        invalid_depth[0, 0] = np.nan
        malformed = reference.measure(color, invalid_depth)
        self.assertFalse(malformed.ok)
        self.assertIsNone(malformed.T_reference_current)
        self.assertIn("finite", malformed.reason or "")


class RightWristMountTests(unittest.TestCase):
    @unittest.skipUnless(RIGHT_OPEN.is_dir(), "consented, fully-open right wrist capture is unavailable")
    def test_right_reference_self_registers_on_both_jaws_without_scene_features(self) -> None:
        manifest = json.loads((RIGHT_OPEN / "capture.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["arm"], "right")
        self.assertEqual(manifest.get("gripper_position_after"), 254)
        K = np.asarray(manifest["camera"]["intrinsics"]["matrix_px"], float)
        color = cv2.imread(str(RIGHT_OPEN / "ref.png"))
        depth = np.load(RIGHT_OPEN / "ref_depth_temporal_median_mm.npy", allow_pickle=False)
        reference = WristMountReference.from_frame(color, depth, K, arm="right")

        cv2.setRNGSeed(0)
        measured = reference.measure(color, depth, arm="right")

        self.assertTrue(measured.ok, measured.reason)
        self.assertGreaterEqual(measured.left_inliers, 2)
        self.assertGreaterEqual(measured.right_inliers, 2)
        self.assertGreaterEqual(measured.inliers, 8)
        assert measured.T_reference_current is not None
        np.testing.assert_allclose(measured.T_reference_current, np.eye(4), atol=1e-5)
        # The view's upper checkerboard, center gap/blue feature, and left-side cup stay outside
        # jaw-only support. Both actual pad corridors still contain depth-backed pixels.
        self.assertEqual(int(measured.jaw_mask[:168].sum()), 0)
        self.assertEqual(int(measured.jaw_mask[:, 390:483].sum()), 0)
        self.assertEqual(int(measured.jaw_mask[250, 100]), 0)
        self.assertGreater(int(measured.jaw_mask[300, 220]), 0)
        self.assertGreater(int(measured.jaw_mask[350, 740]), 0)

        refined = refine_jaw_motion(color, depth, color, depth, K,
                                    measured.T_reference_current, arm="right")
        np.testing.assert_allclose(refined, np.eye(4), atol=1e-5)

    @unittest.skipUnless(RIGHT_OPEN.is_dir() and RIGHT_HALF_OPEN.is_dir(),
                         "right wrist full-open and half-open captures are unavailable")
    def test_right_half_open_geometry_is_rejected_as_a_rigid_camera_match(self) -> None:
        manifest = json.loads((RIGHT_OPEN / "capture.json").read_text(encoding="utf-8"))
        half_open_manifest = json.loads((RIGHT_HALF_OPEN / "capture.json").read_text(encoding="utf-8"))
        K = np.asarray(manifest["camera"]["intrinsics"]["matrix_px"], float)
        color = cv2.imread(str(RIGHT_OPEN / "ref.png"))
        depth = np.load(RIGHT_OPEN / "ref_depth_temporal_median_mm.npy", allow_pickle=False)
        reference = WristMountReference.from_frame(color, depth, K, arm="right")
        half_open_color = cv2.imread(str(RIGHT_HALF_OPEN / "ref.png"))
        half_open_depth = np.load(RIGHT_HALF_OPEN / "ref_depth_temporal_median_mm.npy", allow_pickle=False)

        measured = reference.measure(half_open_color, half_open_depth)

        self.assertFalse(measured.ok)
        self.assertIsNone(measured.T_reference_current)
        with self.assertRaisesRegex(ValueError, "bidirectional jaw tracks"):
            refine_jaw_motion(color, depth, half_open_color, half_open_depth, K, np.eye(4), arm="right")

    @unittest.skipUnless(RIGHT_OPEN.is_dir(), "consented, fully-open right wrist capture is unavailable")
    def test_right_closed_and_absent_fingers_return_no_transform(self) -> None:
        manifest = json.loads((RIGHT_OPEN / "capture.json").read_text(encoding="utf-8"))
        K = np.asarray(manifest["camera"]["intrinsics"]["matrix_px"], float)
        color = cv2.imread(str(RIGHT_OPEN / "ref.png"))
        depth = np.load(RIGHT_OPEN / "ref_depth_temporal_median_mm.npy", allow_pickle=False)
        reference = WristMountReference.from_frame(color, depth, K, arm="right")

        closed = np.full_like(color, 220)
        cv2.fillPoly(closed, [np.asarray([[180, 479], [345, 479], [415, 190], [405, 190]], np.int32)], (0, 0, 0))
        cv2.fillPoly(closed, [np.asarray([[428, 190], [438, 190], [700, 479], [550, 479]], np.int32)], (0, 0, 0))
        absent = np.full_like(color, 220)

        for invalid in (closed, absent):
            measured = reference.measure(invalid, depth)
            self.assertFalse(measured.ok)
            self.assertIsNone(measured.T_reference_current)
            self.assertIn("open gripper", measured.reason or "")
            with self.assertRaisesRegex(ValueError, "two visible open fingers"):
                refine_jaw_motion(color, depth, invalid, depth, K, np.eye(4), arm="right")


if __name__ == "__main__":
    unittest.main()
