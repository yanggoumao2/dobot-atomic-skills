"""Offline checks for open wrist-gripper pad-tip measurement."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from chem_demo.wrist_landmarks import CameraTemplate, measure_open_pad_tips


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DATA = ROOT / "calibration-data" / "chem-demo" / "wrist-calibration"


def synthetic_gripper(left_tip: tuple[int, int], right_tip: tuple[int, int]) -> np.ndarray:
    """Simple high-contrast view with two separate pad silhouettes entering from above."""
    image = np.full((480, 848, 3), 255, dtype=np.uint8)
    left = np.array([[145, 0], [255, 0], left_tip], dtype=np.int32)
    right = np.array([[560, 0], [680, 0], right_tip], dtype=np.int32)
    cv2.fillConvexPoly(image, left, (0, 0, 0))
    cv2.fillConvexPoly(image, right, (0, 0, 0))
    return image


class WristLandmarkTests(unittest.TestCase):
    def test_synthetic_open_gripper_reports_two_pixel_landmarks(self) -> None:
        image = synthetic_gripper((292, 171), (554, 190))
        result = measure_open_pad_tips(image)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.state, "open")
        self.assertIsNotNone(result.points)
        assert result.left is not None and result.right is not None
        self.assertLess(abs(result.left[0] - 292), 3)
        self.assertLess(abs(result.left[1] - 171), 2)
        self.assertLess(abs(result.right[0] - 554), 3)
        self.assertLess(abs(result.right[1] - 190), 2)
        self.assertGreater(result.confidence, 0.65)

        gray_result = measure_open_pad_tips(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.assertTrue(gray_result.ok, gray_result.reason)
        self.assertEqual(gray_result.left, result.left)
        self.assertEqual(gray_result.right, result.right)

    def test_synthetic_closed_gripper_is_detected_and_rejected(self) -> None:
        image = synthetic_gripper((418, 160), (430, 160))
        result = measure_open_pad_tips(image)
        self.assertFalse(result.ok)
        self.assertEqual(result.state, "closed")
        self.assertIsNotNone(result.gap_px)
        self.assertLess(result.gap_px, 35)

    def test_missing_or_low_confidence_pad_is_rejected(self) -> None:
        one_pad = np.full((480, 848, 3), 255, dtype=np.uint8)
        cv2.fillConvexPoly(one_pad, np.array([[130, 0], [245, 0], [292, 171]], np.int32), (0, 0, 0))
        missing = measure_open_pad_tips(one_pad)
        self.assertFalse(missing.ok)
        self.assertEqual(missing.state, "ambiguous")
        self.assertIsNone(missing.right)

        strict = measure_open_pad_tips(
            synthetic_gripper((292, 171), (554, 190)),
            template=CameraTemplate(min_confidence=0.99),
        )
        self.assertFalse(strict.ok)
        self.assertEqual(strict.state, "ambiguous")
        self.assertIn("confidence", strict.reason or "")

    def test_real_calibration_frames_are_open_and_repeatable(self) -> None:
        sessions = (
            ("left-20261121_141715", 9),
            ("left-20261121_135739", 7),
        )
        if not all((CALIBRATION_DATA / name).is_dir() for name, _ in sessions):
            self.skipTest("local wrist calibration images are unavailable")
        positions: list[tuple[list[float], list[float], list[float]]] = []
        for folder_name, expected_count in sessions:
            folder = CALIBRATION_DATA / folder_name
            images = sorted(path for path in folder.glob("*.png")
                            if not path.stem.endswith("_depth") and path.stem != "gripper_mask")
            self.assertEqual(len(images), expected_count, folder_name)

            left_x: list[float] = []
            right_x: list[float] = []
            gaps: list[float] = []
            for path in images:
                result = measure_open_pad_tips(cv2.imread(str(path)))
                self.assertTrue(result.ok, f"{path.name}: {result.reason}")
                self.assertEqual(result.state, "open")
                self.assertGreaterEqual(result.confidence, 0.65)
                assert result.left is not None and result.right is not None and result.gap_px is not None
                left_x.append(result.left[0])
                right_x.append(result.right[0])
                gaps.append(result.gap_px)

            # The sampled poses move and yaw the arm, but the two image-space tip tracks remain
            # tightly clustered horizontally. The left/right vertical offset remains about 19 px.
            self.assertLess(max(left_x) - min(left_x), 8.0, folder_name)
            self.assertLess(max(right_x) - min(right_x), 15.0, folder_name)
            self.assertLess(min(gaps), 275.0)
            self.assertGreater(min(gaps), 250.0)
            positions.append((left_x, right_x, gaps))

        self.assertLess(abs(float(np.median(positions[0][2])) - float(np.median(positions[1][2]))), 5.0)


if __name__ == "__main__":
    unittest.main()
