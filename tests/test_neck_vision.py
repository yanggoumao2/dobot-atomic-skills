"""Offline replay and synthetic checks for the model-free bottle neck detector."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from chem_demo.neck_vision import (
    detect_wrist_rim,
    compare_wrist_to_hover,
    locate_top_rim,
)


ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "calibration-data" / "chem-demo" / "bottle-observations"
BENCH = CAPTURES / "bench-20261121_164404"
HOVER = CAPTURES / "hover-20261121_170459"
FAST_TOP = ROOT / "calibration-data" / "chem-demo" / "neck-pick" / "preflight-20261121_172110" / "top_fast.png"


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise AssertionError(f"could not read replay image: {path}")
    return image


def draw_blue_ring(image: np.ndarray, center: tuple[int, int], radius: tuple[int, int] = (36, 36)) -> None:
    cv2.ellipse(image, center, radius, 0, 0, 360, (180, 80, 20), -1)
    inner = (round(radius[0] * 0.88), round(radius[1] * 0.88))
    cv2.ellipse(image, center, inner, 0, 0, 360, (235, 235, 235), -1)


# The replayed bench images were taken before the left wrist camera bracket was re-tightened on
# 2026-11-22; they are checked against the reference captured with them.
ORIGINAL_HOVER_REFERENCE = {
    "HOVER_REFERENCE_CENTER_PX": (422.3333549499512, 220.28026962280273),
    "HOVER_REFERENCE_AXES_PX": (66.82316589355469, 62.71836471557617),
    "HOVER_REFERENCE_JAWS_PX": ((297.0952380952381, 128.0), (557.8666666666667, 147.0)),
}
HOVER_REFERENCE_CENTER_PX = ORIGINAL_HOVER_REFERENCE["HOVER_REFERENCE_CENTER_PX"]


class NeckVisionReplayTests(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        import chem_demo.neck_vision as module
        patcher = mock.patch.multiple(module, **ORIGINAL_HOVER_REFERENCE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_visible_bottle_rim_between_closed_pads(self):
        directory = CAPTURES.parent / "neck-pick" / "run-20261121_174904-924f7a2c38"
        if not directory.exists():
            self.skipTest("Local closed-grasp capture unavailable")
        for stem, center in (("05_lift", (431, 236)), ("06_lift", (430, 231))):
            ring = detect_wrist_rim(cv2.imread(str(directory / (stem + ".png"))))
            self.assertTrue(ring.ok, ring.reason)
            self.assertLess(math.dist(ring.rim.center_px, center), 3)

    def test_real_bench_and_hover_top_frames_localize_the_ring_plane(self):
        paths = [BENCH / "top.png", HOVER / "top_before.png", HOVER / "hovered" / "top.png"]
        if FAST_TOP.is_file():
            paths.append(FAST_TOP)
        if not all(path.is_file() for path in paths):
            self.skipTest("bench and hover top-camera replay images are unavailable")
        results = [(path.name, locate_top_rim(read_image(path))) for path in paths]
        for name, result in results:
            with self.subTest(image=name):
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.candidate_count, 1)
                self.assertEqual(result.left_base_xyz_mm[2], 377.49)
                self.assertLess(math.dist(result.left_base_xyz_mm[:2], (265.86, -381.82)), 8.0)
                self.assertGreater(result.seconds, 0.0)
        print("[neck-vision replay] top bench/hover: " + ", ".join(
            f"{name}={result.seconds:.3f}s" for name, result in results))

    def test_real_bench_hover_and_raised_wrist_frames_detect_and_gate(self):
        paths = [BENCH / "left_wrist.png", HOVER / "hovered" / "ref.png", HOVER / "raised" / "ref.png"]
        if not all(path.is_file() for path in paths):
            self.skipTest("bench and hover wrist-camera replay images are unavailable")
        results = [(path.name, detect_wrist_rim(read_image(path))) for path in paths]
        for name, result in results:
            with self.subTest(image=name):
                self.assertTrue(result.ok, result.reason)
                self.assertEqual(result.candidate_count, 1)
                self.assertGreater(result.seconds, 0.0)
        gate = compare_wrist_to_hover(read_image(paths[1]))
        self.assertTrue(gate.ok, gate.reason)
        self.assertEqual(gate.jaw_landmarks.state, "open")
        self.assertLess(gate.center_error_px, 1.0)
        self.assertEqual(gate.jaw_errors_px, (0.0, 0.0))
        print("[neck-vision replay] wrist bench/hover/raised: " + ", ".join(
            f"{name}={result.seconds:.3f}s" for name, result in results))

    def test_shifted_top_and_masked_wrist_images_follow_the_ring(self):
        top_path = FAST_TOP if FAST_TOP.is_file() else BENCH / "top.png"
        wrist_path = HOVER / "hovered" / "ref.png"
        if not top_path.is_file() or not wrist_path.is_file():
            self.skipTest("top and wrist replay images are unavailable")

        top = read_image(top_path)
        dx, dy = 12, -7
        shifted_top = cv2.warpAffine(top, np.float32([[1, 0, dx], [0, 1, dy]]), (1280, 720))
        cv2.rectangle(shifted_top, (752 + dx - 17, 527 + dy - 17),
                      (752 + dx - 13, 527 + dy - 13), (235, 235, 235), -1)
        top_result = locate_top_rim(shifted_top)
        self.assertTrue(top_result.ok, top_result.reason)
        self.assertLess(math.dist(top_result.rim.center_px, (752 + dx, 527 + dy)), 3.0)
        self.assertGreater(math.dist(top_result.left_base_xyz_mm[:2], (265.86, -381.82)), 5.0)

        wrist = read_image(wrist_path)
        cx, cy = (round(value) for value in HOVER_REFERENCE_CENTER_PX)
        cv2.rectangle(wrist, (cx - 27, cy - 27), (cx - 23, cy - 23), (235, 235, 235), -1)
        shifted_wrist = cv2.warpAffine(wrist, np.float32([[1, 0, 13], [0, 1, -8]]), (848, 480))
        wrist_result = detect_wrist_rim(shifted_wrist)
        self.assertTrue(wrist_result.ok, wrist_result.reason)
        self.assertLess(math.dist(wrist_result.rim.center_px,
                                  (HOVER_REFERENCE_CENTER_PX[0] + 13, HOVER_REFERENCE_CENTER_PX[1] - 8)), 2.0)
        print(f"[neck-vision synthetic] top shift+mask={top_result.seconds:.3f}s; "
              f"wrist shift+mask={wrist_result.seconds:.3f}s")

    def test_missing_and_ambiguous_wrist_rims_are_rejected(self):
        blank = np.full((480, 848, 3), 235, np.uint8)
        missing = detect_wrist_rim(blank)
        self.assertFalse(missing.ok)
        self.assertIn("no plausible", missing.reason)

        duplicate = blank.copy()
        draw_blue_ring(duplicate, (300, 230))
        draw_blue_ring(duplicate, (560, 230))
        ambiguous = detect_wrist_rim(duplicate)
        self.assertFalse(ambiguous.ok)
        self.assertIn("multiple plausible", ambiguous.reason)
        self.assertGreaterEqual(ambiguous.candidate_count, 2)

    def test_wrist_hover_gate_rejects_a_displaced_rim_with_unchanged_jaws(self):
        path = HOVER / "hovered" / "ref.png"
        if not path.is_file():
            self.skipTest("accepted hover wrist replay image is unavailable")
        image = read_image(path)
        cv2.circle(image, tuple(round(v) for v in HOVER_REFERENCE_CENTER_PX), 55, (235, 235, 235), -1)
        draw_blue_ring(image, (446, 220))
        gate = compare_wrist_to_hover(image)
        self.assertFalse(gate.ok)
        self.assertIn("rim center", gate.reason)
        self.assertEqual(gate.jaw_errors_px, (0.0, 0.0))

    def test_multiple_top_rims_in_work_area_are_rejected(self):
        image = np.full((720, 1280, 3), 235, np.uint8)
        for center in ((700, 500), (800, 540)):
            cv2.ellipse(image, center, (18, 18), 0, 0, 360, (50, 50, 50), -1)
            cv2.ellipse(image, center, (14, 14), 0, 0, 360, (235, 235, 235), -1)
        result = locate_top_rim(image)
        self.assertFalse(result.ok)
        self.assertIn("multiple plausible", result.reason)
        self.assertEqual(result.candidate_count, 2)

    def test_missing_top_rim_is_rejected(self):
        blank = np.full((720, 1280, 3), 235, np.uint8)
        result = locate_top_rim(blank)
        self.assertFalse(result.ok)
        self.assertIsNone(result.left_base_xyz_mm)
        self.assertIsNone(result.rim)


if __name__ == "__main__":
    unittest.main()
