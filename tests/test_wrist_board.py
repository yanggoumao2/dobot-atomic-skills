"""Offline checks for the plain-board wrist-camera fallback."""
from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from chem_demo import check_wrist_board as check


class WristBoardTests(unittest.TestCase):
    @staticmethod
    def checker_image(cols: int = 11, rows: int = 8, pixels_per_square: int = 50) -> np.ndarray:
        margin = 30
        image = np.full((rows * pixels_per_square + 2 * margin,
                         cols * pixels_per_square + 2 * margin), 255, np.uint8)
        for row in range(rows + 1):
            for col in range(cols + 1):
                value = 0 if (row + col) % 2 else 255
                y0, x0 = margin + row * pixels_per_square, margin + col * pixels_per_square
                image[y0:y0 + pixels_per_square, x0:x0 + pixels_per_square] = value
        return image

    def test_full_board_reports_repeatable_coverage_local_corners_and_pnp(self):
        image = self.checker_image()
        K = np.array([[1000.0, 0, 80.0], [0, 1000.0, 80.0], [0, 0, 1.0]])
        depth = np.full(image.shape, 600.0, np.float32)

        first = check.detect_checkerboard(image, K=K, depth_mm=depth)
        second = check.detect_checkerboard(image, K=K, depth_mm=depth)

        self.assertTrue(first["detected"])
        self.assertEqual(first["corner_count"], 11 * 8)
        self.assertTrue(first["board"]["full_board_detected"])
        self.assertTrue(first["orientation_ambiguous"])
        self.assertIn("no absolute board IDs", first["coordinate_convention"])
        self.assertEqual(first["coverage"], second["coverage"])
        self.assertEqual(first["corner_count"], second["corner_count"])
        self.assertEqual(first["metric_pnp"]["status"], "PASS")
        self.assertAlmostEqual(first["metric_pnp"]["tvec_mm_local_origin_in_camera"][2], 600.0, delta=1.0)
        self.assertEqual(first["stereo_check"]["status"], "PASS")
        self.assertFalse(first["calibration_accepted"])

    def test_partial_board_is_local_and_keeps_ambiguity_explicit(self):
        full = self.checker_image()
        partial = full[80:340, 75:475]

        result = check.detect_checkerboard(partial, 11, 8, 30.0)

        self.assertTrue(result["detected"])
        self.assertFalse(result["board"]["full_board_detected"])
        self.assertLess(result["corner_count"], 11 * 8)
        self.assertTrue(result["orientation_ambiguous"])
        self.assertEqual(result["metric_pnp"]["status"], "NOT_RUN")
        self.assertEqual(result["stereo_check"]["status"], "NOT_RUN")

    def test_stereo_failure_does_not_turn_pnp_into_calibration_acceptance(self):
        image = self.checker_image()
        K = np.array([[1000.0, 0, 80.0], [0, 1000.0, 80.0], [0, 0, 1.0]])
        result = check.detect_checkerboard(image, K=K,
                                           depth_mm=np.full(image.shape, 650.0, np.float32))

        self.assertEqual(result["metric_pnp"]["status"], "PASS")
        self.assertEqual(result["stereo_check"]["status"], "FAIL")
        self.assertFalse(result["calibration_accepted"])

    def test_cli_reads_metadata_depth_and_writes_only_diagnostic_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            image_path = folder / "snapshot.png"
            metadata_path = folder / "capture.json"
            depth_path = folder / "depth.npy"
            out_path = folder / "diagnostic.json"
            image = self.checker_image()
            cv2.imwrite(str(image_path), image)
            metadata_path.write_text(json.dumps({"camera": {"intrinsics": {
                "matrix_px": [[1000, 0, 80], [0, 1000, 80], [0, 0, 1]],
                "distortion_coefficients": [0, 0, 0, 0, 0],
                "distortion_removed_from_output": True}}}), encoding="utf-8")
            np.save(depth_path, np.full(image.shape, 600.0, np.float32), allow_pickle=False)

            with redirect_stdout(StringIO()):
                code = check.main([str(image_path), "--metadata", str(metadata_path),
                                   "--depth", str(depth_path), "--out", str(out_path)])
            self.assertEqual(code, 0)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["corner_count"], 88)
            self.assertFalse(report["calibration_accepted"])
            self.assertFalse((folder / "left_wrist.json").exists())

    def test_cli_refuses_active_calibration_output_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            image_path = folder / "snapshot.png"
            cv2.imwrite(str(image_path), self.checker_image())
            active_path = folder / "left_wrist.json"
            active_path.write_text("sentinel", encoding="utf-8")
            alias = folder / "LEFT_WRIST.JSON"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                check.main([str(image_path), "--out", str(alias)])
            self.assertEqual(active_path.read_text(encoding="utf-8"), "sentinel")

    def test_invalid_board_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive finite"):
            check.detect_checkerboard(np.zeros((100, 100), np.uint8), square_mm=0)

    def test_depth_must_be_pixel_aligned_with_color(self):
        image = np.zeros((120, 160), np.uint8)
        depth = np.ones((60, 80), np.float32)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            check.detect_checkerboard(image, depth_mm=depth)

    def test_low_support_larger_hypothesis_cannot_hide_valid_smaller_chart(self):
        image = np.zeros((600, 600), np.uint8)
        K = np.array([[1000.0, 0, 300.0], [0, 1000.0, 300.0], [0, 0, 1.0]])

        def candidate(cols, rows, corrupt=0):
            grid = np.stack(np.meshgrid(np.arange(cols), np.arange(rows)), axis=-1).reshape(-1, 2)
            object_points = np.column_stack((grid * 30.0, np.zeros(len(grid))))
            pixels = np.column_stack((
                K[0, 0] * (object_points[:, 0] - 60.0) / 600.0 + K[0, 2],
                K[1, 1] * (object_points[:, 1] - 60.0) / 600.0 + K[1, 2],
            ))
            if corrupt:
                rng = np.random.default_rng(8)
                pixels[:corrupt] += rng.normal(0, 80, size=(corrupt, 2))
            return cols, rows, pixels.astype(np.float32), "synthetic"

        misleading = [candidate(8, 7, corrupt=30)]
        coherent = candidate(6, 6)
        with mock.patch.object(check, "_detect_grids",
                               side_effect=[misleading, misleading + [coherent]]) as detect:
            result = check.detect_checkerboard(image, 11, 8, 30.0, K=K)

        self.assertEqual(detect.call_count, 2)
        self.assertEqual(result["board"]["detected_grid_shape"], [6, 6])
        self.assertEqual(result["flat_m_total"], 36)
        self.assertEqual(result["flat_m_used"], 36)
        self.assertTrue(result["handeye_support"]["passed"])
        self.assertEqual(result["metric_pnp"]["status"], "PASS")
        self.assertFalse(result["calibration_accepted"])


if __name__ == "__main__":
    unittest.main()
