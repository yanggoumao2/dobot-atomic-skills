"""Focused checks for right-wrist registration of untagged partial checker views."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from chem_demo.check_wrist_board import detect_checkerboard
from chem_demo.right_wrist_grid import (
    register_augmented_checkerboard_sites,
    register_checkerboard_sites,
)


def _render_board(*, camera_x_mm: float, with_label: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a planar 11x8 inner-corner chart plus an optional unique off-grid label."""
    width, height = 540, 360
    square_px = 30
    checker_x, checker_y = 150, 30
    chart = np.full((height, width), 255, np.uint8)
    for row in range(9):
        for col in range(12):
            if (row + col) % 2 == 0:
                x0 = checker_x + col * square_px
                y0 = checker_y + row * square_px
                chart[y0:y0 + square_px, x0:x0 + square_px] = 0
    if with_label:
        cv2.putText(chart, "DP-VISION", (5, 158), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, 0, 2, cv2.LINE_AA)
        cv2.putText(chart, "012-400-30", (5, 198), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, 0, 2, cv2.LINE_AA)
        # A small manufactured logo block supplies repeatable non-text keypoints too.
        cv2.rectangle(chart, (20, 218), (85, 268), 0, 2)
        cv2.circle(chart, (38, 243), 7, 0, 2)
        cv2.line(chart, (54, 224), (75, 260), 0, 2)

    K = np.array([[500.0, 0.0, 424.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    z = 700.0
    H = np.array([[K[0, 0] / z, 0.0,
                   K[0, 2] + K[0, 0] * (camera_x_mm - width / 2) / z],
                  [0.0, K[1, 1] / z,
                   K[1, 2] - K[1, 1] * height / (2 * z)],
                  [0.0, 0.0, 1.0]])
    output_size = (848, 480)
    image = cv2.warpPerspective(chart, H, output_size,
                                flags=cv2.INTER_LINEAR, borderValue=255)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    depth = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                       dtype=np.float32).reshape(1, -1, 2)
    polygon = cv2.perspectiveTransform(corners, H).reshape(-1, 2).round().astype(np.int32)
    cv2.fillConvexPoly(depth, polygon, z)
    return image_bgr, depth, K


class RightWristGridTests(unittest.TestCase):
    @staticmethod
    def _physical_site_ids(pixels: np.ndarray, camera_x_mm: float) -> np.ndarray:
        """Recover synthetic board IDs from the renderer's known pinhole projection."""
        fx = fy = 500.0
        cx, cy = 424.0, 240.0
        z = 700.0
        board_x = (pixels[:, 0] - cx) * z / fx + 540.0 / 2 - camera_x_mm
        board_y = (pixels[:, 1] - cy) * z / fy + 360.0 / 2
        coordinates = np.column_stack(((board_x - 150.0) / 30.0 - 1.0,
                                        (board_y - 30.0) / 30.0 - 1.0))
        sites = np.rint(coordinates).astype(np.int32)
        if np.any(np.linalg.norm(coordinates - sites, axis=1) > 0.15):
            raise AssertionError("synthetic detector corner does not match a known physical site")
        return sites

    def test_unique_off_grid_board_features_register_all_current_corners(self):
        # The reference crops the right edge of the printed grid; the shifted current view sees
        # additional physical columns while retaining the off-grid board label in both views.
        reference_image, reference_depth, K = _render_board(camera_x_mm=430.0, with_label=True)
        current_image, current_depth, _ = _render_board(camera_x_mm=330.0, with_label=True)
        reference_detection = detect_checkerboard(reference_image, K=K, depth_mm=reference_depth)
        current_detection = detect_checkerboard(current_image, K=K, depth_mm=current_depth)

        self.assertTrue(reference_detection["detected"])
        self.assertTrue(current_detection["detected"])
        result = register_checkerboard_sites(
            reference_image, reference_depth, current_image, current_depth, K,
            reference_detection=reference_detection, current_detection=current_detection,
        )

        self.assertEqual(result["status"], "PASS", result)
        sites = np.asarray(result["current_site_ids_in_reference"])
        current_pixels = np.asarray(current_detection["used_corner_pixels"])
        self.assertEqual(sites.shape, (len(current_pixels), 2))
        self.assertTrue(np.issubdtype(sites.dtype, np.integer))
        self.assertEqual(len({tuple(site) for site in sites}), len(sites))
        reference_sites = {tuple(site) for site in reference_detection["used_grid_indices_col_row_local"]}
        self.assertTrue(any(tuple(site) not in reference_sites for site in sites),
                        "current corners beyond the reference patch still receive registered IDs")
        physical_ref = self._physical_site_ids(
            np.asarray(reference_detection["used_corner_pixels"]), camera_x_mm=430.0)
        physical_current = self._physical_site_ids(current_pixels, camera_x_mm=330.0)
        design = np.column_stack((physical_ref, np.ones(len(physical_ref))))
        coefficients, _, _, _ = np.linalg.lstsq(
            design, np.asarray(reference_detection["used_grid_indices_col_row_local"]), rcond=None)
        reference_basis = np.rint(coefficients[:2, :]).astype(np.int32).T
        reference_offset = np.rint(coefficients[2, :]).astype(np.int32)
        expected_sites = physical_current @ reference_basis.T + reference_offset
        np.testing.assert_array_equal(sites, expected_sites)
        self.assertGreaterEqual(result["evidence"]["pnp_inlier_count"], 6)
        self.assertGreaterEqual(len(result["matched_intersections"]), 6)

    def test_checker_only_views_are_rejected_without_off_grid_evidence(self):
        reference_image, reference_depth, K = _render_board(camera_x_mm=0.0, with_label=False)
        current_image, current_depth, _ = _render_board(camera_x_mm=-30.0, with_label=False)
        reference_detection = detect_checkerboard(reference_image, K=K, depth_mm=reference_depth)
        current_detection = detect_checkerboard(current_image, K=K, depth_mm=current_depth)

        result = register_checkerboard_sites(
            reference_image, reference_depth, current_image, current_depth, K,
            reference_detection=reference_detection, current_detection=current_detection,
        )

        self.assertEqual(result["status"], "REJECTED")
        self.assertFalse(result["accepted"])
        self.assertIn("off-grid", result["reason"])
        self.assertEqual(result["current_site_ids_in_reference"], [])

    def test_exclusion_masks_covering_the_only_anchor_cause_rejection(self):
        reference_image, reference_depth, K = _render_board(camera_x_mm=0.0, with_label=True)
        current_image, current_depth, _ = _render_board(camera_x_mm=-28.0, with_label=True)
        reference_detection = detect_checkerboard(reference_image, K=K, depth_mm=reference_depth)
        current_detection = detect_checkerboard(current_image, K=K, depth_mm=current_depth)
        ref_mask = np.zeros(reference_depth.shape, dtype=np.uint8)
        cur_mask = np.zeros(current_depth.shape, dtype=np.uint8)
        # This covers the projected label/logo margin; the remaining texture is checker only.
        ref_mask[190:330, 220:350] = 255
        cur_mask[190:330, 220:350] = 255

        result = register_checkerboard_sites(
            reference_image, reference_depth, current_image, current_depth, K,
            reference_detection=reference_detection, current_detection=current_detection,
            reference_exclude_mask=ref_mask, current_exclude_mask=cur_mask,
        )

        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("off-grid", result["reason"])

    def test_augmented_sites_inherit_only_the_registered_base_chart_d4_phase(self):
        reference_image, reference_depth, K = _render_board(camera_x_mm=430.0, with_label=True)
        current_image, current_depth, _ = _render_board(camera_x_mm=330.0, with_label=True)
        reference_detection = detect_checkerboard(reference_image, K=K, depth_mm=reference_depth)
        current_detection = detect_checkerboard(current_image, K=K, depth_mm=current_depth)

        full_grid = np.asarray(current_detection["used_grid_indices_col_row_local"], dtype=np.int32)
        full_pixels = np.asarray(current_detection["used_corner_pixels"], dtype=np.float64)
        min_col, min_row = full_grid.min(axis=0)
        max_col, max_row = full_grid.max(axis=0)
        base_keep = ((full_grid[:, 0] > min_col) & (full_grid[:, 0] < max_col) &
                     (full_grid[:, 1] > min_row) & (full_grid[:, 1] < max_row))
        self.assertGreaterEqual(int(base_keep.sum()), 30)
        current_base = dict(current_detection)
        current_base["used_grid_indices_col_row_local"] = full_grid[base_keep].tolist()
        current_base["used_corner_pixels"] = full_pixels[base_keep].tolist()
        base_sites = {tuple(site) for site in full_grid[base_keep]}
        added_indices = [index for index, site in enumerate(full_grid) if tuple(site) not in base_sites]
        added_grid = full_grid[added_indices]
        added_pixels = full_pixels[added_indices]
        augmented = dict(current_detection)
        augmented["corner_augmentation"] = {
            "status": "PASS",
            "base_corner_count": int(base_keep.sum()),
            "added_corner_count": int(len(added_grid)),
            "phase_support_count": 12,
            "phase_agreement": 1.0,
            "added_grid_indices_col_row_local": added_grid.tolist(),
            "added_corner_pixels": added_pixels.tolist(),
            "candidates": [
                {"grid_index_col_row_local": site.tolist(), "status": "ACCEPTED",
                 "source": "goodFeaturesToTrack+cornerSubPix",
                 "checker_saddle_contrast": 100.0, "expected_saddle_polarity": 1,
                 "checker_diagonal_pair_error": 0.0,
                 "checker_bidirectional_contrast": 100.0,
                 "grid_neighbor_support_count": 2}
                for site in added_grid
            ],
        }

        fallback_site = augmented["corner_augmentation"]["candidates"][0]
        fallback_site["source"] = "cornerSubPix+corner-response"
        result = register_augmented_checkerboard_sites(
            reference_image, reference_depth, current_image, current_depth, K,
            reference_base_detection=reference_detection,
            current_base_detection=current_base,
            current_augmented_detection=augmented,
        )

        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual(result["phase_extension"]["status"], "PASS")
        self.assertEqual(len(result["current_site_ids_in_reference"]), len(full_grid))
        direct = register_checkerboard_sites(
            reference_image, reference_depth, current_image, current_depth, K,
            reference_detection=reference_detection, current_detection=current_detection,
        )
        self.assertEqual(direct["status"], "PASS", direct)
        np.testing.assert_array_equal(result["current_site_ids_in_reference"],
                                      direct["current_site_ids_in_reference"])

        bad_augmented = dict(augmented)
        bad_sites = [list(site) for site in full_grid]
        bad_sites[added_indices[0]][0] += 1
        bad_augmented["used_grid_indices_col_row_local"] = bad_sites
        rejected = register_augmented_checkerboard_sites(
            reference_image, reference_depth, current_image, current_depth, K,
            reference_base_detection=reference_detection,
            current_base_detection=current_base,
            current_augmented_detection=bad_augmented,
        )
        self.assertEqual(rejected["status"], "REJECTED")

        fallback_site["source"] = "projected-only"
        rejected = register_augmented_checkerboard_sites(
            reference_image, reference_depth, current_image, current_depth, K,
            reference_base_detection=reference_detection, current_base_detection=current_base,
            current_augmented_detection=augmented,
        )
        self.assertFalse(rejected["accepted"])


if __name__ == "__main__":
    unittest.main()
