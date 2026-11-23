"""Checks for measured per-capture checker site augmentation."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from chem_demo import wrist_checker_sites
from chem_demo.wrist_checker_sites import augment_right_capture


def _synthetic_patch(*, occluded_site: tuple[int, int] | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Render a front-facing checkerboard and expose only a 5 by 7 measured patch."""
    width, height = 640, 480
    origin_x, origin_y = 100.0, 60.0
    spacing_px = 26.25
    square_mm = 30.0
    depth_mm_value = 800.0
    image = np.full((height, width), 245, dtype=np.uint8)
    board_mask = np.zeros((height, width), dtype=np.uint8)
    for row in range(9):
        for col in range(12):
            if (row + col) % 2 == 0:
                polygon = np.rint([
                    [origin_x + col * spacing_px, origin_y + row * spacing_px],
                    [origin_x + (col + 1) * spacing_px, origin_y + row * spacing_px],
                    [origin_x + (col + 1) * spacing_px, origin_y + (row + 1) * spacing_px],
                    [origin_x + col * spacing_px, origin_y + (row + 1) * spacing_px],
                ]).astype(np.int32)
                cv2.fillConvexPoly(image, polygon, 18, lineType=cv2.LINE_AA)
    board_polygon = np.rint([
        [origin_x, origin_y], [origin_x + 12 * spacing_px, origin_y],
        [origin_x + 12 * spacing_px, origin_y + 9 * spacing_px],
        [origin_x, origin_y + 9 * spacing_px],
    ]).astype(np.int32)
    cv2.fillConvexPoly(board_mask, board_polygon, 1)
    depth = board_mask.astype(np.float32) * depth_mm_value

    grid = np.stack(np.meshgrid(np.arange(5), np.arange(7)), axis=-1).reshape(-1, 2)
    # The exposed patch starts at physical checker index (2, 1), leaving real adjacent corners
    # on its left and right while the top/bottom proposals land at the printed board border.
    pixels = np.column_stack((origin_x + (grid[:, 0] + 2) * spacing_px,
                              origin_y + (grid[:, 1] + 1) * spacing_px)).astype(np.float32)
    corners = pixels.reshape(-1, 1, 2).copy()
    cv2.cornerSubPix(image, corners, (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    pixels = corners.reshape(-1, 2).astype(np.float64)
    camera = np.array([[700.0, 0.0, width / 2], [0.0, 700.0, height / 2],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    detection: dict[str, object] = {
        "detected": True,
        "board": {"internal_corners": [11, 8], "square_mm": square_mm,
                  "detected_grid_shape": [5, 7], "full_board_detected": False},
        "corner_count": len(pixels),
        "used_corner_count": len(pixels),
        "used_corner_pixels": pixels.tolist(),
        "used_grid_indices_col_row_local": grid.astype(int).tolist(),
        "metric_pnp": {"status": "NOT_RUN"},
    }
    if occluded_site is not None:
        col, row = occluded_site
        x = int(round(origin_x + (col + 2) * spacing_px))
        y = int(round(origin_y + (row + 1) * spacing_px))
        cv2.rectangle(image, (x - 10, y - 10), (x + 10, y + 10), 0, -1)
        # A foreground jaw has valid depth, but it is not on the checker plane.
        cv2.rectangle(depth, (x - 10, y - 10), (x + 10, y + 10), 500.0, -1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), depth, camera, detection


class WristCheckerSiteTests(unittest.TestCase):
    def test_adds_only_subpixel_measured_saddles_with_camera_pnp_support(self):
        image, depth, camera, detection = _synthetic_patch()
        result = augment_right_capture(image, depth, camera, detection)

        self.assertEqual(result["corner_augmentation"]["status"], "PASS")
        self.assertGreaterEqual(result["corner_augmentation"]["added_corner_count"], 10)
        self.assertEqual(result["metric_pnp"]["status"], "PASS")
        self.assertLess(result["metric_pnp"]["rms_px"], 0.5)
        self.assertTrue(result["board"]["full_board_detected"])
        pixels = np.asarray(result["used_corner_pixels"], dtype=np.float64)
        grid = np.asarray(result["used_grid_indices_col_row_local"], dtype=np.int32)
        self.assertEqual(len(pixels), len(grid))
        self.assertEqual(len({tuple(site) for site in grid}), len(grid))
        self.assertEqual(len(np.unique(np.round(pixels, 1), axis=0)), len(pixels))

        added = result["corner_augmentation"]["candidates"]
        accepted = [candidate for candidate in added if candidate["status"] == "ACCEPTED"]
        self.assertTrue(all(candidate["source"] in {
            "goodFeaturesToTrack+cornerSubPix", "cornerSubPix+corner-response"
        } for candidate in accepted))
        self.assertTrue(all(candidate["grid_neighbor_support_count"] >= 1
                            for candidate in accepted))
        # The one-cell top extension meets the printed board's outer white border. It must not be
        # counted as a measured internal corner even if the border has a strong edge.
        accepted_sites = {tuple(candidate["grid_index_col_row_local"]) for candidate in accepted}
        self.assertFalse(any(row < 0 for _, row in accepted_sites))

    def test_foreground_jaw_with_valid_depth_is_excluded_from_checker_sites(self):
        jaw_site = (5, 3)
        image, depth, camera, detection = _synthetic_patch(occluded_site=jaw_site)
        mask = np.zeros(depth.shape, dtype=np.uint8)
        x = int(round(100.0 + (jaw_site[0] + 2) * 26.25))
        y = int(round(60.0 + (jaw_site[1] + 1) * 26.25))
        cv2.rectangle(mask, (x - 11, y - 11), (x + 11, y + 11), 255, -1)

        result = augment_right_capture(image, depth, camera, detection, exclude_mask=mask)

        used_sites = {tuple(site) for site in result["used_grid_indices_col_row_local"]}
        self.assertNotIn(jaw_site, used_sites)
        candidate = next(row for row in result["corner_augmentation"]["candidates"]
                         if tuple(row["grid_index_col_row_local"]) == jaw_site)
        self.assertEqual(candidate["status"], "REJECTED")

    def test_single_and_multi_sample_radii_keep_same_border_rejection(self):
        image, depth, camera, detection = _synthetic_patch()
        original_offsets = wrist_checker_sites.QUADRANT_SAMPLE_OFFSETS
        accepted_sets = []
        try:
            for offsets in ((0.16,), (0.12, 0.14, 0.16, 0.18)):
                wrist_checker_sites.QUADRANT_SAMPLE_OFFSETS = offsets
                result = augment_right_capture(image, depth, camera, detection)
                accepted = [candidate for candidate in result["corner_augmentation"]["candidates"]
                            if candidate["status"] == "ACCEPTED"]
                accepted_sets.append({tuple(candidate["grid_index_col_row_local"])
                                      for candidate in accepted})
                self.assertEqual(result["metric_pnp"]["status"], "PASS")
                self.assertFalse(any(row < 0 for col, row in accepted_sets[-1]))
        finally:
            wrist_checker_sites.QUADRANT_SAMPLE_OFFSETS = original_offsets

        self.assertEqual(accepted_sets[0], accepted_sets[1])

    def test_remeasures_rejected_raw_site_without_erasing_its_chart_flag(self):
        image, depth, camera, detection = _synthetic_patch()
        raw_grid = np.asarray(detection["used_grid_indices_col_row_local"], dtype=np.int32)
        raw_pixels = np.asarray(detection["used_corner_pixels"], dtype=np.float64)
        rejected_site = (2, 3)
        rejected_index = next(index for index, site in enumerate(raw_grid)
                              if tuple(site) == rejected_site)
        raw_object_points = np.column_stack((raw_grid[:, 0] * 30.0,
                                             raw_grid[:, 1] * 30.0,
                                             np.zeros(len(raw_grid)))).tolist()
        chart_mask = [True] * len(raw_grid)
        chart_mask[rejected_index] = False
        detection.update({
            "grid_indices_col_row_local": raw_grid.tolist(),
            "corner_pixels": raw_pixels.tolist(),
            "object_points_local_mm": raw_object_points,
            "chart_inlier_mask": chart_mask,
            "chart_residual_px": [0.0] * len(raw_grid),
            "chart_filter": {"status": "PASS", "inlier_count": len(raw_grid) - 1,
                             "outlier_count": 1},
            "used_corner_pixels": np.delete(raw_pixels, rejected_index, axis=0).tolist(),
            "used_grid_indices_col_row_local": np.delete(raw_grid, rejected_index, axis=0).tolist(),
            "corner_count": len(raw_grid),
            "used_corner_count": len(raw_grid) - 1,
        })

        result = augment_right_capture(image, depth, camera, detection)

        self.assertEqual(result["chart_inlier_mask"], chart_mask)
        self.assertEqual(result["grid_indices_col_row_local"], raw_grid.tolist())
        self.assertEqual(result["chart_filter"]["outlier_count"], 1)
        used_sites = [tuple(site) for site in result["used_grid_indices_col_row_local"]]
        self.assertEqual(used_sites.count(rejected_site), 1)
        candidate = next(row for row in result["corner_augmentation"]["candidates"]
                         if tuple(row["grid_index_col_row_local"]) == rejected_site)
        self.assertEqual(candidate["status"], "ACCEPTED")


if __name__ == "__main__":
    unittest.main()
