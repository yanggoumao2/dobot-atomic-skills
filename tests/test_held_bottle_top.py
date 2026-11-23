"""Top-camera fit of the bottle hanging from the left jaws (synthetic masks, real camera model)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from chem_demo import gripper_geometry as gg
from chem_demo.held_bottle_top import (BOTTLE_HEIGHT_MM, TIPS_ABOVE_BOTTOM_MM, _fill, _silhouette,
                                       fit_hanging_cylinder, predicted_hang)
from vision_pick.cup_locate import DEFAULT_SESSION

try:
    from chem_demo.camera_drift import load_corrected_camera
    CAMERA = load_corrected_camera(DEFAULT_SESSION, "left")
except Exception:  # noqa: BLE001 - calibration data not present
    CAMERA = None

POSE = [52.09, -295.44, 610.0, -180.0, 0.0, -90.0]   # the stored handoff station


class GeometryTest(unittest.TestCase):
    def test_hang_numbers_follow_from_the_commissioned_grasp(self):
        hang = gg.bottle_hang(120.0)
        self.assertAlmostEqual(hang["tips_above_bottom_mm"], 358.86 - 243.2)
        self.assertAlmostEqual(hang["bottom_below_flange_mm"], 115.66 + 120.0)
        self.assertAlmostEqual(hang["grasp_drop_mm"], 115.66 + 120.0 - 40.0)
        self.assertAlmostEqual(hang["total_height_mm"], 377.49 - 243.2)
        self.assertEqual(gg.pad_offset_flange(120.0), (3.0, 0.0, 104.0))
        self.assertAlmostEqual(gg.table_z_real("right", 120.0), 120.18)

    def test_prediction_is_in_the_fingertip_convention(self):
        xy, bottom = predicted_hang(POSE)
        np.testing.assert_allclose(xy, POSE[:2])
        self.assertAlmostEqual(bottom, 610.0 - TIPS_ABOVE_BOTTOM_MM)


@unittest.skipIf(CAMERA is None, "top-camera calibration not present")
class FitTest(unittest.TestCase):
    def render(self, x, y, bottom, hide_top_px=0):
        mask = _fill(_silhouette(CAMERA, x, y, bottom), (720, 1280))
        if hide_top_px:   # the jaws hide the upper part of the outline
            ys = np.nonzero(mask)[0]
            mask[:ys.min() + hide_top_px] = 0
        return mask

    def test_recovers_a_shifted_hang(self):
        xy, bottom = predicted_hang(POSE)
        truth = np.array([xy[0] + 8.0, xy[1] - 6.0, bottom - 12.0])
        params, rms, _ = fit_hanging_cylinder(CAMERA, self.render(*truth), [xy[0], xy[1], bottom])
        np.testing.assert_allclose(params, truth, atol=4.0)   # 1 px of outline ~ 3 mm along the ray here
        self.assertLess(rms, 1.5)

    def test_occluded_top_does_not_pull_the_fit_along_the_ray(self):
        xy, bottom = predicted_hang(POSE)
        truth = np.array([xy[0], xy[1], bottom])
        mask = self.render(*truth, hide_top_px=40)
        params, _, _ = fit_hanging_cylinder(CAMERA, mask, [xy[0] + 10, xy[1] + 10, bottom + 15])
        np.testing.assert_allclose(params, truth, atol=4.0)

    def test_station_bottle_is_inside_the_top_image(self):
        xy, bottom = predicted_hang(POSE)
        outline = _silhouette(CAMERA, xy[0], xy[1], bottom)
        self.assertTrue(((outline[:, 0] > 0) & (outline[:, 0] < 1280) & (outline[:, 1] > 0) &
                         (outline[:, 1] < 720)).all())


@unittest.skipIf(CAMERA is None, "top-camera calibration not present")
class RayGateTest(unittest.TestCase):
    """Real hang checks 2026-11-22/23 (handoff_cycle.json): the offset lies along the camera ray."""

    CASES = [  # predicted axis xy, predicted bottom, measured xy offset, bottom offset
        ((0.0, -380.0), 454.34, (21.70, -6.36), 36.35),
        ((129.97, -357.08), 454.34, (14.07, -4.98), 30.95),
        ((72.93, -413.62), 424.34, (13.57, -2.90), 28.86),
        ((72.93, -413.62), 424.34, (30.45, -7.32), 54.90),   # left beaker behind the bottle; refused before
    ]

    def test_real_offsets_are_mostly_along_the_ray(self):
        from chem_demo.held_bottle_top import MAX_ACROSS_RAY_MM, MAX_ALONG_RAY_MM, split_along_ray
        for xy, bottom, dxy, dz in self.CASES:
            along, across = split_along_ray(CAMERA, xy, bottom, [*dxy, dz])
            self.assertLess(float(np.linalg.norm(across)), 12.0)
            self.assertLess(float(np.linalg.norm(across)), MAX_ACROSS_RAY_MM)
            self.assertLess(abs(along), MAX_ALONG_RAY_MM)

    def test_a_sideways_slip_is_still_caught(self):
        from chem_demo.held_bottle_top import MAX_ACROSS_RAY_MM, split_along_ray
        # A bottle 40 mm off sideways (horizontal, across the ray) is refused. A straight drop is mostly
        # ALONG this ray, so this gate cannot see it: the jaw reading and the right wrist's first-look
        # height gate (handoff_cycle, 50 mm) cover that case.
        xy, bottom = (72.93, -413.62), 424.34
        centre = np.array([xy[0], xy[1], bottom + 67.0])
        ray = centre - CAMERA.center
        side = np.cross(ray, [0.0, 0.0, 1.0])
        side = 40.0 * side / np.linalg.norm(side)
        _, across = split_along_ray(CAMERA, xy, bottom, side)
        self.assertGreater(float(np.linalg.norm(across)), MAX_ACROSS_RAY_MM)


if __name__ == "__main__":
    unittest.main()
