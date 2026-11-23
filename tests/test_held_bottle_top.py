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


if __name__ == "__main__":
    unittest.main()
