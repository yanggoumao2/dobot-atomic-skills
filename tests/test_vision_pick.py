"""Cup localisation must recover a known cup from its own projection, not just look plausible.

Needs OpenCV: run with .venv-calibration (python -m unittest tests.test_vision_pick).
"""
import math
import unittest

import cv2
import numpy as np

from tempfile import TemporaryDirectory
from pathlib import Path

from vision_pick.cup_locate import (Camera, CupFit, fit_cup, frustum_silhouette, learn_size, rim_grasp_points,
                                    segment_saturated_cyan, top_ellipse)

TABLE_Z = 240.0


def rig_camera() -> Camera:
    """Roughly the real rig: about 0.8 m above the table, looking down and slightly tilted."""
    K = np.array([[652.0, 0.0, 636.0], [0.0, 651.6, 362.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.05, 0.057, 0.0, 0.0, -0.019])
    tilt = math.radians(20.0)
    # Camera z (optical axis) mostly down (-base z), tilted towards +x.
    z_axis = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    x_axis = np.array([0.0, -1.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = [-500.0, -550.0, 1070.0]
    return Camera(K, D, T)


def render(camera: Camera, params) -> tuple[np.ndarray, np.ndarray]:
    cup = np.zeros((720, 1280), np.uint8)
    opening = np.zeros_like(cup)
    cv2.fillPoly(cup, [np.round(frustum_silhouette(camera, params, TABLE_Z)).astype(np.int32)], 1)
    cv2.fillPoly(opening, [np.round(top_ellipse(camera, params, TABLE_Z)).astype(np.int32)], 1)
    return opening, cup


class CupLocateTests(unittest.TestCase):
    def test_fit_recovers_position_height_and_radii_without_dimensions(self):
        camera = rig_camera()
        for truth in ([-130.0, -530.0, 98.0, 39.0, 32.0], [-300.0, -480.0, 70.0, 30.0, 30.0]):
            opening, cup = render(camera, truth)
            fit = fit_cup(camera, opening, cup, TABLE_Z)
            self.assertLess(math.hypot(fit.x - truth[0], fit.y - truth[1]), 1.5, fit)
            self.assertLess(abs(fit.height - truth[2]), 4.0, fit)
            self.assertLess(abs(fit.top_radius - truth[3]), 1.5, fit)

    def test_rim_point_lies_on_the_jaw_axis_on_the_side_nearer_the_base(self):
        fit = CupFit(-127.8, -532.0, TABLE_Z, 97.7, 38.9, 31.8, 0.95)
        grasp = rim_grasp_points(fit, rpy=(180.0, 0.0, 90.0))
        x, y, z = grasp["rim_point"]
        self.assertAlmostEqual(y, -532.0, places=1)           # jaws close along base X here
        self.assertAlmostEqual(x, -127.8 + 37.9, places=1)    # the +X side is nearer the base
        self.assertAlmostEqual(z, TABLE_Z + 97.7, places=1)
        self.assertAlmostEqual(grasp["hover_point"][2], z + 30.0, places=1)
        turned = rim_grasp_points(fit, rpy=(180.0, 0.0, 0.0))  # jaws along base Y
        self.assertAlmostEqual(turned["rim_point"][0], -127.8, places=1)
        self.assertGreater(turned["rim_point"][1], -532.0)    # the side towards the base


    def test_seed_is_the_most_saturated_blob_inside_the_workspace_not_the_largest(self):
        # 2026-11-04: exposure drift made a large arm cover (S~170) pass the seed threshold.
        image = np.full((720, 1280, 3), 200, np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[200:360, 1100:1270] = (100, 170, 80)            # big, moderately saturated "arm"
        cv2.ellipse(hsv, (620, 440), (30, 25), 0, 0, 360, (95, 255, 60), -1)  # small, deeply saturated "cup"
        cv2.ellipse(hsv, (100, 100), (30, 25), 0, 0, 360, (95, 255, 60), -1)  # saturated, but out of reach
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        opening, _ = segment_saturated_cyan(image, accept=lambda u, v: u > 300)
        ys, xs = np.nonzero(opening)
        self.assertAlmostEqual(xs.mean(), 620, delta=2)
        self.assertAlmostEqual(ys.mean(), 440, delta=2)

    def test_opening_is_the_darker_inside_even_when_the_wall_is_as_saturated(self):
        # 2026-11-04: near the arm the lit wall was as saturated as the inside (opening = whole cup),
        # and elsewhere the shadowed inside bottom was darker than the seed's V>=25 (a bite).
        image = np.full((720, 1280, 3), 200, np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.ellipse(hsv, (620, 470), (40, 34), 0, 0, 360, (95, 230, 150), -1)  # bottom of the lit wall
        cv2.rectangle(hsv, (580, 440), (660, 470), (95, 230, 150), -1)          # lit, saturated wall
        cv2.ellipse(hsv, (620, 440), (40, 32), 0, 0, 360, (95, 255, 55), -1)    # shaded inside = opening
        cv2.ellipse(hsv, (605, 450), (20, 17), 0, 0, 360, (95, 255, 15), -1)    # very dark inside bottom
        opening, cup = segment_saturated_cyan(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        full = np.zeros(opening.shape, np.uint8)
        cv2.ellipse(full, (620, 440), (40, 32), 0, 0, 360, 1, -1)
        overlap = np.logical_and(opening, full).sum() / np.logical_or(opening, full).sum()
        self.assertGreater(overlap, 0.9)
        self.assertGreater(cup.sum(), opening.sum() * 1.3)  # the wall belongs to the cup, not the opening

    def test_size_is_the_median_of_good_fits_one_vote_per_photo(self):
        with TemporaryDirectory() as directory:
            memory = Path(directory) / "cup.json"
            learn_size(memory, CupFit(0, 0, 0, 97.7, 39.0, 32.0, 0.95), Path("a.png"))
            learn_size(memory, CupFit(0, 0, 0, 97.7, 39.0, 32.0, 0.95), Path("a.png"))  # same photo again
            learn_size(memory, CupFit(0, 0, 0, 40.0, 20.0, 20.0, 0.60), Path("bad.png"))  # poor fit: ignored
            size = learn_size(memory, CupFit(0, 0, 0, 88.7, 40.0, 38.0, 0.94), Path("b.png"))
        self.assertEqual(size["n"], 2)
        self.assertAlmostEqual(size["height"], (97.7 + 88.7) / 2, places=1)


    def test_put_back_error_compares_both_photos_at_the_same_rim_height(self):
        from vision_pick.viewer import put_back_error

        camera = rig_camera()
        rim_z = TABLE_Z + 90.0
        start_px = camera.project(np.array([[-120.0, -530.0, rim_z]]))[0]
        end_px = camera.project(np.array([[-117.0, -534.0, rim_z]]))[0]
        error = put_back_error(camera, {"rim_pixel": list(start_px)}, {"rim_pixel": list(end_px), "rim_z": rim_z})
        self.assertAlmostEqual(error["dx"], 3.0, delta=0.05)
        self.assertAlmostEqual(error["dy"], -4.0, delta=0.05)
        self.assertAlmostEqual(error["distance_mm"], 5.0, delta=0.05)

    def test_live_outline_finds_the_cup_and_tolerates_its_absence(self):
        from vision_pick.viewer import live_outline

        camera = rig_camera()
        params = [-130.0, -530.0, 90.0, 39.0, 35.0]
        opening, cup = render(camera, params)
        image = np.full((720, 1280, 3), 200, np.uint8)
        image[cup > 0] = (160, 150, 60)       # paler wall
        image[opening > 0] = (70, 60, 0)      # deeply saturated inside
        contours = live_outline(image, camera, TABLE_Z)
        self.assertTrue(contours)
        self.assertIsNone(live_outline(np.full((720, 1280, 3), 200, np.uint8), camera, TABLE_Z))


    def test_tracking_finds_the_cup_and_flags_a_covered_rim(self):
        from vision_pick.viewer import track_cup

        camera = rig_camera()
        truth = [-130.0, -530.0, 90.0, 39.0, 35.0]
        opening, cup = render(camera, truth)
        image = np.full((720, 1280, 3), 200, np.uint8)
        image[cup > 0] = (160, 150, 60)       # lit wall
        image[opening > 0] = (70, 60, 0)      # shaded inside
        seen = track_cup(image, camera, TABLE_Z, 90.0, 39.0)
        self.assertTrue(seen["ok"], seen)
        self.assertLess(math.hypot(seen["xy"][0] - truth[0], seen["xy"][1] - truth[1]), 1.5)
        ys, xs = np.nonzero(opening)
        covered = image.copy()
        covered[:, : int(xs.mean())] = 200    # a hand over half of the cup
        hidden = track_cup(covered, camera, TABLE_Z, 90.0, 39.0)
        self.assertFalse(hidden["ok"], hidden)


    def test_gripper_over_the_cup_is_flagged_but_beside_it_is_not(self):
        from vision_pick.viewer import arm_overlap, cup_footprint

        camera = rig_camera()
        cup = cup_footprint(camera, -130.0, -530.0, TABLE_Z, 90.0, 39.0, 35.0)
        above = arm_overlap(camera, [-91.0, -530.0, TABLE_Z + 90.0 + 60.0], cup, (720, 1280))
        # Beside the cup on the side away from this camera's foot point (-500, -550).
        away = np.array([-130.0 + 500.0, -530.0 + 550.0])
        away = away / np.linalg.norm(away) * 110.0
        beside = arm_overlap(camera, [-130.0 + away[0], -530.0 + away[1], TABLE_Z + 90.0 + 60.0], cup, (720, 1280))
        self.assertGreater(above, 0.05)
        self.assertLess(beside, 0.02)
        self.assertEqual(arm_overlap(camera, None, cup, (720, 1280)), 0.0)


if __name__ == "__main__":
    unittest.main()
