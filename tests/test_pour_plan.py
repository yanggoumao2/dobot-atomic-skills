"""Offline checks for the pour planner and the full-pose Nova 2 kinematics it relies on."""

from __future__ import annotations

import math
import unittest

import numpy as np

from chem_demo.handoff_plan import HandoffConfig, _rot
from chem_demo.nova2_model import flange_matrix, fk_pose, ik_numeric, link_points
from chem_demo.pour_plan import (
    BOTTLE_HEIGHT_MM,
    GRASP_ABOVE_BOTTOM_MM,
    PourConfig,
    PourGeometry,
    PourInfeasible,
    Vessel,
    plan_pour,
)

# Controller InverseSolution answers recorded in run-20261123_172644 (preflight_plan.json).
CONTROLLER = [
    ([-137.5107, -545.0587, 314.3079], [0.0, 90.0, -84.82899242444398],
     [-93.469742, -73.45905, -45.47673, 119.036304, 98.650448, 90.059541]),
    ([-153.7338, -365.7912, 314.3079], [0.0, 90.0, -84.82899242444398],
     [-98.740467, -50.169026, -107.791222, 158.066084, 103.932599, 90.066224]),
]
HOLD_XYZ = [-141.7327, -523.7518, 276.5405]          # right flange after the 17:30 handoff down-check
HOLD_RPY = [0.0, 90.0, -84.82899242444398]
WRIST_UP = [-94.18, -52.07, -37.67, -90.26, -99.35, -90.0]
WRIST_DOWN = [-94.18, -74.75, -53.9, 128.65, 99.35, 90.0]
L = 180.0
BEAKER = Vessel("left_beaker", 15.0, -691.7, 23.7, 339.73 - L, 420.73 - L)   # 11-21 top-camera fit
OBSTACLES = [Vessel("centre_beaker", -155.7, -520.0, 37.8, 240.18 - L, 329.39 - L),
             Vessel("right_beaker", 30.1, -325.9, 25.2, 339.73 - L, 419.97 - L),
             Vessel("right_balance", 30.1, -325.9, 110.0, 240.18 - L, 339.73 - L)]


def angle_between(R1, R2) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R1.T @ R2)) - 1.0) / 2.0))))


class KinematicsTest(unittest.TestCase):
    def test_fk_matches_controller(self):
        for xyz, rpy, joints in CONTROLLER:
            pose = fk_pose(joints)
            self.assertLess(max(abs(a - b) for a, b in zip(pose["xyz"], xyz)), 0.5)
            self.assertLess(angle_between(flange_matrix(joints), _rot(rpy)), 0.15)  # TOOL_AT_ZERO is rounded

    def test_ik_returns_the_controller_branch_from_a_nearby_seed(self):
        for xyz, rpy, joints in CONTROLLER:
            seed = [j + 4.0 for j in joints]
            q = ik_numeric(xyz, rpy, seed)
            self.assertLess(max(abs(math.remainder(a - b, 360.0)) for a, b in zip(q, joints)), 0.5)

    def test_ik_never_accepts_an_orientation_off_by_180_degrees(self):
        # The first version measured the orientation error with the skew part only, which is zero at
        # 180 deg, and "converged" onto flipped tools from random seeds.
        rng = np.random.default_rng(3)
        target = _rot(HOLD_RPY)
        for _ in range(25):
            try:
                q = ik_numeric(HOLD_XYZ, HOLD_RPY, rng.uniform(-180.0, 180.0, 6).tolist(), iterations=80)
            except RuntimeError:
                continue
            self.assertLess(angle_between(flange_matrix(q), target), 0.05)

    def test_both_wrist_branches_hold_the_same_pose(self):
        for seed in (WRIST_UP, WRIST_DOWN):
            q = ik_numeric(HOLD_XYZ, HOLD_RPY, seed)
            self.assertLess(angle_between(flange_matrix(q), _rot(HOLD_RPY)), 0.05)
        up = link_points(ik_numeric(HOLD_XYZ, HOLD_RPY, WRIST_UP))["wrist1"][2]
        down = link_points(ik_numeric(HOLD_XYZ, HOLD_RPY, WRIST_DOWN))["wrist1"][2]
        self.assertGreater(up, HOLD_XYZ[2] + 100.0)
        self.assertLess(down, HOLD_XYZ[2] - 100.0)


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.geo = PourGeometry(HOLD_XYZ, HOLD_RPY, BEAKER)

    def test_tilt_axis_is_the_tool_axis_and_pour_faces_the_beaker(self):
        self.assertGreater(abs(float(self.geo.k @ self.geo.n)), 0.999)
        to_beaker = np.array([BEAKER.x, BEAKER.y, 0.0]) - np.array([*self.geo.grasp[:2], 0.0])
        self.assertGreater(float(self.geo.d @ to_beaker), 0.0)

    def test_zero_tilt_at_the_hold_lip_is_the_hold_pose(self):
        p, rpy, _ = self.geo.pose(0.0, self.geo.lip_hold)
        np.testing.assert_allclose(p, HOLD_XYZ, atol=1e-6)
        self.assertLess(angle_between(_rot(rpy), _rot(HOLD_RPY)), 1e-6)

    def test_lip_stays_put_while_tilting(self):
        lip = np.array([10.0, -692.0, 270.0])
        for theta in (0.0, 45.0, 90.0, 120.0):
            p, rpy, _ = self.geo.pose(theta, lip)
            # the lip, carried rigidly with the flange, must land back on the same point
            R = _rot(rpy)
            lip_in_flange = _rot(HOLD_RPY).T @ (self.geo.lip_hold - np.asarray(HOLD_XYZ))
            np.testing.assert_allclose(p + R @ lip_in_flange, lip, atol=1e-6)
            # and the bottle mouth top ring stays around it (the lip is the lowest-reaching point of the neck)
            pts = self.geo.bottle_points(theta, lip)
            self.assertLess(float(np.min(np.linalg.norm(pts - lip, axis=1))), 1e-6)

    def test_bottle_is_upright_and_hangs_below_the_grasp_at_rest(self):
        pts = self.geo.bottle_points(0.0, self.geo.lip_hold)
        self.assertAlmostEqual(float(pts[:, 2].min()), self.geo.grasp[2] - GRASP_ABOVE_BOTTOM_MM, places=6)
        self.assertAlmostEqual(float(pts[:, 2].max()), self.geo.grasp[2] + BOTTLE_HEIGHT_MM - GRASP_ABOVE_BOTTOM_MM,
                               places=6)

    def test_yawed_hold_keeps_the_grasp_point(self):
        geo = PourGeometry(HOLD_XYZ, HOLD_RPY, BEAKER, yaw_deg=30.0)
        p, R = geo.upright(geo.grasp, 1.0)
        np.testing.assert_allclose(p, geo.v_p, atol=1e-9)
        np.testing.assert_allclose(R, geo.v_R, atol=1e-9)
        # turning about the vertical through the grasp keeps the grasp point where it is
        pad = np.asarray(geo.grasp) - geo.hold_p
        np.testing.assert_allclose(geo.v_p + geo.v_R @ (geo.hold_R.T @ pad), geo.grasp, atol=1e-9)


FAST = PourConfig(lip_heights_mm=(30.0,), pretilt_candidates_deg=(60.0,), yaw_candidates_deg=(0.0,))


class PlanTest(unittest.TestCase):
    def test_wrist_up_branch_admits_a_pour(self):
        plan = plan_pour(HOLD_XYZ, HOLD_RPY, ik_numeric(HOLD_XYZ, HOLD_RPY, WRIST_UP), BEAKER, OBSTACLES,
                         ik_numeric, fk=fk_pose, cfg=FAST)
        self.assertEqual(plan.steps[0]["phase"], "hold")
        self.assertEqual(plan.steps[-1]["theta_deg"], FAST.theta_max_deg)
        lips = [s["lip"] for s in plan.steps[plan.pour_start_index:]]
        self.assertLess(max(math.dist(a, plan.lip_xyz) for a in lips), 1e-3)
        self.assertGreaterEqual(min(plan.min_clearance.values()), 0.0)
        for a, b in zip(plan.steps, plan.steps[1:]):
            self.assertLessEqual(max(abs(math.remainder(x - y, 360.0)) for x, y in zip(a["joints"], b["joints"])), 15.0)

    def test_wrist_down_branch_sweeps_the_forearm_over_the_balances(self):
        # The branch the 17:30 handoff ended in: J4 hangs 120 mm below the flange.
        with self.assertRaises(PourInfeasible) as ctx:
            plan_pour(HOLD_XYZ, HOLD_RPY, ik_numeric(HOLD_XYZ, HOLD_RPY, WRIST_DOWN), BEAKER, OBSTACLES,
                      ik_numeric, fk=fk_pose, cfg=FAST)
        self.assertTrue(any("forearm" in k or "wrist" in k or "elbow" in k for k in ctx.exception.rejected))

    def test_beaker_too_close_to_reach_round_is_rejected_not_hit(self):
        # Beaker right next to the hold: the bottle cannot get beside it, so the planner must refuse.
        near = Vessel("left_beaker", -100.0, -700.0, 23.7, BEAKER.bottom_z, BEAKER.top_z + 120.0)
        with self.assertRaises(PourInfeasible):
            plan_pour(HOLD_XYZ, HOLD_RPY, ik_numeric(HOLD_XYZ, HOLD_RPY, WRIST_UP), near, OBSTACLES,
                      ik_numeric, fk=fk_pose, cfg=FAST)


if __name__ == "__main__":
    unittest.main()


class RunPourHelpersTest(unittest.TestCase):
    """chem_demo.run_pour pure helpers (no hardware, no cv2)."""

    VESSELS = {  # shape of the vision worker's locate reply (cw heights, right base XY)
        "left_beaker": {"x": 15.0, "y": -691.7, "radius": 23.7, "base_z": 339.73, "rim_z": 420.73,
                        "support": "left_pan", "role": "beaker"},
        "right_beaker": {"x": 30.1, "y": -325.9, "radius": 25.2, "base_z": 339.73, "rim_z": 419.97,
                         "support": "right_pan", "role": "beaker"},
        "centre_beaker": {"x": -155.7, "y": -520.0, "radius": 37.8, "base_z": 240.18, "rim_z": 329.39,
                          "support": "table", "role": "beaker"},
        "bottle": {"x": -240.6, "y": -733.4, "radius": 33.5, "base_z": 240.18, "rim_z": 337.9,
                   "support": "table", "role": "bottle"},
    }

    def test_scene_converts_cw_heights_and_adds_hidden_obstacles(self):
        from chem_demo.run_pour import scene_from_locate
        beaker, obstacles = scene_from_locate(self.VESSELS, rack_xy=(-338.6, -310.7), fingertip_mm=180.0)
        self.assertAlmostEqual(beaker.top_z, 420.73 - 180.0)
        self.assertAlmostEqual(beaker.bottom_z, 339.73 - 180.0)
        names = {o.name for o in obstacles}
        self.assertEqual(names, {"right_beaker", "right_beaker_balance", "centre_beaker", "centre_rod", "rack"})
        rod = next(o for o in obstacles if o.name == "centre_rod")
        self.assertGreater(rod.top_z, next(o for o in obstacles if o.name == "centre_beaker").top_z)

    def test_scene_refuses_without_the_target(self):
        from chem_demo.run_pour import scene_from_locate
        with self.assertRaises(RuntimeError):
            scene_from_locate({k: v for k, v in self.VESSELS.items() if k != "left_beaker"})

    def test_hold_check_refuses_the_wrist_down_branch_and_an_empty_jaw(self):
        from chem_demo.run_pour import check_hold
        up = {"pose": [*HOLD_XYZ, *HOLD_RPY], "joints": ik_numeric(HOLD_XYZ, HOLD_RPY, WRIST_UP)}
        down = {"pose": [*HOLD_XYZ, *HOLD_RPY], "joints": ik_numeric(HOLD_XYZ, HOLD_RPY, WRIST_DOWN)}
        self.assertEqual(check_hold(up, 180, (150, 210)), [])
        self.assertEqual(len(check_hold(down, 180, (150, 210))), 1)
        self.assertEqual(len(check_hold(up, 254, (150, 210))), 1)

    def test_phases_stop_where_asked_and_the_way_back_ends_at_the_hold(self):
        from chem_demo.run_pour import phases, way_back
        plan = plan_pour(HOLD_XYZ, HOLD_RPY, ik_numeric(HOLD_XYZ, HOLD_RPY, WRIST_UP), BEAKER, OBSTACLES,
                         ik_numeric, fk=fk_pose, cfg=FAST)
        self.assertEqual([n for n, _ in phases(plan, "side")], ["lift", "transit"])
        self.assertEqual([n for n, _ in phases(plan, "pretilt")], ["lift", "transit", "pretilt"])
        full = phases(plan, None)
        self.assertEqual([n for n, _ in full], ["lift", "transit", "pretilt", "pour"])
        self.assertEqual(sum(len(s) for _, s in full), len(plan.steps) - 1)
        back = way_back(phases(plan, "pretilt"), plan.steps[0])
        self.assertEqual(back[-1]["joints"], plan.steps[0]["joints"])
        self.assertEqual(back[0]["phase"], "pretilt")
        for a, b in zip(back, back[1:]):
            self.assertLessEqual(max(abs(x - y) for x, y in zip(a["joints"], b["joints"])), 15.0)
