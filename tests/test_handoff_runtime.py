"""Offline checks for the handoff glue: IK cache, refinement and the planner call."""

from __future__ import annotations

import unittest

import numpy as np

from chem_demo.handoff_cycle import HandoffCycle
from chem_demo.handoff_plan import HandoffInfeasible, _rot, rpy_for_tool_axis
from chem_demo.handoff_runtime import LEFT_HANDOFF_Z_LEVELS, CachedIK, handoff_config, make_plan_fn, make_refine_fn
from task_skills.run_cartesian_jog import SimArm
from tests.test_handoff_plan import FakeArm

T_RIGHT_LEFT = np.array([[-1.0, 0.0, 0.0, 16.0], [0.0, -1.0, 0.0, -1062.0], [0.0, 0.0, 1.0, -3.0], [0, 0, 0, 1.0]])


class RuntimeTest(unittest.TestCase):
    def test_cached_ik_counts_and_replays_errors(self):
        calls = []

        def ik(xyz, rpy, seed):
            calls.append(xyz)
            if xyz[0] > 100:
                raise RuntimeError("unreachable")
            return [0.0] * 6

        cached = CachedIK(ik, "t")
        cached([1, 2, 3], [0, 0, 0], [0] * 6)
        cached([1, 2, 3], [0, 0, 0], [0] * 6)
        with self.assertRaises(RuntimeError):
            cached([200, 0, 0], [0, 0, 0], [0] * 6)
        with self.assertRaises(RuntimeError):
            cached([200, 0, 0], [0, 0, 0], [0] * 6)
        self.assertEqual((cached.calls, cached.hits, len(calls)), (2, 2, 2))

    def test_refine_moves_along_tool_axes_and_caps_total_walk(self):
        right = SimArm([-200, -500, 450], [90, 0, 0])
        start = {"name": "approach", "xyz": [-200.0, -500.0, 450.0], "rpy": [90.0, 0.0, 0.0],
                 "joints": right.solve([-200, -500, 450], [90, 0, 0], None)}
        refine = make_refine_fn(right, start)
        moved = refine(start, [5.0, 0.0, 10.0])
        shift = np.array(moved["xyz"]) - np.array(start["xyz"])
        self.assertAlmostEqual(float(np.linalg.norm(shift)), float(np.linalg.norm([5.0, 0.0, 10.0])), places=2)
        with self.assertRaises(HandoffInfeasible):
            refine(start, [0.0, 0.0, 500.0])

    def test_long_refine_is_checked_in_samples_not_as_one_jump(self):
        # 17:23: the smooth approach's ~15 cm MovL was rejected as one 15 degree "jump".
        class Live:
            def __init__(self):
                self.fake = FakeArm(r_range=(150.0, 700.0))
                self.plan, self.forward = self.fake.ik, self.fake.fk

        right = Live()
        rpy = [0.0, 90.0, 0.0]  # tool Z along +X: straight out from the base
        xyz = [250.0, 0.0, 400.0]
        start = {"name": "approach", "xyz": xyz, "rpy": rpy, "joints": right.plan(xyz, rpy, [0] * 6)}
        refine = make_refine_fn(right, start)
        moved = refine(start, [0.0, 0.0, 150.0])
        self.assertAlmostEqual(moved["xyz"][0], 400.0, places=3)
        self.assertGreater(abs(moved["joints"][1] - start["joints"][1]), 15.0)  # would be one big jump

    def test_plan_fn_returns_a_plan_the_cycle_can_use_at_the_photo_height(self):
        class Live:  # adapter: the planner's analytic fake behind the LiveArm method names
            def __init__(self):
                self.fake = FakeArm(r_range=(150.0, 700.0))
                self.plan, self.forward = self.fake.ik, self.fake.fk

        left, right = Live(), Live()
        right_start = [-300.0, -160.0, 535.0, 180.0, 0.0, 89.97]
        right_state = {"pose": right_start, "joints": right.plan(right_start[:3], right_start[3:], [0] * 6)}

        def state_for(pose):
            return {"pose": pose, "joints": left.plan(pose[:3], pose[3:], [0] * 6)}

        plan_fn = make_plan_fn(left, right, T_RIGHT_LEFT, body_model=False, log=lambda _m: None)
        first = plan_fn(state_for([265.0, -381.0, 389.0, 180.0, 0.0, -89.94]), right_state)
        self.assertGreaterEqual(first["left_handoff_xyz"][2], 440.0)
        self.assertEqual(first["right_waypoints"][0]["step_kind"], "start")
        names = [w["name"] for w in first["right_waypoints"]]
        self.assertIn(first["standoff_pose"]["name"], names)
        self.assertEqual(first["handoff_left_tcp"], first["left_handoff_xyz"])
        # Handoff points are absolute (photo height, between the left base and the middle): a different
        # pick position changes the left route to it, not the handoff height.
        second = plan_fn(state_for([240.0, -330.0, 389.0, 180.0, 0.0, -89.94]), right_state)
        self.assertEqual(second["left_handoff_xyz"][2], LEFT_HANDOFF_Z_LEVELS[0])
        self.assertNotEqual(first["left_legs"][0], second["left_legs"][0])
        # A stored station is checked as is: same handoff point from two different lift poses.
        station = {"left_tcp": first["left_handoff_xyz"],
                   "approach_offset_deg": first["candidate"]["approach_offset_deg"],
                   "pitch_down_deg": first["candidate"]["pitch_down_deg"]}
        fixed = make_plan_fn(left, right, T_RIGHT_LEFT, station=station, body_model=False, log=lambda _m: None)
        for lifted in ([265.0, -381.0, 389.0], [250.0, -370.0, 389.0]):
            plan = fixed(state_for([*lifted, 180.0, 0.0, -89.94]), right_state)
            np.testing.assert_allclose(plan["left_handoff_xyz"], first["left_handoff_xyz"], atol=1e-3)
            self.assertEqual(plan["candidate"]["candidates_enumerated"], 1)

    def test_handoff_height_is_fixed_and_whole_arm_checks_are_on(self):
        for lifted_z in (373.9, 388.9):
            cfg = handoff_config([265, -381, lifted_z])
            self.assertEqual({p[2] for p in cfg.left_points_mm}, {LEFT_HANDOFF_Z_LEVELS[0]})
            self.assertTrue(cfg.body_model)

    def test_search_rolls_keep_the_jaw_axis_horizontal_and_tool_x_vertical(self):
        cfg = handoff_config([265, -381, 389.0])
        self.assertTrue(cfg.roll_candidates_deg)
        for roll in cfg.roll_candidates_deg:
            for azimuth in (0.0, 45.0, 90.0, 180.0, 270.0):
                axis = [np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth)), 0.0]
                R = _rot(rpy_for_tool_axis(axis, roll))
                np.testing.assert_allclose(R[:, 2], axis, atol=1e-6)      # tool Z horizontal at the bottle
                self.assertLess(abs(R[2, 1]), 1e-6)                        # tool Y (jaw axis) horizontal
                self.assertAlmostEqual(abs(R[2, 0]), 1.0, places=6)        # tool X vertical


if __name__ == "__main__":
    unittest.main()
