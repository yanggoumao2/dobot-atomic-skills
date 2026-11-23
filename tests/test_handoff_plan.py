"""Offline checks for the left-to-right handoff planner; kinematics are an analytic fake."""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace

import numpy as np

from chem_demo.handoff_plan import (
    BottleHold,
    HandoffConfig,
    HandoffInfeasible,
    plan_handoff,
    refine_contact,
    rpy_for_tool_axis,
)
from chem_demo.handoff_plan import _Geometry, _jaw_axis, _quat, _quat_matrix
from chem_demo.gripper_geometry import bottle_hang
from task_skills.chemistry.geometry import euler_zyx_to_matrix
from task_skills.run_cartesian_jog import MAX_JOINT_STEP_DEG, WRIST_MARGIN_DEG

# The real platform: bases 1062 mm apart, yawed 180 deg; left flange at the photo height (535 mm),
# the right arm waiting at its photo pose. Tool0 is the flange (fingertips FINGERTIP_FROM_FLANGE_MM beyond it).
LEFT_TCP = [200.0, -420.0, 535.0, 180.0, 0.0, -89.94]
RIGHT_PARK = [-300.0, -160.0, 535.0, 180.0, 0.0, 89.97]
T_RIGHT_LEFT = np.array([[-1.0, 0, 0, 16.0], [0, -1.0, 0, -1062.0], [0, 0, 1.0, -3.0], [0, 0, 0, 1.0]])


LIGHT = replace(HandoffConfig(), left_dxy_offsets_mm=(0.0,), left_dz_offsets_mm=(0.0,))  # faster search


class FakeArm:
    """Smooth analytic kinematics: reach annulus, J5 driven by the tangential tool-axis component."""

    def __init__(self, r_range=(180.0, 620.0), z_range=(150.0, 700.0), flip=None):
        self.r_range, self.z_range, self.flip = r_range, z_range, flip
        self.table: dict[tuple, tuple[list, list]] = {}
        self.calls = 0

    def ik(self, xyz, rpy, seed):
        self.calls += 1
        x, y, z = xyz
        r = math.hypot(x, y)
        if not (self.r_range[0] <= r <= self.r_range[1] and self.z_range[0] <= z <= self.z_range[1]):
            raise RuntimeError("unreachable")
        R = np.array(euler_zyx_to_matrix(rpy))
        a, e_t = R[:, 2], np.array([-y / r, x / r, 0.0])
        j1 = math.degrees(math.atan2(y, x))
        if self.flip and self.flip(xyz):
            j1 += 40.0  # simulated IK branch switch
        joints = [j1, 0.15 * (r - 350), -0.12 * (z - 400), 50.0 * float(a @ [x / r, y / r, 0]),
                  90.0 + 88.0 * float(a @ e_t), 50.0 * R[2, 0] + 30.0 * R[2, 1]]
        self.table[tuple(joints)] = (list(xyz), list(rpy))
        return joints

    def fk(self, joints):
        xyz, rpy = self.table[tuple(joints)]
        return {"xyz": xyz, "rpy": rpy}


def scenario(left_tcp=LEFT_TCP, left=None, right=None, config=None, **kwargs):
    left, right = left or FakeArm(), right or FakeArm()
    left_state = {"pose": list(left_tcp), "joints": left.ik(left_tcp[:3], left_tcp[3:], [0] * 6)}
    right_state = {"pose": list(RIGHT_PARK), "joints": right.ik(RIGHT_PARK[:3], RIGHT_PARK[3:], [0] * 6)}
    bottle = BottleHold(tuple(left_tcp))
    return plan_handoff(left_state, bottle, T_RIGHT_LEFT, left.ik, right.ik, config or HandoffConfig(),
                        right_start_state=right_state, **kwargs), left, right


class RpyHelperTests(unittest.TestCase):
    def test_down_reference(self):
        self.assertEqual([round(v, 9) for v in np.array(euler_zyx_to_matrix((180, 0, 90)))[:, 2]], [0, 0, -1])
        R = np.array(euler_zyx_to_matrix(rpy_for_tool_axis((0, 0, -1), 0)))
        np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-9)

    def test_horizontal_axes_and_roll(self):
        for az in range(0, 360, 37):
            axis = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
            for roll in (0, 90, 180, -90, 33):
                R = np.array(euler_zyx_to_matrix(rpy_for_tool_axis(axis, roll)))
                np.testing.assert_allclose(R[:, 2], axis, atol=1e-9)
                np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
                self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
            R0 = np.array(euler_zyx_to_matrix(rpy_for_tool_axis(axis, 0)))
            np.testing.assert_allclose(R0[:, 1], [0, 0, -1], atol=1e-9)   # roll 0: tool Y points down
            np.testing.assert_allclose(R0[2, 0], 0.0, atol=1e-9)          # tool X horizontal
            R180 = np.array(euler_zyx_to_matrix(rpy_for_tool_axis(axis, 180)))
            np.testing.assert_allclose(R180[:, 0], -R0[:, 0], atol=1e-9)   # both jaw signs reachable
            R90 = np.array(euler_zyx_to_matrix(rpy_for_tool_axis(axis, 90)))
            np.testing.assert_allclose(R90[:, 0], R0[:, 1], atol=1e-9)     # +90 about the tool axis

    def test_quaternion_roundtrip_all_branches(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            rpy = rng.uniform(-180, 180, 3)
            R = np.array(euler_zyx_to_matrix(rpy))
            np.testing.assert_allclose(_quat_matrix(_quat(R)), R, atol=1e-9)
        for rpy in ((180, 0, 0), (0, 180, 0), (0, 0, 180), (180, 0, 90)):  # trace <= 0 branch
            R = np.array(euler_zyx_to_matrix(rpy))
            np.testing.assert_allclose(_quat_matrix(_quat(R)), R, atol=1e-9)

    def test_zero_axis_rejected(self):
        with self.assertRaises(ValueError):
            rpy_for_tool_axis((0, 0, 0), 0)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.left, cls.right = scenario()

    def test_every_right_waypoint_obeys_rules(self):
        wp = self.plan.right_waypoints
        self.assertEqual(wp[0]["name"], "park")
        self.assertEqual(wp[-1]["name"], "contact")
        for a, b in zip(wp, wp[1:]):
            jump = max(abs(math.remainder(b["joints"][i] - a["joints"][i], 360)) for i in range(6))
            self.assertLessEqual(jump, MAX_JOINT_STEP_DEG)
            self.assertLessEqual(float(np.linalg.norm(np.subtract(b["xyz"], a["xyz"]))), 10.0 + 1e-3)
            self.assertLessEqual(_angle(a["rpy"], b["rpy"]), 10.0 + 1e-3)
        for w in wp[1:] + self.plan.retreat_poses:
            self.assertGreaterEqual(abs(math.remainder(w["joints"][4], 180)), WRIST_MARGIN_DEG)
        for m in self.plan.step_margins:
            self.assertLessEqual(m["jump_deg"], MAX_JOINT_STEP_DEG)
            self.assertGreaterEqual(m["j5_margin_deg"], WRIST_MARGIN_DEG)

    def test_geometry_from_live_state(self):
        c = self.plan
        R = np.array(euler_zyx_to_matrix(c.contact_pose["rpy"]))
        self.assertAlmostEqual(abs(R[2, 2]), 0.0, places=6)                   # tool axis horizontal
        pad = np.array(c.contact_pose["xyz"]) + R @ np.array(HandoffConfig().right_pad_offset_tool_mm)
        np.testing.assert_allclose(pad, c.bottle_right_xyz, atol=1e-3)         # pads at the body centre
        h = np.array(c.left_handoff_xyz)
        expect = T_RIGHT_LEFT[:3, :3] @ (h - [0, 0, BottleHold(tuple(LEFT_TCP)).grasp_drop_mm]) + T_RIGHT_LEFT[:3, 3]
        np.testing.assert_allclose(c.bottle_right_xyz, expect, atol=1e-3)
        back = np.array(c.contact_pose["xyz"]) - np.array(c.standoff_pose["xyz"])
        self.assertAlmostEqual(float(np.linalg.norm(back)), c.candidate["standoff_mm"], places=2)
        np.testing.assert_allclose(back / np.linalg.norm(back), R[:, 2], atol=1e-6)
        self.assertEqual(c.left_legs[0], LEFT_TCP[:3])
        self.assertEqual(c.left_legs[-1], c.left_handoff_xyz)
        for xyz in c.left_legs:
            cfg = HandoffConfig()
            self.assertTrue(cfg.left_z_range_mm[0] <= xyz[2] <= cfg.left_z_range_mm[1]
                            and cfg.left_x_range_mm[0] <= xyz[0] <= cfg.left_x_range_mm[1]
                            and cfg.left_y_range_mm[0] <= xyz[1] <= cfg.left_y_range_mm[1])
        self.assertEqual([w["name"] for w in c.retreat_poses][-1], "retreat_park")
        np.testing.assert_allclose(c.retreat_poses[-1]["xyz"], RIGHT_PARK[:3], atol=1e-3)

    def test_left_tcp_change_moves_handoff(self):
        other = [250.0, -350.0, 470.0, 180.0, 0.0, -89.94]
        plan_b, _, _ = scenario(other)
        self.assertNotEqual(self.plan.left_handoff_xyz, plan_b.left_handoff_xyz)
        for plan, tcp in ((self.plan, LEFT_TCP), (plan_b, other)):
            offset = plan.candidate["left_offset_mm"]
            np.testing.assert_allclose(plan.left_handoff_xyz, np.add(tcp[:3], offset), atol=1e-3)
        self.assertNotEqual(self.plan.bottle_right_xyz, plan_b.bottle_right_xyz)

    def test_search_moves_off_the_current_point_when_needed(self):
        # The bottle at the current left Z is too low for the right guard; only a raised handoff works.
        # (Right pads ~bottle grasp height below a 535 mm left flange: too low for a 350 mm floor; +80 clears it.)
        drop = bottle_hang()["grasp_drop_mm"]
        self.assertLess(535.0 - drop, 350.0)
        self.assertGreater(535.0 + 80.0 - drop - 3.0, 350.0)
        tight = replace(HandoffConfig(), left_dz_offsets_mm=(0.0, 80.0), left_z_range_mm=(350.0, 640.0),
                        right_z_range_mm=(350.0, 640.0))
        plan, _, _ = scenario(config=tight)
        self.assertEqual(plan.candidate["left_offset_mm"][2], 80.0)
        self.assertGreater(plan.rejected_summary.get("right_guard", 0), 0)

    def test_tangential_approach_rejected_by_wrist_margin(self):
        self.assertLess(abs(self.plan.candidate["approach_offset_deg"]), 90)
        self.assertGreater(self.plan.rejected_summary.get("right_wrist_singularity", 0), 0)

    def test_fk_forward_check_is_used(self):
        right = FakeArm()
        plan, _, _ = scenario(right=right, fk_right=right.fk, config=LIGHT)
        self.assertTrue(plan.right_waypoints)
        with self.assertRaises(HandoffInfeasible) as caught:
            scenario(right=right, config=LIGHT, fk_right=lambda j: {
                "xyz": [v + 1.0 for v in right.fk(j)["xyz"]], "rpy": right.fk(j)["rpy"]})
        self.assertIn("right_fk_mismatch", caught.exception.histogram)

    def test_joint_limit_proxy(self):
        limits = [(-360, 360)] * 6
        plan, _, _ = scenario(config=LIGHT, joint_limits_deg=limits)
        self.assertTrue(all(m["limit_margin_deg"] > 3 for m in plan.step_margins))
        with self.assertRaises(HandoffInfeasible) as caught:
            scenario(config=LIGHT, joint_limits_deg=[(-360, 360), (-360, 360), (-360, 360), (-1, 1), (-360, 360), (-360, 360)])
        self.assertIn("right_joint_limit", caught.exception.histogram)

    def test_deterministic_and_json_roundtrip(self):
        again, _, _ = scenario()  # full default grid, twice
        self.assertEqual(self.plan.to_dict(), again.to_dict())
        data = self.plan.to_dict()
        self.assertEqual(json.loads(json.dumps(data)), data)
        self.assertEqual(json.loads(json.dumps(HandoffConfig().to_dict()))["max_refine_mm"], 25.0)
        self.assertAlmostEqual(data["bottle"]["grasp_drop_mm"], bottle_hang()["grasp_drop_mm"])


class InfeasibleTests(unittest.TestCase):
    def test_unreachable_right_arm(self):
        with self.assertRaises(HandoffInfeasible) as caught:
            scenario(right=_ReachLimited(), config=LIGHT)
        hist = caught.exception.histogram
        self.assertTrue(hist)
        self.assertEqual(sum(hist.values()), caught.exception.total_candidates)
        self.assertIn("right_unreachable", hist)

    def test_left_branch_switch_is_rejected_and_avoided(self):
        left = FakeArm(flip=lambda xyz: xyz[2] < LEFT_TCP[2] - 1.0)   # J1 jumps 40 deg below the start Z
        plan, _, _ = scenario(left=left)
        self.assertEqual(plan.left_handoff_xyz[2], LEFT_TCP[2])
        self.assertGreater(plan.rejected_summary["left_joint_jump"], 0)

    def test_left_jump_everywhere_is_infeasible(self):
        left = FakeArm(flip=lambda xyz: xyz[2] < LEFT_TCP[2] + 1.0 and xyz[0] != LEFT_TCP[0])
        with self.assertRaises(HandoffInfeasible) as caught:
            scenario(left=left, config=replace(HandoffConfig(), left_dxy_offsets_mm=(40.0,)))
        self.assertIn("left_joint_jump", caught.exception.histogram)

    def test_guard_rejects_everything(self):
        cfg = replace(HandoffConfig(), left_z_range_mm=(350.0, 400.0))
        with self.assertRaises(HandoffInfeasible) as caught:
            scenario(config=cfg)
        self.assertEqual(list(caught.exception.histogram), ["left_guard"])


class RefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.left, cls.right = scenario()

    def test_small_delta_accepted(self):
        out = refine_contact(self.plan, (3.0, -2.0, 5.0), self.right.ik, fk_right=self.right.fk)
        R = np.array(euler_zyx_to_matrix(self.plan.contact_pose["rpy"]))
        np.testing.assert_allclose(np.subtract(out["xyz"], self.plan.contact_pose["xyz"]), R @ [3, -2, 5], atol=1e-3)
        np.testing.assert_allclose(out["delta_base_mm"], R @ [3, -2, 5], atol=1e-3)
        jump = max(abs(math.remainder(a - b, 360)) for a, b in zip(out["standoff"]["joints"],
                                                                   self.plan.standoff_pose["joints"]))
        self.assertLessEqual(jump, MAX_JOINT_STEP_DEG)
        self.assertEqual(out["approach"][-1]["xyz"], out["xyz"])

    def test_zero_delta_reproduces_contact(self):
        out = refine_contact(self.plan, (0, 0, 0), self.right.ik)
        np.testing.assert_allclose(out["xyz"], self.plan.contact_pose["xyz"], atol=1e-3)

    def test_large_delta_refused(self):
        with self.assertRaises(HandoffInfeasible) as caught:
            refine_contact(self.plan, (20.0, 20.0, 0.0), self.right.ik)
        self.assertIn("refine_delta_too_large", caught.exception.histogram)

    def test_unreachable_refinement_refused(self):
        with self.assertRaises(HandoffInfeasible):
            refine_contact(self.plan, (0.0, 0.0, 24.0), _ReachLimited().ik)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            refine_contact(self.plan, (1.0, 2.0), self.right.ik)


class _ReachLimited(FakeArm):
    """Right arm that can only reach the neighbourhood of its park pose, so no handoff exists."""

    def ik(self, xyz, rpy, seed):
        if math.dist(xyz, RIGHT_PARK[:3]) > 60.0:
            raise RuntimeError("unreachable")
        return super().ik(xyz, rpy, seed)


def _angle(rpy_a, rpy_b) -> float:
    a, b = np.array(euler_zyx_to_matrix(rpy_a)), np.array(euler_zyx_to_matrix(rpy_b))
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(a.T @ b) - 1) / 2))))


class OpenFingerTests(unittest.TestCase):
    """2026-11-22 23:00 e-stop: an open finger swept toward the bottle while the tool axis stayed clear."""

    def test_open_finger_beside_the_bottle_is_rejected(self):
        cfg = HandoffConfig()
        bottle = BottleHold((0.0, 0.0, 600.0, 180.0, 0.0, -90.0))
        geo = _Geometry(np.eye(4), bottle, cfg)
        left = np.array([0.0, 0.0, 600.0])
        rpy = rpy_for_tool_axis((1.0, 0.0, 0.0), 90.0)
        jaw = _jaw_axis(rpy)
        self.assertAlmostEqual(abs(jaw[1]), 1.0, places=6)    # jaw axis horizontal, across the approach
        z = 600.0 - bottle.grasp_drop_mm
        lateral = 75.0  # tool axis 75 mm beside the bottle axis: a single axial capsule (r 20) clears it
        q = np.array([-150.0, lateral * jaw[1], z])
        self.assertGreater(lateral, bottle.body_diameter_mm / 2 + cfg.right_finger_radius_mm + cfg.min_clearance_mm)
        reason = geo.clearance(left, q, rpy, True)
        self.assertIsNotNone(reason)
        self.assertIn("right_finger", reason)
        self.assertTrue(reason.endswith("_bottle"))
        far = np.array([-150.0, 160.0 * jaw[1], z])
        self.assertIsNone(geo.clearance(left, far, rpy, True))

    def test_final_approach_lets_the_fingers_straddle_the_bottle(self):
        cfg = HandoffConfig()
        bottle = BottleHold((0.0, 0.0, 600.0, 180.0, 0.0, -90.0))
        geo = _Geometry(np.eye(4), bottle, cfg)
        rpy = rpy_for_tool_axis((1.0, 0.0, 0.0), 90.0)
        contact = np.array([-(cfg.right_pad_offset_tool_mm[2]), 0.0, 600.0 - bottle.grasp_drop_mm])
        self.assertIsNotNone(geo.clearance(np.array([0.0, 0.0, 600.0]), contact, rpy, True))
        self.assertIsNone(geo.clearance(np.array([0.0, 0.0, 600.0]), contact, rpy, False))


if __name__ == "__main__":
    unittest.main()
