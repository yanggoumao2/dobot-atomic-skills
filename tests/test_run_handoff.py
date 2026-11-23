"""Launcher wiring that needs no hardware: argument gate and the two-arm idle states."""

from __future__ import annotations

import unittest
from unittest import mock

from chem_demo import run_handoff


class LauncherTest(unittest.TestCase):
    def test_live_requires_the_path_clear_confirmation(self):
        with self.assertRaises(SystemExit):
            run_handoff.main(["--live"])

    def test_idle_states_returns_both_arms(self):
        profile = {"arms": {"left": {"ip": "l"}, "right": {"ip": "r"}}}
        with mock.patch.object(run_handoff, "_check_idle", return_value={"pose": [1] * 6, "joints": [0] * 6}), \
                mock.patch.object(run_handoff, "read_passive_robot_state",
                                  return_value={"pose": [2] * 6, "joints": [0] * 6, "mode": 4}):
            states = run_handoff._idle_states(profile)
        self.assertEqual(set(states), {"left", "right"})
        self.assertEqual(states["right"]["pose"], [2] * 6)

    def test_a_failed_disable_is_reported_not_swallowed(self):
        profile = {"arms": {"left": {"ip": "l"}, "right": {"ip": "r"}}}
        states = iter([{"mode": 4, "enabled": 0}, {"mode": 5, "enabled": 1}])
        with mock.patch.object(run_handoff, "read_passive_robot_state", side_effect=lambda ip: next(states)):
            with self.assertRaisesRegex(RuntimeError, "下使能"):
                run_handoff._verify_both_disabled(profile)



class GripperHoldEvidenceTest(unittest.TestCase):
    def test_loaded_code_confirms_and_empty_or_odd_codes_do_not(self):
        from chem_demo.run_handoff import _gripper_confirms_hold
        gripper = {"loaded_closed_final": 124, "free_closed_final": 0}
        self.assertTrue(_gripper_confirms_hold(124, gripper))
        self.assertTrue(_gripper_confirms_hold(126, gripper))
        self.assertFalse(_gripper_confirms_hold(0, gripper))      # bottle slipped, jaw ran on
        self.assertFalse(_gripper_confirms_hold(90, gripper))     # something else between the jaws
        self.assertFalse(_gripper_confirms_hold(255, gripper))


class CorridorBackTest(unittest.TestCase):
    """2026-11-23 16:07: right stopped 32 mm off the recorded path during the wrist-camera approach."""

    def _log(self):
        import numpy as np
        from chem_demo.handoff_plan import _rot
        rpy = [0.0, 90.0, -84.82899242444398]
        stand = [-153.7338, -365.7912, 314.3079]
        joints = [-98.0, -56.0, -100.0, 156.0, 103.0, 90.0]
        plan = {"standoff_pose": {"name": "approach"},
                "right_waypoints": [{"name": "park", "xyz": [-300.0, -160.0, 535.0], "rpy": [180.0, 0.0, 90.0],
                                     "joints": [-90.0, 0, 0, 0, 90, 0], "step_kind": "start"},
                                    {"name": "pre_approach", "xyz": stand, "rpy": rpy, "joints": joints,
                                     "step_kind": "joint_approach"},
                                    {"name": "approach", "xyz": stand, "rpy": rpy, "joints": joints,
                                     "step_kind": "straight_approach"}]}
        steps = [{"step": "pre_approach"}, {"step": "approach"}, {"step": "aligned"}, {"step": "approach_2"}]
        return {"plan": plan, "steps": steps}, np.array(stand), _rot(rpy)

    def test_stop_in_the_approach_corridor_backs_out_along_the_tool_axis_first(self):
        import numpy as np
        log, stand, R = self._log()
        # The measured 16:07 pose: same orientation written as other Euler angles (gimbal at pitch 90).
        state = {"pose": [-153.756393, -389.891296, 292.667389, 84.828995, 90.0, 0.0], "joints": [0] * 6}
        found = run_handoff._right_corridor_back(log, state)
        self.assertIsNotNone(found)
        targets, standoff_state = found
        back, home = (np.array(t["xyz"]) for t in targets)
        # First move is purely backwards along the tool axis (away from the bottle) ...
        move = back - np.array(state["pose"][:3])
        self.assertLess(float(move @ R[:, 2]), -20.0)
        self.assertLess(np.linalg.norm(move - (move @ R[:, 2]) * R[:, 2]), 1e-6)
        # ... the second only sideways in the tool frame, ending on the recorded standoff.
        side = R.T @ (home - back)
        self.assertLess(abs(side[2]), 1e-6)
        np.testing.assert_allclose(home, stand, atol=1e-6)
        self.assertEqual(standoff_state["joints"], log["plan"]["right_waypoints"][2]["joints"])
        path = run_handoff._right_retreat_path(log, standoff_state)
        self.assertEqual(path[0]["name"], "approach")
        self.assertEqual(path[-1]["name"], "park")

    def test_poses_outside_the_corridor_are_not_handled(self):
        log, stand, R = self._log()
        rotated = {"pose": [-153.7, -389.9, 292.7, 84.828995, 80.0, 0.0], "joints": [0] * 6}
        self.assertIsNone(run_handoff._right_corridor_back(log, rotated))
        sideways = (stand + R[:, 1] * 60.0).tolist()
        self.assertIsNone(run_handoff._right_corridor_back(log, {"pose": [*sideways, 0.0, 90.0, -84.82899242444398],
                                                                 "joints": [0] * 6}))
        behind = (stand - R[:, 2] * 30.0).tolist()
        self.assertIsNone(run_handoff._right_corridor_back(log, {"pose": [*behind, 0.0, 90.0, -84.82899242444398],
                                                                 "joints": [0] * 6}))
        unreached = {"plan": log["plan"], "steps": [{"step": "pre_approach"}]}
        self.assertIsNone(run_handoff._right_corridor_back(unreached, {"pose": [*stand, 0.0, 90.0, -84.82899242444398],
                                                                       "joints": [0] * 6}))


class HandbackTest(unittest.TestCase):
    """16:58: the right held the bottle after the down check, the left stood open at the station."""

    def _log(self):
        rpy = [0.0, 90.0, -84.82899242444398]
        return {"plan": {"standoff_pose": {"name": "approach"}, "right_waypoints": []},
                "steps": [{"step": "left_route_checked", "legs": [[266.0, -377.0, 389.0], [129.97, -357.08, 570.0]]},
                          {"step": "approach_14", "target": [-141.9051, -525.7308, 291.6706], "rpy": rpy,
                           "measured": [-141.905106, -525.730774, 291.670593, 84.828995, 90.0, 0.0]},
                          {"step": "right_closed"}, {"step": "left_released"},
                          {"step": "right_down_check", "target": [-141.9051, -525.7308, 276.6706], "rpy": rpy,
                           "measured": [-141.905106, -525.730774, 276.670593, 84.828995, 90.0, 0.0]}]}

    def test_live_state_after_the_down_check_is_recognised(self):
        right = {"pose": [-141.9, -525.7, 276.6, 171.1, 90.0, 86.3]}
        left = {"pose": [130.0, -357.1, 570.0, 180.0, 0.0, -90.0]}
        contact = run_handoff._handback_contact(self._log(), right, left, 180, 253, (30, 240))
        self.assertIsNotNone(contact)
        self.assertAlmostEqual(contact["xyz"][2], 291.670593, places=4)

    def test_hand_back_uses_the_commissioned_gripper_keys(self):
        # Saved neck_grasp.gripper (2026-11-23): no close_target/close_speed keys (17:07 KeyError).
        gripper = {"free_closed_final": 0, "loaded_closed_final": 124, "open": 255, "speed": 5, "target": 0}
        calls = []

        class Arm:
            def __init__(self, name, jaw):
                self.name, self.jaw = name, jaw

            def read(self):
                return {"pose": [0.0] * 6, "joints": [0.0] * 6}

            def plan(self, xyz, rpy, seed):
                return [0.0] * 6

            def move(self, *a):
                calls.append((self.name, "move"))

            def wait(self, *a, **k):
                return self.read()

            def grip_to(self, position, speed):
                calls.append((self.name, "grip", position, speed))
                self.jaw = 124 if self.name == "left" else 254
                return {"final": self.jaw}

            def gripper_position(self):
                return self.jaw

        right, left = Arm("right", 180), Arm("left", 253)
        with mock.patch.object(run_handoff, "_stop_key", return_value=False):
            run_handoff._hand_back(right, left, {"xyz": [0.0] * 3, "rpy": [0.0] * 3}, gripper, {})
        self.assertEqual(calls, [("right", "move"), ("left", "grip", 0, 5), ("right", "grip", 255, 5)])

    def test_after_a_completed_handoff_the_left_retraces_its_exit(self):
        log = self._log()
        log["steps"].append({"step": "transfer_confirmed"})
        photo = run_handoff.photo_pose("left")[0]
        left = {"pose": [*photo, 180.0, 0.0, -90.0]}
        right = {"pose": [-141.7, -523.8, 276.5, 171.1, 90.0, 86.3]}
        route = run_handoff._left_return_target(log, left, right, 180, 253, (30, 240))
        self.assertEqual(route, [[photo[0], photo[1], 570.0], [129.97, -357.08, 570.0]])
        # Not after an unconfirmed handoff, nor with the left still closed, nor away from the photo pose.
        log2 = self._log()
        self.assertIsNone(run_handoff._left_return_target(log2, left, right, 180, 253, (30, 240)))
        self.assertIsNone(run_handoff._left_return_target(log, left, right, 180, 124, (30, 240)))
        away = {"pose": [200.0, -250.0, 535.0, 180.0, 0.0, -90.0]}
        self.assertIsNone(run_handoff._left_return_target(log, away, right, 180, 253, (30, 240)))
        moved_right = {"pose": [-141.7, -500.0, 276.5, 171.1, 90.0, 86.3]}
        self.assertIsNone(run_handoff._left_return_target(log, left, moved_right, 180, 253, (30, 240)))

    def test_other_states_are_left_to_the_operator(self):
        log = self._log()
        right = {"pose": [-141.9, -525.7, 276.6, 171.1, 90.0, 86.3]}
        left = {"pose": [130.0, -357.1, 570.0, 180.0, 0.0, -90.0]}
        self.assertIsNone(run_handoff._handback_contact(log, right, left, 10, 253, (30, 240)))   # empty close
        self.assertIsNone(run_handoff._handback_contact(log, right, left, 180, 124, (30, 240)))  # left closed
        away = {"pose": [-141.9, -480.0, 276.6, 171.1, 90.0, 86.3]}
        self.assertIsNone(run_handoff._handback_contact(log, away, left, 180, 253, (30, 240)))
        moved_left = {"pose": [200.0, -357.1, 570.0, 180.0, 0.0, -90.0]}
        self.assertIsNone(run_handoff._handback_contact(log, right, moved_left, 180, 253, (30, 240)))
        no_release = {**log, "steps": [s for s in log["steps"] if s["step"] != "left_released"]}
        self.assertIsNone(run_handoff._handback_contact(no_release, right, left, 180, 253, (30, 240)))


if __name__ == "__main__":
    unittest.main()
