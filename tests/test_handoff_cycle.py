"""Offline order-of-operations checks for the left-to-right bottle handoff."""

from __future__ import annotations

import unittest

import numpy as np

from chem_demo.handoff_cycle import HandoffCycle, HandoffRefused
from chem_demo.handoff_plan import _rot, rpy_for_tool_axis
from task_skills.run_cartesian_jog import JogStopped, SimArm

LEFT_START = [300.0, -160.0, 535.0]
RIGHT_START = [-300.0, -160.0, 535.0]
BOTTLE = np.array([-150.0, -420.0, 450.0])  # simulated body point in the right base frame
AXIS = [0.0, -1.0, 0.0]                      # tool Z points at the bottle
RPY = rpy_for_tool_axis(AXIS, 90.0)          # tool Y horizontal (jaw axis), tool X straight down
LOADED = 124
STANDOFF = 200.0


class Arm(SimArm):
    def __init__(self, xyz, rpy, *, close_final=0, events=None, name="arm", open_ok=True):
        super().__init__(xyz, rpy)
        self.close_final, self.events, self.name, self.open_ok = close_final, events if events is not None else [], name, open_ok
        self.stops = 0

    def grip_to(self, position, speed):
        before = self.gripper_pos
        self.gripper_pos = self.close_final if position == 0 else (position if self.open_ok else before)
        self.events.append((self.name, "grip", position))
        return {"start": before, "target": position, "final": self.gripper_pos, "samples": [self.gripper_pos] * 5}

    def stop(self):
        self.stops += 1


def waypoint(arm, name, xyz, rpy, kind="move"):
    return {"name": name, "xyz": list(map(float, xyz)), "rpy": list(rpy), "joints": arm.solve(xyz, rpy, None),
            "step_kind": kind}


def build(*, right_final=190, left_code=LOADED, held=True, observe=None, stop=lambda: False, left_open_ok=True,
          standoff_error=0.0, verify_hold=None):
    events: list = []
    left = Arm(LEFT_START, [180, 0, -89.94], close_final=LOADED, events=events, name="left", open_ok=left_open_ok)
    left.gripper_pos = left_code
    right = Arm(RIGHT_START, [180, 0, 89.97], close_final=right_final, events=events, name="right")
    standoff_xyz = BOTTLE - np.array(AXIS) * (STANDOFF + standoff_error)

    def plan_fn(left_state, right_state):
        return {"handoff_left_tcp": [250.0, -300.0, 500.0], "candidate": {"standoff_mm": STANDOFF},
                "left_legs": [[300.0, -160.0, 500.0], [250.0, -300.0, 500.0]],
                "standoff_pose": {"name": "approach"},
                "right_waypoints": [waypoint(right, "park", RIGHT_START, [180, 0, 89.97], "start"),
                                    waypoint(right, "rotate", [-300, -190, 500], RPY, "rotate"),
                                    waypoint(right, "approach", standoff_xyz, RPY, "approach")]}

    def refine_fn(base, delta):
        xyz = np.array(base["xyz"]) + _rot(base["rpy"]) @ np.asarray(delta, float)
        return waypoint(right, "refined", xyz, base["rpy"])

    def default_observe(state):
        pose = np.array(state["pose"][:3])
        err = _rot(state["pose"][3:]).T @ (BOTTLE - pose)  # tool X (vertical), Y (lateral), Z (range)
        return {"ok": True, "safe_to_close": True, "error_tool_mm": [float(v) for v in err]}

    cycle = HandoffCycle(left, right, left_loaded_code=LOADED, plan_fn=plan_fn, refine_fn=refine_fn,
                         observe_fn=observe or default_observe,
                         transfer_fn=lambda before, after: {"held": held},
                         capture_fn=lambda: events.append(("both", "capture")) or len(events),
                         verify_hold_fn=verify_hold, stop_pressed=stop, out=lambda _l: None)
    return cycle, left, right, events


def grips(events):
    return [e for e in events if e[1] == "grip"]


class HandoffCycleTest(unittest.TestCase):
    def test_success_closes_right_before_left_opens_and_checks_downward_move(self):
        cycle, left, right, events = build()
        log = cycle.run()
        self.assertTrue(log["result"]["success"])
        self.assertEqual(grips(events), [("right", "grip", 0), ("left", "grip", 255)])
        names = [s["step"] for s in log["steps"]]
        self.assertLess(names.index("right_closed"), names.index("left_released"))
        self.assertLess(names.index("left_released"), names.index("right_down_check"))
        # contact: pads on the bottle point; check move: straight down 15 mm
        np.testing.assert_allclose(right.pose[:3], BOTTLE - [0, 0, 15.0], atol=0.05)

    def test_vision_corrects_a_plan_that_is_off_sideways_and_vertically(self):
        truth = BOTTLE + np.array([9.0, 0.0, -7.0])  # the bottle is really elsewhere than the plan assumed

        def seen(state):
            err = _rot(state["pose"][3:]).T @ (truth - np.array(state["pose"][:3]))
            return {"ok": True, "safe_to_close": True, "error_tool_mm": [float(v) for v in err]}

        cycle, left, right, events = build(observe=seen)
        log = cycle.run()
        self.assertTrue(log["result"]["success"])
        np.testing.assert_allclose(right.pose[:3], truth - [0, 0, 15.0], atol=3.5)  # within the 3 mm tolerance
        self.assertTrue(any(s["step"].startswith("align_") for s in log["steps"]))

    def test_right_close_outside_hold_window_never_opens_left(self):
        cycle, left, right, events = build(right_final=3)
        with self.assertRaisesRegex(HandoffRefused, "body-hold window"):
            cycle.run()
        self.assertNotIn(("left", "grip", 255), events)

    def test_left_that_does_not_open_blocks_the_right_move(self):
        cycle, left, right, events = build(left_open_ok=False)
        moves_before = None
        with self.assertRaisesRegex(HandoffRefused, "only reads"):
            cycle.run()
        self.assertEqual(grips(events), [("right", "grip", 0), ("left", "grip", 255)])
        self.assertNotIn("right_down_check", [s["step"] for s in cycle.log["steps"]])

    def test_bottle_that_stayed_with_the_left_is_regrasped_then_right_opens(self):
        cycle, left, right, events = build(held=False)
        log = cycle.run()
        self.assertEqual(log["result"]["status"], "handoff_aborted_left_regrasped")
        self.assertEqual(grips(events), [("right", "grip", 0), ("left", "grip", 255),
                                         ("left", "grip", 0), ("right", "grip", 255)])

    def test_uncertain_transfer_changes_nothing_further(self):
        # Vision undecided AND the right jaw no longer at its close reading: stop, touch nothing.
        cycle, left, right, events = build(held=None)
        right.gripper_position = lambda: 60 if right.gripper_pos != 254 and right.gripper_pos < 250 else right.gripper_pos
        log = cycle.run()
        self.assertEqual(log["result"]["status"], "transfer_uncertain_both_unchanged")
        self.assertEqual(grips(events), [("right", "grip", 0), ("left", "grip", 255)])

    def test_left_without_loaded_gripper_refuses_before_any_motion(self):
        cycle, left, right, events = build(left_code=4)
        with self.assertRaisesRegex(HandoffRefused, "loaded-neck code"):
            cycle.run()
        self.assertEqual((left.sent, right.sent), ([], []))

    def test_failed_alignment_stops_without_touching_grippers(self):
        cycle, left, right, events = build(observe=lambda s: {"ok": False, "reason": "no bottle in view"})
        with self.assertRaisesRegex(HandoffRefused, "no bottle in view"):
            cycle.run()
        self.assertEqual(grips(events), [])
        self.assertGreaterEqual(left.stops + right.stops, 2)

    def test_stop_key_halts_and_leaves_grippers_alone(self):
        cycle, left, right, events = build(stop=lambda: True)
        with self.assertRaises(JogStopped):
            cycle.run()
        self.assertEqual(grips(events), [])

    def test_measured_range_far_from_the_plan_is_refused_before_moving_closer(self):
        cycle, left, right, events = build(standoff_error=90.0)  # real distance 290 vs planned 200
        with self.assertRaisesRegex(HandoffRefused, "sanity limit"):
            cycle.run()
        self.assertEqual(grips(events), [])

    def test_other_vessel_far_below_the_pads_is_refused_on_the_first_look(self):
        truth = BOTTLE + np.array([0.0, 0.0, -110.0])   # e.g. the beaker on the balance under the hand

        def seen(state):
            err = _rot(state["pose"][3:]).T @ (truth - np.array(state["pose"][:3]))
            return {"ok": True, "safe_to_close": True, "error_tool_mm": [float(v) for v in err]}

        cycle, left, right, events = build(observe=seen)
        with self.assertRaisesRegex(HandoffRefused, "another vessel"):
            cycle.run()
        self.assertEqual(grips(events), [])

    def test_hang_not_confirmed_by_the_top_camera_stops_before_the_right_arm_moves(self):
        seen = []
        cycle, left, right, events = build(verify_hold=lambda state: seen.append(state["pose"]) or
                                           {"ok": False, "reason": "bottom +35 mm from the predicted hang"})
        start = list(right.pose)
        with self.assertRaisesRegex(HandoffRefused, "bottle hang"):
            cycle.run()
        self.assertEqual(len(seen), 1)
        np.testing.assert_allclose(seen[0][:3], [250.0, -300.0, 500.0], atol=0.05)   # measured at the station
        self.assertEqual(list(right.pose), start)
        self.assertEqual(grips(events), [])

    def test_confirmed_hang_is_logged_and_the_handoff_continues(self):
        cycle, left, right, events = build(verify_hold=lambda state: {"ok": True, "xy_offset_mm": [1.0, -2.0],
                                                                      "bottom_offset_mm": 3.0})
        log = cycle.run()
        self.assertTrue(log["result"]["success"])
        self.assertTrue(log["hang_check"]["ok"])
        names = [s["step"] for s in log["steps"]]
        self.assertLess(names.index("hang_confirmed"), names.index("rotate"))

    def test_until_hang_stops_with_the_right_arm_untouched(self):
        cycle, left, right, events = build(verify_hold=lambda state: {"ok": True})
        cycle.stop_after_hang = True
        start = list(right.pose)
        log = cycle.run()
        self.assertEqual(log["result"]["status"], "stopped_after_hang_check")
        self.assertEqual(list(right.pose), start)
        self.assertEqual(grips(events), [])

    def test_gates_failing_at_the_end_never_close(self):
        def gated(state):
            base = np.array(state["pose"][:3])
            err = _rot(state["pose"][3:]).T @ (BOTTLE - base)
            return {"ok": True, "safe_to_close": False, "reason": "gates failed: not_touching_pads",
                    "error_tool_mm": [float(v) for v in err]}

        cycle, left, right, events = build(observe=gated)
        with self.assertRaisesRegex(HandoffRefused, "close gates failed"):
            cycle.run()
        self.assertEqual(grips(events), [])

    def test_overshoot_past_the_bottle_axis_never_closes(self):
        def overshoot(state):
            return {"ok": True, "safe_to_close": True, "error_tool_mm": [0.0, 0.0, -20.0]}

        cycle, left, right, events = build(observe=overshoot)
        # first reading is also far from the planned standoff, so either guard may fire; neither may close
        with self.assertRaises(HandoffRefused):
            cycle.run()
        self.assertEqual(grips(events), [])

    def test_one_bad_frame_during_approach_is_retaken_not_refused(self):
        # 2026-11-23 16:07: one frame jumped (+79 mm vertical, range 146 -> 218 mm) because the mask
        # merged another object; the next look at the same pose is fine.
        calls = {"n": 0}

        def glitch(state):
            calls["n"] += 1
            err = _rot(state["pose"][3:]).T @ (BOTTLE - np.array(state["pose"][:3]))
            if calls["n"] == 3:
                return {"ok": True, "safe_to_close": True, "error_tool_mm": [79.0, -2.0, float(err[2]) + 70.0]}
            return {"ok": True, "safe_to_close": True, "error_tool_mm": [float(v) for v in err]}

        cycle, left, right, events = build(observe=glitch)
        log = cycle.run()
        self.assertTrue(log["result"]["success"])
        self.assertEqual(len(log["observation_retries"]), 1)

    def test_range_that_does_not_shrink_is_retaken_then_refused_if_it_persists(self):
        calls = {"n": 0}

        def stuck(state):
            calls["n"] += 1
            err = _rot(state["pose"][3:]).T @ (BOTTLE - np.array(state["pose"][:3]))
            vertical, lateral = float(err[0]), float(err[1])
            if calls["n"] >= 3:  # from the second approach look on, report a far bottle and no centring
                return {"ok": False, "reason": "no bottle"}
            return {"ok": True, "safe_to_close": True, "error_tool_mm": [vertical, lateral, float(err[2])]}

        cycle, left, right, events = build(observe=stuck)
        with self.assertRaisesRegex(HandoffRefused, "range unavailable"):
            cycle.run()
        self.assertEqual(grips(events), [])

    def test_noisy_vertical_reading_during_approach_does_not_oscillate_into_a_refusal(self):
        # 16:48: +-5 mm frame noise with full-gain correction swung to -6.5 mm and was refused.
        calls = {"n": 0}

        def noisy(state):
            calls["n"] += 1
            err = _rot(state["pose"][3:]).T @ (BOTTLE - np.array(state["pose"][:3]))
            noise = 0.0 if calls["n"] <= 2 else (5.5 if calls["n"] % 2 else -5.5)
            return {"ok": True, "safe_to_close": True,
                    "error_tool_mm": [float(err[0]) + noise, float(err[1]), float(err[2])]}

        cycle, left, right, events = build(observe=noisy)
        log = cycle.run()
        self.assertTrue(log["result"]["success"])
        self.assertNotIn("observation_retries", log)

    def test_uncertain_vision_with_unchanged_right_jaw_counts_as_handed_over(self):
        # 16:58: the bottle was in the right jaws (180 before and after the 15 mm down check) but the
        # image check could not decide.
        cycle, left, right, events = build(held=None)
        log = cycle.run()
        self.assertTrue(log["result"]["success"])
        self.assertEqual(log["result"]["status"], "handed_over_to_right")
        self.assertIn("right jaw", log["transfer_verdict"]["source"])

    def test_uncertain_vision_with_a_jaw_that_ran_on_stays_uncertain(self):
        cycle, left, right, events = build(held=None)
        original = right.gripper_position

        def slipped():
            return 40 if any(e[1] == "grip" and e[0] == "left" and e[2] != 0 for e in events) else original()

        right.gripper_position = slipped
        log = cycle.run()
        self.assertFalse(log["result"]["success"])
        self.assertEqual(log["result"]["status"], "transfer_uncertain_both_unchanged")

    def test_bad_depth_just_before_the_axis_is_replaced_by_the_commanded_distance(self):
        # 17:13: 14.7 mm to go, step 12 -> next reading 18.4 mm (depth through the glass); stepping
        # by it put the pads 9 mm past the axis. The near-range reading must not drive the last step.
        def glassy(state):
            err = _rot(state["pose"][3:]).T @ (BOTTLE - np.array(state["pose"][:3]))
            true_rng = float(err[2])
            reported = true_rng + 16.0 if 0.5 < true_rng < 40.0 else true_rng  # the look at FINAL_LOOK_MM
            return {"ok": True, "safe_to_close": True, "error_tool_mm": [float(err[0]), float(err[1]), reported]}

        cycle, left, right, events = build(observe=glassy)
        log = cycle.run()
        self.assertTrue(log["result"]["success"])
        self.assertTrue(log.get("dead_reckoning"))
        final = np.array(right.pose[:3])
        self.assertLess(abs(float(_rot(right.pose[3:]).T[2] @ (BOTTLE - final))), 16.0)

    def test_off_axis_drift_during_approach_is_refused(self):
        calls = {"n": 0}

        def drifting(state):
            calls["n"] += 1
            err = _rot(state["pose"][3:]).T @ (BOTTLE - np.array(state["pose"][:3]))
            lateral = float(err[1]) + (0.0 if calls["n"] <= 1 else 20.0)
            return {"ok": True, "safe_to_close": True, "error_tool_mm": [float(err[0]), lateral, float(err[2])]}

        cycle, left, right, events = build(observe=drifting)
        with self.assertRaisesRegex(HandoffRefused, "drifted off the bottle axis"):
            cycle.run()
        self.assertEqual(grips(events), [])


if __name__ == "__main__":
    unittest.main()
