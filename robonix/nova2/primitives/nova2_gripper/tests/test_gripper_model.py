"""Deterministic tests for the SDK-independent simulated jaw model."""

from __future__ import annotations

import unittest

from nova2_gripper.model import (
    GRASP_UNKNOWN,
    MOTION_MOVING,
    MOTION_STOPPED,
    STROKE_M,
    MockGripper,
)


class FakeClock:
    def __init__(self) -> None:
        self.elapsed_s = 0.0
        self.epoch_ns = 1_800_000_000_000_000_000

    def monotonic(self) -> float:
        return self.elapsed_s

    def time_ns(self) -> int:
        return self.epoch_ns + int(self.elapsed_s * 1_000_000_000)

    def advance(self, seconds: float) -> None:
        self.elapsed_s += seconds


class MockGripperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.gripper = MockGripper(
            "left",
            monotonic=self.clock.monotonic,
            time_ns=self.clock.time_ns,
        )

    def test_info_reports_fixed_mock_stroke_and_unknown_force(self) -> None:
        info = self.gripper.info()

        self.assertTrue(info["ok"])
        self.assertEqual(info["max_opening_width_m"], 0.08)
        self.assertEqual(info["max_opening_width_m"], STROKE_M)
        self.assertTrue(info["opening_width_range_valid"])
        self.assertFalse(info["gripping_force_range_valid"])
        self.assertFalse(info["grip_effort_configurable"])
        self.assertFalse(info["grip_with_expected_object_width_supported"])
        self.assertTrue(info["grip_without_expected_object_width_supported"])

    def test_move_advances_synthetic_width_then_stops(self) -> None:
        result = self.gripper.move(0.0)
        self.assertTrue(result["ok"])
        self.assertEqual(self.gripper.state()["motion_state"], MOTION_MOVING)

        self.clock.advance(0.5)
        halfway = self.gripper.state()
        self.assertAlmostEqual(halfway["opening_width_m"], 0.04)
        self.assertEqual(halfway["motion_state"], MOTION_MOVING)
        self.assertEqual(halfway["grasp_state"], GRASP_UNKNOWN)

        self.clock.advance(0.5)
        complete = self.gripper.state()
        self.assertEqual(complete["opening_width_m"], 0.0)
        self.assertEqual(complete["motion_state"], MOTION_STOPPED)

    def test_stop_freezes_width_without_opening(self) -> None:
        self.gripper.move(0.0)
        self.clock.advance(0.25)
        width_before_stop = self.gripper.state()["opening_width_m"]

        result = self.gripper.stop()
        self.clock.advance(2.0)
        stopped = self.gripper.state()

        self.assertTrue(result["ok"])
        self.assertAlmostEqual(stopped["opening_width_m"], width_before_stop)
        self.assertEqual(stopped["motion_state"], MOTION_STOPPED)

    def test_grip_rejects_expected_width_without_an_object_model(self) -> None:
        result = self.gripper.grip(
            expected_object_width_valid=True,
            expected_object_width_m=0.04,
        )
        self.assertFalse(result["ok"])
        self.assertIn("no object model", result["message"])
        self.assertEqual(self.gripper.state()["opening_width_m"], STROKE_M)
        self.assertEqual(self.gripper.state()["grasp_state"], GRASP_UNKNOWN)

    def test_grip_without_expected_width_ignores_width_value(self) -> None:
        result = self.gripper.grip(
            expected_object_width_valid=False,
            expected_object_width_m=float("inf"),
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self.gripper.state()["motion_state"], MOTION_MOVING)
        self.assertEqual(self.gripper.state()["grasp_state"], GRASP_UNKNOWN)

    def test_invalid_widths_and_unsupported_settings_are_rejected(self) -> None:
        self.assertFalse(self.gripper.move(-0.001)["ok"])
        self.assertFalse(self.gripper.move(STROKE_M + 0.001)["ok"])
        self.assertFalse(self.gripper.move(float("nan"))["ok"])
        self.assertFalse(self.gripper.move(0.04, relative_speed=0.5)["ok"])
        self.assertFalse(self.gripper.open(relative_speed=1.01)["ok"])
        self.assertFalse(self.gripper.grip(relative_effort=0.1)["ok"])
        self.assertEqual(self.gripper.state()["opening_width_m"], STROKE_M)
        self.assertEqual(self.gripper.state()["motion_state"], MOTION_STOPPED)

    def test_width_value_is_ignored_without_valid_flag(self) -> None:
        accepted = self.gripper.grip(
            expected_object_width_valid=False,
            expected_object_width_m=100.0,
        )

        self.assertTrue(accepted["ok"])

    def test_each_side_has_independent_model_state(self) -> None:
        right = MockGripper(
            "right",
            monotonic=self.clock.monotonic,
            time_ns=self.clock.time_ns,
        )
        self.gripper.move(0.0)
        self.clock.advance(0.5)

        self.assertEqual(self.gripper.side, "left")
        self.assertEqual(right.side, "right")
        self.assertAlmostEqual(self.gripper.state()["opening_width_m"], 0.04)
        self.assertEqual(right.state()["opening_width_m"], STROKE_M)

    def test_lifecycle_gates_motion_and_deactivate_freezes_jaws(self) -> None:
        inactive = MockGripper(
            "left",
            active=False,
            monotonic=self.clock.monotonic,
            time_ns=self.clock.time_ns,
        )
        rejected = inactive.move(0.0)
        self.assertFalse(rejected["ok"])
        self.assertEqual(inactive.state()["opening_width_m"], STROKE_M)

        inactive.activate()
        self.assertTrue(inactive.move(0.0)["ok"])
        self.clock.advance(0.25)
        inactive.deactivate()
        stopped_width = inactive.state()["opening_width_m"]
        self.assertEqual(inactive.state()["motion_state"], MOTION_STOPPED)

        self.clock.advance(2.0)
        self.assertFalse(inactive.open()["ok"])
        self.assertFalse(inactive.grip()["ok"])
        self.assertEqual(inactive.state()["opening_width_m"], stopped_width)

        inactive.activate()
        self.assertTrue(inactive.open()["ok"])

    def test_state_timestamp_uses_unix_epoch(self) -> None:
        state = self.gripper.state()

        self.assertEqual(state["stamp_sec"], 1_800_000_000)
        self.assertEqual(state["stamp_nanosec"], 0)

    def test_side_must_be_left_or_right(self) -> None:
        with self.assertRaisesRegex(ValueError, "side must be 'left' or 'right'"):
            MockGripper("center")


if __name__ == "__main__":
    unittest.main()
