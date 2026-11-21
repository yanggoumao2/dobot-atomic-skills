"""Focused tests for the hardware-free Nova2 arm model."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from nova2_arm.model import (
    ArmModel,
    ArmSettings,
    CommandValidationError,
    joint_names,
)


class ArmSettingsTests(unittest.TestCase):
    def test_accepts_both_isolated_sides_and_default_rate(self):
        self.assertEqual(ArmSettings.from_mapping({"side": "left"}).side, "left")
        self.assertEqual(ArmSettings.from_mapping({"side": "right"}).side, "right")
        self.assertEqual(
            ArmSettings.from_mapping({"side": "left"}).feedback_rate_hz,
            10.0,
        )

    def test_rejects_invalid_config(self):
        for config in (
            {},
            {"side": "center"},
            {"side": "left", "feedback_rate_hz": 0},
            {"side": "left", "feedback_rate_hz": math.inf},
            {"side": "left", "feedback_rate_hz": True},
            {"side": "left", "unknown": 1},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ArmSettings.from_mapping(config)


class ArmModelTests(unittest.TestCase):
    def test_names_and_layout_home_are_side_specific(self):
        left = ArmModel("left")
        right = ArmModel("right")

        self.assertEqual(left.feedback().name, tuple(f"left_joint{i}" for i in range(1, 7)))
        self.assertEqual(right.feedback().name, tuple(f"right_joint{i}" for i in range(1, 7)))
        self.assertEqual(
            left.feedback().position,
            tuple(math.radians(value) for value in (90, 0, 90, 0, -90, -90)),
        )
        self.assertEqual(
            right.feedback().position,
            tuple(math.radians(value) for value in (-90, 0, -90, 0, 90, 90)),
        )

    def test_loads_all_six_urdf_joint_ranges(self):
        expected = (
            (-6.283185307180, 6.283185307180),
            (-3.141592653590, 3.141592653590),
            (-2.722713633111, 2.722713633111),
            (-6.283185307180, 6.283185307180),
            (-6.283185307180, 6.283185307180),
            (-6.283185307180, 6.283185307180),
        )
        self.assertEqual(ArmModel("left", home_positions=(0.0,) * 6)._limits, expected)
        self.assertEqual(ArmModel("right", home_positions=(0.0,) * 6)._limits, expected)

    def test_applies_reordered_complete_command_in_canonical_order(self):
        model = ArmModel("left", home_positions=(0.0,) * 6)
        names = tuple(reversed(joint_names("left")))
        positions = (0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

        frame = model.apply_command(names, positions)

        self.assertEqual(frame.name, joint_names("left"))
        self.assertEqual(frame.position, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        self.assertEqual(frame.velocity, ())
        self.assertEqual(frame.effort, ())

    def test_accepts_jointstate_shaped_object(self):
        model = ArmModel("right", home_positions=(0.0,) * 6)
        command = SimpleNamespace(
            name=joint_names("right"),
            position=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            velocity=(),
            effort=(),
        )

        self.assertEqual(
            model.apply_joint_state(command).position,
            command.position,
        )

    def test_invalid_commands_leave_the_full_state_unchanged(self):
        model = ArmModel("left", home_positions=(0.0,) * 6)
        names = joint_names("left")
        model.apply_command(names, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        before = model.feedback()

        invalid_commands = (
            (names[:-1], (0.1, 0.2, 0.3, 0.4, 0.5), (), ()),
            ((names[0],) * 6, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), (), ()),
            (("right_joint1", *names[1:]), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), (), ()),
            (names, (0.1, 0.2, 0.3, 0.4, 0.5), (), ()),
            (names, (0.1, 0.2, math.nan, 0.4, 0.5, 0.6), (), ()),
            (names, (0.1, 0.2, math.inf, 0.4, 0.5, 0.6), (), ()),
            (names, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), (0.0,) * 6, ()),
            (names, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), (), (0.0,) * 6),
            (names, (0.1, 0.2, 3.0, 0.4, 0.5, 0.6), (), ()),
        )
        for command in invalid_commands:
            with self.subTest(command=command), self.assertRaises(CommandValidationError):
                model.apply_command(
                    command[0],
                    command[1],
                    velocity=command[2],
                    effort=command[3],
                )
            self.assertEqual(model.feedback(), before)

    def test_rejects_wrong_side_and_bad_home_state(self):
        with self.assertRaises(ValueError):
            ArmModel("center")
        with self.assertRaises(ValueError):
            ArmModel("left", home_positions=(0.0,))
        with self.assertRaises(CommandValidationError):
            ArmModel("left", home_positions=(0.0, 0.0, 3.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
