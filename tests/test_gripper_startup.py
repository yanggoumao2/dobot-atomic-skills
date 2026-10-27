import unittest
from unittest.mock import Mock, patch

from dobot_control.robots.dobot import DobotRobot


class GripperStartupTests(unittest.TestCase):
    def robot(self):
        robot = DobotRobot.__new__(DobotRobot)
        robot.gripper = Mock()
        return robot

    def test_waits_for_first_endpoint_before_reversing(self):
        robot = self.robot()
        events = []
        positions = iter((50, 0, 100, 255))
        robot.gripper.move.side_effect = lambda *args: events.append(("move", args))
        def read():
            value = next(positions)
            events.append(("read", value))
            return value
        robot.gripper.get_current_position.side_effect = read
        with patch("dobot_control.robots.dobot.time.sleep"):
            robot._initialize_gripper()
        self.assertEqual(events, [("move", (0, 10, 1)), ("read", 50),
                                  ("read", 0), ("move", (255, 10, 1)),
                                  ("read", 100), ("read", 255)])

    def test_timeout_aborts_before_second_target(self):
        robot = self.robot()
        robot.gripper.get_current_position.return_value = 100
        with patch("dobot_control.robots.dobot.time.monotonic", side_effect=[0, 11]):
            with self.assertRaises(TimeoutError):
                robot._initialize_gripper()
        robot.gripper.move.assert_called_once_with(0, 10, 1)

    def test_read_failure_aborts_before_second_target(self):
        robot = self.robot()
        robot.gripper.get_current_position.side_effect = ValueError("read failed")
        with self.assertRaises(ValueError):
            robot._initialize_gripper()
        robot.gripper.move.assert_called_once_with(0, 10, 1)


if __name__ == "__main__":
    unittest.main()
