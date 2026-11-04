import inspect
import sys
import threading
import time
import types
import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

try:
    import serial  # noqa: F401
except ModuleNotFoundError:
    serial_module = types.ModuleType("serial")
    serial_tools_module = types.ModuleType("serial.tools")
    serial_list_ports_module = types.ModuleType("serial.tools.list_ports")
    serial_list_ports_module.comports = lambda: []
    serial_tools_module.list_ports = serial_list_ports_module
    serial_module.tools = serial_tools_module
    sys.modules["serial"] = serial_module
    sys.modules["serial.tools"] = serial_tools_module
    sys.modules["serial.tools.list_ports"] = serial_list_ports_module

try:
    import zmq  # noqa: F401
except ModuleNotFoundError:
    zmq_module = types.ModuleType("zmq")
    zmq_module.Again = type("Again", (Exception,), {})
    zmq_module.ZMQError = type("ZMQError", (Exception,), {})
    sys.modules["zmq"] = zmq_module

from dobot_control.robots.dobot import DobotRobot
from dobot_control.robots.robot import BimanualRobot
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.env import RobotEnv
from dobot_control.async_writer import BoundedAsyncWriter
from dobot_control.teleop_safety import (
    MasterRunawayGuard,
    check_pose_protection,
    consume_servo_warmup,
)
from scripts.manipulate_utils import (
    apply_same_side_gripper_targets,
    master_to_nova_action,
    master_to_nova_flags,
    master_to_nova_gripper_flags,
)


class FakeRobot:
    def __init__(self, observation_tracker=None):
        self.robot_is_err = False
        self._status_failure = None
        self.commands = []
        self.observation_tracker = observation_tracker

    def num_dofs(self):
        return 7

    def command_joint_state(
        self,
        joint_state,
        command_arm=True,
        command_gripper=True,
    ):
        self.commands.append(
            (np.asarray(joint_state).copy(), command_arm, command_gripper)
        )

    def get_observations(self):
        if self.observation_tracker is not None:
            self.observation_tracker.enter()
            time.sleep(0.02)
            self.observation_tracker.leave()
        return {
            "joint_positions": np.zeros(7),
            "joint_velocities": np.zeros(7),
            "ee_pos_quat": np.zeros(7),
            "gripper_position": np.zeros(1),
        }


class SlowAgent:
    def __init__(self, delay):
        self.delay = delay

    def act(self, obs):
        time.sleep(self.delay)
        return np.zeros(7)

    def set_torque(self, flag):
        pass

    def get_keys(self):
        time.sleep(self.delay)
        return [0, 0, 0]


class ObservationTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def enter(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self):
        with self._lock:
            self.active -= 1


class FakeMoveApi:
    def __init__(self):
        self.servo_commands = []

    def ServoJ(self, *args):
        self.servo_commands.append(args)


class FakeGripper:
    def __init__(self):
        self.moves = []

    def move(self, position, speed, acceleration):
        self.moves.append((position, speed, acceleration))


class FakeEnvRobot:
    def __init__(self):
        self.commands = []

    def num_dofs(self):
        return 14

    def command_joint_state(self, joint_state, arm_flags, gripper_flags=None):
        self.commands.append(
            (
                np.asarray(joint_state).copy(),
                np.asarray(arm_flags).copy(),
                np.asarray(gripper_flags).copy(),
            )
        )

    def get_observations(self):
        return {
            "joint_positions": np.zeros(14),
            "joint_velocities": np.zeros(14),
            "ee_pos_quat": np.zeros(14),
            "gripper_position": np.zeros(2),
        }


class BimanualDispatchTests(unittest.TestCase):
    def test_inactive_arm_still_receives_gripper_dispatch(self):
        left = FakeRobot()
        right = FakeRobot()
        robot = BimanualRobot(left, right)

        robot.command_joint_state(
            np.arange(14, dtype=float),
            arm_flags=[0, 1],
            gripper_flags=[1, 0],
        )

        self.assertEqual(len(left.commands), 1)
        self.assertFalse(left.commands[0][1])
        self.assertTrue(left.commands[0][2])
        self.assertEqual(len(right.commands), 1)
        self.assertTrue(right.commands[0][1])
        self.assertFalse(right.commands[0][2])

    def test_gripper_only_dispatch_does_not_send_servoj(self):
        self.assertIn(
            "command_arm",
            inspect.signature(DobotRobot.command_joint_state).parameters,
        )
        robot = object.__new__(DobotRobot)
        robot.robot_is_err = False
        robot.robot = FakeMoveApi()
        robot._use_gripper = True
        robot.gripper = FakeGripper()
        robot._last_gripper_cmd = None

        robot.command_joint_state(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]),
            command_arm=False,
        )

        self.assertEqual(robot.robot.servo_commands, [])
        self.assertEqual(robot.gripper.moves, [(191, 100, 1)])

    def test_arm_only_dispatch_does_not_move_gripper(self):
        robot = object.__new__(DobotRobot)
        robot.robot_is_err = False
        robot.robot = FakeMoveApi()
        robot._use_gripper = True
        robot.gripper = FakeGripper()
        robot._last_gripper_cmd = None

        robot.command_joint_state(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]),
            command_arm=True,
            command_gripper=False,
        )

        self.assertEqual(len(robot.robot.servo_commands), 1)
        self.assertAlmostEqual(robot.robot.servo_commands[0][-1], 0.03)
        self.assertEqual(robot.gripper.moves, [])

    def test_bimanual_observations_are_read_in_parallel(self):
        tracker = ObservationTracker()
        robot = BimanualRobot(
            FakeRobot(observation_tracker=tracker),
            FakeRobot(observation_tracker=tracker),
        )

        observations = robot.get_observations()

        self.assertEqual(tracker.max_active, 2)
        self.assertEqual(observations["joint_positions"].shape, (14,))

    def test_dobot_error_blocks_arm_and_gripper_writes(self):
        robot = object.__new__(DobotRobot)
        robot.robot_is_err = True
        robot.robot_ip = "fake"
        robot._status_failure = "alarm 7"
        robot.robot = FakeMoveApi()
        robot._use_gripper = True
        robot.gripper = FakeGripper()
        robot._last_gripper_cmd = None

        with self.assertRaisesRegex(RuntimeError, "alarm 7"):
            robot.command_joint_state(
                np.zeros(7),
                command_arm=True,
                command_gripper=True,
            )

        self.assertEqual(robot.robot.servo_commands, [])
        self.assertEqual(robot.gripper.moves, [])

    def test_bimanual_error_blocks_all_dispatch(self):
        left = FakeRobot()
        right = FakeRobot()
        right.robot_is_err = True
        right._status_failure = "alarm 42"
        robot = object.__new__(BimanualRobot)
        robot._robot_l = left
        robot._robot_r = right

        with self.assertRaisesRegex(RuntimeError, "right robot error: alarm 42"):
            robot.command_joint_state(
                np.zeros(14),
                arm_flags=[1, 1],
                gripper_flags=[1, 1],
            )

        self.assertEqual(left.commands, [])
        self.assertEqual(right.commands, [])

    def test_bimanual_master_reads_are_parallel(self):
        from dobot_control.agents.agent import BimanualAgent

        agent = BimanualAgent(SlowAgent(0.03), SlowAgent(0.03))
        started = time.monotonic()
        agent.act({})
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.055)


class CommandFlagTransportTests(unittest.TestCase):
    def test_master_arm_flags_cross_but_gripper_flags_stay_same_side(self):
        master_flags = np.array([1, 0])

        np.testing.assert_array_equal(master_to_nova_flags(master_flags), [0, 1])
        np.testing.assert_array_equal(
            master_to_nova_gripper_flags(master_flags),
            [1, 0],
        )

    def test_single_master_updates_only_same_side_nova_gripper(self):
        raw_master = np.zeros(14)
        raw_master[6] = 0.2
        raw_master[13] = 0.8
        crossed_master = master_to_nova_action(raw_master)
        action = np.full(14, -1.0)
        last_action = np.full(14, 0.5)

        result = apply_same_side_gripper_targets(
            action,
            crossed_master,
            last_action,
            gripper_flags=[1, 0],
        )

        self.assertEqual(result[6], 0.2)
        self.assertEqual(result[13], 0.5)

    def test_gripper_target_is_clamped_to_normalized_range(self):
        raw_master = np.zeros(14)
        raw_master[6] = 4.0
        crossed_master = master_to_nova_action(raw_master)
        result = apply_same_side_gripper_targets(
            np.zeros(14), crossed_master, np.zeros(14), gripper_flags=[1, 0]
        )
        self.assertEqual(result[6], 1.0)

    def test_env_forwards_arm_and_gripper_flags_separately(self):
        robot = FakeEnvRobot()
        env = RobotEnv(robot)
        action = np.arange(14, dtype=float)

        env.step(action, arm_flags=[0, 1], gripper_flags=[1, 0])

        self.assertEqual(len(robot.commands), 1)
        np.testing.assert_array_equal(robot.commands[0][0], action)
        np.testing.assert_array_equal(robot.commands[0][1], [0, 1])
        np.testing.assert_array_equal(robot.commands[0][2], [1, 0])

    def test_zmq_request_keeps_arm_and_gripper_flags_separate(self):
        client = object.__new__(ZMQClientRobot)
        requests = []
        client._request = lambda request: requests.append(request) or 1
        action = np.arange(14, dtype=float)

        client.command_joint_state(
            action,
            arm_flags=[0, 1],
            gripper_flags=[1, 0],
        )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["method"], "command_joint_state")
        np.testing.assert_array_equal(requests[0]["args"]["joint_state"], action)
        np.testing.assert_array_equal(requests[0]["args"]["arm_flags"], [0, 1])
        np.testing.assert_array_equal(
            requests[0]["args"]["gripper_flags"],
            [1, 0],
        )


class MasterRunawayGuardTests(unittest.TestCase):
    def test_servo_warmup_holds_follower_and_rebases_master(self):
        follower = np.arange(14, dtype=float)
        master = follower.copy()
        master[2] += 0.2

        held, reference, remaining = consume_servo_warmup(
            follower, master, remaining_cycles=3
        )

        np.testing.assert_array_equal(held, follower)
        np.testing.assert_array_equal(reference, master)
        self.assertEqual(remaining, 2)

    def test_safe_motion_is_compared_to_previous_cycle_not_unlock_pose(self):
        guard = MasterRunawayGuard(max_speed_rad_s=2.0, min_delta_rad=0.05)
        guard.reset(np.zeros(14), timestamp=0.0)

        for timestamp, position in ((0.1, 0.1), (0.2, 0.2), (0.3, 0.3)):
            sample = np.zeros(14)
            sample[0] = position
            result = guard.check(sample, arm_flags=[1, 0], timestamp=timestamp)
            self.assertFalse(result.tripped)

        self.assertAlmostEqual(result.deltas[0], 0.1)

    def test_threshold_accounts_for_a_delayed_control_cycle(self):
        guard = MasterRunawayGuard(max_speed_rad_s=2.0, min_delta_rad=0.05)
        guard.reset(np.zeros(14), timestamp=0.0)
        sample = np.zeros(14)
        sample[0] = 0.5

        result = guard.check(sample, arm_flags=[1, 0], timestamp=0.5)

        self.assertFalse(result.tripped)
        self.assertAlmostEqual(result.allowed_delta_rad, 1.0)

    def test_pose_protection_uses_crossed_nova_flags(self):
        positions = {
            "left_left": np.array([-0.5, -0.4, 0.1]),
            "left_right": np.array([-0.5, -0.4, 0.1]),
            "right_left": np.array([0.0, -0.4, 0.1]),
            "right_right": np.array([0.0, -0.4, 0.1]),
        }
        velocities = {key: np.zeros(3) for key in positions}

        self.assertFalse(
            check_pose_protection(positions, velocities, nova_flags=[0, 1])
        )

        positions["right_left"][0] = 0.5
        self.assertTrue(
            check_pose_protection(positions, velocities, nova_flags=[0, 1])
        )

    def test_fast_active_joint_jump_trips(self):
        guard = MasterRunawayGuard(
            max_speed_rad_s=2.0, min_delta_rad=0.05, required_violation_cycles=2
        )
        guard.reset(np.zeros(14), timestamp=0.0)
        sample = np.zeros(14)
        sample[2] = 0.5

        result = guard.check(sample, arm_flags=[1, 0], timestamp=0.1)
        self.assertFalse(result.tripped)
        sample[2] = 1.0
        result = guard.check(sample, arm_flags=[1, 0], timestamp=0.2)

        self.assertTrue(result.tripped)
        self.assertEqual(result.joint_indices, [2])

    def test_inactive_arm_jump_is_ignored(self):
        guard = MasterRunawayGuard(max_speed_rad_s=2.0, min_delta_rad=0.05)
        guard.reset(np.zeros(14), timestamp=0.0)
        sample = np.zeros(14)
        sample[9] = 2.0

        result = guard.check(sample, arm_flags=[1, 0], timestamp=0.1)

        self.assertFalse(result.tripped)


class BoundedAsyncWriterTests(unittest.TestCase):
    def test_full_queue_does_not_block_control_thread(self):
        write_started = threading.Event()
        release_write = threading.Event()
        written = []

        def slow_write(item):
            write_started.set()
            release_write.wait(timeout=1.0)
            written.append(item)

        writer = BoundedAsyncWriter(slow_write, max_queue_size=1)
        try:
            self.assertTrue(writer.submit(1))
            self.assertTrue(write_started.wait(timeout=1.0))
            self.assertTrue(writer.submit(2))

            started = time.monotonic()
            self.assertFalse(writer.submit(3))
            self.assertLess(time.monotonic() - started, 0.05)
            self.assertEqual(writer.dropped_count, 1)
        finally:
            release_write.set()
            writer.close(timeout=1.0)

        self.assertEqual(written, [1, 2])

    def test_full_queue_does_not_build_a_dropped_item(self):
        write_started = threading.Event()
        release_write = threading.Event()
        build_calls = []

        def slow_write(item):
            write_started.set()
            release_write.wait(timeout=1.0)

        writer = BoundedAsyncWriter(slow_write, max_queue_size=1)
        try:
            self.assertTrue(writer.submit(1))
            self.assertTrue(write_started.wait(timeout=1.0))
            self.assertTrue(writer.submit(2))

            submitted = writer.submit_factory(
                lambda: build_calls.append(True) or 3
            )

            self.assertFalse(submitted)
            self.assertEqual(build_calls, [])
        finally:
            release_write.set()
            writer.close(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
