"""Exercise the real main-loop branches with every hardware boundary mocked."""
import threading
import unittest
import configparser
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from experiments import run_control as control


class EndSimulation(BaseException):
    pass


class ServoEnableCycleTests(unittest.TestCase):
    def simulate(self, hand, moving=False, real_safety=False, recording=False, dispatch_fault=False, pose_fault=False, unverified=False, unsafe_alignment=False):
        states = np.zeros((2, 3), dtype=int)
        held_pose = np.zeros(14)
        held_pose[[0, 2, 4, 5, 7, 9, 11, 12]] = [1.57, 1.57, -1.57, -1.57, -1.57, -1.57, 1.57, 1.57]
        commands, durations, dispatch_cycles = [], [], []
        measured = held_pose.copy()
        clock = [0.0]
        def monotonic():
            clock[0] += 0.005
            return clock[0]
        agent, env = Mock(), Mock()
        writer = Mock(dropped_count=0)
        recorded = []
        writer.submit_factory.side_effect = lambda factory: recorded.append(factory()) or True
        cycles = 0
        selected_hands = (0, 1) if hand == 2 else (hand,)

        def master_read(_obs):
            nonlocal cycles
            cycles += 1
            if cycles == 26:
                raise EndSimulation()
            if cycles == 13:
                states[:, 1] = 0
            elif cycles in (1, 14):
                for side in selected_hands:
                    states[side, 1] = 1
            states[0, 2] = 1  # exercise B/record branch, even in top-only mode
            if cycles > 30:
                raise AssertionError("simulation failed to dispatch expected cycles")
            sample = np.concatenate([held_pose[7:], held_pose[:7]])
            for side in selected_hands:
                sample[side * 7] += 0.004
            if moving:
                for side in selected_hands:
                    sample[side * 7] += max(0, cycles - 7) * 0.0005
            return sample

        def dispatch(action, arm_flags, gripper_flags):
            commands.append(action.copy())
            dispatch_cycles.append(cycles)
            if dispatch_fault and cycles == 11:
                raise RuntimeError("simulated transport failure")
            measured[:] = action
            return {"joint_positions": measured.copy()}

        def velocities(action, previous, elapsed, robot_type):
            durations.append(elapsed)
            if real_safety:
                return control.calculate_vel_pos_original(action, previous, elapsed, robot_type)
            return ({key: np.zeros(3) for key in ("left_left", "left_right", "right_left", "right_right")},
                    {key: np.zeros(3) for key in ("left_left", "left_right", "right_left", "right_right")})

        agent.act.side_effect = master_read
        env.get_obs.side_effect = lambda: {"joint_positions": measured.copy()}
        env.step.side_effect = dispatch

        class FakeButtonThread:
            def __init__(self, target, args, daemon):
                self.ready = args[3]
            def start(self):
                self.ready.set()
            def is_alive(self):
                return True

        healthy_camera = Mock()
        healthy_camera.is_alive.return_value = True
        fake_ini = configparser.ConfigParser()
        for section, physical in (("HAND_LEFT","right"),("HAND_RIGHT","left")):
            fake_ini.add_section(section)
            fake_ini.set(section,"absolute_mapping_verified","0" if unverified else "1")
            fake_ini.set(section,"physical_side",physical)
        replacements = {
            "mk_dir": Mock(), "BoundedAsyncWriter": Mock(return_value=writer),
            "load_camera_settings": Mock(return_value=SimpleNamespace(serials={"top": "111"}, fps=15, recording_enabled=recording)),
            "start_configured_cameras": Mock(return_value=({"top": healthy_camera}, threading.Event(), threading.Event(), Mock())),
            "load_ini_data_hands": Mock(return_value=(fake_ini, {"HAND_LEFT": Mock(port="COM6"), "HAND_RIGHT": Mock(port="COM5")})),
            "DobotAgent": Mock(), "BimanualAgent": Mock(return_value=agent),
            "ZMQClientRobot": Mock(), "RobotEnv": Mock(return_value=env),
            "check_firmware_version": Mock(return_value=True),
            "get_robot_type": Mock(return_value="Nova 2"), "robot_pose_init": Mock(),
            "runtime_log": Mock(), "set_light": Mock(return_value="yellow"),
            "servo_action_check": Mock(side_effect=lambda action, previous, flags: (1, action)),
            "calculate_vel_pos": Mock(side_effect=velocities),
            "check_pose_protection": Mock(side_effect=lambda *args: pose_fault and cycles == 11),
            "check_joint_safety": Mock(return_value=False),
            "validate_absolute_alignment": Mock(return_value=(not unsafe_alignment,{"samples":2})),
            "what_to_do": states, "is_falling": np.array([0]),
        }
        if real_safety:
            for name in ("servo_action_check", "check_pose_protection", "check_joint_safety"):
                replacements.pop(name)
        with ExitStack() as stack:
            stack.enter_context(patch.object(control, "calculate_vel_pos_original", control.calculate_vel_pos, create=True))
            for name, value in replacements.items():
                stack.enter_context(patch.object(control, name, value))
            stack.enter_context(patch.object(control.threading, "Thread", FakeButtonThread))
            stack.enter_context(patch.object(control.atexit, "register"))
            stack.enter_context(patch.object(control.time, "monotonic", side_effect=monotonic))
            stack.enter_context(redirect_stdout(StringIO()))
            with self.assertRaises(EndSimulation):
                control.main(control.Args())
        if unverified or unsafe_alignment:
            self.assertEqual(commands, [])
            event = "ALIGN_REJECTED_UNCALIBRATED" if unverified else "ALIGN_REJECTED_UNSAFE"
            self.assertTrue(any(call.args[0] == event for call in replacements["runtime_log"].call_args_list))
            for side in (0,1):
                agent.set_torque.assert_any_call(side,True)
            self.assertFalse(any(call.args[0] == "ALIGN_START" for call in replacements["runtime_log"].call_args_list))
            return
        self.assertEqual(len(commands), 22 if pose_fault else 23 if dispatch_fault else 24)
        for side in selected_hands:
            target_index = 7 if side == 0 else 0
            self.assertGreater(commands[0][target_index], held_pose[target_index])
            self.assertLessEqual(commands[0][target_index] - held_pose[target_index], 0.15 * 0.05)
            expected = held_pose[target_index] + 0.004 + (0.009 if moving else 0)
            self.assertAlmostEqual(commands[-1][target_index], expected)
        self.assertGreaterEqual(len(durations), len(commands))
        self.assertTrue(any(call.args[0] == "ALIGN_DONE" for call in replacements["runtime_log"].call_args_list))
        self.assertTrue(all(np.isfinite(dt) and dt >= 1e-3 for dt in durations))
        if moving:
            for side in selected_hands:
                target_index = 7 if side == 0 else 0
                self.assertGreater(commands[-1][target_index], held_pose[target_index])
            if len(selected_hands) == 1:
                inactive_index = 0 if hand == 0 else 7
                self.assertEqual(commands[-1][inactive_index], held_pose[inactive_index])
        if recording:
            self.assertTrue(recorded)
            self.assertEqual(recorded[-1].action.shape, (14,))
        else:
            writer.submit_factory.assert_not_called()
        if dispatch_fault:
            self.assertNotIn(12, dispatch_cycles)
            self.assertTrue(any(call.args[0] == "SERVO_FAILURE"
                                for call in replacements["runtime_log"].call_args_list))
        if pose_fault:
            self.assertNotIn(11, dispatch_cycles)
            self.assertNotIn(12, dispatch_cycles)
            resets = [call.kwargs["reason"] for call in replacements["runtime_log"].call_args_list
                      if call.args[0] == "SERVO_STATE_RESET"]
            self.assertIn("pose_protection", resets)
            self.assertNotIn("joint_safety", resets)

    def test_left_enable_align_follow_and_reenable(self):
        self.simulate(0)

    def test_right_enable_align_follow_and_reenable(self):
        self.simulate(1)

    def test_left_motion_reaches_crossed_follower(self):
        self.simulate(0, moving=True)

    def test_right_motion_reaches_crossed_follower(self):
        self.simulate(1, moving=True)

    def test_both_hands_move_with_real_kinematics_and_safety(self):
        self.simulate(2, moving=True, real_safety=True)

    def test_optional_recording_snapshot_branch(self):
        self.simulate(1, moving=True, recording=True)

    def test_transport_failure_stops_until_explicit_reenable(self):
        self.simulate(1, moving=True, dispatch_fault=True)

    def test_workspace_rejection_stops_and_logs_real_reason(self):
        self.simulate(1, moving=True, pose_fault=True)

    def test_unverified_mapping_never_dispatches(self):
        self.simulate(1, unverified=True)

    def test_unsafe_alignment_never_dispatches(self):
        self.simulate(1, unsafe_alignment=True)


if __name__ == "__main__":
    unittest.main()
