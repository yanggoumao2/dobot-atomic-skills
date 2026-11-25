"""Regression checks for A-owned serial gripper recovery, without hardware."""
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dobot_control import gripper_recovery as recovery
from dobot_control.act_session import ACTSession
from dobot_control.robots.robot_node import ZMQServerRobot


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.right = Mock(robot_ip='192.168.5.1', _use_gripper=True)
        self.left = Mock(robot_ip='192.168.5.2', _use_gripper=True)
        self.robot = SimpleNamespace(_robot_l=self.right, _robot_r=self.left)

    @patch.object(recovery, 'require_teleop_stopped')
    def test_both_physical_sides_use_speed_10_and_update_command_cache(self, stopped):
        for side, action, target, robot, other in (
            ('left', 'open', 255, self.left, self.right),
            ('right', 'close', 0, self.right, self.left),
        ):
            self.left.reset_mock()
            self.right.reset_mock()
            robot.gripper.get_current_position.return_value = target
            result = recovery.recover_gripper(self.robot, side, action)
            robot.gripper.move.assert_called_once_with(target, 10, 1)
            self.assertEqual(robot._last_gripper_cmd, target)
            self.assertEqual(result['position'], target)
            robot.robot.ServoJ.assert_not_called()
            robot.gripper.disconnect.assert_not_called()
            other.gripper.move.assert_not_called()

    def test_active_b_prevents_any_gripper_write(self):
        with patch.object(recovery, 'require_teleop_stopped', side_effect=RuntimeError('B running')):
            with self.assertRaisesRegex(RuntimeError, 'B running'):
                recovery.recover_gripper(self.robot, 'left', 'open')
        self.left.gripper.move.assert_not_called()

    def test_process_query_detects_b_and_query_failure(self):
        for code in (2, 1):
            with patch.object(recovery.os, 'name', 'nt'), patch.object(recovery.subprocess, 'run', return_value=SimpleNamespace(returncode=code)):
                with self.assertRaises(RuntimeError):
                    recovery.require_teleop_stopped()

    def test_active_act_session_prevents_recovery(self):
        server = object.__new__(ZMQServerRobot)
        server._act_session = ACTSession()
        server._act_session.begin()
        server._robot = self.robot
        with patch.object(recovery, 'recover_gripper') as operation:
            with self.assertRaisesRegex(RuntimeError, 'ACT'):
                server.recover_gripper('left', 'open')
            operation.assert_not_called()

    @patch.object(recovery, 'require_teleop_stopped')
    def test_timeout_does_not_repeat_move(self, stopped):
        self.left.gripper.get_current_position.return_value = 0
        with patch.object(recovery.time, 'monotonic', side_effect=[0, 11]):
            with self.assertRaises(TimeoutError):
                recovery.recover_gripper(self.robot, 'left', 'open')
        self.left.gripper.move.assert_called_once_with(255, 10, 1)

    def test_invalid_side_never_opens_device(self):
        with patch.object(recovery, 'require_teleop_stopped') as stopped:
            with self.assertRaises(ValueError):
                recovery.recover_gripper(self.robot, 'other', 'open')
            stopped.assert_not_called()

    def test_rpc_dispatch_reaches_recovery_and_returns_feedback(self):
        import threading
        import zmq
        server = object.__new__(ZMQServerRobot)
        server._act_session = ACTSession()
        server._robot = self.robot
        server._stop_event = threading.Event()
        server._socket = Mock()
        replies = []
        request = dict(method='recover_gripper', args=dict(side='left', action='open'))
        def receive():
            if replies:
                server.stop()
                raise zmq.Again()
            return pickle.dumps(request)
        server._socket.recv.side_effect = receive
        server._socket.send.side_effect = lambda data: replies.append(pickle.loads(data))
        with patch.object(recovery, 'require_teleop_stopped'):
            self.left.gripper.get_current_position.return_value = 255
            server.serve()
        self.assertEqual(replies[0]['position'], 255)
        self.left.gripper.move.assert_called_once_with(255, 10, 1)


if __name__ == '__main__':
    unittest.main()
