"""Dashboard framing and observation validation; no hardware connections."""
import socket
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from dobot_control.robots.dobot_api import DobotApi
from dobot_control.robots.dobot import DobotRobot


class DashboardReplyTests(unittest.TestCase):
    def api(self, chunks, port=29999):
        api = DobotApi.__new__(DobotApi)
        api.ip, api.port = '192.168.5.2', port
        api._DobotApi__globalLock = threading.Lock()
        api._dashboard_reply_buffer = bytearray()
        api._dashboard_protocol_failed = False
        api.socket_dobot = Mock()
        api.socket_dobot.gettimeout.return_value = 3.
        api.socket_dobot.recv.side_effect = chunks
        return api

    def test_fragmented_reply_is_received_in_full(self):
        api = self.api([b'0,{1,2,',b'3,4,5,6},Get',b'Angle();'])
        self.assertEqual(api.sendRecvMsg('GetAngle()'),'0,{1,2,3,4,5,6},GetAngle();')
        api.socket_dobot.sendall.assert_called_once_with(b'GetAngle()')
        self.assertEqual(api.socket_dobot.recv.call_count,3)

    def test_combined_replies_preserve_next_frame(self):
        api = self.api([b'0,{},DO(1,1);0,{1,2,3,4,5,6},GetAngle();'])
        self.assertEqual(api.wait_reply(),'0,{},DO(1,1);')
        self.assertEqual(api.wait_reply(),'0,{1,2,3,4,5,6},GetAngle();')
        self.assertEqual(api.socket_dobot.recv.call_count,1)

    def test_partial_timeout_invalidates_stream_without_resending(self):
        api = self.api([b'0,{1,2',socket.timeout('timed out')])
        with self.assertRaisesRegex(RuntimeError,'192.168.5.2.*partial='):
            api.sendRecvMsg('GetAngle()')
        with self.assertRaisesRegex(RuntimeError,'reconnect required'):
            api.sendRecvMsg('GetAngle()')
        self.assertEqual(api.socket_dobot.sendall.call_count,1)

    def test_closed_connection_is_not_an_empty_observation(self):
        api = self.api([b''])
        with self.assertRaisesRegex(RuntimeError,'connection closed'):
            api.sendRecvMsg('GetAngle()')

    def test_motion_port_keeps_existing_no_reply_timeout_behavior(self):
        api = self.api([socket.timeout()],port=30003)
        self.assertEqual(api.sendRecvMsg('ServoJ(...)'),'')
        api.socket_dobot.send.assert_called_once()
        api.socket_dobot.sendall.assert_not_called()

    def robot(self, reply):
        robot = DobotRobot.__new__(DobotRobot)
        robot.robot_ip = '192.168.5.2'
        robot.robot_is_err = False
        robot._use_gripper = True
        robot.r_inter = Mock()
        robot.r_inter.GetAngle.return_value = reply
        robot.r_inter.GetPose.return_value = reply
        return robot

    def test_valid_angle_units_and_gripper_are_unchanged(self):
        robot = self.robot('0,{90,0,-90,0,90,90},GetAngle();')
        np.testing.assert_allclose(robot.get_joint_state(),np.r_[np.deg2rad([90,0,-90,0,90,90]),1.])

    def test_invalid_angle_replies_fail_with_ip_command_and_raw(self):
        for reply in ('0,{},GetAngle();','0,{},DO(1,1);','-1,{},GetAngle();',
                      '0,{1,2,3,4,5},GetAngle();','0,{1,2,3,4,5,nan},GetAngle();',
                      '0,{1,2,3,4,5,6},GetAngle()',None):
            with self.subTest(reply=reply), self.assertRaisesRegex(RuntimeError,'192.168.5.2 GetAngle invalid response:.*raw='):
                self.robot(reply).get_observations()

    def test_pose_uses_same_validation(self):
        robot = self.robot('0,{1,2,3,4,5,6},GetPose();')
        self.assertEqual(robot.get_XYZrxryrz_state(),[1,2,3,4,5,6])
        robot.r_inter.GetPose.return_value = '0,{},DO(1,1);'
        with self.assertRaisesRegex(RuntimeError,'GetPose invalid response'):
            robot.get_XYZrxryrz_state()
