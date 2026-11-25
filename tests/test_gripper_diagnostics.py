import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO

from dobot_control.gripper.dobot_gripper import COMM_SUCCESS, DobotGripper, GripperReadTimeout


class FakeServo:
    def __init__(self, response):
        self.response = response
        self.servo_ids = []

    def ReadPos(self, servo_id):
        self.servo_ids.append(servo_id)
        return self.response


def make_gripper(response):
    gripper = DobotGripper.__new__(DobotGripper)
    gripper.command_lock = threading.RLock()
    gripper.servo = FakeServo(response)
    gripper._servo_id = 22
    gripper._min_servo_pos = 2000
    gripper._max_servo_pos = 4000
    gripper._min_position = 0
    gripper._max_position = 255
    return gripper


class GripperDiagnosticsTests(unittest.TestCase):
    def test_delayed_serial_reply_respects_receive_budget(self):
        from third_party.feetech.scservo_sdk import PortHandler, protocol_packet_handler
        for budget, expected in [(20, -6), (50, COMM_SUCCESS)]:
            port = PortHandler('unused')  # Construction does not open a port.
            clock, remaining = [0], [255, 255, 21, 2, 0, (~23) & 255]
            def read(length):
                clock[0] += 1
                if clock[0] < 25:
                    return []
                data = remaining[:length]
                del remaining[:length]
                return data
            port.readPort = read
            port.getCurrentTime = lambda: clock[0]
            port.setPacketTimeoutMillis(budget)
            packet = protocol_packet_handler(portHandler=port, protocol_end=0)
            _, result = packet.rxPacket()
            self.assertEqual(result, expected)

    def test_feedback_waits_for_busy_serial_bus_and_updates_cache(self):
        from unittest.mock import Mock
        from dobot_control.act_recording import GripperFeedback
        ready, sampled, stop = threading.Event(), threading.Event(), threading.Event()
        bus = threading.RLock()
        class ObservedLock:
            def acquire(self, *args, **kwargs):
                ready.set()
                return bus.acquire(*args, **kwargs)
            def release(self):
                bus.release()
        gripper = Mock()
        gripper.command_lock = ObservedLock()
        def read():
            sampled.set()
            return 102
        gripper.get_current_position.side_effect = read
        bus.acquire()
        cache = GripperFeedback(gripper, stop)
        try:
            self.assertTrue(ready.wait(1))
            # A pending reader waits for the current write to finish.
            self.assertFalse(sampled.is_set())
            bus.release()
            self.assertTrue(sampled.wait(.08))
            stop.set()
            cache.thread.join(1)
            value, timestamp, valid = cache.snapshot()
            self.assertAlmostEqual(value, .4)
            self.assertTrue(valid)
            self.assertGreater(timestamp, 0)
        finally:
            stop.set()
            cache.thread.join(1)

    def test_startup_retries_only_timeout_reads_without_repeating_move(self):
        from unittest.mock import Mock, patch, call
        from dobot_control.robots.dobot import DobotRobot
        robot = DobotRobot.__new__(DobotRobot)
        robot.robot_ip = '192.168.5.1'
        robot.gripper = Mock()
        robot.gripper.get_current_position.side_effect = [GripperReadTimeout('timeout'), 0, 255]
        with patch('dobot_control.robots.dobot.time.sleep'):
            robot._initialize_gripper()
        self.assertEqual(robot.gripper.move.call_args_list, [call(0, 10, 1), call(255, 10, 1)])
        self.assertEqual(robot.gripper.get_current_position.call_count, 3)

    def test_startup_aborts_persistent_timeout_or_device_error(self):
        from unittest.mock import Mock, patch
        from dobot_control.robots.dobot import DobotRobot
        for error, reads in [(GripperReadTimeout('timeout'), 3), (ValueError('device error'), 1)]:
            robot = DobotRobot.__new__(DobotRobot)
            robot.robot_ip = '192.168.5.1'
            robot.gripper = Mock()
            robot.gripper.get_current_position.side_effect = error
            with patch('dobot_control.robots.dobot.time.sleep'):
                with self.assertRaises(type(error)):
                    robot._initialize_gripper()
            robot.gripper.move.assert_called_once_with(0, 10, 1)
            self.assertEqual(robot.gripper.get_current_position.call_count, reads)

    def test_read_timeout_has_specific_exception_type(self):
        with self.assertRaises(GripperReadTimeout):
            make_gripper((0, -6, 0)).get_current_position()

    def test_communication_failure_reports_both_codes_and_servo_id(self):
        gripper = make_gripper((2500, -3001, 0))

        with self.assertRaises(ValueError) as raised:
            gripper.get_current_position()

        message = str(raised.exception)
        self.assertIn("servo ID 22", message)
        self.assertIn("communication failure", message)
        self.assertIn("comm_result=-3001", message)
        self.assertIn("servo_error=0", message)
        self.assertNotIn("servo/device error", message)
        self.assertEqual(gripper.servo.servo_ids, [22])

    def test_servo_error_is_reported_even_when_communication_succeeds(self):
        gripper = make_gripper((2500, COMM_SUCCESS, 4))

        with self.assertRaises(ValueError) as raised:
            gripper.get_current_position()

        message = str(raised.exception)
        self.assertIn("servo ID 22", message)
        self.assertIn("servo/device error", message)
        self.assertIn(f"comm_result={COMM_SUCCESS}", message)
        self.assertIn("servo_error=4", message)
        self.assertNotIn("communication failure", message)

    def test_successful_read_keeps_existing_position_conversion(self):
        gripper = make_gripper((3000, COMM_SUCCESS, 0))

        self.assertEqual(gripper.get_current_position(), 127)

    def test_disconnect_message_does_not_claim_servo_release(self):
        class FakePortHandler:
            def __init__(self):
                self.closed = False

            def closePort(self):
                self.closed = True

        gripper = DobotGripper.__new__(DobotGripper)
        port_handler = FakePortHandler()
        gripper._portHandler = port_handler
        gripper._packHandler = object()
        gripper.servo = object()
        output = StringIO()

        with redirect_stdout(output):
            gripper.disconnect()

        self.assertTrue(port_handler.closed)
        self.assertEqual(output.getvalue().strip(), "Closing gripper serial port (no open command sent)")


if __name__ == "__main__":
    unittest.main()
