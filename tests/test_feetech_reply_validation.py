import unittest
from unittest.mock import Mock

from third_party.feetech.scservo_sdk.protocol_packet_handler import protocol_packet_handler
from third_party.feetech.scservo_sdk.scservo_def import COMM_SUCCESS, COMM_RX_CORRUPT
from dobot_control.gripper.dobot_gripper import DobotGripper


def reply(data=(), error=0, sid=21):
    packet = [255, 255, sid, len(data) + 2, error, *data]
    return packet + [(~sum(packet[2:])) & 255]


class FakePort:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.is_using = False
        self.sent = []

    def clearPort(self):
        pass

    def writePort(self, packet):
        self.sent.append(list(packet))
        return len(packet)

    def readPort(self, size):
        data = self.incoming[:size]
        del self.incoming[:size]
        return data

    def setPacketTimeout(self, size):
        pass

    def isPacketTimeout(self):
        return not self.incoming


class ReplyValidationTests(unittest.TestCase):
    def test_write_ack_followed_by_position_uses_position_not_ack(self):
        port = FakePort(reply() + reply((0x34, 0x12)))
        handler = protocol_packet_handler(port, 0)
        self.assertEqual(handler.read2ByteTxRx(21, 56), (0x1234, COMM_SUCCESS, 0))
        self.assertEqual(len(port.sent), 1)  # no resend of the read or motion

    def test_short_ack_is_never_used_as_value_for_any_read_width(self):
        for width in (1, 2, 4):
            for suffix in ("Rx", "TxRx"):
                with self.subTest(width=width, suffix=suffix):
                    handler = protocol_packet_handler(FakePort(reply()), 0)
                    method = getattr(handler, f"read{width}Byte{suffix}")
                    args = (21,) if suffix == "Rx" else (21, 56)
                    value, comm, error = method(*args)
                    self.assertEqual(value, 0)
                    self.assertNotEqual(comm, COMM_SUCCESS)

    def test_truncated_payload_checksum_is_not_position_data(self):
        handler = protocol_packet_handler(FakePort(reply((0x34,))), 0)
        value, comm, _ = handler.read2ByteTxRx(21, 56)
        self.assertEqual(value, 0)
        self.assertEqual(comm, COMM_RX_CORRUPT)

    def test_device_error_without_payload_is_preserved(self):
        handler = protocol_packet_handler(FakePort(reply(error=32)), 0)
        value, comm, error = handler.read2ByteTxRx(21, 56)
        self.assertEqual(value, 0)
        self.assertNotEqual(comm, COMM_SUCCESS)
        self.assertEqual(error, 32)

    def test_corrupt_checksum_is_rejected(self):
        packet = reply((0x34, 0x12))
        packet[-1] ^= 1
        handler = protocol_packet_handler(FakePort(packet), 0)
        self.assertEqual(handler.read2ByteTxRx(21, 56)[1], COMM_RX_CORRUPT)

    def test_wrong_id_only_reply_does_not_loop_forever(self):
        handler = protocol_packet_handler(FakePort(reply((0x34, 0x12), sid=22)), 0)
        self.assertEqual(handler.read2ByteTxRx(21, 56)[1], COMM_RX_CORRUPT)

    def test_move_checks_ack_and_keeps_low_speed_conversion(self):
        gripper = DobotGripper.__new__(DobotGripper)
        gripper.port = "COM3"
        gripper._servo_id = 21
        gripper._min_position, gripper._max_position = 0, 255
        gripper._min_speed, gripper._max_speed = 0, 100
        gripper._min_servo_pos, gripper._max_servo_pos = 2048, 3052
        gripper._min_servo_speed, gripper._max_servo_speed = 0, 4096
        gripper.servo = Mock()
        gripper.servo.WritePosEx.return_value = (0, 0)
        self.assertEqual(gripper.move(0, 10, 1), (0, 0))
        gripper.servo.WritePosEx.assert_called_once_with(21, 2048, 409, 0)
        gripper.servo.WritePosEx.return_value = (-6, 0)
        with self.assertRaisesRegex(RuntimeError, "WritePosEx failed.*COM3"):
            gripper.move(255, 10, 1)


if __name__ == "__main__":
    unittest.main()
