import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

calibration = importlib.import_module("scripts.2_get_offset")


class MasterCalibrationTests(unittest.TestCase):
    def config(self):
        return SimpleNamespace(joint_ids=[1,2,4,5,6,7],port="COM6",baud_rate=2000000,
            start_joints=[-1.57,0,-1.57,0,1.57,1.57],
            joint_signs=[1,1,-1,-1,-1,1],joint_offsets=[6.99,.92,3.16,2.17,3.85,3.01])

    def run_capture(self, replies, reference=False):
        port, handler = Mock(), Mock()
        handler.read4ByteTxRx.side_effect=replies
        with patch.object(calibration,"ReadOnlyPositionPort",return_value=port), \
             patch.object(calibration,"PacketHandler",return_value=handler), \
             patch.object(calibration.time,"sleep") as sleep:
            result=calibration.capture("HAND_LEFT",self.config(),reference)
        port.closePort.assert_called_once_with()
        self.assertEqual(handler.read4ByteTxRx.call_count,30)
        self.assertEqual(sleep.call_count,4)
        return result

    def test_candidate_formula_and_signed_raw_do_not_claim_verified(self):
        words=[2048,0,4096,1024,0xFFFFFC00,512]
        result=self.run_capture([(v,0,0) for v in words]*5,reference=True)
        raw=np.array([np.pi,0,2*np.pi,np.pi/2,-np.pi/2,np.pi/4])
        np.testing.assert_allclose(result["raw_rad"],raw)
        cfg=self.config()
        np.testing.assert_allclose((raw-result["candidate_offsets_rad"])*cfg.joint_signs,cfg.start_joints)
        self.assertFalse(result["mapping_verified"])

    def test_ordinary_snapshot_does_not_calibrate(self):
        result=self.run_capture([(2048,0,0)]*30)
        self.assertNotIn("candidate_offsets_rad",result)
        np.testing.assert_allclose(result["mapped_deg"],
            np.rad2deg((np.pi-np.array(self.config().joint_offsets))*self.config().joint_signs))

    def test_failed_read_closes_port_without_retrying_writes(self):
        for reply in ((0,-1,0),(0,0,1)):
            port,handler=Mock(),Mock()
            handler.read4ByteTxRx.return_value=reply
            with patch.object(calibration,"ReadOnlyPositionPort",return_value=port), \
                 patch.object(calibration,"PacketHandler",return_value=handler):
                with self.assertRaisesRegex(RuntimeError,"READ failed"):
                    calibration.capture("HAND_LEFT",self.config())
            port.closePort.assert_called_once_with()
            handler.read4ByteTxRx.assert_called_once()

    def test_motion_during_capture_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,"moved"):
            self.run_capture([(2048,0,0)]*24+[(2100,0,0)]*6,reference=True)

    def test_port_filters_motor_writes_and_other_addresses(self):
        port=calibration.ReadOnlyPositionPort("COM_DO_NOT_OPEN",[1])
        read=[255,255,253,0,1,7,0,2,140,0,4,0,0,0]
        with patch.object(calibration.PortHandler,"writePort",return_value=14) as write:
            self.assertEqual(port.writePort(read),14)
            write.assert_called_once()
            for index,value in ((7,3),(4,2),(8,64),(10,1)):
                packet=read.copy();packet[index]=value
                with self.assertRaisesRegex(RuntimeError,"Blocked packet"):
                    port.writePort(packet)
            write.assert_called_once()


if __name__ == "__main__": unittest.main()
