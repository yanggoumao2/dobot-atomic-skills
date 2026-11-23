"""Left wrist D405 dropping off the shared USB hub (2026-11-23): the camera waits and reopens."""

from __future__ import annotations

import unittest
from unittest import mock

from chem_demo import wrist_camera
from chem_demo.wrist_camera import LiveWristCamera, device_lost


class FakeDevice:
    def __init__(self, serial):
        self.serial = serial

    def get_info(self, _key):
        return self.serial


class FakeRS:
    class camera_info:
        serial_number = "serial"

    def __init__(self, present_after):
        self.calls, self.present_after, self.pipelines = 0, present_after, []

    def context(self):
        rs = self

        class Ctx:
            def query_devices(self_inner):
                rs.calls += 1
                return [FakeDevice("LEFT")] if rs.calls >= rs.present_after else []
        return Ctx()

    def pipeline(self):
        p = mock.Mock()
        self.pipelines.append(p)
        return p


def camera(rs):
    cam = LiveWristCamera.__new__(LiveWristCamera)
    cam.rs, cam.serial, cam._config = rs, "LEFT", object()
    cam.pipeline, cam.streaming, cam.reconnects, cam.restarts = mock.Mock(), True, 0, 0
    cam._last_sdk_sample = None
    return cam


class ReconnectTest(unittest.TestCase):
    def test_lost_markers(self):
        self.assertTrue(device_lost(RuntimeError("Device disconnected. Failed to reconnect: No device connected5000")))
        self.assertTrue(device_lost(RuntimeError("No device connected")))
        self.assertFalse(device_lost(RuntimeError("Frame didn't arrive within 5000")))

    def test_grab_waits_for_the_device_and_reopens_once(self):
        rs = FakeRS(present_after=3)
        cam = camera(rs)
        results = [RuntimeError("Device disconnected. Failed to reconnect: No device connected5000"), {"color": 1}]

        def once(*_a, **_k):
            item = results.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch.object(LiveWristCamera, "_grab_once", side_effect=once), \
                mock.patch.object(wrist_camera.time, "sleep"):
            self.assertEqual(cam.grab(flush=2, require_fresh_pair=True), {"color": 1})
        self.assertEqual(cam.reconnects, 1)
        self.assertEqual(len(rs.pipelines), 1)
        rs.pipelines[0].start.assert_called_once()

    def test_device_that_never_returns_raises_a_clear_error(self):
        rs = FakeRS(present_after=10 ** 9)
        cam = camera(rs)
        clock = iter(range(0, 10 ** 6))
        with mock.patch.object(LiveWristCamera, "_grab_once", side_effect=RuntimeError("No device connected")), \
                mock.patch.object(wrist_camera.time, "sleep"), \
                mock.patch.object(wrist_camera.time, "time", side_effect=lambda: float(next(clock))):
            with self.assertRaisesRegex(RuntimeError, "did not come back"):
                cam.grab()

    def test_slow_frame_is_not_treated_as_a_lost_device(self):
        cam = camera(FakeRS(present_after=1))
        cam.restarts = 3
        with mock.patch.object(LiveWristCamera, "_grab_once", side_effect=RuntimeError("Frame didn't arrive within 5000")):
            with self.assertRaisesRegex(RuntimeError, "didn't arrive"):
                cam.grab()
        self.assertEqual(cam.reconnects, 0)


if __name__ == "__main__":
    unittest.main()
