import time
import unittest
from unittest.mock import mock_open, patch

import numpy as np
from dobot_control.cameras.settings import CameraSettings, load_camera_settings
from experiments import run_control


class FakeCamera:
    instances = []

    def __init__(self, device_id, flip, fps):
        self.args = (device_id, flip, fps)
        self.closed = False
        self.instances.append(self)

    def read(self):
        time.sleep(0.005)
        return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 1))

    def close(self):
        self.closed = True


class CameraSelectionTests(unittest.TestCase):
    def read_config(self, text):
        with patch("builtins.open", mock_open(read_data=text)):
            return load_camera_settings("test-settings.ini")

    def test_legacy_defaults_keep_three_views(self):
        settings = self.read_config("[CAMERA]\ntop=111\nleft=222\nright=333\n")
        self.assertEqual(settings.fps, 90)
        self.assertTrue(settings.recording_enabled)

    def test_disabled_views_need_no_serials_and_cannot_record(self):
        settings = self.read_config("[CAMERA]\nenabled=top\nfps=30\ntop=111\n")
        self.assertEqual(settings.serials, {"top": "111"})
        self.assertFalse(settings.recording_enabled)

    def test_invalid_selection_rejected(self):
        for names in ("", "top,top", "top,unknown"):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.read_config(f"[CAMERA]\nenabled={names}\ntop=111\n")

    def test_duplicate_active_serials_rejected(self):
        with self.assertRaises(ValueError):
            self.read_config("[CAMERA]\nenabled=top,left\ntop=111\nleft=111\n")

    def test_top_only_starts_one_stream_and_cleans_up(self):
        FakeCamera.instances = []
        with patch.object(run_control, "RealSenseCamera", FakeCamera):
            threads, failure, stop, close = run_control.start_configured_cameras(CameraSettings({"top": "111"}, 30))
            try:
                self.assertEqual(set(threads), {"top"})
                self.assertFalse(failure.is_set())
                self.assertEqual([c.args for c in FakeCamera.instances], [("111", True, 30)])
                self.assertTrue(run_control.jpeg_list[0])
            finally:
                close()
            self.assertTrue(stop.is_set())
            self.assertFalse(threads["top"].is_alive())
            self.assertTrue(FakeCamera.instances[0].closed)

    def test_missing_requested_camera_fails_and_releases_previous_stream(self):
        FakeCamera.instances = []
        def factory(**kwargs):
            if kwargs["device_id"] == "222":
                raise RuntimeError("camera missing")
            return FakeCamera(**kwargs)
        with patch.object(run_control, "RealSenseCamera", side_effect=factory):
            with self.assertRaisesRegex(RuntimeError, "camera missing"):
                run_control.start_configured_cameras(CameraSettings({"top": "111", "left": "222"}, 30))
        self.assertTrue(FakeCamera.instances[0].closed)

    def test_stream_read_failure_prevents_startup(self):
        class BrokenCamera(FakeCamera):
            def read(self):
                raise RuntimeError("frame timeout")
        with patch.object(run_control, "RealSenseCamera", BrokenCamera):
            with self.assertRaises(RuntimeError):
                run_control.start_configured_cameras(CameraSettings({"top": "111"}, 30))


if __name__ == "__main__":
    unittest.main()
