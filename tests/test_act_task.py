import configparser
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dobot_control.act_task import load_act_task
from dobot_control.cameras.settings import load_camera_settings


class TaskTests(unittest.TestCase):
    def test_config_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.ini'
            path.write_text('[CAMERA]\nenabled=top,left,right\ntop=123\nleft=456\nright=789\nfps=15\nleft_flip=true\n', encoding='utf-8')
            self.assertEqual(load_act_task(path), {'task': 'bottle_cap_open'})
            settings = load_camera_settings(path)
            self.assertEqual(list(settings.serials), ['top','left','right'])
            self.assertTrue(settings.flips['left'])
            with path.open('a') as f:
                f.write('\n[ACT_TASK]\ntask=test_tube_insert\nfixture=narrow_mouth_cup\n')
            task = load_act_task(path)
            self.assertEqual(task['required_camera_names'], ['top','left','right'])
            self.assertNotIn('expected_arm_flags', task)
            text = path.read_text()
            path.write_text(text.replace('left=456','left=123'))
            with self.assertRaises(ValueError): load_camera_settings(path)
            path.write_text(text.replace('test_tube_insert','unknown'))
            with self.assertRaises(ValueError): load_act_task(path)

    def test_device_listing_without_reset_or_pipeline(self):
        preview = importlib.import_module('scripts.5_camera_read')
        device = Mock()
        device.supports.return_value = True
        device.get_info.return_value = '123'
        with patch('pyrealsense2.context') as context, patch.object(preview,'RealSenseCamera') as camera:
            context.return_value.query_devices.return_value = [device]
            with patch('sys.argv',['camera','--list-devices']):
                preview.main()
            device.hardware_reset.assert_not_called()
            camera.assert_not_called()
