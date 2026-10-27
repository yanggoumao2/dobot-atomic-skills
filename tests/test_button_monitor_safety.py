import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from dobot_control.agents.agent import BimanualAgent
from dobot_control.dynamixel.driver import DynamixelDriver
from experiments import run_control as control


class ButtonSafetyTests(unittest.TestCase):
    def driver(self):
        driver = DynamixelDriver.__new__(DynamixelDriver)
        driver._lock = threading.Lock()
        driver._reading_thread = Mock()
        driver._reading_thread.is_alive.return_value = True
        driver._joint_keys = [1, 1, 0]
        driver._keys_updated_at = time.monotonic()
        return driver

    def packet(self, data, sensor=False):
        driver = self.driver()
        driver.using_sensor = sensor
        driver._portHandler = Mock()
        driver._portHandler.readPort.side_effect = [data[:2], data[2:]]
        driver._portHandler.isPacketTimeout.return_value = len(data) < (4 if sensor else 2)
        return driver._internal_getKey()

    def test_short_or_invalid_frames_are_not_button_states(self):
        for packet in ([], [17], [17, 17]):
            self.assertIsNone(self.packet(packet))
        self.assertIsNone(self.packet([17, 238], sensor=True))

    def test_valid_normal_and_sensor_frames_have_three_fields(self):
        self.assertEqual(self.packet([17, 238]), [1, 1, 0])
        self.assertEqual(self.packet([17, 238, 0, 255], sensor=True), [1, 1, 0])

    def test_recent_state_is_copied(self):
        driver = self.driver()
        result = driver.get_keys()
        result[0] = 0
        self.assertEqual(driver._joint_keys, [1, 1, 0])

    def test_stale_state_is_not_reused(self):
        driver = self.driver()
        driver._keys_updated_at -= 1
        with self.assertRaisesRegex(RuntimeError, "stale"):
            driver.get_keys()

    def test_dead_reader_is_reported_even_with_cached_state(self):
        driver = self.driver()
        driver._reading_thread.is_alive.return_value = False
        with self.assertRaisesRegex(RuntimeError, "reader stopped"):
            driver.get_keys()

    def test_initial_missing_state_has_bounded_wait(self):
        driver = self.driver()
        driver._joint_keys = None
        with patch("dobot_control.dynamixel.driver.time.monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(RuntimeError, "within 1 second"):
                driver.get_keys()

    def test_empty_hand_reply_names_the_failed_side(self):
        left, right = Mock(), Mock()
        left.get_keys.return_value = [1, 1, 0]
        right.get_keys.return_value = []
        with self.assertRaisesRegex(RuntimeError, "RIGHT.*shape"):
            BimanualAgent(left, right).get_keys()

    def test_worker_exception_is_propagated(self):
        left, right = Mock(), Mock()
        left.get_keys.side_effect = RuntimeError("COM6 disconnected")
        right.get_keys.return_value = [1, 1, 0]
        with self.assertRaisesRegex(RuntimeError, "LEFT.*COM6 disconnected"):
            BimanualAgent(left, right).get_keys()

    def test_valid_bimanual_state_has_fixed_shape(self):
        left, right = Mock(), Mock()
        left.get_keys.return_value = [1, 1, 0]
        right.get_keys.return_value = [0, 1, 0]
        self.assertEqual(BimanualAgent(left, right).get_keys().shape, (2, 3))

    def test_failure_clears_servo_and_record_flags_and_blocks_dispatch(self):
        agent = Mock()
        agent.get_keys.side_effect = RuntimeError("button read failed")
        failure, stop, ready = threading.Event(), threading.Event(), threading.Event()
        errors = []
        states = np.ones((2, 3), dtype=int)
        with patch.object(control, "what_to_do", states), patch.object(control, "runtime_log"):
            control.run_button_monitor_guarded(agent, failure, stop, ready, errors)
            self.assertTrue(failure.is_set())
            self.assertFalse(ready.is_set())
            self.assertFalse(states[:, 1:].any())
            dispatch = Mock()
            thread = Mock()
            thread.is_alive.return_value = True
            with self.assertRaisesRegex(RuntimeError, "Button monitoring unavailable"):
                control.require_button_monitor_healthy(thread, failure, errors)
                dispatch()
            dispatch.assert_not_called()

    def test_dead_thread_without_report_is_also_fatal(self):
        thread = Mock()
        thread.is_alive.return_value = False
        with self.assertRaisesRegex(RuntimeError, "not running"):
            control.require_button_monitor_healthy(thread, threading.Event(), [])

    def test_first_sample_sets_baseline_without_toggling_buttons(self):
        stop, ready = threading.Event(), threading.Event()
        agent = Mock()
        def read():
            stop.set()
            return np.array([[1, 1, 0], [1, 1, 0]])
        agent.get_keys.side_effect = read
        states = np.zeros((2, 3), dtype=int)
        with patch.object(control, "what_to_do", states), patch.object(control, "is_falling", np.array([0])):
            control.button_monitor_realtime(agent, stop, ready)
        self.assertTrue(ready.is_set())
        self.assertFalse(states.any())


if __name__ == "__main__":
    unittest.main()
