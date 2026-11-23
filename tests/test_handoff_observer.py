"""The observer must hand the cycle an error in right Tool0 X/Y/Z, with gates enforced."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from chem_demo.handoff_observer import PAD_OFFSET_TOOL_MM, RIGHT_MODEL, RightWristObserver
from tests.test_handoff_vision import K, scene


class FakeCamera:
    def __init__(self, frames):
        self.K, self.frames = K, list(frames)

    def grab(self, flush=0, *, require_fresh_pair=False):
        return {"color": self.frames.pop(0), "depth": None}


@unittest.skipUnless(RIGHT_MODEL.exists(), "right wrist calibration file not present")
class ObserverTest(unittest.TestCase):
    def observe(self, error_tool):
        _, image, masks, _ = scene(np.asarray(error_tool, float))
        with tempfile.TemporaryDirectory() as folder:
            observer = RightWristObserver(None, "000", Path(folder), camera=FakeCamera([image]), segment=False)
            observer._segmenter = lambda color, prompt: masks
            from chem_demo.handoff_plan import rpy_for_tool_axis
            # Horizontal tool, roll +90: tool X vertical (along the upright bottle), Y the jaw axis.
            return observer.observe({"pose": [0.0, 0.0, 0.0, *rpy_for_tool_axis([1.0, 0.0, 0.0], 90.0)]})

    def test_error_is_reported_in_tool_x_y_z_order(self):
        result = self.observe([8.0, -5.0, 12.0])
        self.assertTrue(result["ok"], result["reason"])
        x, y, z = result["raw_error_tool_mm"]
        servo = result["error_tool_mm"]
        self.assertAlmostEqual(abs(servo[0]), abs(x), places=6)   # vertical error kept (tool X is vertical here)
        self.assertAlmostEqual(servo[1], y, places=6)
        self.assertAlmostEqual(servo[2], z, places=6)
        # scene() is rendered against a pad midpoint at the Tool0 origin; the observer uses the pad offset
        # estimate, so the measured error is the rendered error minus that offset.
        ex, ey, ez = (8.0 - PAD_OFFSET_TOOL_MM[0], -5.0 - PAD_OFFSET_TOOL_MM[1], 12.0 - PAD_OFFSET_TOOL_MM[2])
        self.assertAlmostEqual(x, ex, delta=6.0)     # along the bottle axis
        self.assertAlmostEqual(y, ey, delta=6.0)     # along the jaw axis
        self.assertAlmostEqual(z, ez, delta=8.0)     # along the tool axis, positive = beyond the pads
        self.assertEqual(result["calibration_validity"], "unverified_pose")

    def test_gate_failure_is_still_a_servo_measurement_but_not_safe_to_close(self):
        result = self.observe([0.0, 40.0, 0.0])      # bottle far outside the pad span
        self.assertTrue(result["ok"])
        self.assertFalse(result["safe_to_close"])
        self.assertIn("gates", result["reason"])
        self.assertGreater(abs(result["error_tool_mm"][1]), 20.0)


if __name__ == "__main__":
    unittest.main()


class ServoErrorTest(unittest.TestCase):
    def test_pitched_tool_reaches_the_grasp_point(self):
        import math
        from chem_demo.handoff_observer import servo_error
        from chem_demo.handoff_plan import _rot, rpy_for_tool_axis
        for pitch in (0.0, 20.0, 35.0):
            axis = np.array([math.cos(math.radians(pitch)), 0.0, -math.sin(math.radians(pitch))])
            rpy = rpy_for_tool_axis(axis, 90.0)
            R = _rot(rpy)
            pad = np.array([100.0, 50.0, 300.0])
            bottle_xy = np.array([180.0, 56.0])          # 80 mm ahead, 6 mm to the side, horizontally
            for height in (250.0, 300.0, 340.0):          # grasp point above or below the pads
                point = np.array([*bottle_xy, height])
                err = R.T @ (point - pad)
                delta = servo_error(err, rpy)
                np.testing.assert_allclose(pad + R @ np.array(delta), point, atol=1e-6)
                # With no vertical error the command is purely horizontal plus the advance along tool Z.
                flat = servo_error(R.T @ (np.array([*bottle_xy, pad[2]]) - pad), rpy)
                np.testing.assert_allclose((pad + R @ np.array(flat))[2], pad[2], atol=1e-6)
