"""Offline checks for the no-motion bottle observation entry point."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chem_demo import bottle_observation, run_bottle_pick


ROOT = Path(__file__).resolve().parents[1]
PAIRED_CAPTURE = (ROOT / "calibration-data" / "chem-demo" / "bottle-observations" /
                  "bench-20261121_164404" / "paired-164736")


def state(**overrides):
    value = {"mode": 4, "enabled": 0, "running": 0, "tool": 0, "user": 0,
             "pose": [0.0] * 6, "joints": [0.0] * 6,
             "tcp_speed": [0.0] * 6, "joint_speed": [0.0] * 6}
    value.update(overrides)
    return value


class BottleObservationTests(unittest.TestCase):
    def test_state_gate_rejects_moving_or_wrong_frame_feedback(self):
        for sample, expected in ((state(running=1), "running command"),
                                 (state(tool=1), "Tool1"),
                                 (state(user=1), "User1"),
                                 (state(tcp_speed=[2.0, 0, 0, 0, 0, 0]), "TCP linear speed"),
                                 (state(pose=[1.0, 0, 0, 0, 0, 0]), "pose changed")):
            _, problems = bottle_observation._state_gate(
                state(), sample, expected_user=0, expected_tool=0, require_full_telemetry=True)
            self.assertTrue(any(expected in problem for problem in problems), problems)

    def test_saved_paired_capture_replays_as_pose_evidence_only(self):
        if not PAIRED_CAPTURE.is_dir():
            self.skipTest("paired left-wrist capture is not available")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = run_bottle_pick.build_parser().parse_args([
                "--observe-only", "--replay", str(PAIRED_CAPTURE), "--out", temporary,
            ])
            report, code = bottle_observation.run_observation(args, output / "observation", out=lambda _: None)
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "recorded pose accepted")
            self.assertTrue(report["model_pose_match"]["accepted_by_model"])
            self.assertEqual(report["robot"]["before"]["tool"], 0)
            self.assertEqual(report["robot"]["before"]["user"], 0)
            self.assertEqual(report["gripper"]["before"], 254)
            self.assertEqual(report["gripper"]["after"], 254)
            self.assertTrue(report["jaw"]["ok"])
            self.assertTrue(Path(report["evidence"]["jaw_overlay"]).is_file())
            self.assertIn("不代表", report["operator_message"])

    def test_left_live_requires_a_passing_fixed_real_camera_calibration_before_setup(self):
        messages = []
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(run_bottle_pick, "load_calibration", return_value={}), \
                patch.object(run_bottle_pick, "LiveArm") as live_arm, \
                patch.object(run_bottle_pick, "VisionClient") as vision_client:
            missing_calibration = str(Path(temporary) / "missing-left-calibration.json")
            code = run_bottle_pick.main(["--live", "--arm", "left", "--calibration", missing_calibration,
                                         "--out", temporary], out=messages.append, ask=lambda _: "n")
        self.assertEqual(code, 2)
        live_arm.assert_not_called()
        vision_client.assert_not_called()
        self.assertTrue(any("未启动" in message for message in messages))

    def test_observation_flags_cannot_accidentally_run_live_motion(self):
        messages = []
        with patch.object(run_bottle_pick, "LiveArm") as live_arm:
            code = run_bottle_pick.main(["--live", "--arm", "right", "--replay", "capture"],
                                        out=messages.append)
        self.assertEqual(code, 2)
        live_arm.assert_not_called()
        self.assertTrue(any("只能与 --observe-only" in message for message in messages))

    def test_observe_only_dispatches_before_legacy_arm_and_calibration_setup(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(run_bottle_pick, "run_observation", return_value=({"status": "recorded pose accepted"}, 0)) as observe, \
                patch.object(run_bottle_pick, "LiveArm") as live_arm, \
                patch.object(run_bottle_pick, "load_calibration") as load_calibration:
            code = run_bottle_pick.main([
                "--observe-only", "--replay", str(PAIRED_CAPTURE), "--out", temporary,
            ], out=lambda _: None)
        self.assertEqual(code, 0)
        observe.assert_called_once()
        live_arm.assert_not_called()
        load_calibration.assert_not_called()


if __name__ == "__main__":
    unittest.main()
