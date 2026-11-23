import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from chem_demo import check_wrist_calibration as check


class OfflineReplayTests(unittest.TestCase):
    def test_recovery_manifest_selects_the_fresh_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            wrong = np.zeros((8, 12, 3), np.uint8)
            recovered = np.full_like(wrong, 91)
            cv2.imwrite(str(folder / "z-.png"), wrong)
            cv2.imwrite(str(folder / "recovered.png"), recovered)
            depth = np.full((8, 12), 298.5, np.float32)
            np.save(folder / "recovered_depth.npy", depth)
            group = {"captures": [{"files": {"color_png": "recovered.png"}}],
                     "representative_depth_file": "recovered_depth.npy"}
            (folder / "acquisition.json").write_text(json.dumps({"pose_groups": {"z-": group}}))
            color, actual_depth = check.observation(folder, "z-")
            np.testing.assert_array_equal(color, recovered)
            np.testing.assert_array_equal(actual_depth, depth)

    def test_manifest_is_sufficient_after_original_solver_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = {"arm": "left", "table_z_reference_mm": 243.2,
                        "camera": {"intrinsics": {"matrix_px": np.eye(3).tolist()}}}
            (folder / "acquisition.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch("sys.argv", ["replay", tmp]), mock.patch.object(
                check, "replay", return_value={"held_out_median_and_height_within_limits": False}
            ) as replay, mock.patch("builtins.print"):
                self.assertEqual(check.main(), 1)
            self.assertEqual(replay.call_args.args[0], folder)
            np.testing.assert_equal(replay.call_args.args[1], np.eye(3))
            self.assertEqual(replay.call_args.args[2], 243.2)
            self.assertTrue((folder / "open_jaw_replay.json").exists())
            self.assertFalse((folder / "left_wrist.json").exists())

    def test_replay_prefers_full_precision_burst_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cv2.imwrite(str(folder / "ref.png"), np.zeros((4, 6, 3), np.uint8))
            cv2.imwrite(str(folder / "ref_depth.png"), np.full((4, 6), 111, np.uint16))
            expected = np.full((4, 6), 120.25, np.float32)
            np.save(folder / "ref_depth_temporal_median_mm.npy", expected)
            np.testing.assert_equal(check.observation(folder, "ref")[1], expected)
