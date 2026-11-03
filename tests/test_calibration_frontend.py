"""Run deterministic JavaScript regressions for the calibration wizard state."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "tests" / "browser_calibration_state.js"


class CalibrationFrontendStateTests(unittest.TestCase):
    def test_reordered_responses_blank_values_and_dirty_drafts(self):
        node = shutil.which("node") or shutil.which("node.exe")
        if node is None:
            windows_node = Path(r"C:\Program Files\nodejs\node.exe")
            if windows_node.is_file():
                node = str(windows_node)
        self.assertIsNotNone(node, "Node.js is required for the calibration JS regressions")
        completed = subprocess.run(
            [node, str(HARNESS)],
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(completed.returncode, 0,
                         f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["count"], 11, payload)
        self.assertEqual(set(payload["passed"]), {
            "reordered load responses",
            "reordered refresh/poll responses",
            "captured-session mutation guard",
            "late mutation response session guard",
            "blank measurements",
            "dirty draft preservation",
            "same-arm teach draft preservation",
            "same-arm verification draft preservation",
            "query sequence/session binding",
            "query/export dependency invalidation",
            "dirty session switch policy",
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
