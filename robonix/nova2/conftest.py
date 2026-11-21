"""Make both provider packages importable for the deployment's offline tests."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "primitives/nova2_arm"), str(ROOT / "primitives/nova2_gripper")]
collect_ignore_glob = ["primitives/*/rbnx-build"]
