"""A wrist view must distinguish a lifted rim from one that stays on the bench."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from chem_demo.neck_lift import verify_neck_lift
from tests.test_neck_vision import draw_blue_ring


class NeckLiftTests(unittest.TestCase):
    def check_lift(self, rim_shift, bench_shift):
        rng = np.random.default_rng(12)
        texture = rng.integers(30, 220, size=(480, 848), dtype=np.uint8)
        depth = np.full(texture.shape, 260, np.uint16)
        depth[:230] = 110

        def frame(z, rim_dx, bench_dx):
            gray = texture.copy()
            gray[230:] = np.roll(texture[230:], bench_dx, axis=1)
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            draw_blue_ring(color, (422 + rim_dx, 220))
            return {"pose": [0, 0, z], "depth": depth, "color": color}

        return verify_neck_lift(frame(380, 0, 0), frame(395, rim_shift, bench_shift))["held"]

    def test_rim_travels_with_jaws(self):
        self.assertIs(self.check_lift(0, 12), True)

    def test_rim_stays_on_bench(self):
        self.assertIsNone(self.check_lift(12, 12))

    def test_reused_image_after_lift(self):
        self.assertIsNone(self.check_lift(0, 0))


class DepthLiftCheckTest(unittest.TestCase):
    """Replays real 30 mm lifts; a static bottle is simulated by moving the whole scene's depth."""

    RUN = Path(__file__).resolve().parents[1] / "calibration-data" / "chem-demo" / "handoff" / "run-20261122_204916-82aed11b3c"

    def frames(self):
        import json
        import cv2
        out = []
        for stem in ("04_lift", "05_lift"):
            meta = json.loads((self.RUN / f"{stem}.json").read_text(encoding="utf-8"))
            out.append({"color": cv2.imread(str(self.RUN / meta["color"])),
                        "depth": np.load(self.RUN / meta["depth_mm"]), "pose": meta["pose"], "evidence": stem})
        return out

    def test_held_bottle_keeps_its_distance_while_the_table_drops_away(self):
        if not self.RUN.exists():
            self.skipTest("real lift capture unavailable")
        before, after = self.frames()
        result = verify_neck_lift(before, after)
        self.assertIs(result["held"], True)
        self.assertEqual(result["method"], "depth")

    def test_bottle_left_on_the_table_recedes_with_it(self):
        if not self.RUN.exists():
            self.skipTest("real lift capture unavailable")
        before, after = self.frames()
        lift = after["pose"][2] - before["pose"][2]
        static = dict(after, color=before["color"], depth=np.where(before["depth"] > 0, before["depth"] + lift, 0))
        result = verify_neck_lift(before, static)
        self.assertIs(result["held"], False)
