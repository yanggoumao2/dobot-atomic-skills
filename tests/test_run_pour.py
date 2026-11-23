"""Offline checks for how run_pour groups the planned samples into controller motions."""

from __future__ import annotations

import unittest

from chem_demo.run_pour import segments, way_back


def _forward():
    names = [("lift", 2), ("transit", 3), ("pretilt", 4), ("pour", 2)]
    out, i = [], 0
    for name, n in names:
        steps = []
        for _ in range(n):
            i += 1
            steps.append({"index": i, "phase": name})
        out.append((name, steps))
    return out


class SegmentsTest(unittest.TestCase):
    def test_full_pour_is_one_blended_motion_in_plan_order(self):
        forward = _forward()
        groups = segments(forward, continuous=True)
        self.assertEqual(len(groups), 1)
        name, steps = groups[0]
        self.assertEqual(name, "lift+transit+pretilt+pour")
        self.assertEqual([s["index"] for s in steps], list(range(1, 12)))

    def test_rehearsals_stop_after_every_phase(self):
        forward = _forward()[:3]  # --until pretilt
        groups = segments(forward, continuous=False)
        self.assertEqual([g[0] for g in groups], ["lift", "transit", "pretilt"])

    def test_way_back_retraces_every_sample_to_the_hold(self):
        forward = _forward()
        back = way_back(forward, {"index": 0, "phase": "hold"})
        self.assertEqual([s["index"] for s in back], list(range(10, -1, -1)))


if __name__ == "__main__":
    unittest.main()
