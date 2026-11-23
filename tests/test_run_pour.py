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



class GraspOffsetTest(unittest.TestCase):
    """chem_demo.grasp_offset: the predicted mouth ring moves rigidly with the in-hand offset."""

    def _plan(self):
        return {"pour_dir": [1.0, 0.0, 0.0], "theta_pre_deg": 60.0, "lip_xyz": [0.0, -700.0, 280.0],
                "steps": [{"phase": "transit", "lip": [-80.0, -700.0, 300.0]}]}

    def test_upright_ring_shifts_by_the_offset(self):
        import numpy as np
        from chem_demo.grasp_offset import ring_points
        tool = np.array([0.0, -1.0, 0.0])
        a, _ = ring_points(self._plan(), tool, np.zeros(3))
        b, _ = ring_points(self._plan(), tool, np.array([10.0, 20.0, 5.0]))
        np.testing.assert_allclose(b.mean(axis=0) - a.mean(axis=0), [5.0, -10.0, 20.0], atol=1e-6)

    def test_tilted_ring_keeps_the_lip_on_its_rim(self):
        import numpy as np
        from chem_demo.grasp_offset import ring_points
        _, t = ring_points(self._plan(), np.array([0.0, -1.0, 0.0]), np.zeros(3))
        lip = np.array([0.0, -700.0, 280.0 + 180.0])   # ring points are in the fingertip convention
        self.assertLess(float(np.min(np.linalg.norm(t - lip, axis=1))), 1e-6)


if __name__ == "__main__":
    unittest.main()
