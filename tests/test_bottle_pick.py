"""Offline checks of the teach-free bottle pick: synthetic wrist camera, simulated arm and vision.

What matters and is tested here: the board-free hand-eye recovers a known camera mount; the closed
loop lines the fingertips up with the bottle axis from a 15 mm coarse error; a wrong reference
height in the calibration (metal table instead of the white platform) is corrected from the
platform depth; a bottle wider than the open jaws is refused before the arm goes down.

    C:/software/anaconda3/envs/dobot-xtrainer/python.exe -X utf8 -m unittest tests.test_bottle_pick
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from chem_demo import run_bottle_pick, run_wrist_calib
from chem_demo.wrist_camera import SIM_T_FLANGE_CAMERA, SimVision, SimWristCamera
from chem_demo.wrist_geometry import rotation_angle_deg
from task_skills.run_cartesian_jog import Jog, JogRefused, Limits, SimArm, snap_down_rpy

HOME = [117.6, -344.8, 415.5]
TABLE_Z = 243.2


class GraspSimArm(SimArm):
    """Closing on a bottle stops the jaws at its width instead of at the target."""

    camera: SimWristCamera | None = None

    def set_gripper(self, position: int) -> dict:
        before = self.gripper_pos
        final = position
        if self.camera is not None and self.camera.bottle is not None and position < 128:
            b = self.camera.bottle
            pose = self.read()["pose"]
            tip_above_table = pose[2] - self.camera.table_z
            if math.hypot(pose[0] - b["x"], pose[1] - b["y"]) < b["r"] and 0 < tip_above_table < b["h"]:
                final = 90
        self.gripper_pos = final
        return {"start": before, "target": position, "final": final}


def make_rig(table_z: float, bottle: dict | None) -> tuple[GraspSimArm, SimWristCamera]:
    arm = GraspSimArm(HOME, snap_down_rpy([180.0, 0.0, 90.0]))
    camera = SimWristCamera(arm, table_z=table_z, bottle=bottle)
    arm.camera = camera
    return arm, camera


def calibrate(folder: Path, true_table_z: float, reference_z: float = TABLE_Z) -> Path:
    arm, camera = make_rig(true_table_z, None)
    lines: list[str] = []
    code = run_wrist_calib.main(["--sim", "--yes", "--arm", "left", "--table-z", str(reference_z),
                                 "--out", str(folder)], arm=arm, camera=camera, out=lines.append)
    if code != 0:
        raise AssertionError("calibration failed:\n" + "\n".join(lines))
    return next(folder.glob("left-*/left_wrist.json"))


def pick(folder: Path, calibration: Path, bottle: dict, coarse_error=(12.0, -9.0)) -> tuple[int, dict, list[str]]:
    arm, camera = make_rig(TABLE_Z, bottle)
    lines: list[str] = []
    code = run_bottle_pick.main(["--sim", "--yes", "--arm", "left", "--calibration", str(calibration),
                                 "--out", str(folder)], arm=arm, camera=camera,
                                vision=SimVision(camera, coarse_error), out=lines.append, top_frame=lambda: np.zeros((4, 4, 3), np.uint8))
    trial = json.loads(next(folder.glob("left-*/trial.json")).read_text(encoding="utf-8"))
    trial["_arm_sent"] = arm.sent
    return code, trial, lines


class WristCalibrationTest(unittest.TestCase):
    def test_y_minus_to_z_minus_returns_horizontally_then_descends(self):
        rpy = snap_down_rpy([180.0, 0.0, 90.0])
        start = [HOME[0], HOME[1] - 30.0, HOME[2]]
        arm = SimArm(start, rpy)
        jog = Jog(arm, Limits(HOME, 80.0, HOME[2] - 60.0, HOME[2] + 20.0),
                  record_path=Path("unused.csv"), ask=lambda _: "y", out=lambda _: None)

        waypoints, _ = run_wrist_calib._precheck_transition(
            jog, "y-", "z-", [HOME[0], HOME[1], HOME[2] - 50.0], rpy, lambda: False)

        self.assertEqual([step[0] for step in waypoints], ["horizontal"] * 3 + ["vertical descent"] * 5)
        self.assertTrue(all(abs(step[1][2] - HOME[2]) < 1e-6 for step in waypoints[:3]))
        self.assertTrue(all(abs(step[1][0] - HOME[0]) < 1e-6 and
                            abs(step[1][1] - HOME[1]) < 1e-6 for step in waypoints[3:]))
        self.assertTrue(all(math.dist(a[1], b[1]) <= 10.0 for a, b in zip([("start", start)] + waypoints[:-1], waypoints)))
        self.assertEqual(arm.sent, [])  # planning and checks never send a partial transition

    def test_retreat_reverses_vertical_and_horizontal_order(self):
        rpy = snap_down_rpy([180.0, 0.0, 90.0])
        start = [HOME[0], HOME[1], HOME[2] - 50.0]
        arm = SimArm(start, rpy)
        jog = Jog(arm, Limits(HOME, 80.0, HOME[2] - 60.0, HOME[2] + 20.0),
                  record_path=Path("unused.csv"), ask=lambda _: "y", out=lambda _: None)

        waypoints, _ = run_wrist_calib._precheck_transition(
            jog, "z-", "y-", [HOME[0], HOME[1] - 30.0, HOME[2]], rpy, lambda: False)

        self.assertEqual([step[0] for step in waypoints], ["vertical ascent"] * 5 + ["horizontal"] * 3)
        self.assertEqual(arm.sent, [])

    def test_refused_transition_is_rejected_during_precheck(self):
        class RefusingArm(SimArm):
            def plan(self, xyz, rpy, seed):
                if xyz[1] >= HOME[1] - 20.0:
                    raise RuntimeError("synthetic unreachable waypoint")
                return super().plan(xyz, rpy, seed)

        rpy = snap_down_rpy([180.0, 0.0, 90.0])
        start = [HOME[0], HOME[1] - 30.0, HOME[2]]
        arm = RefusingArm(start, rpy)
        jog = Jog(arm, Limits(HOME, 80.0, HOME[2] - 60.0, HOME[2] + 20.0),
                  record_path=Path("unused.csv"), ask=lambda _: "y", out=lambda _: None)

        with self.assertRaisesRegex(JogRefused, "rejected before motion"):
            run_wrist_calib._precheck_transition(
                jog, "y-", "z-", [HOME[0], HOME[1], HOME[2] - 50.0], rpy, lambda: False)

        self.assertEqual(arm.sent, [])

    def test_stop_is_polled_after_confirmation_before_sending_each_waypoint(self):
        rpy = snap_down_rpy([180.0, 0.0, 90.0])
        arm = SimArm(HOME, rpy)
        jog = Jog(arm, Limits(HOME, 80.0, HOME[2] - 60.0, HOME[2] + 20.0),
                  record_path=Path("unused.csv"), ask=lambda _: "y", out=lambda _: None)
        calls = 0

        def stop_pressed() -> bool:
            nonlocal calls
            calls += 1
            return calls == 3  # precheck, confirmation, then immediately before the send

        def confirm(_: str) -> None:
            self.assertFalse(stop_pressed())

        with self.assertRaises(run_wrist_calib.JogStopped):
            run_wrist_calib._execute_transition(
                jog, "ref", "x+", [HOME[0] + 10.0, HOME[1], HOME[2]], rpy,
                confirm=confirm, stop_pressed=stop_pressed)

        self.assertEqual(calls, 3)
        self.assertEqual(arm.sent, [])

    def test_each_sim_transition_shows_checked_motion_for_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            arm, camera = make_rig(TABLE_Z, None)
            prompts: list[str] = []
            args = run_wrist_calib.build_parser().parse_args(
                ["--sim", "--arm", "left", "--table-z", str(TABLE_Z)])
            try:
                with patch("chem_demo.run_wrist_calib.process", return_value={}):
                    run_wrist_calib.run(
                        arm, camera, args, Path(tmp), ask=lambda prompt: prompts.append(prompt) or "y",
                        out=lambda _: None, stop_pressed=lambda: False)
            finally:
                camera.close()

        transitions = [prompt for prompt in prompts if " 到 " in prompt or " 返回 ref" in prompt]
        self.assertEqual(len(transitions), 9)
        y_to_z = next(prompt for prompt in transitions if "y- 到 z-" in prompt)
        self.assertIn("水平移动", y_to_z)
        self.assertIn("垂直下降", y_to_z)
        retreat = next(prompt for prompt in transitions if "z-yaw+ 返回 ref" in prompt)
        self.assertIn("垂直上升", retreat)
        yaw = next(prompt for prompt in transitions if "原地将 yaw" in prompt)
        self.assertIn("每步不超过 5 度", yaw)

    def test_yes_is_rejected_for_live_mode_before_hardware_setup(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                run_wrist_calib.main(["--live", "--yes", "--out", tmp])
        self.assertEqual(error.exception.code, 2)

    def test_recovers_the_camera_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = json.loads(calibrate(Path(tmp), TABLE_Z).read_text(encoding="utf-8"))
        T = np.array(record["T_flange_camera"])
        self.assertLess(rotation_angle_deg(T[:3, :3].T @ SIM_T_FLANGE_CAMERA[:3, :3]), 0.3)
        self.assertLess(np.linalg.norm(T[:3, 3] - SIM_T_FLANGE_CAMERA[:3, 3]), 1.5)
        self.assertTrue(record["checks"]["jaw_gap"]["ok"])


class BottlePickTest(unittest.TestCase):
    BOTTLE = {"x": 248.4, "y": -334.9, "h": 140.0, "r": 33.0}

    def test_lines_up_and_grasps_from_a_coarse_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            calibration = calibrate(Path(tmp) / "cal", TABLE_Z)
            code, trial, lines = pick(Path(tmp) / "pick", calibration, self.BOTTLE)
        self.assertEqual(code, 0, "\n".join(lines))
        plan = trial["plan"]
        self.assertLess(math.dist(plan["bottle_xy"], [self.BOTTLE["x"], self.BOTTLE["y"]]), 1.0)
        self.assertAlmostEqual(plan["diameter"], 2 * self.BOTTLE["r"], delta=3.0)
        self.assertAlmostEqual(plan["bottle_height"], self.BOTTLE["h"], delta=6.0)
        self.assertTrue(trial["held_check"]["held"])
        lowest = min(target[2] for target, _ in trial["_arm_sent"])
        self.assertGreaterEqual(lowest, TABLE_Z + 20.0)  # fingertips never below the floor

    def test_wrong_reference_height_is_corrected_on_the_platform(self):
        # Calibrated over a metal table 8 mm below the platform while assuming the platform height.
        with tempfile.TemporaryDirectory() as tmp:
            calibration = calibrate(Path(tmp) / "cal", TABLE_Z - 8.0)
            code, trial, lines = pick(Path(tmp) / "pick", calibration, self.BOTTLE)
        self.assertEqual(code, 0, "\n".join(lines))
        looks = [s for s in trial["steps"] if s["step"] == "look"]
        self.assertAlmostEqual(looks[0]["height_offset"], 8.0, delta=1.0)
        self.assertLess(math.dist(trial["plan"]["bottle_xy"], [self.BOTTLE["x"], self.BOTTLE["y"]]), 1.0)
        self.assertAlmostEqual(trial["plan"]["bottle_height"], self.BOTTLE["h"], delta=6.0)

    def test_refuses_a_bottle_wider_than_the_jaws_before_descending(self):
        wide = {**self.BOTTLE, "r": 55.0}  # 110 mm + margins > the ~114 mm sim jaw gap
        with tempfile.TemporaryDirectory() as tmp:
            calibration = calibrate(Path(tmp) / "cal", TABLE_Z)
            code, trial, lines = pick(Path(tmp) / "pick", calibration, wide)
        self.assertEqual(code, 1)
        self.assertIn("夹爪开口", trial["result"])
        self.assertNotIn("plan", trial)
        lowest = min(target[2] for target, _ in trial["_arm_sent"])
        self.assertGreater(lowest, TABLE_Z + wide["h"])  # never went down beside the bottle


if __name__ == "__main__":
    unittest.main()
