import csv
import hashlib
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from task_skills.calibration.state import validate_verify_rows
from task_skills.calibration_targets import TargetError, load_arm_fit, main, verify_rows
from task_skills.run_cartesian_jog import Jog, Limits, SimArm, snap_down_rpy

DOWN = snap_down_rpy([180.0, 0.0, 90.0])
# Board flipped onto the table as in the rig: board z points down in base; origin 250 mm up.
T_BASE_BOARD = [[1.0, 0.0, 0.0, -0.2], [0.0, -1.0, 0.0, -0.3], [0.0, 0.0, -1.0, 0.25],
                [0.0, 0.0, 0.0, 1.0]]
COLS, ROWS = 11, 8


def make_session(directory: Path, *, usable=True, fit_ids=(0, 1, 2)) -> Path:
    session = directory / "cal-test"
    (session / "board" / "v001").mkdir(parents=True)
    (session / "jobs" / "job-1").mkdir(parents=True)
    observation = {"corners_uv": [[100.0 + 10 * (i % COLS), 50.0 + 10 * (i // COLS)]
                                  for i in range(COLS * ROWS)]}
    obs_bytes = json.dumps(observation).encode("utf-8")
    (session / "board" / "v001" / "observation_normal.json").write_bytes(obs_bytes)
    (session / "board" / "v001" / "observation_reverse.json").write_text(
        json.dumps({"corners_uv": observation["corners_uv"][::-1]}), encoding="utf-8")
    extrinsics = {"robot_id": "NOVA2_LEFT", "tcp_name": "tip",
                  "board": {"cols": COLS, "rows": ROWS, "square_m": 0.03},
                  "observation_sha256": hashlib.sha256(obs_bytes).hexdigest(),
                  "T_base_board": T_BASE_BOARD,
                  "points": [{"corner_id": i} for i in fit_ids]}
    ext_bytes = json.dumps(extrinsics).encode("utf-8")
    (session / "jobs" / "job-1" / "extrinsics.json").write_bytes(ext_bytes)
    fit = {"job_id": "job-1", "usable": usable, "stale": False, "validation_passed": True,
           "artifact_path": "jobs/job-1/extrinsics.json",
           "artifact_sha256": hashlib.sha256(ext_bytes).hexdigest()}
    (session / "session.json").write_text(json.dumps({"arms": {"left": {"fit": fit}}}), encoding="utf-8")
    return session


class CheckedArm(SimArm):
    """Records every point the controller is asked to solve, i.e. every check along a leg."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.checked = []

    def plan(self, xyz, rpy, seed):
        self.checked.append([round(v, 4) for v in xyz])
        return super().plan(xyz, rpy, seed)


def assert_straight_checked_legs(case, arm, start):
    """Each MovL is one axis-aligned straight leg, and the leg was checked at most 10 mm apart."""
    previous = list(start)
    for target, _ in arm.sent:
        moving = [abs(target[i] - previous[i]) > 1e-6 for i in range(3)]
        case.assertTrue(moving == [False, False, True] or not moving[2], (previous, target))  # vertical or level
        on_leg = [p for p in arm.checked if math.dist(previous, p) + math.dist(p, target)
                  <= math.dist(previous, target) + 1e-6]
        stops = sorted([previous, *on_leg], key=lambda p: math.dist(previous, p))
        case.assertTrue(all(math.dist(a, b) <= 10.0 + 1e-6 for a, b in zip(stops, stops[1:])), (previous, target))
        case.assertIn(target, on_leg)
        previous = target


class Targets:
    """Stand-in for ArmFit with a corner far from the start."""
    fit_ids = frozenset({0})

    def corner_base_mm(self, corner_id):
        if corner_id > 87:
            raise RuntimeError("角点编号必须在 0-87 之间")
        return [-100.0, -400.0, 120.0]


def make_jog(arm=None, answers=(), targets=Targets(), z_min=100.0):
    arm = arm or SimArm([-150.0, -350.0, 200.0], DOWN)
    replies = iter(answers)
    lines = []
    jog = Jog(arm, Limits(arm.pose[:3], 200.0, z_min, 350.0), record_path=Path("unused.csv"),
              ask=lambda _: next(replies), out=lines.append, targets=targets)
    return jog, arm, lines


class SessionTargetTests(unittest.TestCase):
    """The corners must come from the fit the wizard will verify, not from a stale or wrong file."""

    def test_corner_position_follows_the_fitted_board_pose(self):
        with TemporaryDirectory() as directory:
            fit = load_arm_fit(make_session(Path(directory)), "left")
        self.assertEqual(fit.corner_base_mm(0), [-200.0, -300.0, 250.0])
        # id 13 = col 2, row 1 -> board (60, 30) mm; board y maps to base -y.
        self.assertEqual(fit.corner_base_mm(13), [-140.0, -330.0, 250.0])
        self.assertEqual(fit.corners_uv[13], (120.0, 60.0))  # the normal-order observation
        with self.assertRaises(TargetError):
            fit.corner_base_mm(88)

    def test_an_unusable_or_tampered_fit_is_refused(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(TargetError):
                load_arm_fit(make_session(Path(directory), usable=False), "left")
        with TemporaryDirectory() as directory:
            session = make_session(Path(directory))
            path = session / "jobs" / "job-1" / "extrinsics.json"
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(TargetError):
                load_arm_fit(session, "left")

    def test_verify_rows_skip_fitted_corners_and_import_into_the_wizard(self):
        with TemporaryDirectory() as directory:
            fit = load_arm_fit(make_session(Path(directory)), "left")
        touches = [{"corner_id": "1", "x": "-170", "y": "-300", "z": "250"},
                   {"corner_id": "13", "x": "-139.5", "y": "-330", "z": "250.1"},
                   {"corner_id": "40", "x": "-20", "y": "-390", "z": "250"}]
        rows, notes = verify_rows(fit, touches)
        self.assertEqual(len(rows), 2)
        self.assertIn("跳过", notes[0])
        self.assertEqual((rows[0]["u_px"], rows[0]["v_px"]), (120.0, 60.0))
        self.assertAlmostEqual(rows[0]["reference_x_m"], -0.1395)
        self.assertEqual(rows[0]["board_z_m"], 0.0)
        self.assertEqual(len(validate_verify_rows(rows, (1280, 720))), 2)

    def test_command_writes_the_step7_csv_only_with_three_independent_points(self):
        with TemporaryDirectory() as directory:
            session = make_session(Path(directory))
            touches = Path(directory) / "touches.csv"
            out = Path(directory) / "verify.csv"
            with touches.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["corner_id", "split", "x", "y", "z"])
                writer.writeheader()
                for corner in (0, 13, 40):
                    writer.writerow({"corner_id": corner, "split": "val", "x": 0, "y": 0, "z": 250})
            args = ["--session", str(session), "--arm", "left", "--touches", str(touches), "--out", str(out)]
            self.assertEqual(main(args), 2)  # 0 was fitted, so only two independent points
            self.assertFalse(out.exists())
            with touches.open("a", newline="", encoding="utf-8") as handle:
                handle.write("50,val,0,0,250\n")
            self.assertEqual(main(args), 0)
            with out.open(encoding="utf-8") as handle:
                header = next(csv.reader(handle))
        # Same required columns as CalibrationService.import_verify_csv.
        self.assertEqual(set(header), {"u_px", "v_px", "board_z_m", "reference_x_m",
                                       "reference_y_m", "reference_z_m"})


class GotoCornerTests(unittest.TestCase):
    """t must hover above the target, never cut through the board, and send nothing if any piece fails."""

    def test_goto_moves_in_continuous_straight_legs_checked_every_10_mm(self):
        arm = CheckedArm([-150.0, -350.0, 200.0], DOWN)
        jog, arm, lines = make_jog(arm=arm, answers=("5",))
        jog.handle("t")
        targets = [xyz for xyz, _ in arm.sent]
        # Start height 200 = hover 140 + 60, so: across at 200 in one MovL, then down in one MovL.
        self.assertEqual(targets, [[-100.0, -400.0, 200.0], [-100.0, -400.0, 140.0]])
        self.assertTrue(all(rpy == DOWN for _, rpy in arm.sent))
        assert_straight_checked_legs(self, arm, [-150.0, -350.0, 200.0])
        self.assertIn("对齐", lines[-1])

    def test_goto_from_below_lifts_first(self):
        jog, arm, _ = make_jog(arm=SimArm([-150.0, -350.0, 130.0], DOWN), answers=("5",))
        jog.handle("t")
        first = arm.sent[0][0]
        self.assertEqual(first[:2], [-150.0, -350.0])
        self.assertGreater(first[2], 130.0)

    def test_goto_from_home_height_descends_first_and_falls_back_to_a_lower_crossing(self):
        # Like the rig: home is high, and the far corner is only reachable low down.
        class LowReach(SimArm):
            def plan(self, xyz, rpy, seed):
                if xyz[0] > -120.0 and xyz[2] > 170.0:
                    raise RuntimeError("InverseSolution: status -1")
                return super().plan(xyz, rpy, seed)

        jog, arm, lines = make_jog(arm=LowReach([-150.0, -350.0, 300.0], DOWN), answers=("5",))
        jog.handle("t")
        targets = [xyz for xyz, _ in arm.sent]
        # Straight down over the start, across at 160 (60 and 40 mm crossings unreachable), down.
        self.assertEqual(targets, [[-150.0, -350.0, 160.0], [-100.0, -400.0, 160.0], [-100.0, -400.0, 140.0]])
        self.assertTrue(any("下降到 Z=160.0" in line for line in lines))

    def test_any_failing_piece_means_nothing_moves(self):
        class Wall(SimArm):
            def plan(self, xyz, rpy, seed):
                if xyz[0] > -110.0:
                    raise RuntimeError("InverseSolution: status -1")
                return super().plan(xyz, rpy, seed)

        jog, arm, lines = make_jog(arm=Wall([-150.0, -350.0, 200.0], DOWN), answers=("5",))
        jog.handle("t")
        self.assertEqual(arm.sent, [])
        self.assertIn("拒绝", lines[-1])

        low, low_arm, low_lines = make_jog(answers=("5",), z_min=150.0)  # hover below the floor
        low.handle("t")
        self.assertEqual(low_arm.sent, [])
        self.assertIn("下限", low_lines[-1])

    def test_x_goes_to_a_typed_point_by_the_same_checked_route(self):
        arm = CheckedArm([-150.0, -350.0, 200.0], DOWN)
        jog, arm, lines = make_jog(arm=arm, answers=("-120, -380 180",), targets=None)
        jog.handle("x")
        self.assertEqual(arm.sent[-1][0], [-120.0, -380.0, 180.0])
        self.assertEqual(len(arm.sent), 3)  # up to 240, across, down: three continuous legs
        self.assertTrue(all(rpy == DOWN for _, rpy in arm.sent))
        assert_straight_checked_legs(self, arm, [-150.0, -350.0, 200.0])
        bad, bad_arm, bad_lines = make_jog(answers=("1 2",), targets=None)
        bad.handle("x")
        low, low_arm, low_lines = make_jog(answers=("-120 -380 90",), targets=None)  # below the floor
        low.handle("x")
        self.assertEqual(bad_arm.sent + low_arm.sent, [])
        self.assertIn("三个数字", bad_lines[-1])
        self.assertIn("下限", low_lines[-1])

    def test_goto_needs_a_session_a_down_tool_and_a_valid_id(self):
        none, none_arm, none_lines = make_jog(targets=None)
        none.handle("t")
        self.assertIn("--session", none_lines[-1])
        tilted, tilted_arm, tilted_lines = make_jog(arm=SimArm([-150.0, -350.0, 200.0], [175.0, 0.0, 90.0]))
        tilted.handle("t")
        self.assertIn("按 v", tilted_lines[-1])
        bad, bad_arm, bad_lines = make_jog(answers=("99",))
        bad.handle("t")
        self.assertIn("0-87", bad_lines[-1])
        fitted, fitted_arm, fitted_lines = make_jog(answers=("0",))
        fitted.handle("t")
        self.assertTrue(any("参与了本臂拟合" in line for line in fitted_lines))
        self.assertEqual(none_arm.sent + tilted_arm.sent + bad_arm.sent, [])


if __name__ == "__main__":
    unittest.main()
