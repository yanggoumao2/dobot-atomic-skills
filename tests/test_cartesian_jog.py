import csv
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from task_skills import run_cartesian_jog as jogmod
from task_skills.chemistry import driver as drivermod
from task_skills.chemistry.registry import load_profile
from task_skills.run_cartesian_jog import (Jog, Limits, LiveArm, SimArm, down_tilt_deg,
                                          snap_down_rpy, start_refusal)

DOWN = snap_down_rpy([180.0, 0.0, 90.0])


def make(arm=None, radius=200.0, z_min=150.0, z_max=350.0, record=None, answers=(), held=set):
    arm = arm or SimArm([-150.0, -350.0, 200.0], DOWN)
    replies = iter(answers)
    lines = []
    jog = Jog(arm, Limits(arm.pose[:3], radius, z_min, z_max),
              record_path=record or Path("unused.csv"), ask=lambda _: next(replies),
              out=lines.append, held_moves=held)
    return jog, arm, lines


class CartesianJogTests(unittest.TestCase):
    """What matters on the rig: one axis per key, fixed orientation, refusals send nothing."""

    def test_each_key_moves_one_axis_and_never_changes_orientation(self):
        jog, arm, _ = make()
        for key in ("d", "4", "f", "1", "w", "RIGHT", "PGUP"):
            self.assertTrue(jog.handle(key))
        targets = [xyz for xyz, _ in arm.sent]
        self.assertEqual(targets, [[-149.0, -350.0, 200.0], [-149.0, -350.0, 190.0],
                                   [-149.0, -349.9, 190.0], [-148.9, -349.9, 190.0],
                                   [-148.9, -349.9, 190.1]])
        self.assertTrue(all(rpy == DOWN for _, rpy in arm.sent))
        self.assertFalse(jog.handle("q"))

    def test_held_keys_combine_into_one_diagonal_step(self):
        held = set()

        def once():  # keys are reported at the key event, then released (a tap)
            keys = set(held)
            held.clear()
            return keys

        jog, arm, lines = make(held=once)
        held.add("w")
        jog.handle("a")                 # a pressed while w is held
        jog.handle("wa")                # line-mode combination
        held.update({"d", "RIGHT"})     # same direction twice counts once
        jog.handle("d")
        held.update({"a", "d"})         # opposite keys cancel on that axis
        jog.handle("a")
        held.update({"r", "a", "w"})
        jog.handle("r")                 # three axes at once
        self.assertEqual([xyz for xyz, _ in arm.sent],
                         [[-151.0, -349.0, 200.0], [-152.0, -348.0, 200.0],
                          [-151.0, -348.0, 200.0], [-152.0, -347.0, 201.0]])
        self.assertTrue(any("抵消" in line for line in lines))
        self.assertTrue(all(rpy == DOWN for _, rpy in arm.sent))

    def test_refused_targets_send_no_motion(self):
        class Jumpy(SimArm):
            def plan(self, xyz, rpy, seed):
                joints = super().plan(xyz, rpy, seed)
                joints[3] += 40.0  # an IK branch switch
                return joints

        class Unreachable(SimArm):
            def plan(self, xyz, rpy, seed):
                raise RuntimeError("InverseSolution on right: status -1")

        class Drifting(SimArm):
            def forward(self, joints):
                result = super().forward(joints)
                result["xyz"][0] += 0.5
                return result

        cases = [("below z floor", make(z_min=199.5), "f"),
                 ("outside radius", make(radius=0.5), "d"),
                 ("ik jump", make(arm=Jumpy([-150.0, -350.0, 200.0], DOWN)), "d"),
                 ("unreachable", make(arm=Unreachable([-150.0, -350.0, 200.0], DOWN)), "d"),
                 ("fk mismatch", make(arm=Drifting([-150.0, -350.0, 200.0], DOWN)), "d")]
        for label, (jog, arm, lines), key in cases:
            with self.subTest(label):
                self.assertTrue(jog.handle(key))
                self.assertEqual(arm.sent, [])
                self.assertIn("拒绝", lines[-1])
                self.assertEqual(jog.xyz, [-150.0, -350.0, 200.0])

    def test_snap_down_points_tool_z_down_and_keeps_heading(self):
        tilted = [172.0, 5.0, 37.0]
        self.assertGreater(down_tilt_deg(tilted), 5.0)
        jog, arm, _ = make(arm=SimArm([-150.0, -350.0, 200.0], tilted))
        jog.handle("v")
        (xyz, rpy), = arm.sent
        self.assertEqual(xyz, [-150.0, -350.0, 200.0])
        self.assertLess(down_tilt_deg(rpy), 1e-4)
        self.assertAlmostEqual(rpy[2], 37.0, delta=1.0)

        far, far_arm, lines = make(arm=SimArm([-150.0, -350.0, 200.0], [140.0, 0.0, 0.0]))
        far.handle("v")
        self.assertEqual(far_arm.sent, [])
        self.assertIn("拒绝", lines[-1])

    def test_recorded_points_import_into_the_calibration_wizard(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "points.csv"
            jog, _, _ = make(record=path, answers=("12", "", "40", "v", "", "x7"))
            jog.handle("p")
            jog.handle("d")
            jog.handle("p")
            jog.handle("p")  # empty id cancels
            jog.handle("p")  # non-numeric id refused
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        # Same required columns as CalibrationService.import_teach_csv.
        self.assertTrue({"corner_id", "split", "x", "y", "z"}.issubset(rows[0]))
        self.assertEqual([(r["corner_id"], r["split"], r["x"]) for r in rows],
                         [("12", "train", "-150.0000"), ("40", "val", "-149.0000")])

    def test_a_recorded_corner_can_be_overwritten_or_deleted_and_history_is_kept(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "points.csv"
            jog, _, lines = make(record=path, answers=(
                "5", "",            # record 5 as train
                "16", "v",          # record 16 as val
                "5", "",            # 5 again: Enter cancels, nothing changes
                "5", "y", "",       # overwrite 5 at the new position, keeping train
                "16", "d"))         # delete 16
            jog.handle("p")
            jog.handle("p")
            jog.handle("d")
            jog.handle("p")
            jog.handle("p")
            jog.handle("p")
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            history = [json.loads(line)["event"] for line in
                       (Path(directory) / "points_history.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["corner_id"], r["split"], r["x"]) for r in rows], [("5", "train", "-149.0000")])
        self.assertEqual(history, ["record", "record", "overwrite", "delete"])
        self.assertTrue(any("未修改" in line for line in lines))

    def test_live_without_typed_confirmation_never_opens_a_session(self):
        class FakeLive(SimArm):
            name = "live"
            opened = closed = False

            def __init__(self, profile, arm, speed, **kwargs):
                super().__init__([-150.0, -350.0, 200.0], DOWN)

            def open_session(self):
                FakeLive.opened = True

            def close(self):
                FakeLive.closed = True

        lines = []
        with mock.patch.object(jogmod, "LiveArm", FakeLive):
            code = jogmod.main(["--arm", "right", "--live"], ask=lambda _: "yes", out=lines.append,
                               keys=(lambda: "q", lambda: False, lambda: None, set))
        self.assertEqual(code, 1)
        self.assertFalse(FakeLive.opened)
        self.assertTrue(FakeLive.closed)

    def test_sim_session_runs_from_the_command_line(self):
        keys = iter(["d", "f", "q"])
        lines = []
        code = jogmod.main(["--arm", "left", "--sim"], out=lines.append,
                           keys=(lambda: next(keys), lambda: False, lambda: None, set))
        self.assertEqual(code, 0)
        self.assertTrue(any("到达" in line for line in lines))


class HoldStreamingTests(unittest.TestCase):
    """A held key must keep the arm moving without per-step stops, and a tap stays one step."""

    def _jog(self, held_sequence, radius=200.0, clock=None):
        polls = iter(held_sequence)
        jog, arm, lines = make(radius=radius, held=lambda: next(polls, set()))
        jog.hold_delay = 0.0 if clock is None else jog.hold_delay
        jog.clock = clock or (lambda: 0.0)
        jog.sleep = lambda _: None
        return jog, arm, lines

    def test_single_axis_hold_uses_native_jog_and_stops_on_release(self):
        jog, arm, lines = self._jog([{"d"}] * 7)  # first poll is the key event
        jog.handle("d")
        self.assertEqual(arm.sent, [])           # no MovL queue to drain after release
        self.assertEqual(arm.jogs, ["X+", "stop"])
        self.assertGreater(jog.xyz[0], -150.0)  # position taken from the arm, not a target
        self.assertEqual(jog.xyz[1:], [-350.0, 200.0])
        self.assertEqual(jog.rpy, DOWN)
        self.assertIn("松手", lines[-1])

    def test_changing_the_axis_mid_jog_stops_it(self):
        jog, arm, _ = self._jog([{"r"}] * 3 + [{"r", "a"}] * 3)
        jog.handle("r")
        self.assertEqual(arm.jogs, ["Z+", "stop"])

    def test_jog_stops_before_the_soft_limit(self):
        jog, arm, lines = self._jog([{"d"}] * 100, radius=10.0)
        jog.handle("d")
        self.assertEqual(arm.jogs[-1], "stop")
        self.assertLessEqual(jog.xyz[0] + 150.0, 10.0 - 2.0)  # stopped with margin to spare
        self.assertIn("软限位", lines[-1])

    def test_jog_is_stopped_even_when_the_controller_alarms(self):
        class Alarming(SimArm):
            def read(self):
                state = super().read()
                return {**state, "mode": 9} if self.jogging else state

        jog, _, _ = make(arm=Alarming([-150.0, -350.0, 200.0], DOWN), held=lambda: {"f"})
        jog.hold_delay, jog.sleep = 0.0, (lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "报警"):
            jog.handle("f")
        self.assertEqual(jog.arm.jogs, ["Z-", "stop"])

    def test_space_during_a_jog_stops_it_and_ends_the_program(self):
        presses = iter([False, False, True])
        jog, arm, _ = self._jog([{"w"}] * 50)
        jog.stop_pressed = lambda: next(presses, True)
        with self.assertRaises(jogmod.JogStopped):
            jog.handle("w")
        self.assertEqual(arm.jogs, ["Y+", "stop"])

    def test_a_tap_shorter_than_the_hold_delay_is_exactly_one_step(self):
        jog, arm, _ = self._jog([{"d"}] * 5, clock=lambda: 0.0)  # never reaches the delay
        jog.handle("d")
        self.assertEqual([xyz for xyz, _ in arm.sent], [[-149.0, -350.0, 200.0]])
        self.assertEqual(arm.jogs, [])

    def test_diagonal_hold_follows_with_servo_and_stops_on_release(self):
        jog, arm, lines = self._jog([{"d", "w"}] * 6)  # key event + hold decision + 4 ticks
        jog.handle("d")
        points = [point for point in arm.servos if point != "end"]
        self.assertEqual(arm.servos[-1], "end")
        self.assertEqual(len(points), 4)
        self.assertEqual(arm.sent, [])  # no queued MovL to drain after release
        self.assertEqual(arm.jogs, [])
        step = 20.0 * jogmod.SERVO_PERIOD_S / math.sqrt(2)  # 20 mm/s along the diagonal
        for index, (x, y, z) in enumerate(points, start=1):
            self.assertAlmostEqual(x, -150.0 + index * step, places=3)
            self.assertAlmostEqual(y, -350.0 + index * step, places=3)
            self.assertEqual(z, 200.0)
        self.assertEqual(jog.rpy, DOWN)
        self.assertEqual(jog.xyz, [round(v, 4) for v in arm.pose[:3]])
        self.assertIn("松手", lines[-1])

    def test_diagonal_follow_stops_at_the_soft_limit(self):
        jog, arm, lines = self._jog([{"d", "w"}] * 500, radius=3.0)
        jog.handle("d")
        points = [point for point in arm.servos if point != "end"]
        self.assertTrue(points and arm.servos[-1] == "end")
        self.assertTrue(all(math.hypot(x + 150.0, y + 350.0) <= 3.0 for x, y, _ in points))
        self.assertIn("被拒绝", lines[-1])

    def test_space_during_a_diagonal_follow_ends_the_stream(self):
        presses = iter([False, True])
        jog, arm, _ = self._jog([{"a", "f"}] * 50)
        jog.stop_pressed = lambda: next(presses, True)
        with self.assertRaises(jogmod.JogStopped):
            jog.handle("a")
        self.assertEqual(arm.servos[-1], "end")


class FloorTests(unittest.TestCase):
    def test_z_key_sets_the_floor_just_below_the_touch_height(self):
        jog, arm, _ = make(z_min=0.0)
        jog.handle("z")
        self.assertEqual(jog.limits.z_min, 199.5)
        jog.handle("f")              # 1 mm down would go below the floor
        self.assertEqual(arm.sent, [])

    def test_default_floor_reaches_far_below_home(self):
        lines = []
        keys = iter(["q"])
        jogmod.main(["--arm", "right", "--sim"], out=lines.append,
                    keys=(lambda: next(keys), lambda: False, lambda: None, set))
        self.assertTrue(any("Z ∈ [-200.0" in line for line in lines))


class GripperTests(unittest.TestCase):
    """Closing is direct; anything that opens the jaws needs a typed y."""

    def test_close_then_open_with_confirmation(self):
        jog, arm, lines = make(answers=("y",))
        jog.handle("c")
        self.assertEqual(arm.gripper_pos, 0)
        jog.handle("o")
        self.assertEqual(arm.gripper_pos, 255)

    def test_opening_without_y_does_not_move_the_jaws(self):
        jog, arm, lines = make(answers=("", "n"))
        arm.gripper_pos = 0
        jog.handle("o")
        jog.handle("o")
        self.assertEqual(arm.gripper_pos, 0)
        self.assertIn("未确认", lines[-1])

    def test_set_value_validates_and_only_asks_when_opening(self):
        jog, arm, lines = make(answers=("300", "120", "200", "y"))
        jog.handle("g")
        self.assertIn("拒绝", lines[-1])
        jog.handle("g")  # 255 -> 120 closes: no confirmation needed
        self.assertEqual(arm.gripper_pos, 120)
        jog.handle("g")  # 120 -> 200 opens: needs y
        self.assertEqual(arm.gripper_pos, 200)


class _FakeDashboard:
    def __init__(self, state):
        self.state, self.calls = state, []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            if name == "DisableRobot":
                self.state["mode"] = 4
            return f"0,{{}},{name}();"
        return call


def _session_arm(arm_name, joints, mode=4):
    state = {"mode": mode, "joints": list(joints)}
    arm = LiveArm.__new__(LiveArm)
    arm.arm, arm.profile, arm.speed = arm_name, {}, 10
    arm.log, arm.session_open = (lambda _: None), False
    arm.driver = mock.Mock()
    arm.rig = mock.Mock()
    arm.rig.dashboard = _FakeDashboard(state)

    def ensure(*args, **kwargs):
        state["mode"] = 5
        return arm.rig
    arm.driver._ensure_session.side_effect = ensure
    arm.read = lambda: {"pose": [0.0] * 6, "joints": list(state["joints"]), "mode": state["mode"]}
    return arm, state


class SessionLifecycleTests(unittest.TestCase):
    """Home is the calibrated layout joints, and every enabled session ends disabled."""

    def test_home_target_is_the_calibrated_layout_pose(self):
        # These values have been mistaken before; pin them so a silent edit fails here.
        from task_skills.layout import physical_nova_home_degrees
        self.assertEqual(tuple(physical_nova_home_degrees("right")), (-90, 0, -90, 0, 90, 90))
        self.assertEqual(tuple(physical_nova_home_degrees("left")), (90, 0, 90, 0, -90, -90))

    def test_open_session_enables_without_clear_error_then_homes(self):
        arm, _ = _session_arm("right", [-60.0, 10.0, -80.0, 0.0, 90.0, 90.0])
        note = arm.open_session()
        arm.driver._ensure_session.assert_called_once_with({}, "right", 10, clear_errors=False)
        arm.driver._move_to_home.assert_called_once_with(arm.rig, "right", None)
        self.assertIn("[-90, 0, -90, 0, 90, 90]", note)
        self.assertNotIn("CP", [name for name, _ in arm.rig.dashboard.calls])  # no global side effect

    def test_open_session_skips_the_move_when_already_home_or_told_not_to(self):
        parked, _ = _session_arm("left", [90.2, 0.0, 90.0, 0.0, -90.0, -90.0])
        self.assertIn("已在初始位", parked.open_session())
        parked.driver._move_to_home.assert_not_called()
        stay, _ = _session_arm("left", [10.0, 0.0, 90.0, 0.0, -90.0, -90.0])
        stay.open_session(home=False)
        stay.driver._move_to_home.assert_not_called()

    def test_close_disables_only_an_opened_session_and_always_releases(self):
        opened, state = _session_arm("right", [-90.0, 0.0, -90.0, 0.0, 90.0, 90.0])
        opened.open_session()
        opened.close()
        self.assertIn("DisableRobot", [name for name, _ in opened.rig.dashboard.calls])
        self.assertEqual(state["mode"], 4)
        opened.driver.close.assert_called_once()

        never, _ = _session_arm("right", [-90.0, 0.0, -90.0, 0.0, 90.0, 90.0])
        never.close()
        self.assertNotIn("DisableRobot", [name for name, _ in never.rig.dashboard.calls])
        never.driver.close.assert_called_once()


class AlarmHandlingTests(unittest.TestCase):
    """The jog never clears an alarm: it refuses to start on one and stops on one."""

    def test_only_disabled_or_enabled_idle_arms_are_taken_over(self):
        self.assertIsNone(start_refusal(4))
        self.assertIsNone(start_refusal(5))
        self.assertIn("示教器", start_refusal(9))
        self.assertIn("上电", start_refusal(3))
        for mode in (6, 7, 10, 11, None):
            self.assertIsNotNone(start_refusal(mode))

    def test_driver_session_can_skip_clear_error_and_keeps_it_by_default(self):
        class Dashboard:
            def __init__(self):
                self.calls = []

            def __getattr__(self, name):
                return lambda *args: (self.calls.append(name), f"0,{{}},{name}();")[1]

        class Rig:
            def __init__(self, arm):
                self.arm, self.configured, self.dashboard = arm, False, Dashboard()
                self.motion = self.feedback = self.gripper = None

        class Lease:
            def __init__(self, path=None, purpose=None):
                self.owner = "fake"

            def acquire(self):
                pass

        calls = {}
        for clear in (False, True):
            with TemporaryDirectory() as directory, mock.patch.object(drivermod, "DeviceLease", Lease):
                driver = drivermod.LiveDriver(load_profile(), allow_live=True,
                                              lock_path=Path(directory) / "owner")
                driver._rig = lambda arm, prof, with_motion=True: driver._rigs.setdefault(arm, Rig(arm))
                rig = driver._ensure_session(load_profile(), "right", 10, clear_errors=clear)
                calls[clear] = rig.dashboard.calls
        self.assertNotIn("ClearError", calls[False])
        self.assertEqual(calls[False][0], "EnableRobot")
        self.assertEqual(calls[True][:2], ["ClearError", "EnableRobot"])

    def _live_arm(self, mode):
        arm = LiveArm.__new__(LiveArm)
        arm.arm, arm.profile, arm.speed = "right", {}, 10
        arm.read = lambda: {"pose": [0.0] * 6, "joints": [0.0] * 6, "mode": mode}
        arm.stopped = False
        arm.stop = lambda: setattr(arm, "stopped", True)
        arm.driver = mock.Mock()
        return arm

    def test_live_arm_refuses_to_enable_on_alarm_and_stops_on_alarm_mid_move(self):
        alarmed = self._live_arm(9)
        with self.assertRaisesRegex(RuntimeError, "示教器"):
            alarmed.open_session()
        alarmed.driver._ensure_session.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "报警"):
            alarmed.wait([0.0, 0.0, 0.0], [0.0] * 3, [0.0] * 6, lambda: False)
        alarmed.driver.ClearError.assert_not_called()

    def test_live_start_on_alarm_sends_nothing_and_asks_nothing(self):
        class AlarmedLive(SimArm):
            name = "live"
            opened = False

            def __init__(self, profile, arm, speed, **kwargs):
                super().__init__([-150.0, -350.0, 200.0], DOWN)

            def read(self):
                return {**super().read(), "mode": 9}

            def open_session(self):
                AlarmedLive.opened = True

        lines = []
        with mock.patch.object(jogmod, "LiveArm", AlarmedLive):
            code = jogmod.main(["--arm", "right", "--live"], ask=lambda _: self.fail("asked"),
                               out=lines.append, keys=(lambda: "q", lambda: False, lambda: None, set))
        self.assertEqual(code, 5)
        self.assertFalse(AlarmedLive.opened)
        self.assertIn("拒绝启动", lines[-1])


if __name__ == "__main__":
    unittest.main()
