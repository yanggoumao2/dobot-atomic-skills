"""The one-command pick must follow its plan exactly and stop cleanly whenever the operator declines.

Runs in the robot env (no OpenCV needed): the vision step is replaced by a recorded report.
"""
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from task_skills.run_cartesian_jog import SimArm, snap_down_rpy
from vision_pick.run_cup_pick import main

HOME = [-117.6, -344.8, 415.5]
REPORT = {"overlay": "x.png", "camera_xy": [-504.5, -547.3],  # the rig's camera foot point (right base)
          "fit": {"x": -127.8, "y": -532.0, "table_z": 240.18, "height": 97.7, "top_radius": 38.9,
                  "bottom_radius": 31.8, "iou": 0.95},
          "grasp": {"rim_point": [-89.9, -532.0, 337.9]}}


class RecordingArm(SimArm):
    def __init__(self):
        super().__init__(HOME, snap_down_rpy([180.0, 0.0, 89.97]))
        self.events = []
        self.closed = False

    def move(self, xyz, rpy, joints):
        super().move(xyz, rpy, joints)
        self.events.append(("move", [round(v, 3) for v in xyz]))
        self.rpys = getattr(self, "rpys", set()) | {tuple(round(v, 6) for v in rpy)}

    def set_gripper(self, position):
        self.events.append(("gripper", position))
        return super().set_gripper(position)

    def stop(self):
        self.events.append(("stop",))

    def close(self):
        self.closed = True


class FakeLink:
    """Stands in for the viewer process: canned locate report, records what the window would show."""

    def __init__(self, report, stop_after_status=None):
        self.report, self.shown, self.closed = report, [], False
        self.stop_after_status = stop_after_status

    def locate(self, rpy):
        self.rpy = list(rpy)
        return self.report

    def status(self, step, tcp=None, plan=None):
        self.shown.append((step, tcp, plan))

    def relocate(self, rpy, reference_xy=None):
        self.relocated, self.reference_xy = True, reference_xy
        return {"report": self.report, "put_back": {"distance_mm": 0.8, "dx": 0.5, "dy": -0.6}}

    def stop_requested(self):
        return self.stop_after_status is not None and any(self.stop_after_status in s for s, _, _ in self.shown)

    def close(self):
        self.closed = True


def run(answers, report=REPORT, link=None, arm=None, extra=()):
    arm = arm or RecordingArm()
    link = link or FakeLink(report)
    replies = iter(answers)
    lines = []

    def ask(prompt):
        arm.events.append(("ask", prompt))
        return next(replies)

    with TemporaryDirectory() as directory:
        code = main(["--sim", "--out", directory, *extra], ask=ask, out=lines.append,
                    arm=arm, link=link, stop_pressed=lambda: False)
        trials = (Path(directory) / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        trial_dirs = [p for p in Path(directory).iterdir() if p.is_dir()]
        log = json.loads((trial_dirs[0] / "trial.json").read_text(encoding="utf-8"))
    return code, arm, lines, trials, log


class CupPickTests(unittest.TestCase):
    def test_full_trial_follows_the_plan_and_puts_the_cup_back(self):
        code, arm, lines, trials, log = run(["y", "y", "y", "y", "s", "", "y", "y"])
        self.assertEqual(code, 0)
        grippers = [e[1] for e in arm.events if e[0] == "gripper"]
        self.assertEqual(grippers, [100, 0, 100])  # pre-close, grasp, release
        moves = [e[1] for e in arm.events if e[0] == "move"]
        hover, grasp_z, lift_z = [-89.9, -532.0, 367.9], 322.9, 352.9
        self.assertIn(hover, moves)
        grasp_at = arm.events.index(("gripper", 0))
        before_grasp = [e[1] for e in arm.events[:grasp_at] if e[0] == "move"]
        self.assertEqual(before_grasp[-1], [-89.9, -532.0, grasp_z])  # closes only at the grasp depth
        # The descent and the lift are straight up/down over the rim point.
        after_hover = moves[moves.index(hover):]
        self.assertTrue(all(p[:2] == [-89.9, -532.0] for p in after_hover[:6]))
        self.assertIn([-89.9, -532.0, lift_z], moves)
        self.assertEqual(moves[-1], [round(v, 3) for v in HOME])  # back to the start pose
        self.assertEqual(len(arm.rpys), 1)  # the wrist never turns
        self.assertTrue(arm.closed)
        self.assertTrue(log["outcome"]["success"])
        self.assertEqual(json.loads(trials[0])["result"], "成功")
        self.assertEqual(json.loads(trials[0])["put_back_mm"], 0.8)  # measured after returning home
        self.assertEqual(log["put_back"]["distance_mm"], 0.8)
        self.assertGreaterEqual(min(p[2] for p in moves), grasp_z)  # never below the grasp depth

    def test_the_window_sees_the_plan_and_the_fingertips_move_towards_the_grasp(self):
        link = FakeLink(REPORT)
        run(["y", "y", "y", "y", "s", "", "y", "y"], link=link)
        planned = [plan for _, _, plan in link.shown if plan]
        self.assertEqual(planned[0], {"hover": [-89.9, -532.0, 367.9], "grasp": [-89.9, -532.0, 322.9]})
        descending = [tcp for step, tcp, _ in link.shown if step == "descending into the cup"]
        self.assertEqual([round(t[2], 1) for t in descending][-1], 322.9)  # last update is at the grasp depth
        self.assertTrue(all(s.isascii() for s, _, _ in link.shown))  # OpenCV cannot draw Chinese
        self.assertTrue(link.closed)

    def test_space_in_the_window_stops_the_arm(self):
        from task_skills.run_cartesian_jog import JogStopped

        class StoppableArm(RecordingArm):
            def wait(self, xyz, rpy, joints, stop_pressed, progress=None):
                if stop_pressed():
                    raise JogStopped("空格：已停止")
                return super().wait(xyz, rpy, joints, stop_pressed, progress)

        arm = StoppableArm()
        code, arm, lines, trials, log = run(["y", "y"], link=FakeLink(REPORT, stop_after_status="moving to hover"),
                                            arm=arm)
        self.assertEqual(code, 3)
        self.assertNotIn(("gripper", 0), arm.events)
        self.assertTrue(arm.closed)

    def test_auto_mode_runs_without_confirmations_and_asks_the_result_at_the_end(self):
        code, arm, lines, trials, log = run(["s", ""], extra=["--auto", "--auto-delay", "0.2", "--hold", "0.1"])
        self.assertEqual(code, 0)
        asks = [i for i, e in enumerate(arm.events) if e[0] == "ask"]
        last_move = max(i for i, e in enumerate(arm.events) if e[0] == "move")
        self.assertTrue(all(i > last_move for i in asks))  # nothing is asked until the arm is back
        self.assertEqual([e[1] for e in arm.events if e[0] == "gripper"], [100, 0, 100])
        self.assertTrue(log["auto"])
        self.assertTrue(log["outcome"]["success"])

    def test_auto_stop_during_the_countdown_sends_no_motion(self):
        code, arm, lines, trials, log = run([], link=FakeLink(REPORT, stop_after_status="AUTO start"),
                                            extra=["--auto", "--auto-delay", "0.3"])
        self.assertEqual(code, 3)
        self.assertEqual([e for e in arm.events if e[0] in ("move", "gripper")], [])
        self.assertIn(("stop",), arm.events)
        self.assertTrue(arm.closed)

    def test_auto_stop_between_steps_is_honoured_before_the_next_motion(self):
        # Space pressed while descending: the descent finishes in the sim, the grasp must not follow.
        code, arm, lines, trials, log = run([], link=FakeLink(REPORT, stop_after_status="descending into the cup"),
                                            extra=["--auto", "--auto-delay", "0.1"])
        self.assertEqual(code, 3)
        moves = [e[1] for e in arm.events if e[0] == "move"]
        self.assertEqual(moves[-1], [-89.9, -532.0, 322.9])
        self.assertNotIn(("gripper", 0), arm.events)
        self.assertTrue(arm.closed)

    def test_declining_at_the_hover_check_stops_there_and_still_disables(self):
        code, arm, lines, trials, log = run(["y", "y", ""])
        self.assertEqual(code, 1)
        self.assertNotIn(("gripper", 0), arm.events)
        moves = [e[1] for e in arm.events if e[0] == "move"]
        self.assertEqual(moves[-1][2], 367.9)  # stayed at the hover point
        self.assertTrue(arm.closed)
        self.assertIn("中止", json.loads(trials[0])["result"])

    def test_an_implausible_fit_sends_no_motion(self):
        bad = json.loads(json.dumps(REPORT))
        bad["fit"]["height"] = 12.0
        code, arm, lines, trials, log = run([], report=bad)
        self.assertEqual(code, 1)
        self.assertEqual([e for e in arm.events if e[0] in ("move", "gripper")], [])
        self.assertIn("杯高", json.loads(trials[0])["result"])


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(seconds, 0.001)


class TrackingLink(FakeLink):
    """The cup position over (fake) time, as the viewer would stream it, one new frame per tick.

    path(t, link) / visible(t, link) may look at link.views (banners shown so far) to script events
    like "the hand nudges the cup once the arm starts descending". arm_bias: while the fingertips
    are within 70 mm of the cup, the viewer would mark the frame arm_overlap and its position is
    shifted by this many mm (what the gripper over the cup did on the real rig).
    """

    def __init__(self, clock, path, visible=lambda t, link: True, noise=0.0, arm_bias=0.0, seed=0):
        super().__init__(REPORT)
        self.clock, self.path, self.visible = clock, path, visible
        self.noise, self.arm_bias = noise, arm_bias
        self.rng = __import__("random").Random(seed)
        self.tracking, self.views, self.tcp = False, [], None

    def status(self, step, tcp=None, plan=None):
        super().status(step, tcp, plan)
        self.views.append(step)
        if tcp is not None:
            self.tcp = tcp

    def seen(self, text):
        return any(text in v for v in self.views)

    def start_tracking(self, table_z, height, top_radius, bottom_radius=None):
        self.tracking, self.t0 = True, self.clock()

    def latest_track(self):
        t = self.clock() - self.t0
        x, y = self.path(t, self)
        x += self.rng.uniform(-self.noise, self.noise)
        y += self.rng.uniform(-self.noise, self.noise)
        overlap = bool(self.tcp) and math.dist(self.tcp[:2], (x, y)) < 70.0
        if overlap:
            x += self.arm_bias
        return {"t": t, "age": 0.0, "ok": self.visible(t, self), "xy": [x, y], "quality": 1.0,
                "arm_overlap": overlap, "overlap": 0.3 if overlap else 0.0}

    def stop_tracking(self):
        self.tracking = False


def follow_run(link, clock, extra=(), answers=("s", "")):
    arm = RecordingArm()
    replies = iter(answers)
    lines = []
    with TemporaryDirectory() as directory:
        code = main(["--sim", "--auto", "--auto-delay", "0", "--hold", "0", "--follow", "--out", directory, *extra],
                    ask=lambda _: next(replies), out=lines.append, arm=arm, link=link,
                    stop_pressed=lambda: False, clock=clock, sleep=clock.sleep)
        log = json.loads(next(Path(directory).glob("trial-*/trial.json")).read_text(encoding="utf-8"))
        telemetry = [json.loads(line) for line in
                     next(Path(directory).glob("trial-*/follow_log.jsonl")).read_text(encoding="utf-8").splitlines()]
    return code, arm, log, lines, telemetry


START = (-127.8, -532.0)
OFFSET = (-89.9 - -127.8, 0.0)  # rim point - cup centre in REPORT


def glide(points, speed=50.0, after=0.0):
    """Cup moved by hand along straight segments at `speed` mm/s (starting at `after` s), then left still."""
    def at(t, link=None):
        t -= after
        if t <= 0:
            return tuple(points[0])
        for (a, b) in zip(points, points[1:]):
            seconds = math.dist(a, b) / speed
            if t <= seconds:
                return tuple(a[i] + (b[i] - a[i]) * t / seconds for i in range(2))
            t -= seconds
        return tuple(points[-1])
    return at


def still(t, link=None):
    return START


class FollowTests(unittest.TestCase):
    def test_arm_follows_the_moved_cup_and_grasps_where_it_finally_stands(self):
        clock = FakeClock()
        end = (-100.0, -490.0)
        code, arm, log, lines, _ = follow_run(TrackingLink(clock, glide([START, end])), clock)
        self.assertEqual(code, 0)
        followed = log["plan"]["followed_to"]["rim_point"]
        self.assertAlmostEqual(followed[0], end[0] + OFFSET[0], delta=1.0)
        self.assertAlmostEqual(followed[1], end[1] + OFFSET[1], delta=1.0)
        servos = [p for p in arm.servos if p != "end"]
        speeds = [math.dist(a, b) / 0.03 for a, b in zip(servos, servos[1:])]
        self.assertLessEqual(max(speeds), 100.0 + 0.01)  # speed cap
        accels = [abs(b - a) / 0.03 for a, b in zip(speeds, speeds[1:])]
        self.assertLessEqual(max(accels), 500.0 * 1.05 + 1.0)  # no jerks: acceleration capped
        self.assertAlmostEqual(min(p[2] for p in servos), 322.9, delta=0.11)  # exactly to the grasp depth
        self.assertEqual([e[1] for e in arm.events if e[0] == "gripper"], [100, 0, 100])

    def test_noise_and_the_gripper_over_the_cup_do_not_make_the_arm_go_back_and_forth(self):
        # The real failure: still cup, +-1.5 mm frame noise, and 6 mm bias while the gripper is over it.
        clock = FakeClock()
        link = TrackingLink(clock, still, noise=1.5, arm_bias=6.0)
        code, arm, log, lines, telemetry = follow_run(link, clock)
        self.assertEqual(code, 0)
        self.assertFalse(any("回到旁边" in line or "抬回" in line or "又动了" in line for line in lines), lines)
        phases = [row[1] for row in telemetry[1:]]
        changes = [b for a, b in zip(phases, phases[1:]) if a != b]
        self.assertEqual(changes, ["hands_off", "approach", "descend"])  # one clean pass, no cycling
        followed = log["plan"]["followed_to"]["rim_point"]
        self.assertAlmostEqual(followed[0], START[0] + OFFSET[0], delta=1.5)

    def test_reacts_quickly_when_the_cup_is_moved(self):
        clock = FakeClock()
        jump = (START[0] + 150.0, START[1] + 40.0)
        # Cup still while the arm gets beside it, then carried 155 mm in 0.5 s and left there.
        link = TrackingLink(clock, glide([START, jump], speed=310.0, after=4.0))
        code, arm, log, lines, telemetry = follow_run(link, clock)
        self.assertEqual(code, 0)
        rows = telemetry[1:]
        moved_at = 4.0 + 0.5
        foot = REPORT["camera_xy"]
        away = [jump[0] - foot[0], jump[1] - foot[1]]
        n = math.hypot(*away)
        rim = [jump[0] + OFFSET[0], jump[1] + OFFSET[1]]
        standoff = [rim[0] + 45 * away[0] / n, rim[1] + 45 * away[1] / n]
        arrive = next(r[0] for r in rows if r[0] > moved_at and math.dist(r[2][:2], standoff) < 5.0)
        self.assertLess(arrive - moved_at, 2.5)  # beside the cup's new place within 2.5 s

    def test_no_descent_while_the_cup_keeps_moving_and_a_move_during_descent_lifts_back(self):
        clock = FakeClock()
        moved = (START[0] + 30.0, START[1])  # a real push, well beyond the gripper's few-mm bias

        def path(t, link):
            if link.seen("DESCENDING") and not hasattr(link, "nudged_at"):
                link.nudged_at = t + 0.5
            return moved if hasattr(link, "nudged_at") and t >= link.nudged_at else START

        link = TrackingLink(clock, path)
        code, arm, log, lines, telemetry = follow_run(link, clock)
        self.assertEqual(code, 0)
        self.assertTrue(any("抬回" in line for line in lines))
        nudge_row = next(i for i, r in enumerate(telemetry[1:]) if r[1] == "retreat")
        deepest_before = min(r[2][2] for r in telemetry[1:nudge_row + 1])
        self.assertGreater(deepest_before, 322.9 + 1.0)  # lifted back before reaching the old grasp depth
        followed = log["plan"]["followed_to"]["rim_point"]
        self.assertAlmostEqual(followed[0], moved[0] + OFFSET[0], delta=1.0)  # grasped at the new spot

    def test_a_cup_pushed_next_to_the_waiting_gripper_is_still_followed(self):
        # Every frame shows the gripper over the cup after the push; a big persistent shift must still
        # move the cup estimate, or the arm waits beside the old place forever.
        clock = FakeClock()
        pushed = (START[0] + 70.0, START[1])  # pushed right under the 45 mm waiting point

        def path(t, link):
            if link.seen("FOLLOWING") and not hasattr(link, "pushed_at"):
                link.pushed_at = t + 2.0
            return pushed if hasattr(link, "pushed_at") and t >= link.pushed_at else START

        code, arm, log, lines, telemetry = follow_run(TrackingLink(clock, path), clock,
                                                      extra=["--follow-timeout", "30"])
        self.assertEqual(code, 0, lines[-3:])
        followed = log["plan"]["followed_to"]["rim_point"]
        self.assertAlmostEqual(followed[0], pushed[0] + OFFSET[0], delta=1.5)

    def test_an_invisible_cup_means_hold_still_and_time_out(self):
        clock = FakeClock()
        link = TrackingLink(clock, still, visible=lambda t, link: False)
        arm = RecordingArm()
        replies = iter(())
        with TemporaryDirectory() as directory:
            code = main(["--sim", "--auto", "--auto-delay", "0", "--follow", "--follow-timeout", "3", "--out",
                         directory], ask=lambda _: next(replies), out=lambda _: None, arm=arm, link=link,
                        stop_pressed=lambda: False, clock=clock, sleep=clock.sleep)
        self.assertEqual(code, 1)  # aborted
        self.assertNotIn(("gripper", 0), arm.events)
        servos = [p for p in arm.servos if p != "end"]
        self.assertTrue(all(p[2] >= 387.0 - 0.5 for p in servos))  # never went below the safe height
        self.assertFalse(link.tracking)

    def test_while_following_the_arm_waits_beside_the_cup_away_from_the_camera(self):
        clock = FakeClock()
        end = (-100.0, -490.0)
        link = TrackingLink(clock, glide([START, end]))
        code, arm, log, lines, telemetry = follow_run(link, clock)
        self.assertEqual(code, 0)
        last_follow = [r for r in telemetry[1:] if r[1] == "follow"][-1]
        rim = [end[0] + OFFSET[0], end[1] + OFFSET[1]]
        to_arm = [last_follow[2][0] - rim[0], last_follow[2][1] - rim[1]]
        foot = REPORT["camera_xy"]
        away = [end[0] - foot[0], end[1] - foot[1]]
        self.assertAlmostEqual(math.hypot(*to_arm), 45.0, delta=2.5)           # 45 mm out from the grasp point
        self.assertGreater(away[0] * to_arm[0] + away[1] * to_arm[1], 0.0)    # away from the camera
        approach = [r for r in telemetry[1:] if r[1] == "approach"]
        detour = math.dist(approach[0][2][:2], approach[-1][2][:2])
        self.assertLess(detour, 47.5)  # the trip back to the grasp point is the 45 mm, not ~70 mm

    def test_hidden_cup_during_approach_and_descent_is_still_grasped(self):
        clock = FakeClock()
        link = TrackingLink(clock, still, visible=lambda t, link: not link.seen("APPROACHING"))
        code, arm, log, lines, _ = follow_run(link, clock)
        self.assertEqual(code, 0)
        followed = log["plan"]["followed_to"]["rim_point"]
        self.assertAlmostEqual(followed[0], START[0] + OFFSET[0], delta=1.0)
        self.assertIn(("gripper", 0), arm.events)

    def test_a_moved_then_hidden_cup_does_not_freeze_the_arm(self):
        clock = FakeClock()
        end = (-110.0, -480.0)
        link = TrackingLink(clock, glide([START, end]), visible=lambda t, link: not link.seen("HANDS OFF"))
        code, arm, log, lines, _ = follow_run(link, clock)
        self.assertEqual(code, 0)
        followed = log["plan"]["followed_to"]["rim_point"]
        self.assertAlmostEqual(followed[0], end[0] + OFFSET[0], delta=1.0)
        self.assertAlmostEqual(followed[1], end[1] + OFFSET[1], delta=1.0)


if __name__ == "__main__":
    unittest.main()
