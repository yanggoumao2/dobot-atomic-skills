"""Focused mock tests for the lease-guarded MobileArm adapter."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from task_skills import mobile_arm as mobile


class SimMobileArmTests(unittest.TestCase):
    def _open(self):
        adapter = mobile.MobileArm("right", live=False)
        snapshot = adapter.open()
        self.addCleanup(adapter.close)
        return adapter, snapshot

    def test_constructor_is_inert_and_open_returns_snapshot(self):
        with mock.patch.object(mobile, "SimArm", side_effect=AssertionError("constructed early")):
            adapter = mobile.MobileArm("right", live=False)
        self.assertIsNone(adapter._backend)

        adapter, snapshot = self._open()
        self.assertEqual(set(("pose", "joints", "mode")), set(snapshot))
        self.assertEqual(len(snapshot["pose"]), 6)
        self.assertEqual(len(snapshot["joints"]), 6)
        self.assertEqual(snapshot["mode"], 5)
        self.assertEqual(adapter.workspace["horizontal_radius_mm"], 200.0)
        self.assertEqual(adapter.workspace["z_from_enable_mm"], [-150.0, 150.0])

    def test_step_accepts_only_finite_base_steps(self):
        adapter, initial = self._open()
        cancel = threading.Event()

        result = adapter.step("x", 1, 0.5, cancel)
        self.assertAlmostEqual(result["pose"][0], initial["pose"][0] + 0.5)
        result = adapter.step("Y", -1, 1.0, cancel)
        self.assertAlmostEqual(result["pose"][1], initial["pose"][1] - 1.0)
        result = adapter.step("Z", 1, 5, cancel)
        self.assertAlmostEqual(result["pose"][2], initial["pose"][2] + 5.0)

        sent_before = len(adapter._backend.sent)
        for args in (("A", 1, 1), ("X", 0, 1), ("X", True, 1),
                     ("X", 1, 2), ("X", 1, float("nan"))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                adapter.step(*args, cancel)
        self.assertEqual(len(adapter._backend.sent), sent_before)

    def test_pre_cancel_and_unsafe_mode_send_no_move(self):
        adapter, _ = self._open()
        backend = adapter._backend
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(mobile.JogStopped):
            adapter.step("X", 1, 1, cancelled)
        self.assertEqual(backend.sent, [])

        original_read = backend.read
        backend.read = lambda: {**original_read(), "mode": 7}
        with self.assertRaisesRegex(mobile.MobileArmError, "RobotMode=5"):
            adapter.step("X", 1, 1, threading.Event())
        self.assertEqual(backend.sent, [])

    def test_missing_state_and_soft_limit_fail_closed(self):
        adapter, _ = self._open()
        backend = adapter._backend
        backend.read = lambda: {"pose": [0.0] * 6, "joints": [0.0] * 6}
        with self.assertRaisesRegex(mobile.MobileArmError, "RobotMode"):
            adapter.snapshot()

        # A fresh reopen is unnecessary: restore valid state, then narrow the
        # existing Limits instance to exercise its refusal path.
        backend.read = lambda: {"pose": list(backend.pose),
                                "joints": list(backend.joints), "mode": 5}
        adapter._limits.radius = 0.0
        with self.assertRaises(mobile.JogRefused):
            adapter.step("X", 1, 0.5, threading.Event())
        self.assertEqual(backend.sent, [])

    def test_drive_scales_diagonal_and_proportional_samples_then_settles_on_ttl(self):
        adapter, initial = self._open()
        now = [0.0]
        adapter._clock = lambda: now[0]
        adapter._wait_tick = lambda cancel, period=0.03, tick_start=None: now.__setitem__(
            0, now[0] + period)
        samples = iter([
            (0.5, 0.5, 0.0), (0.5, 0.5, 0.0),
            (0.25, 0.0, 0.0), (0.25, 0.0, 0.0),
            None,  # release or TTL expiry ends the stream
        ])

        result = adapter.drive(lambda: next(samples), threading.Event())
        points = [point for point in adapter._backend.servos if point != "end"]
        self.assertEqual(adapter._backend.servos[-1], "end")
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0] - initial["pose"][0], 0.45, places=3)
        self.assertAlmostEqual(points[0][1] - initial["pose"][1], 0.45, places=3)
        self.assertAlmostEqual(points[1][0] - points[0][0], 0.225, places=3)
        self.assertAlmostEqual(points[1][1], points[0][1], places=3)
        self.assertEqual(result["mode"], 5)
        self.assertAlmostEqual(result["pose"][0], initial["pose"][0] + 0.675, places=3)

    def test_drive_keeps_streaming_while_thumb_moves(self):
        # A changing thumb must not drop points (that caused stop-start motion);
        # release (None) still ends the stream before dispatch.
        adapter, initial = self._open()
        samples = iter([(0.5, 0, 0), (0.45, 0, 0), (0.4, 0, 0), (0.35, 0, 0), None])
        result = adapter.drive(lambda: next(samples), threading.Event())
        points = [p for p in adapter._backend.servos if p != "end"]
        self.assertEqual(len(points), 2)  # each tick reads two samples; none dropped
        self.assertGreater(points[1][0], points[0][0])
        self.assertGreater(points[0][0], initial["pose"][0])
        self.assertEqual(result["mode"], 5)
        samples = iter([(0.5, 0, 0), None])
        adapter.drive(lambda: next(samples), threading.Event())
        self.assertEqual(len([p for p in adapter._backend.servos if p != "end"]), 2)

    def test_drive_stops_when_arm_lags_far_behind_setpoint(self):
        adapter, initial = self._open()
        backend = adapter._backend
        now = [0.0]
        adapter._clock = lambda: now[0]
        def wait_tick(cancel, period=0.03, tick_start=None):
            now[0] += period
            backend.pose[0] = initial["pose"][0] - 50  # arm did not follow at all
        adapter._wait_tick = wait_tick
        with self.assertRaises(mobile.MobileArmError):
            adapter.drive(lambda: (1.0, 0.0, 0.0), threading.Event())

    def test_drive_soft_limit_settles_and_leaves_adapter_ready_to_steer_away(self):
        adapter, initial = self._open()
        backend = adapter._backend
        backend.pose[0] = initial["pose"][0] + 199.95
        backend.joints = backend.solve(backend.pose[:3], backend.pose[3:], None)
        samples = iter([(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        result = adapter.drive(lambda: next(samples), threading.Event())
        self.assertEqual([p for p in backend.servos if p != "end"], [])
        self.assertEqual(result["mode"], 5)
        self.assertAlmostEqual(result["pose"][0], initial["pose"][0] + 199.95)

        samples = iter([(-1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), None])
        corrected = adapter.drive(lambda: next(samples), threading.Event())
        self.assertLess(corrected["pose"][0], result["pose"][0])
        self.assertEqual(corrected["mode"], 5)

    def test_cancel_during_drive_stops_before_close(self):
        adapter, _ = self._open()
        backend = adapter._backend
        events = []
        backend.stop = lambda: events.append("stop")
        backend.close = lambda: events.append("close")
        now = [0.0]
        adapter._clock = lambda: now[0]

        def cancel_tick(cancel, period=0.03, tick_start=None):
            cancel.set()
            raise mobile.JogStopped("cancelled in test")

        adapter._wait_tick = cancel_tick
        samples = iter([(0.5, 0, 0), (0.5, 0, 0)])
        cancel = threading.Event()
        with self.assertRaises(mobile.JogStopped):
            adapter.drive(lambda: next(samples), cancel)
        adapter.close()
        self.assertEqual(events, ["stop", "close"])


class _FakeLease:
    instances = []
    busy = False

    def __init__(self, path=None, purpose=None):
        self.path, self.purpose = path, purpose
        self.acquired = False
        self.released = False
        self.__class__.instances.append(self)

    def acquire(self):
        if self.__class__.busy:
            raise RuntimeError("lease busy")
        self.acquired = True

    def release(self):
        self.released = True
        self.acquired = False


class _FakeDashboard:
    def __init__(self, state):
        self.state = state
        self.calls = []
        self.fail_disable = False
        self.stop_reply = "0,{},StopScript();"

    def EnableRobot(self):
        self.calls.append("EnableRobot")
        self.state["mode"] = 5
        return "0,{},EnableRobot();"

    def DisableRobot(self):
        self.calls.append("DisableRobot")
        if self.fail_disable:
            return "1,{},DisableRobot();"
        self.state["mode"] = 4
        return "0,{},DisableRobot();"

    def StopScript(self):
        self.calls.append("StopScript")
        if self.state["mode"] in (7, 10):
            self.state["mode"] = 5
        return self.stop_reply

    def PowerOn(self):
        self.calls.append("PowerOn")
        if self.state["mode"] == 3:
            self.state["mode"] = 4
        return "0,{},PowerOn();"

    def ClearError(self):
        self.calls.append("ClearError")
        if self.state["mode"] == 9:
            self.state["mode"] = getattr(self, "mode_after_clear", 5)
        return "0,{},ClearError();"


class _FakeRig:
    def __init__(self, state):
        self.dashboard = _FakeDashboard(state)
        self.motion = None


class _FakeDriver:
    instances = []

    def __init__(self, profile, **kwargs):
        self.profile = profile
        self._lock_path = Path("fake-device-owner")
        self._lease = None
        self.state = {"pose": [-150.0, -350.0, 200.0, 180.0, 0.0, 90.0],
                      "joints": [-15.0, -35.0, 20.0, 0.0, 90.0, 90.0],
                      "mode": 4, "source": "dashboard"}
        self.rig = _FakeRig(self.state)
        self.rig_requested = False
        self.fail_after_enable = False
        self.cancel_on_enable = None
        self.fail_motion = False
        self.cancel_on_motion = None
        self.last_plan = None
        self.motion_commands = []
        self.closed = False
        self.configurations = []
        self.__class__.instances.append(self)

    def _rig(self, arm, profile, with_motion=False):
        self.rig_requested = True
        return self.rig

    def _read_state(self, rig):
        return {**self.state, "pose": list(self.state["pose"]),
                "joints": list(self.state["joints"])}

    def _ensure_session(self, profile, arm, speed, *, clear_errors=True):
        self.configurations.append((arm, speed, clear_errors))
        if self.cancel_on_enable is not None:
            self.cancel_on_enable.set()
        self.rig.dashboard.EnableRobot()
        if self.fail_after_enable:
            raise RuntimeError("configuration failed after enable")
        return self.rig

    def _configure_speed(self, rig, speed, *, linear):
        self.configurations.append(("linear-speed", speed, linear))

    def _send_motion(self, rig, command, label):
        self.motion_commands.append((command, label))
        if self.cancel_on_motion is not None and command.startswith("MovL("):
            self.cancel_on_motion.set()
        if self.fail_motion:
            raise RuntimeError("missing motion echo")
        return {"result": "acknowledged"}

    def _inverse_solution(self, rig, xyz, rpy, seed, profile):
        self.last_plan = (list(xyz), list(rpy))
        return [xyz[0] / 10.0, xyz[1] / 10.0, xyz[2] / 10.0,
                0.0, 90.0, float(rpy[2])]

    def _forward_solution(self, rig, joints, profile):
        xyz, rpy = self.last_plan
        return {"xyz": list(xyz), "rpy": list(rpy)}

    def stop(self, arms):
        results = {}
        for arm in arms:
            reply = self.rig.dashboard.StopScript()
            results[arm] = {"stop_script": reply, "disable_robot_called": False}
        return {"stopped": results}

    def close(self):
        self.closed = True
        if self._lease is not None:
            self._lease.release()
            self._lease = None
            return {"lease_released": True, "closed": []}
        return {"lease_released": False, "closed": []}


_original_driver_init = _FakeDriver.__init__


def _powered_off_init(self, profile, **kwargs):
    _original_driver_init(self, profile, **kwargs)
    self.state["mode"] = 3


class LiveMobileArmTests(unittest.TestCase):
    def setUp(self):
        _FakeLease.instances = []
        _FakeLease.busy = False
        _FakeDriver.instances = []

    def _environment(self, *, busy=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        profile = Path(temporary.name) / "profile.json"
        profile.write_text(json.dumps({"arms": {"right": {"ip": "127.0.0.1"}}}),
                           encoding="utf-8")
        _FakeLease.busy = busy
        patches = [
            mock.patch.object(mobile, "DEFAULT_PROFILE", profile),
            mock.patch.object(mobile, "LiveDriver", _FakeDriver),
            mock.patch.object(mobile, "DeviceLease", _FakeLease),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        return _FakeDriver.instances

    def test_busy_lease_fails_before_rig_sockets_or_stop_and_names_q(self):
        self._environment(busy=True)
        adapter = mobile.MobileArm("right", live=True)
        with self.assertRaises(mobile.MobileArmError) as raised:
            adapter.open()
        driver = _FakeDriver.instances[0]
        self.assertIn("q", str(raised.exception))
        self.assertFalse(driver.rig_requested)
        self.assertNotIn("EnableRobot", driver.rig.dashboard.calls)
        self.assertNotIn("DisableRobot", driver.rig.dashboard.calls)
        self.assertNotIn("StopScript", driver.rig.dashboard.calls)

    def test_cancel_before_enable_releases_lease_without_enable_or_disable(self):
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        cancel = threading.Event()
        driver = None

        class CancellingDriver(_FakeDriver):
            def __init__(self, *args, **kwargs):
                nonlocal driver
                super().__init__(*args, **kwargs)
                driver = self
                self.cancel_on_enable = cancel

        with mock.patch.object(mobile, "LiveDriver", CancellingDriver):
            with self.assertRaises(mobile.JogStopped):
                adapter.open(cancel)
        self.assertNotIn("EnableRobot", driver.rig.dashboard.calls)
        self.assertNotIn("DisableRobot", driver.rig.dashboard.calls)
        self.assertTrue(_FakeLease.instances[-1].released)

    def test_partial_enable_failure_is_disabled_during_open_rollback(self):
        self._environment()
        driver = None

        class FailingDriver(_FakeDriver):
            def __init__(self, *args, **kwargs):
                nonlocal driver
                super().__init__(*args, **kwargs)
                driver = self
                self.fail_after_enable = True

        with mock.patch.object(mobile, "LiveDriver", FailingDriver):
            adapter = mobile.MobileArm("right", live=True)
            with self.assertRaisesRegex(RuntimeError, "after enable"):
                adapter.open()
        self.assertEqual(driver.rig.dashboard.calls[:2], ["EnableRobot", "DisableRobot"])
        self.assertTrue(_FakeLease.instances[-1].released)

    def test_partial_open_rollback_surfaces_disable_failure(self):
        self._environment()

        class PartialShutdownFailureDriver(_FakeDriver):
            def _ensure_session(self, profile, arm, speed, *, clear_errors=True):
                self.fail_after_enable = True
                self.rig.dashboard.fail_disable = True
                return super()._ensure_session(
                    profile, arm, speed, clear_errors=clear_errors)

        with mock.patch.object(mobile, "LiveDriver", PartialShutdownFailureDriver):
            adapter = mobile.MobileArm("right", live=True)
            with self.assertRaises(mobile.MobileCleanupError) as raised:
                adapter.open()
        driver = PartialShutdownFailureDriver.instances[0]
        self.assertIn("DisableRobot", str(raised.exception))
        self.assertIn("DisableRobot", driver.rig.dashboard.calls)
        self.assertTrue(_FakeLease.instances[-1].released)

    def test_close_surfaces_disable_failure_and_is_idempotent(self):
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        adapter.open()
        driver = _FakeDriver.instances[0]
        driver.rig.dashboard.fail_disable = True

        with self.assertRaises(mobile.MobileCleanupError):
            adapter.close()
        self.assertTrue(driver.closed)
        self.assertTrue(_FakeLease.instances[-1].released)
        adapter.close()  # a second close cannot hide or repeat the failure

    def test_live_stop_ack_failure_is_reported_and_close_still_disables(self):
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        adapter.open()
        driver = _FakeDriver.instances[0]
        driver.rig.dashboard.stop_reply = "1,{},StopScript();"
        adapter._motion_pending = True

        with self.assertRaisesRegex(RuntimeError, "StopScript.*status 0"):
            adapter._safe_stop()
        adapter.close()
        self.assertIn("DisableRobot", driver.rig.dashboard.calls)
        self.assertTrue(_FakeLease.instances[-1].released)

    def test_step_cancellation_uses_checked_stop_ack_once(self):
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        adapter.open()
        driver = _FakeDriver.instances[0]
        cancel = threading.Event()
        driver.cancel_on_motion = cancel

        with self.assertRaises(mobile.JogStopped):
            adapter.step("X", 1, 0.5, cancel)
        self.assertEqual(driver.rig.dashboard.calls.count("StopScript"), 1)
        adapter.close()
        self.assertIn("DisableRobot", driver.rig.dashboard.calls)

    def test_live_servoj_uses_one_strict_driver_write_without_retry(self):
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        adapter.open()
        driver = _FakeDriver.instances[0]
        adapter._send_servo([1, 2, 3], [180, 0, 90], [10, 20, 30, 40, 90, 60], 0.03)
        self.assertEqual(len(driver.motion_commands), 1)
        command, label = driver.motion_commands[0]
        self.assertEqual(command,
                         "ServoJ(10.000000,20.000000,30.000000,40.000000,90.000000,60.000000,0.030000,gain=500)")
        self.assertEqual(label, "mobile arm ServoJ")
        driver.fail_motion = True
        with self.assertRaisesRegex(RuntimeError, "missing motion echo"):
            adapter._send_servo([1, 2, 3], [180, 0, 90], [10, 20, 30, 40, 90, 60], 0.03)
        self.assertEqual(len(driver.motion_commands), 2)  # never retries a lost echo

    def test_no_live_command_when_dashboard_freshness_is_missing(self):
        self._environment()

        class StaleDriver(_FakeDriver):
            def _read_state(self, rig):
                state = super()._read_state(rig)
                state["source"] = "30004 broadcast"
                return state

        with mock.patch.object(mobile, "LiveDriver", StaleDriver):
            adapter = mobile.MobileArm("right", live=True)
            with self.assertRaisesRegex(mobile.MobileArmError, "dashboard state"):
                adapter.open()
        driver = StaleDriver.instances[0]
        self.assertNotIn("EnableRobot", driver.rig.dashboard.calls)
        self.assertTrue(_FakeLease.instances[-1].released)

    def test_alarm_during_session_still_releases_without_cleanup_failure(self):
        # 2026-11-21 on site: an alarm (9) made DisableRobot fail, the release
        # was reported as a cleanup failure and the remote locked itself out.
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        adapter.open()
        driver = _FakeDriver.instances[0]
        driver.state["mode"] = 9
        driver.rig.dashboard.fail_disable = True
        adapter.close()  # no MobileCleanupError
        self.assertTrue(_FakeLease.instances[-1].released)

    def test_recover_aborts_a_paused_queue_and_reenables(self):
        # After an alarm the controller can sit paused (10) with the old queue;
        # ClearError alone never returns it to idle, so nothing moves again.
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        adapter.open()
        driver = _FakeDriver.instances[0]
        dashboard = driver.rig.dashboard
        driver.state["mode"] = 9
        dashboard.mode_after_clear = 10
        driver._stop_requested = True
        adapter._stop_sent = True
        adapter.recover(threading.Event())
        self.assertEqual(driver.state["mode"], 5)
        self.assertIn("ClearError", dashboard.calls)
        self.assertIn("StopScript", dashboard.calls[dashboard.calls.index("ClearError"):])
        self.assertFalse(driver._stop_requested)  # gripper usable again
        self.assertFalse(adapter._stop_sent)

    def test_clear_and_enable_powers_on_an_unpowered_arm(self):
        # 2026-11-22: after an emergency stop both arms sat at RobotMode=3 and
        # the operator wants the phone, not the pendant, to bring them back.
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        with mock.patch.object(_FakeDriver, "__init__", _powered_off_init):
            adapter.open(threading.Event(), clear_errors=True)
        dashboard = _FakeDriver.instances[0].rig.dashboard
        self.assertLess(dashboard.calls.index("PowerOn"), dashboard.calls.index("EnableRobot"))
        self.assertEqual(_FakeDriver.instances[0].state["mode"], 5)

    def test_plain_connect_never_powers_on(self):
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        with mock.patch.object(_FakeDriver, "__init__", _powered_off_init):
            with self.assertRaises(mobile.MobileArmError):
                adapter.open()
        self.assertNotIn("PowerOn", _FakeDriver.instances[0].rig.dashboard.calls)

    def test_close_aborts_running_queue_before_disable(self):
        self._environment()
        adapter = mobile.MobileArm("right", live=True)
        adapter.open()
        driver = _FakeDriver.instances[0]
        driver.state["mode"] = 7
        adapter.close()
        calls = driver.rig.dashboard.calls
        self.assertLess(calls.index("StopScript"), calls.index("DisableRobot"))
        self.assertEqual(driver.state["mode"], 4)


if __name__ == "__main__":
    unittest.main()


class MobileArmActionTests(unittest.TestCase):
    def test_sim_preset_gripper_and_speed(self):
        import threading
        from task_skills.mobile_arm import MobileArm, GRIPPER_CLOSED
        arm = MobileArm("right", False)
        arm.open()
        cancel = threading.Event()
        before = arm.snapshot()
        arm.set_speed(150)
        self.assertEqual((arm.drive_speed, arm.preset_percent), (150.0, 50))
        arm.set_speed(5)
        self.assertEqual(arm.preset_percent, 5)
        for bad in ("fast", 151, 0, True):
            with self.assertRaises(ValueError):
                arm.set_speed(bad)
        result = arm.move_joints([-10.0, -30.0, 25.0, 0.0, 90.0, 90.0], "test", cancel)
        self.assertEqual(result["joints"][:3], [-10.0, -30.0, 25.0])
        self.assertNotEqual(result["pose"][:3], before["pose"][:3])
        # The soft workspace follows the preset, so a small step is still allowed.
        arm.step("Z", 1, 1, cancel)
        before_photo = arm.snapshot()["pose"]
        sent_before = len(arm._backend.sent)
        photo = arm.go_photo(cancel)
        self.assertEqual([round(v, 3) for v in photo["pose"][:3]], [-300.0, -160.0, 535.0])
        # Straight MovL legs with the orientation kept: rise in place, then cross at Z 535.
        legs = arm._backend.sent[sent_before:]
        self.assertEqual([leg[0] for leg in legs],
                         [[before_photo[0], before_photo[1], 535.0], [-300.0, -160.0, 535.0]])
        self.assertTrue(all(leg[1] == before_photo[3:] for leg in legs))
        arm.step("X", 1, 1, cancel)
        arm.set_gripper(GRIPPER_CLOSED, cancel)
        self.assertEqual(arm.gripper_position, 0)
        with self.assertRaises(ValueError):
            arm.set_gripper(128, cancel)
        arm.close()
