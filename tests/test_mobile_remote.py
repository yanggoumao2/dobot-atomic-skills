"""The command gate, cancellation and network lease can be exercised without a rig."""
import threading
import time
import unittest

from task_skills.run_mobile_remote import Remote, RequestError
from task_skills.run_cartesian_jog import JogStopped


class FakeDevice:
    def __init__(self, *args):
        self.closed = 0
        self.moves = []
        self.accept = threading.Event()
        self.entered = threading.Event()

    def open(self, cancel=None, clear_errors=False):
        self.cleared = getattr(self, "cleared", 0) + bool(clear_errors)
        return {"pose": [0] * 6, "mode": 5}

    def recover(self, cancel):
        self.cleared = getattr(self, "cleared", 0) + 1
        return self.open()

    def close(self):
        self.closed += 1

    def step(self, axis, sign, mm, cancel):
        self.moves.append((axis, sign, mm))
        self.entered.set()
        self.accept.wait(2)
        if cancel.is_set():
            raise JogStopped("operator stopped the move")
        return self.open()

    def drive(self, sample, cancel):
        self.entered.set()
        while not cancel.is_set() and sample() is not None:
            time.sleep(.002)
        return self.open()


class RemoteTests(unittest.TestCase):
    def setUp(self):
        self.now = 50
        self.device = FakeDevice()
        self.remote = Remote(False, clock=lambda: self.now, factory=lambda *args: self.device)
        self.token = self.remote.api("/api/pair", {"code": self.remote.code}, "")["token"]
        self.addCleanup(self.remote.shutdown)

    def post(self, endpoint, **body):
        body.setdefault("issued", self.now)
        return self.remote.api("/api/" + endpoint, body, self.token)

    def ready(self):
        self.post("open", revision=0, arm="left")
        self.remote.worker.join(1)
        self.assertEqual(self.remote.phase, "ready")

    def start_drag(self):
        self.post("drive", seq=1, revision=self.remote.revision, x=.3, y=.4, z=0)
        self.assertTrue(self.device.entered.wait(1))
        self.assertEqual(self.remote.phase, "driving")

    def test_pairing_and_single_operator(self):
        with self.assertRaises(RequestError):
            self.remote.api("/api/open", {"arm": "left", "revision": 0}, "")
        with self.assertRaises(RequestError):
            self.remote.api("/api/pair", {"code": self.remote.code}, "")
        for _ in range(8):
            with self.assertRaises(RequestError):
                self.remote.api("/api/pair", {"code": ""}, "")
        with self.assertRaises(RequestError) as limited:
            self.remote.api("/api/pair", {"code": self.remote.code}, "")
        self.assertEqual(limited.exception.status, 429)

    def test_step_rejects_duplicate_busy_and_invalid_values(self):
        self.ready()
        rev = self.remote.revision
        for extra in ({"axis": "J1"}, {"mm": True}, {"mm": 10}, {"direction": False}):
            body = dict(axis="X", direction=1, mm=1, revision=rev)
            body.update(extra)
            with self.assertRaises(RequestError):
                self.post("step", **body)
        self.post("step", revision=rev, axis="X", direction=1, mm=1)
        self.assertTrue(self.device.entered.wait(1))
        with self.assertRaises(RequestError):
            self.post("step", revision=rev, axis="X", direction=1, mm=1)
        self.post("stop")
        self.device.accept.set()
        self.remote.worker.join(1)
        self.assertEqual((self.remote.phase, self.device.closed, len(self.device.moves)), ("idle", 1, 1))

    def test_release_overtakes_first_drive(self):
        self.ready()
        self.post("release", seq=2)
        with self.assertRaises(RequestError):
            self.post("drive", seq=1, revision=self.remote.revision, x=1, y=0, z=0)
        self.assertFalse(self.device.entered.is_set())

    def test_network_delayed_start_rejected_with_fresh_heartbeat(self):
        self.ready()
        self.now += 2
        self.post("state")
        with self.assertRaises(RequestError):
            self.post("drive", seq=1, issued=self.now - 1, revision=self.remote.revision, x=1, y=0, z=0)
        self.assertFalse(self.device.entered.is_set())

    def test_analog_update_and_release(self):
        self.ready()
        self.start_drag()
        revision = self.remote.revision
        self.post("drive", seq=2, revision=revision, x=0, y=0, z=-.2)
        self.assertEqual(self.remote.sample(), (0, 0, -.2))
        self.post("release", seq=4)
        self.remote.worker.join(1)
        self.assertEqual(self.remote.phase, "ready")
        with self.assertRaises(RequestError):
            self.post("drive", seq=3, revision=revision, x=0, y=0, z=1)

    def test_thumb_timeout_independent_of_page_heartbeat(self):
        self.ready()
        self.start_drag()
        revision = self.remote.revision
        self.now += .5
        self.post("state")
        self.remote.worker.join(1)
        self.assertEqual(self.remote.phase, "ready")
        with self.assertRaises(RequestError):
            self.post("drive", seq=2, revision=revision, x=0, y=0, z=1)
        self.now += 6
        self.remote.expire()
        self.remote.worker.join(1)
        self.assertEqual((self.remote.phase, self.device.closed), ("idle", 1))

    def test_explicit_stop_during_drag(self):
        self.ready()
        self.start_drag()
        self.post("stop")
        self.remote.worker.join(1)
        self.assertEqual((self.remote.phase, self.device.closed), ("idle", 1))


from task_skills.run_mobile_remote import Settings, to_base


class ActionDevice(FakeDevice):
    def __init__(self, *args):
        super().__init__(*args)
        self.calls = []
        self.speed = None
        self.gripper_position = None
        self.fail_gripper = False

    def set_speed(self, level):
        self.speed = level

    def go_home(self, cancel):
        self.calls.append("home")
        return self.open()

    def go_photo(self, cancel):
        self.calls.append("photo")
        return self.open()

    def set_gripper(self, position, cancel):
        if self.fail_gripper:
            raise RuntimeError("serial timeout")
        self.gripper_position = position
        return self.open()


class OperatorFrameTests(unittest.TestCase):
    def test_forward_points_away_from_operator(self):
        # Robot front = base -Y. Standing in front, forward moves towards +Y
        # (towards the robot) and the operator's right is +X.
        self.assertEqual(to_base("front", 270, 0, 1, 0), (0, 1, 0))
        self.assertEqual(to_base("front", 270, 1, 0, 0), (1, 0, 0))
        # Behind the robot everything flips.
        self.assertEqual(to_base("back", 270, 0, 1, 0), (0, -1, 0))
        self.assertEqual(to_base("back", 270, 1, 0, 0), (-1, 0, 0))
        # On the robot's left (+X when front is -Y), forward is -X.
        self.assertEqual(to_base("left", 270, 0, 1, .5), (-1, 0, .5))
        self.assertEqual(to_base("right", 270, 0, 1, 0), (1, 0, 0))

    def test_default_front_is_minus_y_on_both_arms(self):
        # Home and work poses sit at Y ~ -345..-385: front is base -Y. Standing
        # in front, forward (towards the robot) is +Y and the operator's right
        # (the robot's left, the left arm's side) is +X, on both arms.
        settings = Settings()
        for arm in ("left", "right"):
            self.assertEqual(to_base("front", settings.front(arm), 0, 1, 0), (0, 1, 0))
            self.assertEqual(to_base("front", settings.front(arm), 1, 0, 0), (1, 0, 0))


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.now = 50
        self.device = ActionDevice()
        import tempfile, pathlib
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "settings.json"
        self.remote = Remote(False, clock=lambda: self.now, factory=lambda *a: self.device,
                             settings=Settings(self.path))
        self.token = self.remote.api("/api/pair", {"code": self.remote.code}, "")["token"]
        self.addCleanup(self.remote.shutdown)
        self.post("open", revision=0, arm="left")
        self.remote.worker.join(1)

    def post(self, endpoint, **body):
        body.setdefault("issued", self.now)
        return self.remote.api("/api/" + endpoint, body, self.token)

    def act(self, endpoint, **body):
        self.post(endpoint, revision=self.remote.revision, **body)
        self.remote.worker.join(1)

    def test_drive_and_step_follow_operator_side(self):
        self.post("settings", view="back")
        self.post("drive", seq=1, revision=self.remote.revision, x=0, y=1, z=0)
        self.assertEqual(self.remote.vector, (0, -1, 0))  # behind the robot
        self.post("release", seq=2)
        self.remote.worker.join(1)
        self.act("step", axis="X", direction=1, mm=1)
        self.device.accept.set()
        self.assertEqual(self.device.moves[-1], ("X", -1, 1))

    def test_speed_reaches_device_and_front_is_persisted(self):
        self.post("settings", speed=85)
        self.assertEqual(self.device.speed, 85.0)
        for bad in ("fast", 2, 500, True):
            with self.assertRaises(RequestError):
                self.post("settings", speed=bad)
        self.post("front", arm="left", deg=90)
        self.assertEqual(Settings(self.path).front("left"), 90)
        self.assertEqual(self.remote.state()["front"]["left"], 90)

    def test_home_camera_and_gripper(self):
        self.act("home")
        self.act("camera")
        self.assertEqual(self.device.calls, ["home", "photo"])
        self.act("gripper", action="close")
        self.assertEqual((self.remote.phase, self.remote.gripper), ("ready", 0))
        with self.assertRaises(RequestError):
            self.post("gripper", revision=self.remote.revision, action="wiggle")

    def test_recover_in_session_and_from_idle(self):
        self.act("recover")
        self.assertEqual((self.remote.phase, self.device.cleared), ("ready", 1))
        self.post("stop")
        self.remote.worker.join(1)
        self.assertEqual(self.remote.phase, "idle")
        self.act("recover", arm="right")
        self.assertEqual((self.remote.phase, self.remote.arm, self.device.cleared), ("ready", "right", 2))

    def test_failed_release_can_be_stopped_and_recovered(self):
        from task_skills.mobile_arm import MobileCleanupError

        def bad_close():
            raise MobileCleanupError("DisableRobot refused")
        self.device.close = bad_close
        self.device.step = lambda *a: (_ for _ in ()).throw(RuntimeError("alarm"))
        self.act("step", axis="X", direction=1, mm=1)
        self.assertEqual(self.remote.phase, "fault")
        self.assertTrue(self.remote.cleanup_failed)
        self.post("stop")  # no longer a silent no-op
        self.assertEqual(self.remote.phase, "idle")
        self.device.close = lambda: None
        self.act("recover", arm="left")
        self.assertEqual(self.remote.phase, "ready")
        self.assertFalse(self.remote.cleanup_failed)

    def test_gripper_fault_keeps_arm_session(self):
        self.device.fail_gripper = True
        self.act("gripper", action="open")
        self.assertEqual((self.remote.phase, self.device.closed), ("ready", 0))
        self.assertIn("serial timeout", self.remote.message)
