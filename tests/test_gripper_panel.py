"""Gripper-panel tests use only fake drivers and temporary device leases."""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from tools.gripper_panel import Endpoint, GripperControl, GripperPanel, load_endpoints
from dobot_control.device_lease import DeviceLease


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.endpoints = {"left": Endpoint("COM3",21,(2048,3052)),
                          "right": Endpoint("COM4",22,(2048,3052))}

    def test_open_close_each_physical_side_at_low_speed(self):
        for side, target, action in ((s,p,a) for s in self.endpoints
                                    for p,a in ((255,"open"),(0,"close"))):
            with self.subTest(side=side, action=action), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/"owner"
                gripper = Mock()
                gripper.get_current_position.side_effect = [128,target]
                def construct(endpoint):
                    self.assertTrue((path/"owner.json").is_file())
                    self.assertEqual(endpoint,self.endpoints[side])
                    return gripper
                control = GripperControl(self.endpoints,construct,
                    lambda: DeviceLease(path=path,purpose="legacy-control"),sleep=lambda _:None)
                self.assertEqual(control.execute(side,action),target)
                gripper.move.assert_called_once_with(target,10,1)
                gripper.disconnect.assert_called_once()
                self.assertFalse(path.exists())
                self.assertTrue(path.with_name("owner.scene-invalid").exists())

    def test_existing_owner_prevents_driver_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"owner"
            with DeviceLease(path):
                factory = Mock()
                control = GripperControl(self.endpoints,factory,lambda:DeviceLease(path))
                with self.assertRaisesRegex(RuntimeError,"设备被占用"):
                    control.execute("left","open")
                factory.assert_not_called()
                self.assertTrue(path.exists())

    def test_write_failure_is_not_retried_and_resources_close(self):
        gripper, lease = Mock(), Mock()
        gripper.move.side_effect = RuntimeError("comm_result=-3001")
        control = GripperControl(self.endpoints,lambda _:gripper,lambda:lease)
        with self.assertRaisesRegex(RuntimeError,"comm_result"):
            control.execute("right","close")
        gripper.move.assert_called_once()
        gripper.get_current_position.assert_not_called()
        gripper.disconnect.assert_called_once()
        lease.acquire.assert_called_once()
        lease.release.assert_called_once()

    def test_timeout_reports_failure_and_releases_resources(self):
        gripper, lease = Mock(), Mock()
        gripper.get_current_position.return_value = 100
        control = GripperControl(self.endpoints,lambda _:gripper,lambda:lease,
                                 clock=Mock(side_effect=[0,11]))
        with self.assertRaises(TimeoutError):
            control.execute("left","open")
        gripper.move.assert_called_once()
        gripper.disconnect.assert_called_once()
        lease.release.assert_called_once()

    def test_read_failure_and_constructor_failure_release_lease(self):
        for stage in ("construct","read"):
            gripper, lease = Mock(), Mock()
            gripper.get_current_position.side_effect = RuntimeError("read failed")
            factory = Mock(side_effect=RuntimeError("connect failed")) if stage=="construct" else Mock(return_value=gripper)
            control = GripperControl(self.endpoints,factory,lambda:lease)
            with self.assertRaises(RuntimeError):
                control.execute("left","open")
            lease.release.assert_called_once()
            if stage=="read":
                gripper.disconnect.assert_called_once()

    def test_invalid_or_concurrent_action_sends_no_command(self):
        factory = Mock()
        control = GripperControl(self.endpoints,factory)
        with self.assertRaises(ValueError):
            control.execute("left","half")
        with control._busy:
            with self.assertRaisesRegex(RuntimeError,"正在执行"):
                control.execute("left","open")
        factory.assert_not_called()

    def test_configuration_uses_physical_sides_and_validates_endpoints(self):
        endpoints = load_endpoints()
        self.assertEqual(endpoints["left"].servo_id,21)
        self.assertEqual(endpoints["right"].servo_id,22)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"settings.ini"
            path.write_text("[GRIPPER_LEFT]\nport=COM4\nid=21\npos=2048,3052\n"
                            "[GRIPPER_RIGHT]\nport=COM4\nid=22\npos=2048,3052\n")
            with self.assertRaises(ValueError):
                load_endpoints(path)


class PanelTests(unittest.TestCase):
    def test_home_buttons_choose_their_own_side_and_stop_without_gripper_motion(self):
        import tkinter as tk
        for side in ("left","right"):
            root=tk.Tk(); root.withdraw()
            entered,finish=threading.Event(),threading.Event()
            control=Mock(endpoints={"left":Endpoint("COM3",21,(2048,3052)),
                                   "right":Endpoint("COM4",22,(2048,3052))})
            def home(chosen):
                entered.set(); finish.wait(2)
                return [0]*6
            control.home.side_effect=home
            panel=GripperPanel(root,control)
            try:
                self.assertEqual(panel.side.get(),"")
                panel.home_buttons[side].invoke()
                self.assertTrue(entered.wait(1))
                self.assertTrue(panel.home_buttons["left"].instate(["disabled"]))
                self.assertTrue(panel.home_buttons["right"].instate(["disabled"]))
                panel.stop_button.invoke()
                control.cancel_home.assert_called_once()
                control.prepare_home.assert_called_once()
                finish.set()
                result=panel.results.get(timeout=2); panel.results.put(result)
                root.after_cancel(panel._poll_id); panel.poll()
                self.assertFalse(panel.busy)
                self.assertTrue(panel.stop_button.instate(["disabled"]))
                self.assertTrue(panel.buttons["open"].instate(["disabled"]))
                self.assertIn("已归位",panel.status.get())
                control.home.assert_called_once_with(side)
                control.execute.assert_not_called()
            finally:
                finish.set()
                root.after_cancel(panel._poll_id); root.destroy()

    def test_window_selection_dispatch_and_busy_close(self):
        import tkinter as tk
        root=tk.Tk(); root.withdraw()
        control=Mock(endpoints={"left":Endpoint("COM3",21,(2048,3052)),
                               "right":Endpoint("COM4",22,(2048,3052))})
        entered, finish=threading.Event(),threading.Event()
        def execute(side,action):
            entered.set(); finish.wait(2)
            return 255
        control.execute.side_effect=execute
        panel=GripperPanel(root,control)
        try:
            control.execute.assert_not_called()
            self.assertTrue(panel.buttons["open"].instate(["disabled"]))
            panel.selectors[0].invoke()
            self.assertEqual(panel.side.get(),"left")
            control.execute.assert_not_called()
            panel.buttons["open"].invoke()
            self.assertTrue(entered.wait(1))
            self.assertTrue(panel.busy)
            self.assertTrue(panel.buttons["close"].instate(["disabled"]))
            panel.close()
            self.assertTrue(root.winfo_exists())
            finish.set()
            # Wait on the worker result without issuing any device commands.
            result=panel.results.get(timeout=2)
            panel.results.put(result)
            root.after_cancel(panel._poll_id)
            panel.poll()
            self.assertFalse(panel.busy)
            self.assertIn("已全开",panel.status.get())
            control.execute.assert_called_once_with("left","open")
        finally:
            finish.set()
            if panel.busy:
                root.destroy()
            else:
                panel.close()


if __name__=="__main__":
    unittest.main()
