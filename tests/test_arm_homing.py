"""Single-arm homing tests; all TCP and motion boundaries are fake."""
import math
import threading
import unittest
from unittest.mock import Mock

from tools.arm_homing import ArmHoming, StopUnconfirmed, values, validate_path
from tools.gripper_panel import GripperControl
from dobot_control.layout import physical_nova_home_degrees, physical_nova_ip


class ArmHomingTests(unittest.TestCase):
    def runner(self, side, delta=1):
        target = list(physical_nova_home_degrees(side))
        actual = target[:]; actual[0] += delta
        dashboard, motion = Mock(), Mock()
        dashboard.RobotMode.return_value = "0,{5},RobotMode();"
        dashboard.EnableRobot.return_value = "0,{},EnableRobot();"
        dashboard.ResetRobot.return_value = "0,{},ResetRobot();"
        dashboard.GetAngle.side_effect = lambda: "0,{"+",".join(map(str,actual))+"},GetAngle();"
        commands, pauses = [], []
        def send(*angles, t):
            commands.append(list(angles))
            actual[:] = angles
        motion.ServoJ.side_effect = send
        df, mf, check = Mock(return_value=dashboard), Mock(return_value=motion), Mock()
        runner = ArmHoming(df,mf,check,sleep=pauses.append)
        return runner,dashboard,motion,df,mf,check,commands,pauses,target

    def test_each_side_only_connects_its_own_ip_and_obeys_low_speed(self):
        for side in ("left","right"):
            with self.subTest(side=side):
                runner,d,m,df,mf,check,commands,pauses,target = self.runner(side)
                result = runner.run(side,threading.Event())
                df.assert_called_once_with(physical_nova_ip(side))
                mf.assert_called_once_with(physical_nova_ip(side))
                self.assertEqual(result,target)
                previous=target[:]; previous[0]+=1
                for command in commands:
                    self.assertLessEqual(max(abs(a-b) for a,b in zip(previous,command)),math.degrees(.15*.05)+1e-9)
                    previous=command
                self.assertEqual(commands[-1],target)
                self.assertTrue(all(t >= .05 for t in pauses))
                d.EnableRobot.assert_not_called()
                d.ResetRobot.assert_not_called()
                d.close.assert_called_once(); m.close.assert_called_once()

    def test_already_home_does_not_enable_or_open_motion_port(self):
        runner,d,m,df,mf,*_ = self.runner("left",0)
        runner.run("left",threading.Event())
        mf.assert_not_called(); d.EnableRobot.assert_not_called()

    def test_preflight_rejection_never_enables_or_sends_motion(self):
        runner,d,m,df,mf,check,*_ = self.runner("right")
        check.side_effect=RuntimeError("unsafe path")
        with self.assertRaisesRegex(RuntimeError,"unsafe path"):
            runner.run("right",threading.Event())
        mf.assert_not_called(); d.EnableRobot.assert_not_called()

    def test_alarm_and_invalid_replies_fail_without_motion(self):
        for reply in ("0,{9},RobotMode();", "", "-1,{},RobotMode();", "0,{nan},RobotMode();"):
            runner,d,m,df,mf,*_ = self.runner("right")
            d.RobotMode.return_value=reply
            with self.assertRaises(RuntimeError):
                runner.run("right",threading.Event())
            mf.assert_not_called(); d.EnableRobot.assert_not_called()

    def test_disabled_controller_enables_selected_side_before_movement(self):
        runner,d,m,*_ = self.runner("left")
        replies=iter(["0,{4},RobotMode();"])
        d.RobotMode.side_effect=lambda: next(replies,"0,{5},RobotMode();")
        runner.run("left",threading.Event())
        d.EnableRobot.assert_called_once()

    def test_stop_request_after_first_command_stops_selected_arm(self):
        runner,d,m,df,mf,check,commands,pauses,target = self.runner("left")
        cancel=threading.Event()
        runner.sleep=lambda _: cancel.set()
        with self.assertRaisesRegex(RuntimeError,"已停止"):
            runner.run("left",cancel)
        self.assertEqual(len(commands),1)
        d.ResetRobot.assert_called_once()

    def test_cancel_before_first_send_does_not_move(self):
        runner,d,m,df,mf,*_ = self.runner("right")
        cancel=threading.Event(); cancel.set()
        with self.assertRaisesRegex(RuntimeError,"取消"):
            runner.run("right",cancel)
        mf.assert_not_called()

    def test_transport_failure_is_not_retried_and_attempts_stop(self):
        runner,d,m,*_ = self.runner("right")
        m.ServoJ.side_effect=OSError("connection lost")
        with self.assertRaises(OSError):
            runner.run("right",threading.Event())
        m.ServoJ.assert_called_once(); d.ResetRobot.assert_called_once()

    def test_stop_failure_is_explicit_and_lease_is_retained(self):
        runner,d,m,*_ = self.runner("right")
        m.ServoJ.side_effect=OSError("lost")
        d.ResetRobot.return_value=""
        lease=Mock()
        control=GripperControl({},lease_factory=lambda:lease,homing_factory=lambda:runner)
        with self.assertRaises(StopUnconfirmed):
            control.home("right")
        lease.release.assert_not_called()

    def test_shared_busy_lock_and_lease_prevent_overlapping_operations(self):
        runner=Mock(); runner.run.return_value=[0]*6
        lease=Mock()
        control=GripperControl({},lease_factory=lambda:lease,homing_factory=lambda:runner)
        with control._busy:
            with self.assertRaises(RuntimeError): control.home("left")
        lease.acquire.assert_not_called()
        self.assertEqual(control.home("left"),[0]*6)
        lease.acquire.assert_called_once(); lease.release.assert_called_once()

    def test_real_fk_preflight_uses_physical_side_home(self):
        for side in ("left","right"):
            home=list(physical_nova_home_degrees(side))
            validate_path(side,home,home)

    def test_arrival_timeout_does_not_claim_success(self):
        runner,d,m,*_ = self.runner("left",.3)
        m.ServoJ.side_effect=lambda *args,**kwargs: None
        ticks=iter([0,11])
        runner.clock=lambda:next(ticks)
        with self.assertRaises(TimeoutError): runner.run("left",threading.Event())
        d.ResetRobot.assert_called_once()


if __name__=="__main__":
    unittest.main()
