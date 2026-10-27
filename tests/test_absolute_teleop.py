import unittest
from unittest.mock import patch
import numpy as np
from dobot_control.absolute_teleop import AbsoluteTeleopPlanner, CommandInterval
from experiments.run_control import (limit_tcp_descent, calculate_vel_pos,
                                     check_pose_protection, validate_absolute_alignment)


class AbsolutePlannerTests(unittest.TestCase):
    def test_slow_alignment_then_exact_absolute_target_and_normal_speed(self):
        planner = AbsoluteTeleopPlanner()
        actual = np.zeros(14)
        target = np.zeros(14)
        target[0] = 0.3
        command = planner.enable([1, 0], actual, actual)
        for _ in range(120):
            previous = command.copy()
            was_aligned = planner.aligned[0]
            command, _ = planner.step(target, actual, command, 0.02)
            if not was_aligned and not planner.aligned[0]:
                self.assertLessEqual(abs(command[0]-previous[0]), 0.003000001)
            actual = command.copy()
        self.assertTrue(planner.aligned[0])
        self.assertAlmostEqual(command[0], 0.3)
        target[0] = 0.4
        command, _ = planner.step(target, actual, command, 0.02)
        self.assertAlmostEqual(command[0], 0.38)

    def test_command_reaching_target_does_not_fake_actual_alignment(self):
        planner = AbsoluteTeleopPlanner()
        zero = np.zeros(14)
        target = zero.copy(); target[0] = 0.3
        command = planner.enable([1, 0], zero, zero)
        for _ in range(150):
            command, _ = planner.step(target, zero, command, 0.02)
        self.assertAlmostEqual(command[0], 0.3)
        self.assertFalse(planner.aligned[0])

    def test_disabling_and_new_arm_enable_keep_other_arm_reference(self):
        planner = AbsoluteTeleopPlanner()
        zero = np.zeros(14)
        command = planner.enable([1, 0], zero, zero)
        for _ in range(3): command, _ = planner.step(zero, zero, command, .02)
        measured = zero.copy(); measured[7] = 0.7
        command = planner.enable([1, 1], measured, command)
        self.assertTrue(planner.aligned[0]); self.assertFalse(planner.aligned[1])
        self.assertEqual(command[7], .7)
        command = planner.enable([0, 1], measured, command)
        target = np.ones(14)
        new, _ = planner.step(target, measured, command, .02)
        np.testing.assert_array_equal(new[:7], command[:7])

    def test_long_pause_cannot_cause_large_catchup_step(self):
        planner = AbsoluteTeleopPlanner()
        zero = np.zeros(14)
        planner.enable([1, 0], zero, zero)
        command, _ = planner.step(np.ones(14), zero, zero, 8)
        self.assertLessEqual(np.max(command[:6]), .15 * .05)

    def test_invalid_values_fail_closed(self):
        with self.assertRaises(ValueError): AbsoluteTeleopPlanner(align_speed=float("nan"))
        planner = AbsoluteTeleopPlanner()
        with self.assertRaises(ValueError): planner.step(np.full(14,np.nan), np.zeros(14),np.zeros(14),.02)
        with self.assertRaises(ValueError): planner.step(np.zeros(14),np.zeros(14),np.zeros(14),0)

    def test_interval_includes_prior_network_roundtrip_and_does_not_commit_rejections(self):
        clock = CommandInterval(); clock.reset(10)
        clock.committed(10.01)
        self.assertAlmostEqual(clock.elapsed(10.03), .02)
        self.assertAlmostEqual(clock.elapsed(10.04), .03)
        clock.committed(10.04)
        self.assertAlmostEqual(clock.elapsed(10.06), .02)

    def test_tcp_speed_limit_scales_step_without_modifying_other_arm(self):
        def fk(action, previous, dt, robot):
            return {}, {"left_left":np.array([0,0,-(action[0]-previous[0])/dt]),
                        "left_right":np.array([0,0,-(action[0]-previous[0])/dt]),
                        "right_left":np.zeros(3),"right_right":np.zeros(3)}
        previous = np.zeros(14); action = previous.copy(); action[0]=.04; action[7]=.1
        with patch("experiments.run_control.calculate_vel_pos",side_effect=fk):
            limited, details = limit_tcp_descent(action, previous, .02, "Nova 2", [1,0])
        self.assertGreater(limited[0],0)
        self.assertLessEqual(limited[0]/.02, .951)
        self.assertEqual(limited[7], .1)
        self.assertEqual(len(details),1)

    def test_real_fk_descent_is_limited_without_disabling_workspace_guard(self):
        previous = np.array([np.pi/2,0,np.pi/2,0,-np.pi/2,-np.pi/2,0,
                             -np.pi/2,0,-np.pi/2,0,np.pi/2,np.pi/2,0])
        found = False
        for joint in range(8,13):
            for delta in (-.03,.03):
                action=previous.copy(); action[joint]+=delta
                _,vel=calculate_vel_pos(action,previous,.005,"Nova 2")
                if min(vel["right_left"][2],vel["right_right"][2]) >= -1:
                    continue
                found=True
                limited,details=limit_tcp_descent(action,previous,.005,"Nova 2",[0,1])
                pos,vel=calculate_vel_pos(limited,previous,.005,"Nova 2")
                self.assertTrue(details)
                self.assertGreaterEqual(min(vel["right_left"][2],vel["right_right"][2]),-1)
                self.assertFalse(check_pose_protection(pos,vel,[0,1]))
                np.testing.assert_array_equal(limited[:7],previous[:7])
        self.assertTrue(found)

    def test_logged_unsafe_absolute_target_is_rejected_before_motion(self):
        current=np.array([.729556,.018416,1.6751,-.796963,-2.464388,-.887678,1.,
                          -1.570796,0,-1.570796,0,1.570796,1.570796,1.])
        target=np.array([-.154194,.065573,1.898621,-1.676641,-3.444629,-.156524,1.,
                         -4.172077,1.463806,-1.486428,.099126,.314174,2.615108,1.])
        safe, details = validate_absolute_alignment(current,target,"Nova 2",[1,0])
        self.assertFalse(safe)
        self.assertLess(details["fraction"], .02)
        self.assertEqual(details["nova_side"], "left")

    def test_safe_home_to_home_path_is_accepted(self):
        home=np.array([np.pi/2,0,np.pi/2,0,-np.pi/2,-np.pi/2,0,
                       -np.pi/2,0,-np.pi/2,0,np.pi/2,np.pi/2,0])
        safe, details=validate_absolute_alignment(home,home,"Nova 2",[1,1])
        self.assertTrue(safe)
        self.assertGreaterEqual(details["samples"],2)

    def test_absolute_angles_are_not_silently_wrapped(self):
        planner=AbsoluteTeleopPlanner()
        zero=np.zeros(14); target=zero.copy(); target[0]=2*np.pi-.1
        command=planner.enable([1,0],zero,zero)
        command,_=planner.step(target,zero,command,.02)
        self.assertAlmostEqual(command[0], .003)

    def test_inactive_target_and_gripper_do_not_distort_preflight(self):
        home=np.array([np.pi/2,0,np.pi/2,0,-np.pi/2,-np.pi/2,0,
                       -np.pi/2,0,-np.pi/2,0,np.pi/2,np.pi/2,0])
        target=home.copy(); target[7:]=100; target[6]=1
        safe,details=validate_absolute_alignment(home,target,"Nova 2",[1,0])
        self.assertTrue(safe)
        self.assertEqual(details["samples"],2)
        np.testing.assert_array_equal(details["target"],home)

    def test_invalid_preflight_sampling_step_rejected(self):
        for value in (0,-1,float("nan")):
            with self.assertRaises(ValueError):
                validate_absolute_alignment(np.zeros(14),np.zeros(14),"Nova 2",[1,0],value)


if __name__ == "__main__": unittest.main()
