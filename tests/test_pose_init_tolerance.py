"""Startup pose tolerance tests with no hardware or motion."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts.manipulate_utils import robot_pose_init, _safe_pose_segment


class PoseInitToleranceTests(unittest.TestCase):
    def home(self):
        return np.deg2rad([90,0,90,0,-90,-90,0,-90,0,-90,0,90,90,0])

    def test_small_residual_and_inclusive_boundary_send_no_motion(self):
        for index in (0,1,2,3,4,5,7,8,9,10,11,12):
            for delta in (.004422, .1, -.1):
                with self.subTest(index=index,delta=delta):
                    q=self.home();q[index]+=np.deg2rad(delta)
                    q[[6,13]]=1.  # gripper differences never trigger homing
                    env=Mock();env.get_obs.return_value={'joint_positions':q}
                    with patch.dict('os.environ',{'DOBOT_ALLOW_SAFE_POSE_INIT':'0'}), contextlib.redirect_stdout(io.StringIO()):
                        robot_pose_init(env)
                    env.step.assert_not_called()

    def test_above_boundary_still_requires_explicit_opt_in(self):
        for index in (0,7):
            q=self.home();q[index]+=np.deg2rad(.1001)
            env=Mock();env.get_obs.return_value={'joint_positions':q}
            with patch.dict('os.environ',{'DOBOT_ALLOW_SAFE_POSE_INIT':'0'}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'Safe stop before pose motion'):
                    robot_pose_init(env)
            env.step.assert_not_called()

    def test_authorized_segment_also_skips_small_residual(self):
        q=self.home();q[7]+=np.deg2rad(.1)
        env=Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            _safe_pose_segment(env,q,self.home(),'test segment')
        env.step.assert_not_called()

    def test_log_displays_small_residual_instead_of_zero(self):
        q=self.home();q[10]+=np.deg2rad(.004422)
        env=Mock();env.get_obs.return_value={'joint_positions':q};output=io.StringIO()
        with contextlib.redirect_stdout(output):robot_pose_init(env)
        self.assertIn('max_arm_delta_deg = 0.004422',output.getvalue())
        self.assertIn('match_tolerance_deg = 0.1',output.getvalue())
