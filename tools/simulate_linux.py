"""One offline run: platform branch simulation plus shared teleop regressions."""
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Existing regressions describe the historical Windows rig. Linux-specific
# cases override this with a temporary /dev profile and sys.platform=linux.
os.environ['DOBOT_CONFIG']=str(ROOT/'scripts/dobot_config/dobot_settings.ini')
patterns=['test_linux_portability.py','test_teleop_regressions.py','test_camera_recording_controls.py',
          'test_recording_pipeline.py','test_act_dataset.py','test_act_task.py','test_gripper_diagnostics.py',
          'test_gripper_panel.py','test_arm_homing.py','test_service_gripper.py','test_pose_init_tolerance.py','test_dashboard_reply.py']
suite=unittest.TestSuite()
for pattern in patterns: suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
print('MODE: simulated Linux configuration/process/launcher branches; hardware mocked; host='+sys.platform,flush=True)
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
