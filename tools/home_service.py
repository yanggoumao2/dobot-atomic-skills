"""Explicit operator-requested homing via the existing A service."""
import os

def main():
    from dobot_control.gripper_recovery import require_teleop_stopped
    require_teleop_stopped()
    from dobot_control.env import RobotEnv
    from dobot_control.robots.robot_node import ZMQClientRobot
    from scripts.manipulate_utils import robot_pose_init
    os.environ["DOBOT_ALLOW_SAFE_POSE_INIT"] = "1"
    client = ZMQClientRobot(port=6001, host="127.0.0.1")
    try:
        robot_pose_init(RobotEnv(client))
    finally:
        client.close()

if __name__ == "__main__": main()
