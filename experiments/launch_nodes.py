import sys
import os
# 获取根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from dataclasses import dataclass
import tyro
from dobot_control.robots.dobot import DobotRobot
from dobot_control.robots.robot import BimanualRobot, PrintRobot
from dobot_control.robots.robot_node import ZMQServerRobot
from dobot_control.layout import TELEOP_NOVA_SIDES, physical_nova_ip


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"


def launch_robot_server(args: Args):
    from dobot_control.config import validate_hardware_config
    validate_hardware_config()
    from dobot_control.device_lease import DeviceLease

    # The legacy ZMQ server has no verified shutdown protocol. Keep ownership
    # on exit until an onsite operator verifies the rig and clears the lock.
    DeviceLease(purpose="legacy-control").acquire()
    _launch_owned_robot_server(args)


def _launch_owned_robot_server(args: Args):
    port = args.robot_port
    # Preserve calibrated vector order: physical RIGHT (.1), LEFT (.2).
    first, second = (
        DobotRobot(robot_ip=physical_nova_ip(side), robot_number=2)
        for side in TELEOP_NOVA_SIDES
    )
    robot = BimanualRobot(first, second)
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"Starting robot server on port {port}")
    server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
