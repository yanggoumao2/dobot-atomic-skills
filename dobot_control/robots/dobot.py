from typing import Dict
import numpy as np
import time
from dobot_control.robots.robot import Robot
import struct
import sys
from scripts.manipulate_utils import load_ini_data_hands, load_ini_data_gripper
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, \
    servo_action_check, load_ini_data_hands, set_light, load_ini_data_camera
from threading import Event, Lock, Thread


class DobotRobot(Robot):
    _EXPECTED_STATUS_TEST_VALUE = 0x123456789ABCDEF
    _MAX_CONSECUTIVE_INVALID_STATUS_FRAMES = 3

    def __init__(self, robot_ip: str = "192.168.5.1", no_gripper: bool = False, robot_number: int = 1):
        from dobot_control.robots import dobot_api
        self.robot_number = robot_number
        self.frequency_ = 1 / 0.015
        # Set delta time to be used by receiveCallback
        self.delta_time_ = 1 / self.frequency_
        [print("in dobot robot") for _ in range(4)]
        self.robot = dobot_api.DobotApiMove(robot_ip, 30003)  # 运动指令的端�?
        # try:
        #     self.robot = dobot_api.DobotApiMove(robot_ip, 30003)  # 运动指令的端�?
        # except Exception as e:
        #     print(e)
        #     print("Please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
        #     sys.exit()
        self.robot_ip = robot_ip
        self.r_inter = dobot_api.DobotApiDashboard(robot_ip, 29999)  # 获取信息指令的端�?
        self.r_inter.EnableRobot()  # 上使能机械臂，发送指令前必须执行�?
        # Conservative limits for real-robot bring-up.  Pose initialization
        # additionally applies its own bounded step and period.
        self.r_inter.SpeedFactor(5)  # global speed limit
        self.r_inter.AccJ(5)  # global joint acceleration limit
        self.r_inter.SpeedJ(5)  # global joint speed limit
        self.r_inter.SetTool(1, 0, 0, 197, 0, 0, 0)  # 设置夹爪工具坐标系
        self.r_inter.Tool(1)  # 设置当前工具坐标系，使得Getpose()获取坐标时为此工具下

        self.com_list = {"192.168.5.1": "GRIPPER_LEFT", "192.168.5.2": "GRIPPER_RIGHT"}  # left, right
        _, gripper_dict = load_ini_data_gripper()
        self.gripper_list = gripper_dict[self.com_list[robot_ip]].pos
        self.gripper_id_name = gripper_dict[self.com_list[robot_ip]].id_name

        if not no_gripper:
            from dobot_control.gripper.dobot_gripper import DobotGripper
            self.gripper = DobotGripper(port=gripper_dict[self.com_list[robot_ip]].port,
                                        id_name=self.gripper_id_name,
                                        servo_pos=self.gripper_list)
            print("gripper connected")
            channel = self.com_list[robot_ip]
            print(f"[GRIPPER START] {channel} ip={robot_ip} "
                  f"port={gripper_dict[channel].port} id={self.gripper_id_name}", flush=True)
            self._initialize_gripper()
            print(f"[GRIPPER DONE] {channel}; next channel may now start", flush=True)

        self._free_drive = False  # 拖拽模式
        self.r_inter.StopDrag()  # 关闭拖拽
        self._use_gripper = not no_gripper

        self.robot_status = dobot_api.DobotApiStatus(robot_ip, 30004)
        self._stop_thread = Event()
        self._lock = Lock()
        self.robot_is_err = False
        self._status_failure = None
        self._invalid_error_status_seen = set()
        self._consecutive_invalid_status_frames = 0
        # Cache last commanded gripper position so a static grip does not
        # spam the RS-485 bus on every ServoJ cycle. That write was serial
        # with the ServoJ RPC and dominated the per-arm follow latency.
        self._last_gripper_cmd = None
        self._start_reading_thread()

    def _initialize_gripper(self):
        # launch_nodes constructs LEFT then RIGHT. Wait for each endpoint so
        # the next gripper cannot start while this one's cycle is unfinished.
        for target in (0, 255):
            print(f"[GRIPPER MOVE] target={target} speed=10", flush=True)
            self.gripper.move(target, 10, 1)
            deadline = time.monotonic() + 10.0
            while True:
                position = self.gripper.get_current_position()
                if abs(position - target) <= 5:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Startup gripper did not reach {target} within 10s "
                        f"(position={position}); subsequent startup is aborted"
                    )
                time.sleep(0.1)
            time.sleep(0.5)  # visible pause at each endpoint

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self.get_robot_err)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def get_robot_err(self):
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            try:
                feedback = self.robot_status.get_feedback()
                test_value = int(feedback['test_value'])
                robot_mode = int(feedback['robot_mode'])
                error_status = int(feedback['error_status']) & 0xFF
            except (OSError, TimeoutError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                failure = f"unexpected {type(exc).__name__}: {exc}"
            else:
                if test_value != self._EXPECTED_STATUS_TEST_VALUE:
                    self._consecutive_invalid_status_frames += 1
                    invalid_count = self._consecutive_invalid_status_frames
                    frame_bytes = feedback.tobytes()
                    invalid_details = (
                        f"test_value=0x{test_value:X} robot_mode={robot_mode} "
                        f"error_status={error_status} "
                        f"prefix_0_63={frame_bytes[:64].hex().upper()} "
                        f"status_bytes_1008_1037={frame_bytes[1008:1038].hex().upper()}"
                    )
                    if invalid_count < self._MAX_CONSECUTIVE_INVALID_STATUS_FRAMES:
                        print(
                            f"[{self.robot_ip}] ignored invalid feedback frame "
                            f"{invalid_count}/{self._MAX_CONSECUTIVE_INVALID_STATUS_FRAMES}: "
                            f"{invalid_details}",
                            flush=True,
                        )
                        continue
                    failure = (
                        f"{invalid_count} consecutive invalid feedback frames: "
                        f"{invalid_details}"
                    )
                else:
                    self._consecutive_invalid_status_frames = 0
                    if robot_mode == 9:
                        try:
                            dashboard_error = self.r_inter.GetErrorID()
                        except Exception as exc:
                            dashboard_error = (
                                f"dashboard query failed: {type(exc).__name__}: {exc}"
                            )
                        failure = (
                            f"controller alarm robot_mode={robot_mode} "
                            f"error_status={error_status} raw=0x{error_status:02X} "
                            f"test_value=0x{test_value:X} "
                            f"dashboard_error={dashboard_error}"
                        )
                    else:
                        if error_status != 0 and error_status not in self._invalid_error_status_seen:
                            self._invalid_error_status_seen.add(error_status)
                            status_bytes = feedback.tobytes()[1008:1038].hex().upper()
                            status_kind = (
                                "nonzero" if error_status == 1 else "protocol-invalid"
                            )
                            try:
                                dashboard_error = self.r_inter.GetErrorID()
                            except Exception as exc:
                                dashboard_error = (
                                    f"dashboard query failed: {type(exc).__name__}: {exc}"
                                )
                            print(
                                f"[{self.robot_ip}] status telemetry warning: "
                                f"{status_kind} error_status={error_status} "
                                f"raw=0x{error_status:02X} "
                                f"ignored as an alarm because robot_mode={robot_mode}; "
                                f"test_value=0x{test_value:X} "
                                f"status_bytes_1008_1037={status_bytes} "
                                f"dashboard_error={dashboard_error}",
                                flush=True,
                            )
                        continue

            with self._lock:
                self.robot_is_err = True
                self._status_failure = failure
            print(f"[{self.robot_ip}] status monitor stopped: {failure}", flush=True)
            return

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 7

    def _require_ready(self) -> None:
        if self.robot_is_err:
            failure = getattr(self, "_status_failure", None)
            detail = failure or "status monitor reported an error"
            raise RuntimeError(f"{self.robot_ip}: {detail}")

    def _get_gripper_pos(self) -> float:
        # import time
        self._require_ready()
        gripper_pos = self.gripper.get_current_position()
        # gripper_pos = self.get_current_position()
        # print("gripper_pos:",gripper_pos)
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return np.abs(1 - gripper_pos / 255)
        # return np.abs(gripper_pos / 255)

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        # log_write(str(self.robot_ip) + ": getpose_start")
        robot_joints_angle = list(map(float, self.r_inter.GetAngle().split("{")[1].split("}")[0].split(",")))  # 单位：度�?
        robot_joints = [np.deg2rad(robot_joint) for robot_joint in robot_joints_angle]  # 单位：弧�?
        if self._use_gripper:
            gripper_pos = [1.0]
            pos = np.append(robot_joints, gripper_pos)
        else:
            gripper_pos = [1.0]
            pos = np.append(robot_joints, gripper_pos)
        return pos

    def get_XYZrxryrz_state(self) -> np.ndarray:
        """Get the current X Y Z rx ry rz state of the robot.
        Returns:
            T: The current X Y Z rx ry rz state of the robot.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        pos = list(map(float, self.r_inter.GetPose().split("{")[1].split("}")[0].split(",")))  # 单位：度数
        return pos

    def command_joint_state(
        self,
        joint_state: np.ndarray,
        command_arm: bool = True,
        command_gripper: bool = True,
    ) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        self._require_ready()
        if command_arm:
            robot_joints_angle = joint_state[:6]  # 单位：弧�?
            robot_joints = [np.rad2deg(robot_joint) for robot_joint in robot_joints_angle]
            self.robot.ServoJ(robot_joints[0],
                              robot_joints[1],
                              robot_joints[2],
                              robot_joints[3],
                              robot_joints[4],
                              robot_joints[5],
                              0.03)
        if self._use_gripper and command_gripper:
            gripper_pos = int(np.clip(float(joint_state[-1]), 0.0, 1.0) * 255)
            # Only write to the gripper when its normalized target actually
            # moves. A serial RS-485 write on every cycle added tens of ms
            # inside the same RPC that carries ServoJ, which is the main
            # per-arm follow-lag contributor when the gripper is static.
            if self._last_gripper_cmd is None or gripper_pos != self._last_gripper_cmd:
                self.gripper.move(gripper_pos, 100, 1)
                self._last_gripper_cmd = gripper_pos
        return 1

    def moveJ(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        robot_joints_angle = joint_state[:6]  # ��λ������
        robot_joints = [np.rad2deg(robot_joint) for robot_joint in robot_joints_angle]  # ��λ���Ƕ�
        tic = time.time()
        print(robot_joints)
        self.robot.JointMovJ(robot_joints[0],
                             robot_joints[1],
                             robot_joints[2],
                             robot_joints[3],
                             robot_joints[4],
                             robot_joints[5])
        toc = time.time()
        if self._use_gripper:
            tic = time.time()
            gripper_pos = int(joint_state[-1] * 255)
            # print("gripper_pos:", gripper_pos)
            self.gripper.move(gripper_pos, 100, 1)
            # log_write(str(self.robot_ip) + ": gripper_move")
            toc = time.time()

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        if enable and not self._free_drive:
            self._free_drive = True
            self.r_inter.StartDrag()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.r_inter.StopDrag()

    def get_observations(self) -> Dict[str, np.ndarray]:
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }

    def get_obs(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }

    def get_DI_state(self, index):
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        return int(self.r_inter.DI(index).split('{')[1].split('}')[0])

    def set_do_status(self, which_do):
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        self.r_inter.DO(which_do[0], which_do[1])
        return 1


def main():
    dobot = DobotRobot("192.168.5.2", no_gripper=False)
    dobot.set_do_status([1, 0])
    dobot.set_do_status([2, 0])
    dobot.set_do_status([3, 0])
    # while 1:
    #     dobot.get_joint_state()
    # dobot = DobotRobot("192.168.5.2", no_gripper=False)
    # set_light(dobot, "red", 0)


if __name__ == "__main__":
    main()
