from abc import abstractmethod
from typing import Dict, Protocol

import numpy as np
import time
import threading

class Robot(Protocol):
    """Robot protocol.

    A protocol for a robot that can be controlled.
    """

    @abstractmethod
    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        raise NotImplementedError

    @abstractmethod
    def command_joint_state(
        self,
        joint_state: np.ndarray,
        arm_flags: np.ndarray,
        gripper_flags: np.ndarray = None,
    ) -> None:
        """Command the leader robot to a given state.

        Args:
            arm_flags: Per-arm motion enable flags.
            gripper_flags: Per-gripper enable flags. Defaults to arm_flags.
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        raise NotImplementedError

    def set_do_status(self, which_do):
        """Get the current observations of the robot.

        Args:
            which_do:
        """
        raise NotImplementedError

    def get_DI_state(self, i):
        raise NotImplementedError

    @abstractmethod
    def get_XYZrxryrz_state(self) -> np.ndarray:
        """Get the current X Y Z rx ry rz state of the robot.
        Returns:
            T: The current X Y Z rx ry rz state of the leader robot.
        """
        raise NotImplementedError


class PrintRobot(Robot):
    """A robot that prints the commanded joint state."""

    def __init__(self, num_dofs: int, dont_print: bool = False):
        self._num_dofs = num_dofs
        # self._joint_state = np.zeros(num_dofs)
        self._joint_state = np.deg2rad(
            [90, 0, 90, 0, -90, 0, 90]
        )  # DOBOT

        self._dont_print = dont_print

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(
        self,
        joint_state: np.ndarray,
        command_arm: bool = True,
        command_gripper: bool = True,
    ) -> None:
        assert len(joint_state) == (self._num_dofs), (
            f"Expected joint state of length {self._num_dofs}, "
            f"got {len(joint_state)}."
        )
        if command_arm:
            self._joint_state[:6] = joint_state[:6]
        if command_gripper:
            self._joint_state[-1] = joint_state[-1]
        if not self._dont_print:
            print(self._joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_state = self.get_joint_state()
        pos_quat = np.zeros(7)
        return {
            "joint_positions": joint_state,
            "joint_velocities": joint_state,
            "ee_pos_quat": pos_quat,
            "gripper_position": np.array(0),
        }


class BimanualRobot(Robot):
    @staticmethod
    def _require_ready(robot, side):
        if robot.robot_is_err:
            failure = getattr(robot, "_status_failure", None)
            detail = failure or "status monitor reported an error"
            raise RuntimeError(f"{side} robot error: {detail}")

    def __init__(self, robot_l: Robot, robot_r: Robot):
        self._robot_l = robot_l
        self._robot_r = robot_r

        self.frequency_ = 1/0.015
        # Set delta time to be used by receiveCallback
        self.delta_time_ = 1 / self.frequency_
        self._require_ready(self._robot_l, "left")
        self._require_ready(self._robot_r, "right")

    def num_dofs(self) -> int:
        return self._robot_l.num_dofs() + self._robot_r.num_dofs()

    def get_joint_state(self) -> np.ndarray:
        self._require_ready(self._robot_l, "left")
        self._require_ready(self._robot_r, "right")
        return np.concatenate(
            (self._robot_l.get_joint_state(), self._robot_r.get_joint_state())
        )

    def command_joint_state(
        self,
        joint_state: np.ndarray,
        arm_flags,
        gripper_flags=None,
    ) -> None:
        # Arm channels may be crossed while grippers remain physically
        # same-side. Route the two permissions independently so an inactive
        # Nova arm can still receive its own gripper command without ServoJ.
        self._require_ready(self._robot_l, "left")
        self._require_ready(self._robot_r, "right")

        left_dofs = self._robot_l.num_dofs()
        left_cmd = joint_state[:left_dofs]
        right_cmd = joint_state[left_dofs:]

        arm_flags = np.asarray(arm_flags)
        if arm_flags.shape != (2,):
            raise ValueError(f"Expected two arm flags, got shape {arm_flags.shape}")
        if gripper_flags is None:
            gripper_flags = arm_flags
        gripper_flags = np.asarray(gripper_flags)
        if gripper_flags.shape != (2,):
            raise ValueError(
                f"Expected two gripper flags, got shape {gripper_flags.shape}"
            )

        commands = (
            (
                self._robot_l,
                left_cmd,
                bool(arm_flags[0]),
                bool(gripper_flags[0]),
            ),
            (
                self._robot_r,
                right_cmd,
                bool(arm_flags[1]),
                bool(gripper_flags[1]),
            ),
        )
        errors = [None, None]

        def _send(index, robot, command, command_arm, command_gripper):
            try:
                robot.command_joint_state(
                    command,
                    command_arm=command_arm,
                    command_gripper=command_gripper,
                )
            except Exception as exc:  # noqa: BLE001
                errors[index] = exc

        threads = []
        for index, command_args in enumerate(commands):
            _, _, command_arm, command_gripper = command_args
            if not (command_arm or command_gripper):
                continue
            thread = threading.Thread(
                target=_send,
                args=(index, *command_args),
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        for error in errors:
            if error is not None:
                raise error
        return 1

    def get_observations(self) -> Dict[str, np.ndarray]:
        self._require_ready(self._robot_l, "left")
        self._require_ready(self._robot_r, "right")
        observations = [None, None]
        errors = [None, None]

        def _read(index, robot):
            try:
                observations[index] = robot.get_observations()
            except Exception as exc:  # noqa: BLE001
                errors[index] = exc

        threads = [
            threading.Thread(target=_read, args=(0, self._robot_l), daemon=True),
            threading.Thread(target=_read, args=(1, self._robot_r), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for error in errors:
            if error is not None:
                raise error

        l_obs, r_obs = observations
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError()

        return return_obs

    def set_do_status(self, which_do):
        self._require_ready(self._robot_l, "left")
        self._require_ready(self._robot_r, "right")
        self._robot_l.set_do_status(which_do)

    def get_XYZrxryrz_state(self) -> np.ndarray:
        self._require_ready(self._robot_l, "left")
        self._require_ready(self._robot_r, "right")
        return np.concatenate(
            (self._robot_l.get_XYZrxryrz_state(), self._robot_r.get_XYZrxryrz_state())
        )

def main():
    pass


if __name__ == "__main__":
    main()
