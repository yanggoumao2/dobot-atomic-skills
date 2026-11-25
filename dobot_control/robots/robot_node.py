import pickle
import threading
from typing import Any, Dict

import numpy as np
import zmq
import time
from dobot_control.robots.robot import Robot
from scripts.function_util import log_write

DEFAULT_ROBOT_PORT = 6000

import threading


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self.port = port
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()
        from dobot_control.act_session import ACTSession
        self._act_session = ACTSession()

        self.ON = 1
        self.OFF = 0
        self.Enable = self.OFF
        self.Follow = self.OFF
        self.Record = self.OFF

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1)  # Set timeout to 1000 ms
        print("*"*100)
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == 'begin_act_control':
                    result = self._act_session.begin()
                elif method == 'end_act_control':
                    result = self._act_session.end(args.get('token'))
                elif method == 'act_protocol':
                    result = {'protocol': 1, 'owned': self._act_session.token is not None}
                elif method == 'recover_gripper':
                    result = self.recover_gripper(**args)
                elif method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    # log_write(str(self.port) + ": get_joint_state start")
                    result = self._robot.get_joint_state()
                    # log_write(str(self.port) + ": get_joint_state end")
                elif method == "command_joint_state":
                    args = self._act_session.authorize(method, args)
                    tic = time.time()
                    toc = time.time()
                    # log_write(str(self.port) + ": command_joint_state start")
                    result = self._robot.command_joint_state(**args)
                    # log_write(str(self.port) + ": command_joint_state end")
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "set_do_status":
                    args = self._act_session.authorize(method, args)
                    result = self._robot.set_do_status(**args)
                elif method == "get_XYZrxryrz_state":
                    result = self._robot.get_XYZrxryrz_state()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                continue
            except Exception as exc:
                error = f"Robot RPC {method if 'method' in locals() else 'unknown'} failed: {type(exc).__name__}: {exc}"
                print(error, flush=True)
                try:
                    self._socket.send(pickle.dumps({"__robot_rpc_error__": error}))
                except zmq.ZMQError as send_exc:
                    print(f"Failed to return robot RPC error: {send_exc}", flush=True)

    def recover_gripper(self, side, action):
        if self._act_session.token is not None:
            raise RuntimeError('ACT 执行程序正在占用控制权，请先正常退出该程序，再恢复夹爪')
        from dobot_control.gripper_recovery import recover_gripper
        return recover_gripper(self._robot, side, action)

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def _request(self, request):
        self._socket.send(pickle.dumps(request))
        result = pickle.loads(self._socket.recv())
        if isinstance(result, dict) and "__robot_rpc_error__" in result:
            raise RuntimeError(result["__robot_rpc_error__"])
        return result

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        return self._request(request)

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        return self._request(request)

    def command_joint_state(
        self,
        joint_state: np.ndarray,
        arm_flags,
        gripper_flags=None,
    ) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
            arm_flags: Per-arm motion enable flags.
            gripper_flags: Per-gripper enable flags. Defaults to arm_flags.
        """
        request = {
            "method": "command_joint_state",
            "args": {
                "joint_state": joint_state,
                "arm_flags": arm_flags,
                "gripper_flags": gripper_flags,
            },
        }
        return self._request(request)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        return self._request(request)

    def set_do_status(self, which_do: np.ndarray):
        request = {
            "method": "set_do_status",
            "args": {"which_do": which_do},
        }
        return self._request(request)

    def close(self):
        self._socket.close()

    def get_XYZrxryrz_state(self):
        request = {"method": "get_XYZrxryrz_state"}
        return self._request(request)
