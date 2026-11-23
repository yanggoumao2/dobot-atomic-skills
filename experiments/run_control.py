# $env:DOBOT_ALLOW_SAFE_POSE_INIT="1"
# python experiments\run_control.py --robot-port 6500 --show-img
import sys
import os
import atexit
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
from dobot_control.agents.agent import BimanualAgent
from scripts.format_obs import save_frame
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.function_util import mismatch_data_write, wait_period, log_write, mk_dir
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, master_to_nova_action, master_to_nova_flags, master_to_nova_gripper_flags, apply_same_side_gripper_targets, load_ini_data_hands, set_light, load_ini_data_camera
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.async_writer import BoundedAsyncWriter
from dobot_control.teleop_safety import (
    MasterRunawayGuard,
    check_pose_protection,
    consume_servo_warmup,
)
import datetime
from pathlib import Path
import requests

@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = False
    save_data_path = str(Path(__file__).parent.parent.parent)+"/datasets/"
    project_name = "dataset_package_test"


# Thread button: [lock or nor, servo or not, record or not]
# 0: lock, 1: unlock
# 0: stop servo, 1: servo
# 0: stop recording, 1: recording
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
dt_time = np.array([20240507161455])
using_sensor_protection = False
is_falling = np.array([0])
# Incremented by the control loop after a servo fault so the button thread
# resets its private edge counters instead of interpreting the next press as
# the opposite state.
_button_state_reset_generation = np.array([0])
_runtime_log_lock = threading.Lock()
_runtime_log_path = os.path.join(BASE_DIR, "scripts", "logs", "teleop_runtime.log")
_runtime_trace_enabled = os.environ.get("DOBOT_TELEOP_TRACE", "0") == "1"
_runtime_trace_events = {"SERVO_COMMAND", "ENV_STEP_BEGIN", "ENV_STEP_OK"}


def runtime_log(event, **fields):
    if event in _runtime_trace_events and not _runtime_trace_enabled:
        return
    timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
    parts = [f"{timestamp} [{event}]"]
    for key, value in fields.items():
        if isinstance(value, np.ndarray):
            value = np.array2string(value, precision=6, separator=",", max_line_width=10000)
        parts.append(f"{key}={value}")
    line = " ".join(parts)
    with _runtime_log_lock:
        os.makedirs(os.path.dirname(_runtime_log_path), exist_ok=True)
        with open(_runtime_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
            log_file.flush()
    print(line, flush=True)


@dataclass(frozen=True)
class RecordingSample:
    episode_dir: str
    index: int
    images: tuple
    observations: dict
    action: np.ndarray


def write_recording_sample(sample):
    episode_dir = Path(sample.episode_dir)
    image_dirs = (
        episode_dir / "topImg",
        episode_dir / "leftImg",
        episode_dir / "rightImg",
    )
    obs_dir = episode_dir / "observation"
    for folder in (*image_dirs, obs_dir):
        folder.mkdir(parents=True, exist_ok=True)
    for folder, jpeg_image in zip(image_dirs, sample.images):
        output_path = folder / f"{sample.index}.jpg"
        output_path.write_bytes(jpeg_image)
    save_frame(str(obs_dir) + os.sep, sample.index, sample.observations, sample.action)


def close_record_writer(record_writer):
    try:
        record_writer.close(timeout=10.0)
    except Exception as exc:
        print(f"[RECORD] writer shutdown failed: {type(exc).__name__}: {exc}", flush=True)


def button_monitor_realtime(agent):
    # servo
    last_keys_status = np.array(([0, 0, 0], [0, 0, 0]))
    start_press_status = np.array(([0, 0], [0, 0]))  # start press
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))
    reset_generation = int(_button_state_reset_generation[0])

    while not is_falling[0]:
        # Keep button-thread edge counters synchronized with control-loop
        # fault recovery. Otherwise a forced servo-off leaves odd/even counts
        # stale and the next press appears to do nothing.
        current_generation = int(_button_state_reset_generation[0])
        if current_generation != reset_generation:
            keys_press_count[:, 1] = 0
            start_press_status[:, 0] = 0
            reset_generation = current_generation
            runtime_log("BUTTON_STATE_RESET", generation=reset_generation)
        # time.sleep(0.010)
        now_keys = agent.get_keys()
        dev_keys = now_keys - last_keys_status
        # button a
        for i in range(2):
            if dev_keys[i, 0] == -1:  # button a: start
                tic = time.time()
                start_press_status[i, 0] = 1
            if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # button a: end
                start_press_status[i, 0] = 0
                toc = time.time()
                if toc-tic < 0.5:
                    keys_press_count[i, 0] += 1
                    if keys_press_count[i, 0] % 2 == 1:
                        what_to_do[i, 0] = 1
                        runtime_log("BUTTON_A_SHORT", hand=i, action="unlock", state=what_to_do.copy())
                    else:
                        what_to_do[i, 0] = 0
                        runtime_log("BUTTON_A_SHORT", hand=i, action="lock", state=what_to_do.copy())

                elif toc-tic > 1:
                    keys_press_count[i, 1] += 1
                    if keys_press_count[i, 1] % 2 == 1:
                        what_to_do[i, 1] = 1
                        runtime_log("BUTTON_A_LONG", hand=i, action="servo_on", state=what_to_do.copy())
                    else:
                        what_to_do[i, 1] = 0
                        runtime_log("BUTTON_A_LONG", hand=i, action="servo_off", state=what_to_do.copy())

        # button B
        # more than one start servo
        for i in range(2):
            if dev_keys[i, 1] == -1:  # B button pressed
                start_press_status[i, 1] = 1
            if dev_keys[i, 1] == 1:
                start_press_status[i, 1] = 0
                if keys_press_count[0, 2] % 2 == 1:
                    if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                        what_to_do[0, 2] = 1
                        # log_write(__file__, "ButtonB: [" + str(i) + "] recording")
                        # new recording
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S%f")[:-3])
                        keys_press_count[0, 2] += 1
                else:
                    what_to_do[0, 2] = 0
                    keys_press_count[0, 2] += 1
                    # log_write(__file__, "ButtonB: [" + str(i) + "] stop recording")

        # status fall
        if using_sensor_protection:
            for i in range(2):
                if now_keys[i, 2] and what_to_do[i, 0]:  # button a: lock
                    agent.set_torque(2, True)
                    is_falling[0] = 1

        last_keys_status = now_keys


# Thread: camera
img_list = np.array([np.zeros((480, 640, 3)), np.zeros((480, 640, 3)), np.zeros((480, 640, 3))])
jpeg_list = [b"", b"", b""]
CAMERA_STARTUP_TIMEOUT_S = 30
CAMERA_MAX_CONSECUTIVE_FAILURES = 3


def run_thread_cam(rs_cam, which_cam, ready_event, failure_event, stop_event):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            if which_cam == 1 or which_cam == 2:
                image_cam, _ = rs_cam.read()
                image_cam = image_cam[:, :, ::-1]
            elif which_cam == 0:
                image_top_src, _ = rs_cam.read()
                image_top_src = image_top_src[150:420, 220:480, ::-1]
                image_cam = cv2.resize(image_top_src, (640, 480))

            encoded, jpeg_image = cv2.imencode('.jpg', image_cam, encode_param)
            if not encoded:
                raise RuntimeError(f"camera {which_cam} JPEG encoding failed")
            img_list[which_cam] = image_cam
            jpeg_list[which_cam] = jpeg_image.tobytes()
            ready_event.set()
            consecutive_failures = 0
        except RuntimeError as exc:
            consecutive_failures += 1
            print(
                f"camera {which_cam} frame timeout "
                f"{consecutive_failures}/{CAMERA_MAX_CONSECUTIVE_FAILURES}: {exc}",
                flush=True,
            )
            if consecutive_failures >= CAMERA_MAX_CONSECUTIVE_FAILURES:
                failure_event.set()
                stop_event.set()
                return
            time.sleep(0.1)



def dh_transformation_matrix(theta, d, a, alpha):
    """
    Create the DH transformation matrix
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

def claw_width(coef):
    """
    Calculate the claw width
    """
    claw_servo = 2.3818 - coef * 1.5401
    cos_claw_servo = np.cos(claw_servo)
    claw_wid = 0.03 * cos_claw_servo + 0.5 * np.sqrt(0.0036 * cos_claw_servo ** 2 + 0.0028)
    return claw_wid

def forward_kinematics(q0, q1, q2, q3, q4, q5, y, r_type):
    """
    Compute the forward kinematics
    """
    if r_type == "Nova 2":
        dh_params = [
            (q0, 0.2234, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.280, 0),
            (q2, 0, -0.225, 0),
            (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0)
        ]
    if r_type == "Nova 5":
        dh_params = [
            (q0, 0.240, 0, np.pi / 2),
            (q1 - np.pi / 2, 0, -0.400, 0),
            (q2, 0, -0.330, 0),
            (q3 - np.pi / 2, 0.135, 0, np.pi / 2),
            (q4, 0.120, 0, -np.pi / 2),
            (q5, 0.088, 0, 0)
        ]

    t = np.eye(4)
    for params in dh_params:
        t = np.dot(t, dh_transformation_matrix(*params))
    t_tool = np.eye(4)
    t_tool[:3, 3] = np.array([0, y, 0.2])
    t_final = np.dot(t, t_tool)
    pos = t_final[:3, 3]
    return pos


def calculate_vel_pos(action, last_action, total_time, r_type):
    """
    Calculate the velocity for forward kinematics
    """
    claw_left = claw_width(action[6])
    claw_right = claw_width(action[13])

    positions = {}
    vel = {}

    for side in ['left', 'right']:
        for paw in ['left', 'right']:
            coef = 1 if paw == 'left' else -1
            claw = claw_left if side == 'left' else claw_right
            claw *= coef

            current_fk = forward_kinematics(*action[0:6] if side == 'left' else action[7:13], claw, r_type)
            last_fk = forward_kinematics(*last_action[0:6] if side == 'left' else last_action[7:13], claw, r_type)

            positions[f'{side}_{paw}'] = current_fk
            vel[f'{side}_{paw}'] = (current_fk - last_fk) / total_time

    return positions, vel

def check_joint_safety(action, flag_in=None):
    """Check J3 signs only for the follower half that is actually active.

    After crossed mapping, ``flag_in`` marks which Nova half is being
    commanded this cycle. The inactive half stays frozen at ``last_action``
    and must NOT be safety-checked: its frozen J3 (index 2 for left, 9 for
    right) keeps its enable-time value and would otherwise trip a false
    reject every cycle, producing the "moves a tiny bit then locks" symptom.

    When ``flag_in`` is None (legacy callers) both halves are checked.
    """
    protect_err = False
    if flag_in is None:
        check_left = True
        check_right = True
    else:
        flags = np.asarray(flag_in)
        check_left = bool(flags[0])
        check_right = bool(flags[1])

    # Left Nova starts at J3 = +90 deg (see robot_pose_init), so its safe
    # half-plane is J3 > 0. The old check hard-coded J3 < 0 for both halves,
    # which made any command to the left Nova (or any bimanual command)
    # reject every cycle and silently reset the servo state. That was the
    # true root cause behind both "right master does nothing" and
    # "both arms cannot teleoperate together" under the crossed mapping.
    if check_left and not (action[2] > 0):
        runtime_log("JOINT_SAFETY_REJECT", joint=2, value=float(action[2]), action=action)
        print("[Warn]:The left Nova J3 joint is out of the safe position! ")
        print(action)
        protect_err = True
    if check_right and not (action[9] < 0):
        runtime_log("JOINT_SAFETY_REJECT", joint=9, value=float(action[9]), action=action)
        print("[Warn]:The right Nova J3 joint is out of the safe position! ")
        print(action)
        protect_err = True
    return protect_err

def get_firmware_version_satisfied(robot_ip):
    try:
        response = requests.post("http://"+robot_ip+":22000/settings/version")
        if response.status_code == 200:
            rt_version = response.text.split("{")[1].split("\n\t")[3].split(":")[1].split("\"")[1].split("-")[0].split(".")
            rt_num = "".join(rt_version)
            return 1, int(rt_num)
        else:
            print("Failed to retrieve the version webpage")
            return 0, 0
    except Exception as e:
        print(e)
        return 0, 0

def get_robot_type(robot_ip):
    response = requests.post("http://"+robot_ip+":22000/properties/controllerType")
    if response.status_code == 200:
        print("The type of robot is:", eval(response.text)["name"])
        return eval(response.text)["name"]
    else:
        print("Failed to obtain the type of robot")
        return None

def check_firmware_version():
    left_version = get_firmware_version_satisfied("192.168.5.1")
    right_version = get_firmware_version_satisfied("192.168.5.2")
    if left_version[1] < 3581 or left_version[1]>=4000:
        print("[ERROR]Left hand error[192.168.5.1]:firmware version requires V3 and must >=3.5.8.1 (found: %s),please check and update"%left_version[1])
        return False
    if right_version[1] < 3581 or right_version[1]>=4000:
        print("[ERROR]Right hand error[192.168.5.1]:firmware version requires V3 and must >=3.5.8.1 (found: {%s}),please check and update"% right_version[1])
        return False
    return True

def main(args):
    # create dataset file path
    save_dir = args.save_data_path+args.project_name+"/collect_data"
    mk_dir(save_dir)
    record_queue_size = int(os.environ.get("DOBOT_RECORD_QUEUE_SIZE", "8"))
    record_writer = BoundedAsyncWriter(
        write_recording_sample,
        max_queue_size=record_queue_size,
    )
    atexit.register(close_record_writer, record_writer)

    # camera init
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    camera_stop_event = threading.Event()
    camera_failure_event = threading.Event()
    camera_ready_events = [threading.Event() for _ in range(3)]
    thread_cam_top = threading.Thread(
        target=run_thread_cam,
        args=(rs1, 0, camera_ready_events[0], camera_failure_event, camera_stop_event),
        daemon=True,
    )
    thread_cam_left = threading.Thread(
        target=run_thread_cam,
        args=(rs2, 1, camera_ready_events[1], camera_failure_event, camera_stop_event),
        daemon=True,
    )
    thread_cam_right = threading.Thread(
        target=run_thread_cam,
        args=(rs3, 2, camera_ready_events[2], camera_failure_event, camera_stop_event),
        daemon=True,
    )
    thread_cam_top.start()
    thread_cam_left.start()
    thread_cam_right.start()
    show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
    camera_deadline = time.monotonic() + CAMERA_STARTUP_TIMEOUT_S
    while not all(event.is_set() for event in camera_ready_events):
        if camera_failure_event.is_set() or time.monotonic() >= camera_deadline:
            camera_stop_event.set()
            raise RuntimeError(
                "Camera startup failed: all three cameras must deliver a frame before "
                "the robot is initialized. Check the camera timeout messages above."
            )
        time.sleep(0.05)
    print("camera thread init success...")

    # agent init
    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    # pose init
    print("Waiting to connect the robot...")
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    print("If the robot fails to initialize successfully after 5 seconds,please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
    if check_firmware_version()==False:
        return
    robot_type = get_robot_type("192.168.5.1")
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    obs = env.get_obs()
    start_servo = False
    curr_light = "dark"
    print("robot init success....")

    # button status init
    last_status = np.array(([0, 0, 0], [0, 0, 0]))  # init lock
    thread_button = threading.Thread(target=button_monitor_realtime, args=(agent, ))
    thread_button.start()
    print("button thread init success...")

    print("-------------------------Ok, let's start------------------------")
    idx = 0
    record_source_index = 0
    last_record_queue_warning = 0.0
    safe_limit = 0
    total_time = 0.04
    perf_window_start = time.monotonic()
    perf_cycle_durations = []
    # Bound follower speed in physical units instead of using one fixed step
    # per loop. The loop is variable-rate because it includes network and
    # status RPCs; a fixed step therefore creates lag at low frequency.
    max_servoj_speed_rad_s = float(
        os.environ.get("DOBOT_SERVOJ_MAX_SPEED_RAD_S", "4.0")
    )
    min_servoj_step_rad = float(
        os.environ.get("DOBOT_SERVOJ_MIN_STEP_RAD", "0.0")
    )
    max_servoj_step_rad = float(
        os.environ.get("DOBOT_SERVOJ_MAX_STEP_RAD", "0.35")
    )
    print(
        f"[TELEOP] follower speed={max_servoj_speed_rad_s} rad/s, "
        f"step=[{min_servoj_step_rad}, {max_servoj_step_rad}] rad",
        flush=True,
    )
    # Preserve the old 0.25 rad minimum jump tolerance, but scale the allowed
    # delta by actual monotonic elapsed time when a slow cycle occurs.
    master_drop_delta_rad = float(
        os.environ.get("DOBOT_MASTER_DROP_DELTA_RAD", "0.25")
    )
    master_drop_speed_rad_s = float(
        os.environ.get("DOBOT_MASTER_DROP_SPEED_RAD_S", "12.5")
    )
    master_runaway_guard = MasterRunawayGuard(
        max_speed_rad_s=master_drop_speed_rad_s,
        min_delta_rad=master_drop_delta_rad,
        required_violation_cycles=2,
    )
    print(
        f"[TELEOP] master runaway limits: min_delta={master_drop_delta_rad} rad, "
        f"max_speed={master_drop_speed_rad_s} rad/s",
        flush=True,
    )
    # Relative teleoperation calibration. Absolute master angles are never
    # sent to a follower; the first servo sample holds the current follower
    # pose and subsequent commands apply only master motion deltas.
    master_ref = None
    follower_ref = None
    servo_warmup_remaining = 0

    while 1:
        tic = time.monotonic()

        if camera_failure_event.is_set():
            camera_stop_event.set()
            raise RuntimeError("Camera stream failed; robot commands are stopped.")
        assert thread_cam_top.is_alive(), "Error: please check the top camera!"
        assert thread_cam_left.is_alive(), "Error: please check the left camera!"
        assert thread_cam_right.is_alive(), "Error: please check the right camera!"
        assert not is_falling, "sensor   detection!"

        # Convert the master-hand convention to the Nova convention before
        # safety checks and robot commands. Raw master angles are not sent.
        # Read the master pose exactly once per cycle. The servo branch reuses
        # this sample instead of calling agent.act again; the second read was
        # a serial USB roundtrip that added its own follow lag every cycle.
        master_now = master_to_nova_action(agent.act({}))
        action = master_now.copy()
        dev_what_to_do = what_to_do.copy()-last_status
        last_status = what_to_do.copy()
        master_flags = np.array([what_to_do[0, 1], what_to_do[1, 1]])
        flag_in = master_to_nova_flags(master_flags)
        gripper_flags = master_to_nova_gripper_flags(master_flags)
        # button A: short press event. lock and unlock
        for i in range(2):
            if dev_what_to_do[i, 0] != 0:
                agent.set_torque(i, not what_to_do[i, 0])

        # button A: long press event. servo or not
        if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
            # Unlock the selected master. Do not auto-approach the Nova.
            # The crossed target must already be close to the corresponding
            # follower before ServoJ is enabled.
            for i in range(2):
                if what_to_do[i, 1]:
                    agent.set_torque(i, False)
            master_ref = master_now.copy()
            follower_ref = env.get_obs()["joint_positions"].copy()
            # Seed the guard so the first servo cycle cannot include motion
            # accumulated while servo was off.
            master_runaway_guard.reset(master_ref)
            # The enable event is a hold-point calibration, not a move event.
            # No absolute master pose is sent to the follower here.
            last_action = follower_ref.copy()
            servo_warmup_remaining = 10
            start_servo = True
            if curr_light != "green":
                curr_light = set_light(env, "yellow", 1)
            print(
                "[SERVO READY] relative hold calibrated; first command holds "
                "the current follower pose; no automatic approach.",
                flush=True,
            )

        if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
            # Stopping servo must explicitly restore torque for the master
            # whose long-press state changed. Otherwise it remains limp and
            # the next short/long press appears to be ignored.
            for i in range(2):
                if dev_what_to_do[i, 1] == -1:
                    agent.set_torque(i, True)
                    runtime_log("SERVO_OFF", hand=i, state=what_to_do.copy())
            if what_to_do[0, 1] == 0 and what_to_do[1, 1] == 0:
                start_servo = False
                master_ref = None
                follower_ref = None
                master_runaway_guard.clear()
                servo_warmup_remaining = 0
                last_action = env.get_obs()["joint_positions"].copy()
                set_light(env, "green", 0)

        if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
            # Master-drop / runaway guard. Detect an implausible per-cycle
            # change of the master joints on the currently active side and
            # hard-lock the whole teleop before any command is dispatched.
            # This is the requested safety for the "master falls out of the
            # operator's hand" case: the master goes limp, gravity throws
            # its joints, and without this guard the follower would try to
            # chase that transient.
            warmup_action, master_ref, servo_warmup_remaining = consume_servo_warmup(
                follower_ref,
                master_now,
                servo_warmup_remaining,
            )
            if warmup_action is not None:
                action = warmup_action
                master_runaway_guard.reset(master_now)
            else:
                runaway = master_runaway_guard.check(master_now, flag_in)
            if warmup_action is None and runaway.tripped:
                runtime_log(
                    "MASTER_DROP_LOCK",
                    threshold=runaway.allowed_delta_rad,
                    elapsed_s=runaway.elapsed_s,
                    max_speed_rad_s=master_drop_speed_rad_s,
                    delta=runaway.deltas.tolist(),
                    joints=runaway.joint_indices,
                    flags=flag_in.copy(),
                )
                print(
                    f"[SAFETY LOCK] master speed exceeded {master_drop_speed_rad_s} rad/s "
                    f"on joints {runaway.joint_indices}; both masters relocked, "
                    "servo disabled",
                    flush=True,
                )
                set_light(env, "red", 1)
                for hand in range(2):
                    agent.set_torque(hand, True)
                what_to_do[:, 1] = 0
                what_to_do[:, 2] = 0
                _button_state_reset_generation[0] += 1
                start_servo = False
                master_ref = None
                follower_ref = None
                master_runaway_guard.clear()
                servo_warmup_remaining = 0
                last_action = env.get_obs()["joint_positions"].copy()
                continue

            # Keep the same master-to-Nova convention conversion for every
            # ServoJ cycle. Reuse the sample captured at the top of the loop
            # so the master serial port is polled exactly once per cycle.
            # Relative crossed teleoperation: preserve each follower's pose at
            # enable, then apply only the mapped master displacement.
            if warmup_action is None:
                action = follower_ref.copy()
                cycle_elapsed = max(time.monotonic() - tic, 1e-3)
                cycle_step_rad = min(
                    max_servoj_step_rad,
                    max(min_servoj_step_rad, max_servoj_speed_rad_s * cycle_elapsed),
                )
                for i in range(14):
                    if i in (6, 13):
                        continue
                    if not flag_in[0 if i < 7 else 1]:
                        action[i] = last_action[i]
                    else:
                        master_delta = master_now[i] - master_ref[i]
                        desired = follower_ref[i] + master_delta
                        delta = desired - last_action[i]
                        action[i] = last_action[i] + np.clip(
                            delta, -cycle_step_rad, cycle_step_rad
                        )
                action = apply_same_side_gripper_targets(
                    action,
                    master_now,
                    last_action,
                    gripper_flags,
                )
            runtime_log(
                "SERVO_COMMAND",
                raw_action=master_now,
                command=action,
                last_action=last_action.copy(),
                flags=flag_in.copy(),
            )
            err3, action = servo_action_check(action, last_action, flag_in)
            if err3 == 0:
                runtime_log("SERVO_REJECTED", reason="servo_action_check", action=action)
                set_light(env, "red", 1)
                for hand in range(2):
                    agent.set_torque(hand, True)
                what_to_do[:, 1] = 0
                what_to_do[:, 2] = 0
                _button_state_reset_generation[0] += 1
                start_servo = False
                master_ref = None
                follower_ref = None
                last_action = env.get_obs()["joint_positions"].copy()
                runtime_log("SERVO_STATE_RESET", reason="servo_action_check")
                continue

            # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
            # [Note]: Modify the protection parameters in this section carefully !
            pose_protection = False
            joint_safety = False
            if (safe_limit < 1):
                safe_limit = safe_limit + 1
            else:
                positions, vel = calculate_vel_pos(
                    action, last_action, max(cycle_elapsed, 1e-3), robot_type
                )
                pose_protection = check_pose_protection(positions, vel, flag_in)
                joint_safety = check_joint_safety(action, flag_in)
            if pose_protection or joint_safety:
                reasons = []
                if pose_protection:
                    reasons.append("pose_protection")
                if joint_safety:
                    reasons.append("joint_safety")
                runtime_log(
                    "SERVO_REJECTED",
                    reason="+".join(reasons),
                    action=action,
                    flags=flag_in.copy(),
                )
                set_light(env, "red", 1)
                for hand in range(2):
                    agent.set_torque(hand, True)
                what_to_do[:, 1] = 0
                what_to_do[:, 2] = 0
                _button_state_reset_generation[0] += 1
                start_servo = False
                master_ref = None
                follower_ref = None
                servo_warmup_remaining = 0
                safe_limit = 0
                runtime_log("SERVO_STATE_RESET", reason="joint_safety")
                continue
            # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

            # button B: recording or not
            if dev_what_to_do[0, 2] == 1:
                idx = 0
                record_source_index = 0
                curr_light = set_light(env, "green", 1)
            elif dev_what_to_do[0, 2] == -1:
                curr_light = set_light(env, "yellow", 1)

            # Preserve the behavior-cloning pairing used by the original
            # collector: pre-command observation obs_t with action_t. The
            # queue callback performs disk I/O later, off the control thread.
            if what_to_do[0, 2] == 1:
                capture_time = time.monotonic()

                def _snapshot_recording():
                    observations = {
                        key: np.asarray(value).copy()
                        for key, value in obs.items()
                    }
                    observations["_teleop_capture_monotonic_s"] = np.asarray(
                        capture_time
                    )
                    observations["_teleop_source_index"] = np.asarray(
                        record_source_index
                    )
                    observations["_teleop_dropped_before"] = np.asarray(
                        record_writer.dropped_count
                    )
                    return RecordingSample(
                        episode_dir=os.path.join(save_dir, str(dt_time[0])),
                        index=idx,
                        images=tuple(jpeg_list),
                        observations=observations,
                        action=np.asarray(action).copy(),
                    )

                try:
                    submitted = record_writer.submit_factory(_snapshot_recording)
                except RuntimeError as exc:
                    what_to_do[:, 2] = 0
                    runtime_log("RECORD_WRITER_FAILURE", error=str(exc))
                else:
                    record_source_index += 1
                    if submitted:
                        idx += 1
                    else:
                        now = time.monotonic()
                        if now - last_record_queue_warning >= 1.0:
                            runtime_log(
                                "RECORD_QUEUE_FULL",
                                dropped=record_writer.dropped_count,
                                queue_size=record_queue_size,
                            )
                            last_record_queue_warning = now

            try:
                runtime_log("ENV_STEP_BEGIN", action=action, flags=flag_in.copy())
                obs = env.step(action, flag_in, gripper_flags)
                runtime_log("ENV_STEP_OK", action=action, flags=flag_in.copy())
            except Exception as exc:
                runtime_log(
                    "SERVO_FAILURE",
                    error=f"{type(exc).__name__}: {exc}",
                    action=action,
                    last_action=last_action.copy(),
                    flags=flag_in.copy(),
                )
                set_light(env, "red", 1)
                for hand in range(2):
                    agent.set_torque(hand, True)
                what_to_do[:, 1] = 0
                what_to_do[:, 2] = 0
                _button_state_reset_generation[0] += 1
                start_servo = False
                safe_limit = 0
                try:
                    last_action = env.get_obs()["joint_positions"].copy()
                except Exception as state_exc:
                    runtime_log("RECOVERY_STATE_READ_FAILED", error=f"{type(state_exc).__name__}: {state_exc}")
                continue
            obs["joint_positions"][6] = action[6]
            obs["joint_positions"][13] = action[13]
            last_action = action
        else:
            start_servo = False
            safe_limit = 0

        # img show
        if args.show_img:
            show_canvas[:, :640] = np.asarray(img_list[0], dtype="uint8")
            show_canvas[:, 640:640 * 2] = np.asarray(img_list[1], dtype="uint8")
            show_canvas[:, 640 * 2:640 * 3] = np.asarray(img_list[2], dtype="uint8")
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)

        toc = time.monotonic()
        total_time = toc-tic
        perf_cycle_durations.append(total_time)
        perf_elapsed = toc - perf_window_start
        if perf_elapsed >= 5.0:
            runtime_log(
                "CONTROL_LOOP_PERF",
                hz=len(perf_cycle_durations) / perf_elapsed,
                mean_ms=1000.0 * sum(perf_cycle_durations) / len(perf_cycle_durations),
                max_ms=1000.0 * max(perf_cycle_durations),
                record_dropped=record_writer.dropped_count,
            )
            perf_window_start = toc
            perf_cycle_durations.clear()


if __name__ == "__main__":
    main(tyro.cli(Args))
