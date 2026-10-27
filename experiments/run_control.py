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
from dobot_control.cameras.settings import CAMERA_NAMES, load_camera_settings
from dobot_control.async_writer import BoundedAsyncWriter
from dobot_control.absolute_teleop import AbsoluteTeleopPlanner, CommandInterval
from dobot_control.teleop_safety import (
    MasterRunawayGuard,
    check_pose_protection,
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


def run_button_monitor_guarded(agent, failure_event, stop_event, ready_event, errors):
    try:
        button_monitor_realtime(agent, stop_event, ready_event)
        if not stop_event.is_set():
            raise RuntimeError("Button monitor exited unexpectedly")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        failure_event.set()  # publish failure before logging or any cleanup
        what_to_do[:, 1:] = 0
        runtime_log("BUTTON_MONITOR_FAILURE", error=errors[-1])


def require_button_monitor_healthy(thread, failure_event, errors):
    if failure_event.is_set() or not thread.is_alive():
        what_to_do[:, 1:] = 0
        detail = errors[-1] if errors else "button thread is not running"
        raise RuntimeError(f"Button monitoring unavailable; stop teleoperation: {detail}")


def button_monitor_realtime(agent, stop_event, ready_event):
    # servo
    last_keys_status = None
    press_started_at = np.zeros(2)
    start_press_status = np.array(([0, 0], [0, 0]))  # start press
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))
    reset_generation = int(_button_state_reset_generation[0])

    while not stop_event.is_set() and not is_falling[0]:
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
        if last_keys_status is None:
            last_keys_status = now_keys.copy()
            ready_event.set()
            continue  # first sample is a baseline, not a button edge
        dev_keys = now_keys - last_keys_status
        # button a
        for i in range(2):
            if dev_keys[i, 0] == -1:  # button a: start
                press_started_at[i] = time.monotonic()
                start_press_status[i, 0] = 1
            if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # button a: end
                start_press_status[i, 0] = 0
                held_seconds = time.monotonic() - press_started_at[i]
                if held_seconds < 0.5:
                    keys_press_count[i, 0] += 1
                    if keys_press_count[i, 0] % 2 == 1:
                        what_to_do[i, 0] = 1
                        runtime_log("BUTTON_A_SHORT", hand=i, action="unlock", state=what_to_do.copy())
                    else:
                        what_to_do[i, 0] = 0
                        runtime_log("BUTTON_A_SHORT", hand=i, action="lock", state=what_to_do.copy())

                elif held_seconds > 1:
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
        stop_event.wait(0.005)


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



def start_configured_cameras(settings):
    """Start only selected views; fail before constructing any robot/hand."""
    cameras, threads, ready_events = {}, {}, []
    stop_event, failure_event = threading.Event(), threading.Event()

    def close():
        stop_event.set()
        for thread in threads.values():
            thread.join(timeout=6.0)
        for camera in cameras.values():
            camera.close()

    try:
        for name, serial in settings.serials.items():
            camera = RealSenseCamera(device_id=serial, flip=name != "left", fps=settings.fps)
            cameras[name] = camera
            ready = threading.Event()
            ready_events.append(ready)
            thread = threading.Thread(target=run_thread_cam,
                                      args=(camera, CAMERA_NAMES.index(name), ready,
                                            failure_event, stop_event), daemon=True)
            threads[name] = thread
            thread.start()
        deadline = time.monotonic() + CAMERA_STARTUP_TIMEOUT_S
        while not all(event.is_set() for event in ready_events):
            if failure_event.is_set() or time.monotonic() >= deadline:
                raise RuntimeError("Enabled cameras did not deliver frames before startup deadline")
            time.sleep(0.05)
        if failure_event.is_set() or not all(t.is_alive() for t in threads.values()):
            raise RuntimeError("Enabled camera stream failed during startup")
    except BaseException:
        close()
        raise
    return threads, failure_event, stop_event, close


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

def limit_tcp_descent(action, previous, elapsed, robot_type, flags, limit=1.0):
    """Reduce proposed steps rather than unlatching servo on speed alone.

    The final workspace/velocity checker still independently checks the result.
    This is commanded FK velocity, not measured physical TCP velocity.
    """
    result = np.asarray(action, dtype=float).copy()
    previous = np.asarray(previous, dtype=float)
    limited = []
    for side, name in enumerate(("left", "right")):
        if not flags[side]:
            continue
        def downward(candidate):
            _, velocities = calculate_vel_pos(candidate, previous, elapsed, robot_type)
            return max(0.0, -float(velocities[f"{name}_left"][2]),
                       -float(velocities[f"{name}_right"][2]))
        before = downward(result)
        if before <= limit:
            continue
        arm = slice(side * 7, side * 7 + 6)
        delta = result[arm] - previous[arm]
        low, high = 0.0, 1.0
        for _ in range(16):
            middle = (low + high) / 2.0
            candidate = result.copy()
            candidate[arm] = previous[arm] + delta * middle
            if downward(candidate) <= limit * 0.95:
                low = middle
            else:
                high = middle
        result[arm] = previous[arm] + delta * low
        limited.append({"side": name, "before_m_s": before,
                        "after_m_s": downward(result), "scale": low})
    return result, limited


def validate_absolute_alignment(current, target, robot_type, flags, joint_step=0.01):
    """Sample the enable-time straight joint path; runtime guards still apply."""
    current, target = map(AbsoluteTeleopPlanner.vector, (current, target))
    flags = np.asarray(flags, dtype=bool)
    if flags.shape != (2,):
        raise ValueError("Expected two Nova flags")
    if not np.isfinite(joint_step) or joint_step <= 0:
        raise ValueError("Positive finite sampling step required")
    target = target.copy()
    for side in range(2):
        if not flags[side]:
            target[side*7:side*7+7] = current[side*7:side*7+7]
        # Grippers do not move during alignment.
        target[side*7+6] = current[side*7+6]
    delta = target - current
    samples = max(2, int(np.ceil(np.max(np.abs(delta))/joint_step)) + 1)
    for index, fraction in enumerate(np.linspace(0.0, 1.0, samples)):
        sample = current + delta * fraction
        positions, velocities = calculate_vel_pos(sample, sample, 1.0, robot_type)
        positions_mm = {key: value * 1000 for key, value in positions.items()}
        for side, name, x_range in ((0,"left",(-450,300)), (1,"right",(-250,450))):
            if not flags[side]:
                continue
            for tip in (f"{name}_left", f"{name}_right"):
                p = positions_mm[tip]
                if not (x_range[0] <= p[0] <= x_range[1]
                        and -750 <= p[1] <= -160 and p[2] > 40):
                    return False, {"sample":index, "samples":samples,
                                   "fraction":float(fraction), "nova_side":name,
                                   "tip":tip, "position_mm":p.tolist(),
                                   "target":target.tolist()}
        if check_joint_safety(sample, flags):
            return False, {"sample":index, "samples":samples,
                           "fraction":float(fraction), "reason":"joint_safety",
                           "target":target.tolist()}
    return True, {"samples":samples, "target":target.tolist()}


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
    camera_settings = load_camera_settings()
    camera_threads, camera_failure_event, camera_stop_event, close_cameras = start_configured_cameras(camera_settings)
    atexit.register(close_cameras)
    active_camera_indices = [CAMERA_NAMES.index(name) for name in camera_settings.serials]
    recording_enabled = camera_settings.recording_enabled
    print(f"camera thread init success: {list(camera_settings.serials)}, {camera_settings.fps} fps")
    if not recording_enabled:
        print("[RECORD] Three-view recording disabled; selected-camera teleoperation only.")

    # agent init
    hands_ini, hands_dict = load_ini_data_hands()
    hand_sections = ("HAND_LEFT", "HAND_RIGHT")
    absolute_mapping_verified = np.array([
        hands_ini.getboolean(section, "absolute_mapping_verified", fallback=False)
        for section in hand_sections
    ])
    physical_hand_sides = [
        hands_ini.get(section, "physical_side", fallback="unverified")
        for section in hand_sections
    ]
    print("[MAPPING] " + "; ".join(
        f"hand={index} {section} COM={hands_dict[section].port} "
        f"physical={physical_hand_sides[index]} absolute_verified={bool(absolute_mapping_verified[index])}"
        for index, section in enumerate(hand_sections)
    ), flush=True)
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
    button_failure_event, button_stop_event = threading.Event(), threading.Event()
    button_ready_event, button_errors = threading.Event(), []
    thread_button = threading.Thread(target=run_button_monitor_guarded,
                                    args=(agent, button_failure_event, button_stop_event,
                                          button_ready_event, button_errors), daemon=True)
    thread_button.start()
    atexit.register(button_stop_event.set)
    if not button_ready_event.wait(timeout=3.0):
        button_stop_event.set()
        require_button_monitor_healthy(thread_button, button_failure_event, button_errors)
        raise RuntimeError("Button monitor did not receive initial valid states within 3 seconds")
    require_button_monitor_healthy(thread_button, button_failure_event, button_errors)
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
    max_servoj_step_rad = float(
        os.environ.get("DOBOT_SERVOJ_MAX_STEP_RAD", "0.35")
    )
    align_speed = float(os.environ.get("DOBOT_ALIGN_SPEED_RAD_S", "0.15"))
    planner = AbsoluteTeleopPlanner(align_speed=align_speed,
                                   follow_speed=max_servoj_speed_rad_s,
                                   max_step=max_servoj_step_rad)
    command_interval = CommandInterval()
    last_tcp_limit_log = 0.0
    print(
        f"[TELEOP] absolute joint targets; align={align_speed} rad/s, "
        f"follow={max_servoj_speed_rad_s} rad/s, max_step={max_servoj_step_rad} rad",
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
    last_action = env.get_obs()["joint_positions"].copy()

    while 1:
        tic = time.monotonic()
        require_button_monitor_healthy(thread_button, button_failure_event, button_errors)

        if camera_failure_event.is_set():
            camera_stop_event.set()
            raise RuntimeError("Camera stream failed; robot commands are stopped.")
        if not all(thread.is_alive() for thread in camera_threads.values()):
            camera_stop_event.set()
            raise RuntimeError("Enabled camera thread stopped; robot commands are stopped.")
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
            # Slow alignment is part of this nonblocking loop, so button,
            # camera and workspace checks remain active during alignment.
            obs = env.get_obs()
            new_master_flags = np.asarray(dev_what_to_do[:, 1] == 1, dtype=int)
            new_nova_flags = master_to_nova_flags(new_master_flags)
            unverified = np.flatnonzero(new_master_flags.astype(bool) & ~absolute_mapping_verified)
            if len(unverified):
                for master_side in range(2):
                    agent.set_torque(master_side, True)
                what_to_do[:, 1:] = 0
                start_servo = False
                planner.reset()
                master_runaway_guard.clear()
                _button_state_reset_generation[0] += 1
                runtime_log("ALIGN_REJECTED_UNCALIBRATED", master_indices=unverified.tolist(),
                            sections=[hand_sections[i] for i in unverified],
                            physical_sides=[physical_hand_sides[i] for i in unverified])
                set_light(env, "red", 1)
                print("[ALIGN REJECTED] absolute master mapping is not calibrated. "
                      "Complete manual step 7 calibration; no alignment motion was sent.",
                      flush=True)
                continue
            safe_alignment, alignment_details = validate_absolute_alignment(
                obs["joint_positions"], master_now, robot_type, new_nova_flags)
            if not safe_alignment:
                for master_side in range(2):
                    agent.set_torque(master_side, True)
                what_to_do[:, 1:] = 0
                start_servo = False
                planner.reset()
                master_runaway_guard.clear()
                _button_state_reset_generation[0] += 1
                runtime_log("ALIGN_REJECTED_UNSAFE", master_flags=new_master_flags,
                            nova_flags=new_nova_flags, current=obs["joint_positions"],
                            target=master_now, details=alignment_details)
                set_light(env, "red", 1)
                print("[ALIGN REJECTED] absolute target/path leaves the configured "
                      "workspace; no alignment motion was sent. Calibrate the master "
                      "mapping or place it at a safe compatible pose.", flush=True)
                continue
            for i in range(2):
                if dev_what_to_do[i, 1] == 1:
                    agent.set_torque(i, False)
            last_action = planner.enable(
                flag_in, obs["joint_positions"], last_action)
            master_runaway_guard.reset(master_now)
            command_interval.reset(time.monotonic())
            start_servo = True
            if curr_light != "green":
                curr_light = set_light(env, "yellow", 1)
            runtime_log("ALIGN_START", flags=flag_in.copy(), speed_rad_s=align_speed,
                        current=obs["joint_positions"], target=master_now,
                        path_samples=alignment_details["samples"])

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
                planner.reset()
                master_runaway_guard.clear()
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
            last_action = planner.enable(flag_in, obs["joint_positions"], last_action)
            runaway = master_runaway_guard.check(master_now, flag_in)
            if runaway.tripped:
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
                planner.reset()
                master_runaway_guard.clear()
                last_action = env.get_obs()["joint_positions"].copy()
                continue

            # Use the interval between consecutive command proposals, including
            # the prior network roundtrip, not time spent in this loop prefix.
            command_time = time.monotonic()
            cycle_elapsed = command_interval.elapsed(command_time)
            action, aligned_sides = planner.step(
                master_now, obs["joint_positions"], last_action, cycle_elapsed)
            for side in aligned_sides:
                runtime_log("ALIGN_DONE", nova_side=side, follow_speed_rad_s=max_servoj_speed_rad_s)
            # Gripper flags are in master/same-side order; alignment is Nova/crossed order.
            aligned_gripper_flags = np.asarray(gripper_flags) * planner.aligned[::-1]
            action = apply_same_side_gripper_targets(action, master_now, last_action, aligned_gripper_flags)
            action, limited_sides = limit_tcp_descent(
                action, last_action, cycle_elapsed, robot_type, flag_in)
            if limited_sides and command_time - last_tcp_limit_log >= 1.0:
                runtime_log("TCP_SPEED_LIMITED", details=limited_sides, interval_s=cycle_elapsed)
                last_tcp_limit_log = command_time
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
                planner.reset()
                last_action = env.get_obs()["joint_positions"].copy()
                runtime_log("SERVO_STATE_RESET", reason="servo_action_check")
                continue

            # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
            # [Note]: Modify the protection parameters in this section carefully !
            pose_protection = False
            joint_safety = False
            positions, vel = calculate_vel_pos(action, last_action, cycle_elapsed, robot_type)
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
                    command_interval_s=cycle_elapsed,
                    positions_mm={key: (value * 1000).tolist() for key, value in positions.items()},
                    tcp_z_m_s={key: float(value[2]) for key, value in vel.items()},
                )
                set_light(env, "red", 1)
                for hand in range(2):
                    agent.set_torque(hand, True)
                what_to_do[:, 1] = 0
                what_to_do[:, 2] = 0
                _button_state_reset_generation[0] += 1
                start_servo = False
                planner.reset()
                safe_limit = 0
                runtime_log("SERVO_STATE_RESET", reason="+".join(reasons))
                continue
            # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

            # button B: recording or not
            if dev_what_to_do[0, 2] == 1 and recording_enabled:
                idx = 0
                record_source_index = 0
                curr_light = set_light(env, "green", 1)
            elif dev_what_to_do[0, 2] == -1 and recording_enabled:
                curr_light = set_light(env, "yellow", 1)

            # Preserve the behavior-cloning pairing used by the original
            # collector: pre-command observation obs_t with action_t. The
            # queue callback performs disk I/O later, off the control thread.
            if what_to_do[0, 2] == 1 and recording_enabled:
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

            # Outside the ServoJ recovery handler: a dead button monitor is
            # fatal, and must not be swallowed and retried next cycle.
            require_button_monitor_healthy(thread_button, button_failure_event, button_errors)
            try:
                runtime_log("ENV_STEP_BEGIN", action=action, flags=flag_in.copy())
                obs = env.step(action, flag_in, aligned_gripper_flags)
                command_interval.committed(command_time)
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
                planner.reset()
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
            show_canvas = np.concatenate([np.asarray(img_list[i], dtype="uint8")
                                          for i in active_camera_indices], axis=1)
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
