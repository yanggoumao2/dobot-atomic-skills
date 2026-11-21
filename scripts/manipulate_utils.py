import numpy as np
import time
import configparser
from dobot_control.agents.dobot_agent import DobotRobotConfig
import os
from pathlib import Path
from dataclasses import dataclass

from scripts.function_util import wait_period, log_write
from task_skills.layout import physical_nova_home_degrees


def set_light(env, which_color, which_status):
    print("light change")
    if which_color == "red":
        env.set_do_status([3, 0])
        env.set_do_status([2, 0])
        env.set_do_status([1, which_status])
    elif which_color == "yellow":
        env.set_do_status([3, 0])
        env.set_do_status([2, which_status])
        env.set_do_status([1, 0])
    elif which_color == "green":
        env.set_do_status([3, which_status])
        env.set_do_status([2, 0])
        env.set_do_status([1, 0])
    return which_color

def load_ini_data_camera():
    camera_dict = {"top": None, "left": None, "right": None}
    ini_file_path = str(Path(__file__).parent) + "/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)
    for _cam in camera_dict.keys():
        camera_dict[_cam] = ini_file.get("CAMERA", _cam)
    return camera_dict

def load_ini_data_hands():
    ini_file_path = str(Path(__file__).parent) + "/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)

    hands_dict = {"HAND_LEFT": None, "HAND_RIGHT": None}
    for _hand in hands_dict.keys():
        hands_dict[_hand] = DobotRobotConfig(
            joint_ids=[int(i) for i in ini_file.get(_hand, "joint_ids").split(",")],
            append_id=int(ini_file.get(_hand, "append_id")),
            port=ini_file.get(_hand, "port"),
            joint_offsets=[float(i) for i in ini_file.get(_hand, "joint_offsets").split(",")],
            joint_signs=[int(i) for i in ini_file.get(_hand, "joint_signs").split(",")],
            gripper_config=[int(i) for i in ini_file.get(_hand, "gripper_config").split(",")],
            start_joints=[float(i) for i in ini_file.get(_hand, "start_joints").split(",")],
            baud_rate=int(ini_file.get(_hand, "baud_rate")),
            using_sensor=int(ini_file.get(_hand, "using_sensor")))
    return ini_file, hands_dict


@dataclass
class GripperConfig:
    id_name: int
    pos: tuple
    port: str


def load_ini_data_gripper():
    ini_file_path = str(Path(__file__).parent) + "/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)

    gripper_dict = {"GRIPPER_LEFT": None, "GRIPPER_RIGHT": None}
    for _gripper in gripper_dict.keys():
        gripper_dict[_gripper] = GripperConfig(id_name=int(ini_file.get(_gripper, "id")),
                                               pos=list([int(i) for i in ini_file.get(_gripper, "pos").split(",")]),
                                               port=ini_file.get(_gripper, "port"))
    return ini_file, gripper_dict

# robot init move
_SAFE_POSE_INIT_ENV = "DOBOT_ALLOW_SAFE_POSE_INIT"
_SAFE_POSE_INIT_STEP_RAD = float(os.environ.get("DOBOT_SAFE_POSE_STEP_RAD", "0.02"))
_SAFE_POSE_INIT_PERIOD_S = float(os.environ.get("DOBOT_SAFE_POSE_PERIOD_S", "0.05"))
_SAFE_POSE_INIT_SERVO_T_S = float(os.environ.get("DOBOT_SAFE_POSE_SERVO_T_S", "0.05"))


def _print_pose_check(curr_joints, target_joints, label):
    # The environment vector has 14 entries: 6 joints + gripper for each arm.
    # Gripper values are normalized [0, 1], not radians, so never convert or
    # use them as a six-axis pose difference.
    curr_arm = np.concatenate([curr_joints[:6], curr_joints[7:13]])
    target_arm = np.concatenate([target_joints[:6], target_joints[7:13]])
    curr_deg = np.rad2deg(curr_arm)
    target_deg = np.rad2deg(target_arm)
    delta_deg = target_deg - curr_deg
    print(f"[POSE CHECK] {label}")
    print("  current_arm_deg =", np.round(curr_deg, 2).tolist())
    print("  target_arm_deg  =", np.round(target_deg, 2).tolist())
    print("  delta_arm_deg   =", np.round(delta_deg, 2).tolist())
    print("  gripper_current =", np.round([curr_joints[6], curr_joints[13]], 4).tolist())
    print("  gripper_target  =", np.round([target_joints[6], target_joints[13]], 4).tolist())
    print("  max_arm_delta_deg =", round(float(np.max(np.abs(delta_deg))), 2))


def _safe_pose_segment(env, curr_joints, target_joints, label):
    _print_pose_check(curr_joints, target_joints, label)
    curr_arm = np.concatenate([curr_joints[:6], curr_joints[7:13]])
    target_arm = np.concatenate([target_joints[:6], target_joints[7:13]])
    max_delta = float(np.max(np.abs(curr_arm - target_arm)))
    if max_delta <= 1e-6:
        print(f"[SAFE POSE INIT] {label}: six-axis pose already matches; no motion sent.", flush=True)
        return
    # Preserve the current gripper commands. Pose initialization must not
    # open/close either gripper as a side effect of arm interpolation.
    command_target = np.asarray(target_joints, dtype=float).copy()
    command_target[6] = curr_joints[6]
    command_target[13] = curr_joints[13]
    steps = max(1, int(np.ceil(max_delta / _SAFE_POSE_INIT_STEP_RAD)))
    print(
        f"[SAFE POSE INIT] {label}: steps={steps}, "
        f"step_rad<={_SAFE_POSE_INIT_STEP_RAD}, "
        f"period_s={_SAFE_POSE_INIT_PERIOD_S}, "
        f"servo_t_s={_SAFE_POSE_INIT_SERVO_T_S}",
        flush=True,
    )
    for jnt in np.linspace(curr_joints, command_target, steps + 1)[1:]:
        env.step(jnt, [1, 1], [0, 0])
        time.sleep(_SAFE_POSE_INIT_PERIOD_S)


def robot_pose_init(env):
    """Inspect the reset pose, and move only after explicit opt-in.

    The old implementation sent ServoJ commands as fast as the Python loop
    could run. That is unsafe for a real robot. The default is now a read-only
    pose check that aborts before any initialization motion.
    """
    reset_1 = np.deg2rad([
        *physical_nova_home_degrees("left"), 0,
        *physical_nova_home_degrees("right"), 0,
    ])
    reset_2 = reset_1.copy()

    curr_joints = np.asarray(env.get_obs()["joint_positions"], dtype=float)
    _print_pose_check(curr_joints, reset_1, "reset waypoint 1 (read-only)")
    _print_pose_check(reset_1, reset_2, "reset waypoint 1 -> final")

    curr_arm = np.concatenate([curr_joints[:6], curr_joints[7:13]])
    target_arm = np.concatenate([reset_1[:6], reset_1[7:13]])
    max_arm_delta = float(np.max(np.abs(curr_arm - target_arm)))
    if max_arm_delta <= 1e-6:
        print("[SAFE POSE INIT] six-axis pose already matches target; no motion sent.", flush=True)
        return

    if os.environ.get(_SAFE_POSE_INIT_ENV) != "1":
        raise RuntimeError(
            f"Safe stop before pose motion. Review the arm pose check above, then "
            f"set {_SAFE_POSE_INIT_ENV}=1 for an explicit low-speed opt-in."
        )

    _safe_pose_segment(env, curr_joints, reset_1, "current -> waypoint 1")
    _safe_pose_segment(env, reset_1, reset_2, "waypoint 1 -> final")


# main hand pose dev check
def obs_action_check(env, agent):
    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.6).any():
        print("Action is too big")
        # print which joints are too big
        joint_index = np.where(action - joints > 0.5)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        return 0, 0
    else:
        return 1, action


# Master-to-Nova channel conversion.
# The requested physical wiring is crossed: left master -> right Nova and
# right master -> left Nova. Their verified startup poses already match in
# this crossed pairing, so no additional sign inversion is applied here.


def master_to_nova_action(action):
    """Convert master order to crossed Nova target order.

    Master order: HAND_LEFT (COM6), HAND_RIGHT (COM5).
    Nova target order: left Nova, right Nova.
    Configured routing: HAND_RIGHT -> left Nova and
    HAND_LEFT -> right Nova. Config labels are NOT physical side evidence.
    Gripper entries stay normalized [0, 1].
    Configured reference poses match in this crossed pairing (HAND_LEFT
    and right Nova both start at ~[-90,0,-90,0,+90,+90] deg; right master and
    left Nova both start at ~[+90,0,+90,0,-90,-90] deg), so no sign inversion
    is applied here. A direct straight-through mapping would flip J3/J5/J6.
    """
    action = np.asarray(action, dtype=float).copy()
    if action.shape != (14,):
        raise ValueError(f"Expected 14-D bimanual action, got shape {action.shape}")

    master_left = action[:7].copy()
    master_right = action[7:14].copy()

    # Output is target order: left Nova <- right master,
    # right Nova <- left master.
    return np.concatenate([master_right, master_left])


def master_to_nova_flags(master_flags):
    """Swap active-servo flags to match the crossed Nova target order."""
    flags = np.asarray(master_flags).copy()
    if flags.shape != (2,):
        raise ValueError(f"Expected two master flags, got shape {flags.shape}")
    return flags[::-1]


def master_to_nova_gripper_flags(master_flags):
    """Keep gripper flags in config-channel order; physical routing needs verification."""
    flags = np.asarray(master_flags).copy()
    if flags.shape != (2,):
        raise ValueError(f"Expected two master flags, got shape {flags.shape}")
    return flags


def apply_same_side_gripper_targets(
    action,
    crossed_master_action,
    last_action,
    gripper_flags,
):
    """Apply config-channel gripper mapping; physical placement is verified onsite."""
    result = np.asarray(action, dtype=float).copy()
    crossed_master_action = np.asarray(crossed_master_action, dtype=float)
    last_action = np.asarray(last_action, dtype=float)
    flags = np.asarray(gripper_flags)
    if result.shape != (14,) or crossed_master_action.shape != (14,):
        raise ValueError("Expected 14-D action and crossed master action")
    if last_action.shape != (14,):
        raise ValueError(f"Expected 14-D last action, got shape {last_action.shape}")
    if flags.shape != (2,):
        raise ValueError(f"Expected two gripper flags, got shape {flags.shape}")

    # crossed_master_action[13] is the HAND_LEFT master gripper;
    # crossed_master_action[6] is the HAND_RIGHT master gripper.
    for target_index, source_index, enabled in (
        (6, 13, bool(flags[0])),
        (13, 6, bool(flags[1])),
    ):
        result[target_index] = (
            np.clip(crossed_master_action[source_index], 0.0, 1.0)
            if enabled
            else last_action[target_index]
        )
    return result


# nova2 dev joint check
def servo_action_check(action, last_action, flag_in, step_len=0.9):
    ind_list = [i for i in range(14)]
    assert len(ind_list), "err in servo_action_check"
    if (np.abs(action - last_action) > step_len).any():
        joint_index = np.where(np.abs(action - last_action) > step_len)
        print("action: ", action)
        print("last_action: ", last_action)
        for j in joint_index[0]:
            if j != 6 and j != 13 and (j in ind_list):
                pi_2_cal = (action[j] - last_action[j])/np.pi
                if abs(pi_2_cal) > 1.85 and abs(pi_2_cal) < 2.15:
                    action[j] = action[j] - 2 * np.pi * (pi_2_cal / abs(pi_2_cal))
                else:
                    print("Servo action dev is too big")
                    print(
                        f"Joint [{j}], leader: {action[j]}, follower: {last_action[j]}, "
                        f"diff: {(action[j] - last_action[j])/np.pi}")
                    return 0, action
    return 1, action


# pose check between main hand and the follower
def pose_check(env, agent, flag_in):
    start_pos = master_to_nova_action(agent.act(env.get_obs()))
    obs = env.get_obs()
    joints = obs["joint_positions"]
    err_pose_check, action_return = servo_action_check(start_pos, joints, flag_in, 0.6)
    if err_pose_check:
        return 1, action_return
    else:
        return 0, action_return


def dynamic_approach(env, agent, flag_in):
    err1, action1 = pose_check(env, agent, flag_in)
    assert err1 != 0, set_light(env, "red", 1)
    obs = env.get_obs()
    joints = obs["joint_positions"]
    # log_write(__file__, "joints: " + str(joints))
    # log_write(__file__, "action1: " + str(action1))
    joints[6] = action1[6]
    joints[13] = action1[13]
    if flag_in[0] and not flag_in[1]:
        abs_deltas = max(np.abs(action1[:6] - joints[:6]))
    elif not flag_in[0] and flag_in[1]:
        abs_deltas = max(np.abs(action1[7:13] - joints[7:13]))
    else:
        abs_deltas = max(np.abs(action1 - joints))
    # log_write(__file__,  "abs_deltas: " + str(abs_deltas))
    steps = int(abs_deltas / 0.01)
    # log_write(__file__, "steps: " + str(steps))

    for jnt in np.linspace(joints, action1, steps):
        env.step(jnt, flag_in)
        tic = time.time()
        wait_period(50, tic)

        # log_write(__file__, "flag_in: " + str(flag_in))
        # log_write(__file__, "jnt: " + str(jnt))
    # time.sleep(0.05)
    err1, action1 = pose_check(env, agent, flag_in)
    assert err1 != 0, set_light(env, "red", 1)
    return action1


if __name__ == "__main__":
    print("test")
    # action = [-1.44164974,  0.13345643, -2.07741816,  0.59677646, 1.60714534,  1.91935946,
    #            0.99817288,  0.95174802,  1.03044725,  1.90838818, -0.36576736, -1.41200051, -2.05291206,  1.]
    #
    # print(action[7:13])
    # print(np.rad2deg(0.6))
