"""Pure command checks shared by the ACT test client and offline tests."""
import time

import numpy as np


def measured_state(observation, now=None):
    now = time.perf_counter() if now is None else now
    qpos = np.asarray(observation.get('dataset_joint_positions'), dtype=float)
    if qpos.shape != (14,) or not np.isfinite(qpos).all():
        raise ValueError('A must run with DOBOT_ACT_RECORDING=1 and valid measured grippers')
    for key, limit in [('_act_state_time', .2), ('_act_gripper_time', .3)]:
        stamps = np.asarray(observation.get(key), dtype=float)
        if stamps.shape != (2,) or not np.isfinite(stamps).all() or np.any(now-stamps < 0) or np.any(now-stamps > limit):
            raise ValueError(f'Stale or invalid observation {key}')
    valid = np.asarray(observation.get('_act_gripper_valid'), dtype=bool)
    if valid.shape != (2,) or not valid.all():
        raise ValueError('Gripper feedback invalid')
    if np.any(qpos[[6, 13]] < 0) or np.any(qpos[[6, 13]] > 1):
        raise ValueError('Measured gripper outside 0..1')
    return qpos


def safe_command(qpos, prediction, previous, dt, allow_gripper=False, planner=None):
    from dobot_control.absolute_teleop import configured_teleop_planner
    qpos, prediction, previous = map(np.asarray, (qpos, prediction, previous))
    if any(x.shape != (14,) or not np.isfinite(x).all() for x in (qpos, prediction, previous)):
        raise ValueError('Expected finite 14-D vectors')
    if planner is None:
        planner = configured_teleop_planner()
        planner.enable([0, 1], qpos, previous)
        planner.aligned[1] = True
    command, _ = planner.step(prediction, qpos, previous, dt)
    command[:7] = qpos[:7]
    command[13] = (np.clip(prediction[13], 0, 1)
                   if allow_gripper and planner.aligned[1] else previous[13])
    return command


def command_deadline(observation, prediction_origin, now=None):
    """Validate at send time and bound transport by the input freshness limits."""
    now = time.perf_counter() if now is None else now
    measured_state(observation, now)
    if not np.isfinite(prediction_origin) or now < prediction_origin:
        raise ValueError('Invalid prediction timestamp')
    if now-prediction_origin >= .25:
        return None
    return float(min(now+.1, prediction_origin+.25,
                     np.min(observation['_act_state_time'])+.2,
                     np.min(observation['_act_gripper_time'])+.3))
