"""Collection task metadata; never controls robot enable/follow state."""
import configparser
from dobot_control.cameras.settings import DEFAULT_PATH


def load_act_task(path=DEFAULT_PATH):
    config = configparser.ConfigParser()
    with open(path, encoding='utf-8') as handle:
        config.read_file(handle)
    task = config.get('ACT_TASK', 'task', fallback='bottle_cap_open')
    if task == 'bottle_cap_open':
        return {'task': task}
    fixture = config.get('ACT_TASK', 'fixture', fallback='narrow_mouth_cup')
    if task != 'test_tube_insert' or fixture != 'narrow_mouth_cup':
        raise ValueError('Unsupported ACT task/fixture')
    cameras = [n.strip() for n in config.get('ACT_TASK', 'camera_names', fallback='top,left,right').split(',')]
    arms = config.get('ACT_TASK', 'active_arms', fallback='both')
    if cameras not in (['top','left'], ['top','left','right']) or arms not in ('left','both'):
        raise ValueError('Unsupported ACT camera/active arm profile')
    return dict(task=task, fixture=fixture, task_protocol_version=2 if cameras == ['top','left'] or arms == 'left' else 1,
                required_camera_names=cameras,
                required_arm_flags=[arms != 'left', True],
                required_gripper_flags=[arms != 'left', True],
                max_camera_skew_s=.10,
                completion='insert, release gripper, withdraw left arm; tube remains stable')
