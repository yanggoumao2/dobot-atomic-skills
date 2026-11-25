import time
import sys
from unittest.mock import patch

import numpy as np
import pytest

from dobot_control.act_runtime import measured_state, safe_command, command_deadline
from dobot_control.act_session import ACTSession


def command_args(token, **changes):
    args = dict(joint_state=np.zeros(14), arm_flags=[0,1], gripper_flags=[0,1],
                _act_token=token, _act_deadline=time.perf_counter()+.1)
    return dict(args, **changes)


def test_legacy_and_exclusive_act_ownership():
    session = ACTSession()
    legacy = dict(joint_state=np.zeros(14), arm_flags=[1,1])
    assert session.authorize('command_joint_state', legacy) is not legacy
    token = session.begin()['token']
    with pytest.raises(RuntimeError):
        session.authorize('command_joint_state', legacy)
    with pytest.raises(RuntimeError):
        session.authorize('set_do_status', {})
    assert '_act_token' not in session.authorize('command_joint_state', command_args(token))
    with pytest.raises(RuntimeError):
        session.begin()
    with pytest.raises(RuntimeError):
        session.end('wrong')
    session.end(token)
    session.authorize('command_joint_state', legacy)


def test_expired_and_wrong_side_commands_rejected():
    session = ACTSession()
    token = session.begin()['token']
    with pytest.raises(RuntimeError):
        session.authorize('command_joint_state', command_args(token, _act_deadline=0))
    with pytest.raises(ValueError):
        session.authorize('command_joint_state', command_args(token, arm_flags=[1,1]))
    with pytest.raises(ValueError):
        session.authorize('command_joint_state', command_args(token, gripper_flags=[1,1]))


def test_right_arm_masked_and_left_uses_teleop_follow_speed():
    qpos = np.zeros(14)
    target = np.ones(14)
    target[7:13] = np.deg2rad(5)
    actual = safe_command(qpos,target,qpos,.03)
    np.testing.assert_array_equal(actual[:7],qpos[:7])
    np.testing.assert_allclose(actual[7:13], np.deg2rad(5))
    assert actual[13] == 0
    assert safe_command(qpos,target,qpos,.03,True)[13] == pytest.approx(1.)


@pytest.mark.parametrize('target', [np.full(14,np.nan), np.zeros(13)])
def test_invalid_or_large_prediction_rejected(target):
    with pytest.raises(ValueError):
        safe_command(np.zeros(14),target,np.zeros(14),.03)


@pytest.mark.parametrize('dt', [.01, 1/30, .1])
def test_motion_matches_teleop_for_large_targets_and_follower_lag(dt):
    from dobot_control.absolute_teleop import configured_teleop_planner
    qpos = np.zeros(14)
    previous = qpos.copy()
    previous[7:13] = np.deg2rad(4)
    prediction = np.zeros(14)
    prediction[7:13] = np.deg2rad([9, -10, 20, -30, 45, -60])
    planner = configured_teleop_planner()
    planner.enable([0,1], qpos, previous)
    planner.aligned[1] = True
    expected, _ = planner.step(prediction, qpos, previous, dt)
    result = safe_command(qpos, prediction, previous, dt)
    np.testing.assert_allclose(result[7:13], expected[7:13])
    assert np.max(np.abs(result[7:13]-previous[7:13])) > np.deg2rad(10)*min(dt,.04)


def test_act_reads_the_same_operator_motion_overrides(monkeypatch):
    monkeypatch.setenv('DOBOT_SERVOJ_MAX_SPEED_RAD_S', '.8')
    monkeypatch.setenv('DOBOT_SERVOJ_MAX_STEP_RAD', '.01')
    result = safe_command(np.zeros(14), np.ones(14), np.zeros(14), .04, True)
    np.testing.assert_allclose(result[7:13], .01)
    assert result[13] == 1


def test_act_uses_teleop_alignment_before_gripper_enable():
    from dobot_control.absolute_teleop import configured_teleop_planner
    planner = configured_teleop_planner()
    qpos = np.zeros(14)
    planner.enable([0, 1], qpos, qpos)
    target = np.ones(14)
    result = safe_command(qpos, target, qpos, 1/30, True, planner)
    np.testing.assert_allclose(result[7:13], .15/30)
    assert result[13] == 0


def test_feedback_age_and_validity():
    obs = dict(dataset_joint_positions=np.zeros(14), _act_state_time=[1,1],
               _act_gripper_time=[1,1], _act_gripper_valid=[True,True])
    measured_state(obs,1.1)
    with pytest.raises(ValueError):
        measured_state(obs,1.4)
    obs['_act_gripper_valid'] = [True,False]
    with pytest.raises(ValueError):
        measured_state(obs,1.1)


def test_robot_server_act_route_blocks_legacy_write():
    import pickle
    import zmq
    from dobot_control.robots.robot_node import ZMQServerRobot
    from unittest.mock import Mock
    robot = Mock()
    robot.command_joint_state.return_value = 1
    with patch('dobot_control.robots.robot_node.zmq.Context'):
        server = ZMQServerRobot(robot,port=0)
    server._socket.close()
    token = server._act_session.begin()['token']
    packets = [dict(method='command_joint_state',args=command_args(token)),
               dict(method='command_joint_state',args=dict(joint_state=np.zeros(14),arm_flags=[1,1]))]
    replies = []
    def receive():
        if packets:
            return pickle.dumps(packets.pop(0))
        server.stop()
        raise zmq.Again()
    server._socket = Mock()
    server._socket.recv.side_effect = receive
    server._socket.send.side_effect = lambda data: replies.append(pickle.loads(data))
    server.serve()
    assert robot.command_joint_state.call_count == 1
    assert replies[0] == 1
    assert '__robot_rpc_error__' in replies[1]
    server._context.term()


@pytest.fixture
def nova_fk():
    # Load the production pure FK functions without importing hardware modules.
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'experiments/run_control.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'claw_width', 'dh_transformation_matrix', 'forward_kinematics', 'calculate_vel_pos', 'limit_tcp_descent', 'check_joint_safety', 'validate_absolute_alignment'}
    from dobot_control.absolute_teleop import AbsoluteTeleopPlanner
    from dobot_control.teleop_safety import is_within_workspace
    scope = {'np': np, 'AbsoluteTeleopPlanner': AbsoluteTeleopPlanner,
             'is_within_workspace': is_within_workspace, 'runtime_log': lambda *a, **k: None}
    exec(compile(ast.Module(body=[node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name in names],
                            type_ignores=[]), str(path), 'exec'), scope)
    tree = ast.parse((path.parents[1]/'scripts/manipulate_utils.py').read_text(encoding='utf-8'))
    exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                 and node.name == 'servo_action_check'], type_ignores=[]), str(path), 'exec'), scope)
    scope['calculate_vel_pos'].teleop_functions = scope
    return scope['calculate_vel_pos']


def startup_command():
    # Rounded observation/prediction preceding the 20261116_150927_a976d6 fault.
    qpos = np.zeros(14)
    qpos[7:13] = np.deg2rad([-90, 0, -90.011, -.005, 90, 90])
    qpos[13] = .996078431372549
    prediction = qpos.copy()
    prediction[7:13] += np.deg2rad([-.087, 2.913, -1.544, -1.083, .873, .250])
    return qpos, safe_command(qpos, prediction, qpos, 1/30)


def test_deadline_starts_at_send_and_respects_input_expiry():
    observation = dict(dataset_joint_positions=np.zeros(14), _act_state_time=[1, 1],
                       _act_gripper_time=[1, 1], _act_gripper_valid=[True, True])
    # Preparation used 120ms, leaving the original now+.1 deadline expired.
    assert command_deadline(observation, 1., 1.12) == pytest.approx(1.2)
    with pytest.raises(ValueError, match='Stale'):
        command_deadline(observation, 1., 1.21)
    observation['_act_state_time'] = [1.2, 1.2]
    assert command_deadline(observation, .9, 1.2) is None
    for origin in [np.nan, np.inf, 1.3]:
        with pytest.raises(ValueError, match='timestamp'):
            command_deadline(observation, origin, 1.2)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows live CLI requires foreground-key gate; Linux RPC/data flow is exercised separately')
@pytest.mark.parametrize('model_latency', [.005, .12])
@pytest.mark.parametrize('allow_gripper', [False, True])
def test_full_execute_rpc_pause_resume_and_restart(tmp_path, monkeypatch, nova_fk, model_latency, allow_gripper):
    """Normal CLI/live loop over real ZMQ with synthetic cameras/model/robot."""
    import importlib.util
    import json
    import sys
    import threading
    import types
    from pathlib import Path
    from dobot_control.robots.robot_node import ZMQServerRobot
    path = Path(__file__).resolve().parents[1] / 'tools/act_real_test.py'
    spec = importlib.util.spec_from_file_location('act_test_integration', path)
    platform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(platform)
    monkeypatch.setitem(sys.modules, 'experiments.run_control',
                        types.SimpleNamespace(**{name: nova_fk.teleop_functions[name] for name in
                            ['calculate_vel_pos', 'limit_tcp_descent', 'check_joint_safety',
                             'validate_absolute_alignment', 'servo_action_check']}))
    qpos, _ = startup_command()

    class SimulatedRobot:
        def __init__(self):
            self.qpos = qpos.copy()
            self.commands = []

        def get_observations(self):
            now = time.perf_counter()
            return dict(dataset_joint_positions=self.qpos.copy(), _act_state_time=[now, now],
                        _act_gripper_time=[now, now], _act_gripper_valid=[True, True])

        def command_joint_state(self, joint_state, arm_flags, gripper_flags):
            assert arm_flags == [0, 1] and gripper_flags in ([0, 0], [0, int(allow_gripper)])
            np.testing.assert_array_equal(joint_state[:7], qpos[:7])
            if allow_gripper:
                assert 0 <= joint_state[13] <= 1
            else:
                assert joint_state[13] == qpos[13]
            self.qpos = joint_state.copy()
            self.commands.append(joint_state.copy())
            return 1

    class SimulatedModel:
        def __init__(self, url):
            pass

        def health(self):
            return dict(protocol=1, camera_names=['top', 'left'], image_encoding='rgb8', step=20000)

        def predict(self, joints, images, request_id):
            time.sleep(model_latency)
            target = qpos.copy()
            target[8] += np.deg2rad(.5)
            target[13] = .4
            return np.tile(target, (30, 1))

    class SimulatedCameras:
        def __init__(self):
            self.frames = 0

        def snapshot(self):
            self.frames += 1
            origin = time.perf_counter() - (.26 if self.frames <= 8 else 0)
            return [np.zeros((480, 640, 3), dtype=np.uint8)] * 2, origin

        def close(self):
            pass

    robot = SimulatedRobot()
    server = ZMQServerRobot(robot, port=0)
    import zmq
    port = int(server._socket.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(':', 1)[1])
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    monkeypatch.setattr(platform, 'ModelClient', SimulatedModel)
    monkeypatch.setattr(platform, 'Cameras', SimulatedCameras)
    monkeypatch.setattr(platform, 'ROOT', tmp_path/'repo')
    monkeypatch.setenv('DOBOT_DEVICE_LOCK_PATH', str(tmp_path/'runtime/device-owner'))
    monkeypatch.setattr(platform.cv2, 'getWindowProperty', lambda *a: 1)
    monkeypatch.setattr(platform.cv2, 'destroyAllWindows', lambda: None)
    from unittest.mock import Mock
    monkeypatch.setattr(platform.cv2, 'VideoWriter', Mock())
    try:
        for _ in range(2):
            start_count = len(robot.commands)
            state = {'phase': 'initial'}

            def space(title):
                return state['phase'] in ('held', 'resumed')

            def display(images, lines, title):
                count = len(robot.commands)-start_count
                key = 255
                if state['phase'] == 'initial':
                    key, state['phase'] = ord('r'), 'held'
                elif state['phase'] == 'held' and count >= 3:
                    state['phase'] = 'release'
                elif state['phase'] == 'release':
                    key, state['phase'] = ord('p'), 'paused'
                elif state['phase'] == 'paused':
                    key, state['phase'] = ord('r'), 'resumed'
                elif state['phase'] == 'resumed' and count >= 6:
                    key = 27
                return key, np.zeros((700, 1280, 3), dtype=np.uint8)

            monkeypatch.setattr(platform, 'focused_space', space)
            monkeypatch.setattr(platform, 'display', display)
            argv = ['act_real_test.py', '--mode', 'execute', '--seconds', '3', '--robot-port', str(port)]
            if allow_gripper:
                argv.append('--allow-gripper')
            monkeypatch.setattr(sys, 'argv', argv)
            platform.main()
            assert len(robot.commands)-start_count >= 6
            assert server._act_session.token is None
            assert not (tmp_path/'runtime/act-client-owner').exists()
        logs = list((tmp_path/'act_real_tests').glob('*/events.jsonl'))
        assert len(logs) == 2
        for log in logs:
            events = [json.loads(line) for line in log.read_text().splitlines()]
            assert not any(event.get('event') == 'fault' for event in events)
            assert any(event.get('armed') and event.get('age_s', 0) > .25
                       and event.get('command') is None for event in events)
            commands = [event for event in events if event.get('command') is not None]
            assert len(commands) >= 6
            assert all(event['control']['prepare_ms'] >= 0 for event in commands)
        if allow_gripper:
            assert any(command[13] != qpos[13] for command in robot.commands)
    finally:
        server.stop()
        thread.join(timeout=2)
        server._socket.close()
        server._context.term()
