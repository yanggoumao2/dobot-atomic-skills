"""Laptop ACT preview and explicitly enabled physical-left-arm trials."""
import argparse
import base64
import ctypes
import json
import os
from pathlib import Path
import pickle
import queue
import sys
import threading
import time
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import zmq
from dobot_control.act_runtime import measured_state, safe_command, command_deadline
from dobot_control.cameras.settings import load_camera_settings
from dobot_control.teleop_safety import check_pose_protection, MasterRunawayGuard
from dobot_control.absolute_teleop import configured_teleop_planner
from dobot_control.device_lease import DeviceLease


class RobotRPC:
    def __init__(self, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 500)
        self.socket.setsockopt(zmq.SNDTIMEO, 500)
        self.socket.connect(f'tcp://127.0.0.1:{port}')

    def call(self, method, **args):
        self.socket.send(pickle.dumps({'method':method, 'args':args}))
        result = pickle.loads(self.socket.recv())
        if isinstance(result, dict) and '__robot_rpc_error__' in result:
            raise RuntimeError(result['__robot_rpc_error__'])
        return result

    def close(self):
        self.socket.close()
        self.context.term()


class ModelClient:
    def __init__(self, url):
        self.url = url.rstrip('/')
        if self.url not in ('http://127.0.0.1:8787', 'http://localhost:8787'):
            raise ValueError('Use the loopback SSH tunnel on port8787')
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self):
        with self.opener.open(self.url + '/health', timeout=3) as response:
            return json.load(response)

    def predict(self, qpos, images, request_id):
        encoded = []
        for rgb in images:
            if rgb.shape != (480,640,3) or rgb.dtype != np.uint8:
                raise ValueError('Expected 480x640 RGB uint8')
            encoded.append(base64.b64encode(np.ascontiguousarray(rgb).tobytes()).decode('ascii'))
        payload = dict(qpos=np.asarray(qpos).tolist(), camera_names=['top','left'],
                       rgb_images=encoded, request_id=request_id)
        request = urllib.request.Request(self.url+'/predict', json.dumps(payload).encode(),
                                         {'Content-Type':'application/json'})
        with self.opener.open(request, timeout=.5) as response:
            result = json.load(response)
        if result['request_id'] != request_id:
            raise ValueError('Inference request ID mismatch')
        action = np.asarray(result['action'], dtype=float)
        if action.shape != (30,14) or not np.isfinite(action).all():
            raise ValueError('Invalid model output')
        return action


class InferenceWorker:
    def __init__(self, client):
        self.client = client
        self.pending = queue.Queue(maxsize=1)
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        self.busy = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, packet):
        # Do not queue an observation behind an in-flight HTTP request.
        # The next idle submission will contain newer images and joint state.
        if self.busy.is_set():
            return False
        self.busy.set()
        try:
            self.pending.put_nowait(packet)
        except queue.Full:
            self.busy.clear()
            return False
        return True

    def run(self):
        while not self.stop.is_set():
            try:
                packet = self.pending.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                action = self.client.predict(packet['qpos'], packet['images'], packet['id'])
                self.latest = dict(packet, action=action, received=time.perf_counter())
                self.error = None
            except Exception as error:
                self.latest = None
                self.error = str(error)
            finally:
                self.busy.clear()


class Cameras:
    def __init__(self):
        from dobot_control.cameras.realsense_camera import RealSenseCamera
        settings = load_camera_settings()
        if list(settings.serials) != ['top','left'] or settings.fps != 15:
            raise ValueError('Configure CAMERA.enabled=top,left and fps=15')
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frames = {}
        self.error = None
        self.threads = []
        for name in ['top','left']:
            def capture(name=name):
                camera = None
                try:
                    camera = RealSenseCamera(device_id=settings.serials[name], fps=15,
                                             flip=settings.flips[name], enable_depth=False)
                    while not self.stop.is_set():
                        image, _ = camera.read()
                        stamp = time.perf_counter()
                        with self.lock:
                            self.frames[name] = (image.copy(), stamp)
                except Exception as error:
                    self.error = str(error)
                finally:
                    if camera is not None:
                        camera.close()
            thread = threading.Thread(target=capture, daemon=True)
            thread.start()
            self.threads.append(thread)

    def snapshot(self):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            if len(self.frames) != 2:
                return None
            frames = [self.frames[name] for name in ['top','left']]
        stamps = np.array([x[1] for x in frames])
        now = time.perf_counter()
        if np.any(now-stamps > .2) or np.ptp(stamps) > .1:
            raise RuntimeError('Camera frames stale or out of sync')
        return [x[0] for x in frames], float(stamps.min())

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)


def focused_space(title):
    if os.name != 'nt':
        return False
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    handle = user32.GetForegroundWindow()
    text = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(ctypes.c_void_p(handle), text, len(text))
    return text.value == title and bool(user32.GetAsyncKeyState(0x20) & 0x8000)


def display(images, lines, title):
    canvas = np.concatenate([cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in images], axis=1)
    canvas = cv2.copyMakeBorder(canvas, 0, 220, 0, 0, cv2.BORDER_CONSTANT)
    for index, line in enumerate(lines):
        cv2.putText(canvas, str(line)[:135], (12, 505+26*index),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (240,240,240), 1)
    cv2.imshow(title, canvas)
    return cv2.waitKey(1) & 0xff, canvas


def replay(args, client, log):
    import h5py
    with h5py.File(args.episode, 'r') as data:
        if json.loads(data.attrs['camera_names']) != ['top','left']:
            raise ValueError('Replay camera order mismatch')
        count = min(len(data['action']), args.frames)
        for index in range(count):
            qpos = data['observations/qpos'][index]
            images = [data['observations/images/'+name][index] for name in ['top','left']]
            begin = time.perf_counter()
            action = client.predict(qpos, images, str(index))
            error = np.rad2deg(np.abs(action[0,7:13]-data['action'][index,7:13])).mean()
            row = dict(frame=index, latency_s=time.perf_counter()-begin,
                       left_mae_deg=float(error), prediction=action[0].tolist())
            log.write(json.dumps(row)+'\n')
            if not args.headless:
                key, _ = display(images, ['REPLAY: recorded images, no robot connection',
                                        f'Frame {index} | left error {error:.2f} deg',
                                        'ESC: exit'], 'ACT replay')
                if key == 27:
                    break
            print(json.dumps({k:v for k,v in row.items() if k!='prediction'}), flush=True)


def live(args, client, output, log):
    from experiments.run_control import (calculate_vel_pos, limit_tcp_descent,
        check_joint_safety, validate_absolute_alignment, servo_action_check)
    robot = RobotRPC(args.robot_port)
    cameras = None
    worker = None
    token = None
    armed = False
    armed_at = 0.
    previous = None
    planner = configured_teleop_planner()
    target_guard = MasterRunawayGuard(
        max_speed_rad_s=float(os.environ.get('DOBOT_MASTER_DROP_SPEED_RAD_S', '12.5')),
        min_delta_rad=float(os.environ.get('DOBOT_MASTER_DROP_DELTA_RAD', '0.25')),
        required_violation_cycles=2)
    print(f'[ACT] teleop motion settings: align={planner.align_speed} rad/s, '
          f'follow={planner.follow_speed} rad/s, max_step={planner.max_step} rad', flush=True)
    state_text = 'Preview | R: prepare | hold SPACE: execute | P: pause | ESC: exit'
    title = 'ACT real test'
    start = time.perf_counter()
    period = 1/30
    writer = None
    last_video = 0.
    try:
        if args.mode == 'execute':
            protocol = robot.call('act_protocol')
            if protocol['protocol'] != 1:
                raise RuntimeError('Restart A using the current code')
            token = robot.call('begin_act_control')['token']
            (output/'session.json').write_text(json.dumps(dict(token=token, pid=os.getpid())))
        cameras = Cameras()
        worker = InferenceWorker(client)
        last_submit = 0.
        last_command = time.perf_counter()
        last_prediction_id = None
        last_fresh_prediction = 0.
        while time.perf_counter()-start < args.seconds:
            cycle = time.perf_counter()
            control_trace = {}
            try:
                snapshot = cameras.snapshot()
                if snapshot is None:
                    if cycle-start > 20:
                        raise RuntimeError('Camera startup timeout')
                    time.sleep(.02)
                    continue
                images, image_time = snapshot
                observation = robot.call('get_observations')
                now = time.perf_counter()
                control_trace.update(state_age_ms=((now-np.asarray(observation.get('_act_state_time', [])))*1000).tolist(),
                                     gripper_age_ms=((now-np.asarray(observation.get('_act_gripper_time', [])))*1000).tolist(),
                                     gripper_valid=np.asarray(observation.get('_act_gripper_valid', [])).tolist())
                qpos = measured_state(observation, now)
                origin = min(image_time, float(np.min(observation['_act_state_time'])))
                if now-last_submit >= 1/15:
                    if worker.submit(dict(qpos=qpos.copy(), images=images, origin=origin,
                                          id=uuid4().hex, observed=now)):
                        last_submit = now
                prediction = worker.latest
                command = None
                space = focused_space(title)
                if args.mode == 'execute' and armed and space:
                    if worker.error:
                        raise RuntimeError('Inference failed: '+worker.error)
                    if (prediction is not None and prediction['observed'] >= armed_at
                            and now-prediction['origin'] < .23):
                        last_fresh_prediction = now
                        dt = max(now-last_command, .001)
                        # Advance within the predicted chunk to account for transport/inference time.
                        action_index = min(29, int((now-prediction['observed'])*15))
                        control_trace.update(qpos=qpos.tolist(), previous=(qpos if previous is None else previous).tolist(), dt_s=dt,
                                             action_index=action_index, prediction_id=prediction['id'],
                                             prediction=prediction['action'][action_index].tolist(),
                                             prediction_delta_deg=np.rad2deg(prediction['action'][action_index,7:13]-qpos[7:13]).tolist(),
                                             age_s=now-prediction['origin'])
                        target = prediction['action'][action_index]
                        if previous is None:
                            valid, details = validate_absolute_alignment(qpos, target, 'Nova 2', [0, 1])
                            if not valid:
                                raise RuntimeError(f'Teleop alignment path rejected: {details}')
                            planner.reset()
                            previous = planner.enable([0, 1], qpos, qpos.copy())
                            target_guard.reset(target, now)
                        if target_guard.check(target, [0, 1], now).tripped:
                            raise RuntimeError('Target jump exceeds teleop runaway thresholds')
                        command = safe_command(qpos, target, previous, dt, args.allow_gripper, planner)
                        command, limited = limit_tcp_descent(command, previous, dt, 'Nova 2', [0, 1])
                        control_trace['teleop_tcp_descent_limited'] = limited
                        accepted, command = servo_action_check(command, previous, [0, 1])
                        if not accepted:
                            raise RuntimeError('Teleop servo step protection rejected command')
                        positions, velocities = calculate_vel_pos(command, previous, dt, 'Nova 2')
                        if (check_pose_protection(positions, velocities, [0, 1])
                                or check_joint_safety(command, [0, 1])):
                            raise RuntimeError('Teleop workspace/joint protection rejected command')
                        send_time = time.perf_counter()
                        control_trace.update(send_time=send_time, prepare_ms=(send_time-now)*1000,
                                             aligned=bool(planner.aligned[1]))
                        deadline = command_deadline(observation, prediction['origin'], send_time)
                        control_trace['deadline'] = deadline
                        if deadline is not None and deadline-send_time >= .02:
                            robot.call('command_joint_state', joint_state=command, arm_flags=[0,1],
                                       gripper_flags=[0,int(args.allow_gripper and planner.aligned[1])],
                                       _act_token=token, _act_deadline=deadline)
                            previous = command.copy()
                            last_command = send_time
                            state_text = ('EXECUTING' if planner.aligned[1] else 'ALIGNING') + ' - release SPACE to stop new commands'
                        else:
                            command = None
                            state_text = 'WAITING for fresh prediction - no command sent'
                    else:
                        previous = None
                        state_text = 'WAITING for fresh prediction - no command sent'
                        if now-max(armed_at, last_fresh_prediction) > 3:
                            raise RuntimeError('No fresh prediction for 3 seconds')
                else:
                    previous = None
                    if armed:
                        # A new hold must receive an observation after the key release.
                        armed_at = now
                age = None if prediction is None else now-prediction['origin']
                lines = [f'{args.mode.upper()} | '+state_text,
                         f'Model {args.health["step"]} | prediction age {age!s} | gripper writes {args.allow_gripper}',
                         f'LEFT .2 | Teleop follow {planner.follow_speed:g} rad/s | step {planner.max_step:g} rad',
                         'Measured LEFT deg: '+np.array2string(np.rad2deg(qpos[7:13]),precision=1),
                         'Predicted LEFT deg: '+('pending' if prediction is None else
                            np.array2string(np.rad2deg(prediction['action'][0,7:13]),precision=1)),
                         f'Gripper measured {qpos[13]:.3f} | inference error: {worker.error}',
                         'Use physical emergency stop for immediate hardware stop']
                key, canvas = display(images, lines, title)
                if writer is None:
                    writer = cv2.VideoWriter(str(output/'preview.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                             15, (canvas.shape[1],canvas.shape[0]))
                if now-last_video >= 1/15:
                    writer.write(canvas)
                    last_video = now
                if prediction is not None and (prediction['id'] != last_prediction_id or command is not None):
                    log.write(json.dumps(dict(time=now, qpos=qpos.tolist(), prediction=prediction['action'][0].tolist(),
                                               age_s=age, command=None if command is None else command.tolist(),
                                               action_index=None if command is None else action_index,
                                               armed=armed, space=space, control=control_trace))+'\n')
                    last_prediction_id = prediction['id']
                if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('r'),ord('R')) and args.mode == 'execute':
                    armed = True
                    armed_at = time.perf_counter()
                    previous = None
                    state_text = 'PREPARED - hold SPACE to send; release to stop new commands'
                if key in (ord('p'),ord('P')):
                    armed = False
                    state_text = 'PAUSED - press R to prepare again'
            except Exception as error:
                armed = False
                previous = None
                state_text = 'FAULT: '+str(error)
                log.write(json.dumps(dict(event='fault', message=str(error), time=time.perf_counter(),
                                          control=control_trace))+'\n')
                print(state_text, flush=True)
                # Transport errors leave REQ socket unusable. Exit rather than retry control.
                if isinstance(error, zmq.ZMQError):
                    raise
                if 'images' in locals():
                    key, _ = display(images, [state_text, 'P: pause / R: retry after correction / ESC: exit'], title)
                    if key == 27:
                        break
                time.sleep(.05)
            time.sleep(max(0, period-(time.perf_counter()-cycle)))
    finally:
        if worker:
            worker.stop.set()
        if cameras:
            cameras.close()
        if writer:
            writer.release()
        if token:
            try:
                robot.call('end_act_control', token=token)
            except Exception:
                print('ACT session retained by A. Use session.json recovery after inspecting the rig.', flush=True)
        robot.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['replay','shadow','execute'], default='shadow')
    parser.add_argument('--url', default='http://127.0.0.1:8787')
    parser.add_argument('--robot-port', type=int, default=6001)
    parser.add_argument('--episode', type=Path)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--allow-gripper', action='store_true')
    parser.add_argument('--label-run', type=Path)
    parser.add_argument('--release-session', type=Path)
    parser.add_argument('--result', choices=['success','failure','aborted'])
    parser.add_argument('--stage', choices=['grasp','align','insert','release','withdraw','complete'])
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    if args.release_session:
        session = json.loads(args.release_session.read_text())
        if os.name != 'nt':
            parser.error('Session recovery requires Windows')
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(session['pid']))
        if handle:
            code = ctypes.c_ulong()
            queried = kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
            kernel.CloseHandle(ctypes.c_void_p(handle))
            if not queried:
                parser.error('Cannot determine whether the recorded PID is alive')
            if code.value == 259:
                parser.error('Recorded client PID is still running; close that client first')
        elif kernel.GetLastError() != 87:
            parser.error('Cannot verify recorded PID exit; inspect permissions and process state')
        robot = RobotRPC(args.robot_port)
        try:
            print(robot.call('end_act_control',token=session['token']))
        finally:
            robot.close()
        return
    if args.label_run:
        if args.result is None or args.stage is None:
            parser.error('Label requires --result and --stage')
        (args.label_run/'outcome.json').write_text(json.dumps(dict(result=args.result, stage=args.stage,
                                                                 note=args.note),ensure_ascii=False,indent=2),encoding='utf-8')
        return
    if args.mode == 'replay' and not args.episode:
        parser.error('Replay requires --episode')
    if args.mode != 'replay' and (args.headless or os.name != 'nt'):
        parser.error('Live trials require a visible Windows preview')
    if args.frames < 1 or not 0 < args.seconds <= 120:
        parser.error('frames must be positive; seconds must be 0..120')
    client = ModelClient(args.url)
    args.health = client.health()
    if (args.health.get('protocol') != 1 or args.health.get('camera_names') != ['top','left']
            or args.health.get('image_encoding') != 'rgb8'):
        raise ValueError('Unsupported model service')
    output = ROOT.parent/'act_real_tests'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(vars(args),default=str,indent=2))
    print('TRIAL_OUTPUT',output,flush=True)
    with (output/'events.jsonl').open('w',encoding='utf-8',buffering=1) as log:
        if args.mode == 'replay':
            replay(args,client,log)
        else:
            lock = Path(os.environ.get('DOBOT_DEVICE_LOCK_PATH',ROOT.parent/'.local/runtime/device-owner'))
            with DeviceLease(lock.with_name('act-client-owner'), purpose='act-client'):
                live(args,client,output,log)


if __name__ == '__main__':
    main()
