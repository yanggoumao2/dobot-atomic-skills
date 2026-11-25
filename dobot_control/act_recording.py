"""ACT telemetry helpers; no robot commands or training dependencies."""
import json
import threading
import time
from pathlib import Path

import numpy as np


class GripperFeedback:
    """Serialize read/write transactions; observation reads use only this cache."""

    def __init__(self, gripper, stop):
        self.gripper, self.stop = gripper, stop
        self.value, self.timestamp, self.error = float('nan'), 0.0, 'not sampled'
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            try:
                # Wait briefly behind a write instead of skipping an entire
                # 100ms sampling period each time the serial bus is occupied.
                if not self.gripper.command_lock.acquire(timeout=.03):
                    self.stop.wait(.01)
                    continue
                try:
                    raw = self.gripper.get_current_position()
                finally:
                    self.gripper.command_lock.release()
                if not 0 <= raw <= 255:
                    raise ValueError(f'gripper position out of range: {raw}')
                with self.lock:
                    self.value, self.timestamp, self.error = raw / 255., time.perf_counter(), ''
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
            self.stop.wait(.1)

    def snapshot(self):
        with self.lock:
            return self.value, self.timestamp, not self.error


class CameraSnapshots:
    def __init__(self):
        self.lock, self.frames = threading.Lock(), {}

    def publish(self, index, jpeg, received):
        with self.lock:
            previous = self.frames.get(index, (None, 0, 0))
            self.frames[index] = (jpeg, received, previous[2] + 1)

    def snapshot(self, indices):
        with self.lock:
            return tuple(self.frames[i] for i in indices)


class RecordingCadence:
    """Limit disk sampling without sleeping or changing robot control rate."""
    def __init__(self, fps=15.):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('recording fps must be positive')
        self.period = 1. / fps
        self.reset()

    def reset(self):
        self.next_time = None

    def ready(self, now):
        return self.next_time is None or now + 1e-9 >= self.next_time

    def advance(self, now):
        if self.next_time is None:
            self.next_time = now + self.period
        else:
            self.next_time += max(1, int((now - self.next_time) / self.period) + 1) * self.period


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def enrich_observation(obs, frames, arm_flags, gripper_flags, command_time):
    """Copy pre-command measurements before control code mutates its own obs."""
    result = {k: np.asarray(v).copy() for k, v in obs.items()}
    result['joint_positions'] = result.pop('dataset_joint_positions')
    result['_act_schema'] = np.asarray(2)
    result['_act_camera_time'] = np.array([f[1] for f in frames])
    result['_act_camera_id'] = np.array([f[2] for f in frames])
    result['_act_arm_flags'] = np.asarray(arm_flags).copy()
    result['_act_gripper_flags'] = np.asarray(gripper_flags).copy()
    result['_act_command_time'] = np.asarray(command_time)
    return result


def validate_camera_packet(meta, frames, capture_time):
    names = meta.get('camera_names', [])
    if (not names or len(set(names)) != len(names)
            or any(name not in ('top', 'left', 'right') for name in names)):
        raise ValueError('invalid camera names')
    required = meta.get('required_camera_names')
    if required is not None and names != required:
        raise ValueError(f'task requires camera order {required}')
    if len(frames) != len(names) or any(not frame[0] for frame in frames):
        raise ValueError('missing camera image')
    times = np.asarray([frame[1] for frame in frames], dtype=float)
    age = capture_time - times
    if not np.isfinite(capture_time) or not np.isfinite(times).all() or np.any(age < 0) or np.any(age > .2):
        raise ValueError('camera stale/future')
    if 'max_camera_skew_s' in meta:
        limit = float(meta['max_camera_skew_s'])
        if not np.isfinite(limit) or not 0 < limit <= .2:
            raise ValueError('invalid camera skew limit')
        if np.ptp(times) > limit:
            raise ValueError('camera skew exceeds limit')
