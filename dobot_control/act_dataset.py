"""Validated raw-v2 -> ACT HDF5 conversion. No torch dependency."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from dobot_control.act_recording import atomic_json, validate_camera_packet


def inspect_episode(episode, fps=15.):
    episode = Path(episode)
    errors, warnings, rows = [], [], []
    meta_path = episode / 'episode.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    paths = sorted((episode / 'observation').glob('*.pkl'), key=lambda p: int(p.stem))
    if not paths:
        errors.append('no samples')
    if meta.get('schema') != 2:
        errors.append('requires raw schema 2; legacy telemetry cannot recover measured gripper/time')
    closed_path = episode / 'closed.json'
    closed = json.loads(closed_path.read_text(encoding='utf-8')) if closed_path.exists() else {}
    if not closed.get('closed') or closed.get('samples') != len(paths) or closed.get('aborted'):
        errors.append('episode not cleanly closed or saved sample count mismatch')
    names = meta.get('camera_names', ['top'])
    if not names or len(set(names)) != len(names) or any(n not in ('top','left','right') for n in names):
        errors.append('invalid camera names')
        names = []
    if meta.get('task') == 'test_tube_insert':
        supported = (['top','left'], ['top','left','right'])
        if names not in supported or meta.get('required_camera_names') != names:
            errors.append('tube insertion camera profile mismatch')
        if meta.get('max_camera_skew_s') != .10:
            errors.append('tube insertion requires protocol camera skew limit 0.10s')
    for path in paths:
        try:
            with path.open('rb') as f:
                row = pickle.load(f)  # Trusted local robot recordings only.
            if int(row.get('_act_schema', 0)) != 2:
                raise ValueError('missing schema 2')
            for key in ('joint_positions', 'control'):
                value = np.asarray(row[key])
                if value.shape != (14,) or not np.isfinite(value).all():
                    raise ValueError(f'{key}: expected finite 14-vector')
                if np.any(value[[6, 13]] < 0) or np.any(value[[6, 13]] > 1):
                    raise ValueError(f'{key}: gripper outside [0,1]')
            t = float(row['_act_command_time'])
            state = np.asarray(row['_act_state_time'])
            camera = np.asarray(row['_act_camera_time'])
            grip = np.asarray(row['_act_gripper_time'])
            for key, value, shape in [('state', state, (2,)), ('camera', camera, (len(names),)), ('gripper', grip, (2,))]:
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f'invalid {key} timestamps')
                age = t - value
                if np.any(age < 0) or np.any(age > (.35 if key == 'gripper' else .2)):
                    raise ValueError(f'{key} stale/future: age={age.tolist()}')
            if not np.asarray(row['_act_gripper_valid']).all() or not bool(row['_act_send_ok']):
                raise ValueError('feedback/send invalid')
            for key, requirement in (('_act_arm_flags','required_arm_flags'),
                                     ('_act_gripper_flags','required_gripper_flags')):
                flags = np.asarray(row[key])
                required = np.asarray(meta.get(requirement, [True, True]))
                if (flags.shape != (2,) or required.shape != (2,)
                        or not required[1] or not np.isin(flags, [0,1]).all()
                        or not np.isin(required, [0,1]).all()
                        or not flags[required.astype(bool)].all()):
                    raise ValueError(f'{key}: required arm/gripper must be enabled')
            if '_teleop_camera_names' in row and list(row['_teleop_camera_names']) != names:
                raise ValueError('row camera names differ from episode')
            validate_camera_packet(dict(meta, camera_names=names),
                [(b'present', t0, fid) for t0, fid in zip(camera, row['_act_camera_id'])], t)
            if np.asarray(row['_act_camera_id']).shape != (len(names),):
                raise ValueError('invalid camera frame IDs')
            if not np.isfinite(t) or not t <= float(row['_act_send_end']) <= t + .5:
                raise ValueError('invalid send timing')
            for name in names:
                image_path = episode / f'{name}Img' / f'{path.stem}.jpg'
                if not image_path.is_file():
                    raise ValueError(f'missing/invalid image {name}')
                image = cv2.imread(str(image_path))
                if image is None or image.shape != (480, 640, 3):
                    raise ValueError(f'missing/invalid image {name}')
            rows.append((int(path.stem), t, row))
        except Exception as exc:
            errors.append(f'{path.name}: {exc}')
    times = np.array([r[1] for r in rows])
    if len(times) < 2:
        errors.append('need at least two valid samples')
    elif np.any(np.diff(times) <= 0):
        errors.append('command timestamps must strictly increase')
    elif np.max(np.diff(times)) > .2:
        errors.append('command gap exceeds 0.2s; recollect or annotate a separate valid episode')
    else:
        for c in range(len(names)):
            ids = np.array([r[2]['_act_camera_id'][c] for r in rows])
            if np.any(np.diff(ids) < 0):
                errors.append('camera frame IDs decreased')
            if len(np.unique(ids)) < 2:
                errors.append('camera did not advance')
    gaps = sum(b[2].get('_teleop_source_index', b[0]) - a[2].get('_teleop_source_index', a[0]) > 1
               for a, b in zip(rows, rows[1:]))
    if gaps:
        warnings.append(f'{gaps} source gaps (time continuity checked)')
    telemetry = {}
    if rows:
        q = np.array([r[2]['joint_positions'] for r in rows])
        telemetry['gripper_range'] = np.ptp(q[:, [6, 13]], axis=0).tolist()
        telemetry['camera_unique_frames'] = [len(set(int(r[2]['_act_camera_id'][i]) for r in rows)) for i in range(len(names))]
        camera_times = np.array([r[2]['_act_camera_time'] for r in rows])
        telemetry['camera_age_p50_p95_max'] = np.quantile(times[:,None] - camera_times, [.5,.95,1], axis=0).T.tolist()
        telemetry['camera_skew_p50_p95_max'] = np.quantile(np.ptp(camera_times, axis=1), [.5,.95,1]).tolist()
        if len(times) > 1:
            telemetry['command_interval_p50_p95_max'] = np.quantile(np.diff(times), [.5, .95, 1]).tolist()
    report = dict(episode=episode.name, samples=len(paths), valid_samples=len(rows),
                  duration=float(times[-1] - times[0]) if len(times) else 0.,
                  passed=not errors, errors=errors, warnings=warnings,
                  status=meta.get('status', 'unreviewed'), fps=fps, telemetry=telemetry)
    return report, rows, meta


def convert_episode(episode, destination, fps=15.):
    import h5py
    if not np.isfinite(fps) or not 1 <= fps <= 15:
        raise ValueError('fps must be in [1,15] for current camera')
    report, rows, meta = inspect_episode(episode, fps)
    if not report['passed'] or meta.get('status') != 'success':
        raise ValueError(json.dumps(report, ensure_ascii=False))
    times = np.array([r[1] for r in rows])
    grid = np.arange(times[0], times[-1] + 1e-8, 1. / fps)
    # Causal zero-order hold. Never interpolate a gripper transition or use
    # future images/commands. Training uses exactly the same selected row.
    selected = np.searchsorted(times, grid, side='right') - 1
    if len(grid) < 2 or np.any(grid - times[selected] > .2):
        raise ValueError('insufficient continuous samples')
    data = [rows[i] for i in selected]
    qpos = np.array([r[2]['joint_positions'] for r in data], dtype=np.float32)
    actions = np.array([r[2]['control'] for r in data], dtype=np.float32)
    velocity = np.gradient(qpos, 1. / fps, axis=0).astype(np.float32)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temporary = target.with_suffix('.tmp')
    try:
        with h5py.File(temporary, 'w') as root:
            root.attrs.update(sim=False, compress=False, schema_version=2, fps=fps,
                              action_alignment='pre_command_observation_same_row_action',
                              source_episode=Path(episode).name,
                              metadata_json=json.dumps(meta, ensure_ascii=False))
            root.attrs['camera_names'] = json.dumps(meta['camera_names'])
            for key, value in [('observations/qpos', qpos), ('observations/qvel', velocity),
                               ('action', actions), ('timestamp', grid - grid[0]),
                               ('source_index', np.array([r[0] for r in data]))]:
                root.create_dataset(key, data=value)
            for name in meta['camera_names']:
                ds = root.create_dataset(f'observations/images/{name}', (len(data), 480, 640, 3),
                                         dtype='uint8', chunks=(1, 480, 640, 3), compression='lzf')
                for i, row in enumerate(data):
                    bgr = cv2.imread(str(Path(episode) / f'{name}Img' / f'{row[0]}.jpg'))
                    ds[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def load_act_window(path, start=0, chunk_size=30):
    """Training adapter core; no ALOHA real-data -1 shift. Padding is masked."""
    import h5py
    with h5py.File(path, 'r') as root:
        if root.attrs.get('action_alignment') != 'pre_command_observation_same_row_action':
            raise ValueError('unsupported action alignment')
        n = len(root['action'])
        if not 0 <= start < n or chunk_size < 1:
            raise ValueError('invalid window')
        count = min(chunk_size, n - start)
        action = np.zeros((chunk_size, 14), np.float32)
        action[:count] = root['action'][start:start + count]
        mask = np.arange(chunk_size) >= count
        if 'camera_names' in root.attrs:
            names = json.loads(root.attrs['camera_names'])
        else:
            names = json.loads(root.attrs.get('metadata_json', '{}')).get('camera_names')
            if names is None:
                import warnings
                warnings.warn('Legacy file has no camera order; using sorted names')
                names = sorted(root['observations/images'])
        if not names or len(set(names)) != len(names) or set(names) != set(root['observations/images']):
            raise ValueError('invalid HDF5 camera order')
        images = np.stack([root[f'observations/images/{name}'][start]
                           for name in names])
        return images, root['observations/qpos'][start], action, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'label', 'convert', 'preview'):
        p = sub.add_parser(name)
        p.add_argument('path', type=Path)
        if name == 'label':
            p.add_argument('--status', choices=['success', 'failure', 'unreviewed'], required=True)
            p.add_argument('--note', required=True, help='fixture, object, insertion depth, release/withdrawal outcome and recovery')
        else:
            p.add_argument('--report', type=Path)
            p.add_argument('--fps', type=float, default=15.)
        if name in ('convert', 'preview'):
            p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command != 'label' and (not np.isfinite(args.fps) or not 1 <= args.fps <= 15):
        parser.error('fps must be in [1,15]')
    if args.command == 'preview':
        paths = sorted((args.path / 'observation').glob('*.pkl'), key=lambda p:int(p.stem))
        times = []
        for path in paths:
            with path.open('rb') as f:
                times.append(float(pickle.load(f)['_teleop_capture_monotonic_s']))
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError('preview requires increasing timestamps and at least two samples')
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = args.path/'episode.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
        names = metadata.get('camera_names', ['top'])
        if not names or len(set(names)) != len(names) or any(n not in ('top','left','right') for n in names):
            raise ValueError('invalid preview camera names')
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (640*len(names),480))
        if not writer.isOpened():
            raise RuntimeError('cannot open MP4 writer')
        try:
            for t in np.arange(times[0], times[-1], 1/args.fps):
                i = np.searchsorted(times, t, side='right')-1
                images = []
                for name in names:
                    image = cv2.imread(str(args.path/f'{name}Img'/f'{paths[i].stem}.jpg'))
                    if image is None or image.shape != (480,640,3):
                        raise ValueError(f'missing/invalid image: {name}/{paths[i].stem}')
                    cv2.putText(image, f'{name} {t-times[0]:.2f}s sample={paths[i].stem}', (10,25),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1)
                    images.append(image)
                writer.write(np.concatenate(images, axis=1))
        finally:
            writer.release()
        print(args.output)
        return
    if args.command == 'label':
        path = args.path / 'episode.json'
        meta = json.loads(path.read_text(encoding='utf-8'))
        meta.update(status=args.status, note=args.note)
        atomic_json(path, meta)
        print(f'labeled {args.path.name}: {args.status}')
        return
    episodes = [args.path] if (args.path / 'observation').exists() else sorted(p for p in args.path.iterdir() if p.is_dir())
    reports, entries = [], []
    if args.command == 'convert' and args.output.exists():
        raise FileExistsError('output must be a new directory; use a new dataset version')
    if args.command == 'convert':
        args.output.mkdir(parents=True)
    for ep in episodes:
        report, _, meta = inspect_episode(ep, args.fps)
        reports.append(report)
        if args.command == 'convert' and report['passed'] and meta.get('status') == 'success':
            if entries and any(meta.get(k) != entries[0]['metadata'].get(k)
                               for k in ('config_sha256', 'camera_names', 'channel_order',
                                         'top_crop', 'top_flip', 'image_pipeline_version',
                                         'hardware_layout_version', 'gripper_ports', 'task', 'fixture',
                                         'task_protocol_version', 'camera_serials', 'camera_flips',
                                         'camera_positions', 'camera_fps', 'camera_depth_enabled', 'max_camera_skew_s',
                                         'required_camera_names', 'required_arm_flags', 'required_gripper_flags')):
                raise ValueError('incompatible configuration/camera/channel order/image geometry; export separate dataset versions')
            name = f'episode_{len(entries)}.hdf5' 
            convert_episode(ep, args.output / name, args.fps)
            entries.append(dict(file=name, source=ep.name, metadata=meta))
        print(f"{ep.name}: {'PASS' if report['passed'] else 'FAIL'}, status={report['status']}, errors={len(report['errors'])}")
    report_path = args.report or ((args.output if args.command == 'convert' else args.path) / 'quality_report.json')
    atomic_json(report_path, reports)
    import html
    report_path.with_suffix('.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>ACT data quality</title><h1>ACT 数据质量检查</h1><pre>'
        + html.escape(json.dumps(reports, ensure_ascii=False, indent=2)) + '</pre>', encoding='utf-8')
    if args.command == 'convert':
        rng = np.random.default_rng(0)
        ids = rng.permutation(len(entries)).tolist()
        nval = max(1, round(len(ids) * .2)) if len(ids) >= 2 else 0
        atomic_json(args.output / 'splits.json', dict(train=ids[nval:], validation=ids[:nval], seed=0))
        atomic_json(args.output / 'dataset_manifest.json', dict(schema=2, fps=args.fps, episodes=entries,
                    camera_names=entries[0]['metadata']['camera_names'] if entries else [],
                    action_alignment='same_row_no_shift', image_color='RGB', qvel='finite_difference',
                    normalization='compute only over train episodes; floor std at 0.01'))
        print(f'Exported {len(entries)} episodes to {args.output}')
        if not entries:
            raise SystemExit('No eligible episodes exported; inspect quality_report.json')
    elif any(not r['passed'] for r in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
