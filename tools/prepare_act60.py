"""Package reviewed successful demonstrations with ten held-out validation episodes."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dobot_control.act_dataset import convert_episode
from scripts.act_training_adapter import training_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--since', default='20261116')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--train-count', type=int, default=50)
    parser.add_argument('--top-mask-x', type=int, default=371,
                        help='Black out top image columns x..639; 640 keeps full image')
    args = parser.parse_args()
    if args.train_count < 1:
        parser.error('--train-count must be positive')
    if not 1 <= args.top_mask_x <= 640:
        parser.error('--top-mask-x must be in 1..640')
    candidates = []
    for directory in args.raw:
        for episode in sorted(directory.iterdir()):
            if not episode.is_dir() or episode.name < args.since:
                continue
            metadata = episode / 'episode.json'
            if metadata.exists():
                meta = json.loads(metadata.read_text(encoding='utf-8'))
                if meta.get('status') == 'success':
                    candidates.append((episode, meta))
    if len({p.name for p, _ in candidates}) != len(candidates):
        parser.error('Duplicate source episode names across raw directories')
    if len(candidates) < 11:
        parser.error('Need at least 11 reviewed successes to reserve 10 validation episodes')
    cache = {}
    if args.cache:
        manifest = json.loads((args.cache / 'dataset_manifest.json').read_text(encoding='utf-8'))
        cache = {entry['source']: args.cache / entry['file'] for entry in manifest['episodes']}
    args.output.mkdir(parents=True, exist_ok=False)
    entries, rejected = [], []
    preprocess = dict(top_mask_x=args.top_mask_x, fill_rgb=[0, 0, 0],
                      width=640, height=480, version=1)
    reference = None
    for episode, meta in candidates:
        if len(entries) >= args.train_count + 10:
            break
        profile = [meta.get(k) for k in ('config_sha256', 'camera_names', 'channel_order')]
        if reference is None:
            reference = profile
        if profile != reference:
            raise ValueError(f'Incompatible recording profile: {episode}')
        target = args.output / f'episode_{len(entries)}.hdf5'
        if episode.name in cache:
            with h5py.File(cache[episode.name], 'r') as cached:
                cached_preprocess = json.loads(cached.attrs.get('image_preprocess', '{}'))
            cached_x = cached_preprocess.get('top_mask_x', 640)
            if cached_x < 640 and cached_x != args.top_mask_x:
                raise ValueError('Masked cache has a different boundary; use raw data or a full-image cache')
            shutil.copy2(cache[episode.name], target)
        else:
            try:
                convert_episode(episode, target, fps=15)
            except ValueError as exc:
                rejected.append(dict(source=episode.name, error=str(exc)))
                print('REJECTED', episode.name, flush=True)
                continue
        with h5py.File(target, 'r+') as data:
            data.attrs['metadata_json'] = json.dumps(meta, ensure_ascii=False)
            data.attrs['image_preprocess'] = json.dumps(preprocess)
            images = data['observations/images/top']
            if args.top_mask_x < 640:
                for index in range(len(images)):
                    image = images[index]
                    image[:, args.top_mask_x:] = 0
                    images[index] = image
            frames = len(data['action'])
        entries.append(dict(file=target.name, source=episode.name, raw=str(episode),
                            metadata=meta, frames=frames))
        print('PACKAGED', len(entries), episode.name, flush=True)
    if len(entries) < 11:
        raise ValueError('Fewer than 11 eligible episodes after conversion')
    ids = np.random.default_rng(0).permutation(len(entries)).tolist()
    split = dict(seed=0, train=ids[10:], validation=ids[:10])
    for name, value in [('splits.json', split), ('image_preprocess.json', preprocess),
            ('dataset_manifest.json', dict(schema=2, fps=15, episodes=entries,
                excluded=rejected, camera_names=['top', 'left'],
                action_alignment='same_row_no_shift', image_preprocess=preprocess))]:
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    stats = training_stats(args.output, split['train'])
    (args.output / 'train_normalization_stats.json').write_text(
        json.dumps({key: value.tolist() for key, value in stats.items()}, indent=2), encoding='utf-8')
    support = args.output / 'training_support' / 'dobot_control'
    support.mkdir(parents=True)
    for name in ['__init__.py', 'act_dataset.py', 'act_recording.py']:
        shutil.copy2(ROOT / 'dobot_control' / name, support / name)
    scripts = args.output / 'training_scripts'
    scripts.mkdir()
    for name in ['train_act_host.py', 'evaluate_act_host.py', 'act_training_adapter.py']:
        shutil.copy2(ROOT / 'scripts' / name, scripts / name)
    summary = dict(successes=len(entries), train=len(split['train']), validation=10,
                   labeled_success_candidates=len(candidates),
                   target_train=args.train_count, target_validation=10,
                   additional_successes_needed=max(0, args.train_count+10-len(entries)),
                   ready_for_target=len(split['train']) >= args.train_count, top_mask_x=args.top_mask_x)
    (args.output / 'preparation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
