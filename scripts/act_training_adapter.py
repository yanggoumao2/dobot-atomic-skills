"""Copy alongside ACT on training server; replaces its dataset loading only."""
import json
from pathlib import Path

import numpy as np

from dobot_control.act_dataset import load_act_window


def training_stats(directory, episode_ids):
    import h5py
    directory = Path(directory)
    if not episode_ids:
        raise ValueError('empty training split')
    stats = {}
    for key, name in [('observations/qpos', 'qpos'), ('action', 'action')]:
        count, total, square = 0, np.zeros(14), np.zeros(14)
        for i in episode_ids:
            with h5py.File(directory / f'episode_{i}.hdf5', 'r') as root:
                values = root[key][()].astype(np.float64)
            count += len(values)
            total += values.sum(0)
            square += np.square(values).sum(0)
        mean = total / count
        stats[name + '_mean'] = mean.astype(np.float32)
        stats[name + '_std'] = np.sqrt(np.maximum(square / count - mean ** 2, .0001)).astype(np.float32)
    return stats


def load_data(directory, batch_size=8, chunk_size=30):
    import torch
    from torch.utils.data import Dataset, DataLoader
    directory = Path(directory)
    split = json.loads((directory / 'splits.json').read_text(encoding='utf-8'))
    stats = training_stats(directory, split['train'])

    class ACTDataset(Dataset):
        def __init__(self, ids):
            self.ids = ids

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, index):
            import h5py
            path = directory / f'episode_{self.ids[index]}.hdf5'
            with h5py.File(path, 'r') as root:
                start = np.random.randint(len(root['action']))
            image, qpos, action, mask = load_act_window(path, start, chunk_size)
            return (torch.from_numpy(image.transpose(0, 3, 1, 2).copy()).float() / 255.,
                    torch.from_numpy((qpos - stats['qpos_mean']) / stats['qpos_std']),
                    torch.from_numpy((action - stats['action_mean']) / stats['action_std']),
                    torch.from_numpy(mask))

    train = DataLoader(ACTDataset(split['train']), batch_size=batch_size, shuffle=True, num_workers=0)
    val = DataLoader(ACTDataset(split['validation']), batch_size=batch_size, num_workers=0)
    return train, val, stats, False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    train, val, stats, _ = load_data(args.directory)
    image, qpos, action, mask = next(iter(train))
    print('image/qpos/action/mask:', image.shape, qpos.shape, action.shape, mask.shape)
    print('train/validation episodes:', len(train.dataset), len(val.dataset))
