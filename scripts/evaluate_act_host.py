"""Evaluate ACT prior inference on the held-out episodes and fixed windows."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_act_host import make_policy


class EvaluationFrames(Dataset):
    def __init__(self, directory, episode_ids, stats):
        self.directory = directory
        self.stats = stats
        self.frames = []
        self.anchors = set()
        for episode in episode_ids:
            with h5py.File(directory / f'episode_{episode}.hdf5', 'r') as data:
                length = len(data['action'])
            self.frames.extend((episode, frame) for frame in range(length))
            for frame in sorted({0, length // 4, length // 2, 3 * length // 4, length - 1}):
                self.anchors.add((episode, frame))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        from dobot_control.act_dataset import load_act_window
        episode, frame = self.frames[index]
        images, qpos, actions, padding = load_act_window(
            self.directory / f'episode_{episode}.hdf5', frame, 30)
        image_tensor = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).float() / 255
        normalized_qpos = (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        return image_tensor, normalized_qpos, qpos, actions, padding, episode, frame


def summarize(prediction, target):
    error = np.abs(prediction - target)
    left = error[:, 7:13]
    right = error[:, :6]
    return dict(
        left_joint_mae_deg=float(np.rad2deg(left).mean()),
        left_joint_p95_deg=float(np.rad2deg(np.percentile(left, 95))),
        left_joint_per_joint_mae_deg=np.rad2deg(left.mean(axis=0)).tolist(),
        left_gripper_mae=float(error[:, 13].mean()),
        left_gripper_binary_accuracy=float(((prediction[:, 13] >= .5) == (target[:, 13] >= .5)).mean()),
        right_joint_mae_deg=float(np.rad2deg(right).mean()),
        right_gripper_mae=float(error[:, 6].mean()),
        valid_actions=len(target))


def transitions(signal):
    return np.flatnonzero(np.diff((signal >= .5).astype(np.int8))) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((args.run / 'config.json').read_text())
    directory = Path(settings['dataset'])
    sys.path.insert(0, str(directory / 'training_support'))
    split = json.loads((args.run / 'splits.json').read_text())
    stats = {key: np.asarray(value, dtype=np.float32)
             for key, value in json.loads((args.run / 'normalization.json').read_text()).items()}
    dataset = EvaluationFrames(directory, split['validation'], stats)
    loader = DataLoader(dataset, batch_size=8, num_workers=4, pin_memory=True)
    policy = make_policy(settings['policy'], args.run)
    mean = stats['action_mean']
    std = stats['action_std']
    results = {}
    plot_data = {}
    for name in ['best', 'last']:
        checkpoint = torch.load(args.run / f'{name}.pt', map_location='cpu', weights_only=False)
        policy.load_state_dict(checkpoint['policy'])
        step = checkpoint['step']
        del checkpoint
        policy.eval()
        predictions, targets, baselines, episodes, frames = [], [], [], [], []
        anchor_prediction, anchor_target, anchor_baseline = [], [], []
        with torch.inference_mode():
            for images, normalized, qpos, actions, padding, episode, frame in loader:
                # Future actions are used only as targets after the model call.
                predicted = policy(normalized.cuda(), images.cuda()).cpu().numpy() * std + mean
                truth = actions.numpy()
                measured = qpos.numpy()
                predictions.append(predicted[:, 0])
                targets.append(truth[:, 0])
                baselines.append(measured)
                episodes.extend(episode.tolist())
                frames.extend(frame.tolist())
                for index, pair in enumerate(zip(episode.tolist(), frame.tolist())):
                    if pair in dataset.anchors:
                        valid = ~padding[index].numpy()
                        anchor_prediction.append(predicted[index][valid])
                        anchor_target.append(truth[index][valid])
                        anchor_baseline.append(np.repeat(measured[index][None], valid.sum(), axis=0))
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        baseline = np.concatenate(baselines)
        episode_array = np.asarray(episodes)
        per_episode = {}
        for episode in split['validation']:
            selection = episode_array == episode
            per_episode[str(episode)] = dict(
                **summarize(prediction[selection], target[selection]),
                target_gripper_transition_frames=transitions(target[selection, 13]).tolist(),
                predicted_gripper_transition_frames=transitions(prediction[selection, 13]).tolist())
        results[name] = dict(
            checkpoint_step=step,
            first_action_all_frames=summarize(prediction, target),
            fixed_50_window_actions=summarize(np.concatenate(anchor_prediction), np.concatenate(anchor_target)),
            per_episode=per_episode)
        np.savez_compressed(args.output / f'{name}_predictions.npz', prediction=prediction,
                            target=target, qpos=baseline, episode=episode_array, frame=frames)
        plot_data[name] = prediction
        if name == 'best':
            results['hold_current_qpos'] = dict(
                first_action_all_frames=summarize(baseline, target),
                fixed_50_window_actions=summarize(np.concatenate(anchor_baseline), np.concatenate(anchor_target)))
        print(name, json.dumps({k: v for k, v in results[name].items() if k != 'per_episode'}), flush=True)
    results['protocol'] = dict(validation_episodes=split['validation'], frames=len(dataset),
                               fixed_windows=len(dataset.anchors), inference='policy(qpos,image), no future actions',
                               padding='excluded from window metric denominator',
                               gripper_threshold=.5, joint_unit='degree', gripper_unit='0..1',
                               predictions_clipped=False)
    (args.output / 'evaluation.json').write_text(json.dumps(results, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    worst = max(results['last']['per_episode'], key=lambda e: results['last']['per_episode'][e]['left_joint_mae_deg'])
    for episode in sorted({split['validation'][0], int(worst)}):
        selection = episode_array == episode
        time = np.asarray(frames)[selection] / 15
        figure, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        for axis, dimension in zip(axes, range(7, 14)):
            scale = 180 / np.pi if dimension < 13 else 1
            axis.plot(time, target[selection, dimension] * scale, label='Demonstration', color='black')
            for name in ['best', 'last']:
                axis.plot(time, plot_data[name][selection, dimension] * scale, label=name, alpha=.8)
            axis.set_ylabel(f'Left J{dimension-6} (deg)' if dimension < 13 else 'Gripper 0..1')
            axis.grid(alpha=.25)
        axes[0].legend()
        axes[-1].set_xlabel('Time (s)')
        figure.suptitle(f'Episode {episode}: observed-frame one-step predictions')
        figure.tight_layout()
        figure.savefig(args.output / f'episode_{episode}_curves.png', dpi=140)
        plt.close(figure)
    print('EVALUATION_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
