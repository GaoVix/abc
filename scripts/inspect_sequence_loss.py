"""Inspect one original training sequence, including all its retained segments.

Uses the training data transforms and model.forward loss without optimizer updates.
Run from the repository root with: python -m scripts.inspect_sequence_loss --help
"""

import argparse
import copy
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from abc_minimal.config import ClipConfig, DiTConfig, FlowConfig, PromptConfig
from abc_minimal.dataloader import EpisodeDataset, collate, decode_frame, read_state_action_rows
from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained
from abc_minimal.kinematics import EEFPoseFK
from abc_minimal.operator import load_operator_label_maps
from abc_minimal.preprocess import augment_and_normalize, normalize, parse_norm_stats
from abc_minimal.train_loop import batch_to_device


class SequenceViews(Dataset):
    """Decode each observation once, then make clean and training-augmented views."""

    def __init__(self, dataset, episode=None, stride=1, max_chunks=0):
        self.clean = dataset
        groups = {}
        for record in dataset.episodes:
            folder = record[0]
            metadata = json.loads((folder / 'episode_metadata.json').read_text())
            source = metadata.get('sequence_ik', {}).get('source_episode', folder.name)
            groups.setdefault(source, []).append((record, metadata))
        matches = [name for name, entries in groups.items()
                   if episode is None or episode in name
                   or any(episode in r[0].name for r, _ in entries)]
        if len(matches) != 1 and episode is not None:
            raise ValueError(f'--episode must identify one original sequence; matches: {matches[:10]}')
        self.episode = sorted(matches)[0]
        selected = sorted(groups[self.episode], key=lambda entry: entry[0][0].name)
        dataset.episodes = [record for record, _ in selected]
        dataset.cum = np.cumsum([record[2] for record in dataset.episodes])
        self.augmented = copy.copy(dataset)
        self.augmented.train = True
        self.chunks = []
        for segment, (record, metadata) in enumerate(selected):
            source_start = metadata.get('source_frame_range', [0])[0]
            for frame in range(0, record[2], stride):
                self.chunks.append((segment, frame, source_start + frame))
        if max_chunks:
            self.chunks = self.chunks[:max_chunks]
        self.metadata = [metadata for _, metadata in selected]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        segment, frame, source_frame = self.chunks[index]
        folder, length, _, cameras, task, timeline = self.clean.episodes[segment]
        config = self.clean.model_config
        rows = read_state_action_rows(folder, frame, frame + config.chunk_length, config)
        images = decode_frame(folder, frame, length, cameras, config.camera_keys)
        views = []
        for dataset in (self.clean, self.augmented):
            state = normalize(rows[0, :config.state_dim], dataset.norm_stats['state']).astype(np.float32)
            masked = bool(dataset.train and torch.rand(1).item() < dataset.mask_state_ratio)
            if masked:
                state = np.zeros_like(state)
            views.append(dict(
                state=torch.from_numpy(state),
                actions=torch.from_numpy(normalize(
                    rows[:, config.state_dim:], dataset.norm_stats['actions']).astype(np.float32)),
                images=augment_and_normalize(images, dataset.train,
                                             norm_preset=dataset.norm_preset, image_size=dataset.image_size),
                state_is_masked=masked,
                prompt=dataset.build_prompt(folder, frame, task, timeline),
            ))
        return views[0], views[1], (segment, frame, source_frame)


def collate_views(samples):
    clean, augmented, locations = zip(*samples)
    cameras = tuple(clean[0]['images'])
    return collate(clean, cameras), collate(augmented, cameras), locations


@torch.no_grad()
def reconstruction_metrics(prediction, target, action_stats, fk):
    """Per-chunk metrics; joint deltas are direct radian differences, without wrapping."""
    scale = torch.as_tensor(action_stats['std'], device=target.device) + 1e-6
    difference = (prediction - target) * scale
    positions, rotations = fk(torch.stack((prediction, target)))
    distance_mm = (positions[0] - positions[1]).norm(dim=-1) * 1000
    relative = rotations[0].double().transpose(-1, -2) @ rotations[1].double()
    sine = (relative - relative.transpose(-1, -2)).norm(dim=(-1, -2)) / (8 ** 0.5)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    angle_deg = torch.atan2(sine, cosine).rad2deg()
    metrics = {'recon_mse_normalized': (prediction - target).square().mean(dim=(1, 2))}
    for arm, side in enumerate(('left', 'right')):
        joints = difference[..., 7 * arm:7 * arm + 6]
        metrics.update({
            f'{side}_joint_rmse_rad': joints.square().mean(dim=(1, 2)).sqrt(),
            f'{side}_joint_mae_rad': joints.abs().mean(dim=(1, 2)),
            f'{side}_joint_max_rad': joints.abs().amax(dim=(1, 2)),
            f'{side}_gripper_rmse': difference[..., 7 * arm + 6].square().mean(dim=1).sqrt(),
            f'{side}_eef_position_rmse_mm': distance_mm[..., arm].square().mean(dim=1).sqrt(),
            f'{side}_eef_position_mean_mm': distance_mm[..., arm].mean(dim=1),
            f'{side}_eef_position_max_mm': distance_mm[..., arm].amax(dim=1),
            f'{side}_eef_rotation_mean_deg': angle_deg[..., arm].mean(dim=1),
            f'{side}_eef_rotation_max_deg': angle_deg[..., arm].amax(dim=1),
        })
        for joint in range(6):
            metrics[f'{side}_j{joint+1}_rmse_rad'] = joints[..., joint].square().mean(dim=1).sqrt()
    return {key: value.cpu().numpy() for key, value in metrics.items()}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(output, chunks, batches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 1, figsize=(13, 19), constrained_layout=True)
    for key in ('train_flow_loss', 'val_flow_loss'):
        axes[0].plot([r['chunk_index'] for r in batches], [r[key] for r in batches], label=key)
    plots = [
        ('Normalized action reconstruction MSE', ['recon_mse_normalized']),
        ('Joint RMSE (rad)', [f'{s}_joint_rmse_rad' for s in ('left', 'right')]),
        ('Gripper RMSE (opening fraction)', [f'{s}_gripper_rmse' for s in ('left', 'right')]),
        ('EEF position RMSE (mm)', [f'{s}_eef_position_rmse_mm' for s in ('left', 'right')]),
        ('EEF rotation mean error (deg)', [f'{s}_eef_rotation_mean_deg' for s in ('left', 'right')]),
    ]
    axes[0].set_title('Flow loss: batch means, averaged over noise/time/prefix draws')
    for axis, (title, keys) in zip(axes[1:], plots):
        for key in keys:
            axis.plot([r[key] for r in chunks], label=key)
        axis.set_title(title)
    boundaries = [i for i in range(1, len(chunks)) if chunks[i]['segment'] != chunks[i-1]['segment']]
    for axis in axes:
        for boundary in boundaries:
            axis.axvline(boundary, color='gray', alpha=0.4, linestyle='--')
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_xlabel('Retained action chunk index (original source frame is recorded in chunks.csv)')
    figure.savefig(output / 'metrics.png', dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--split', choices=('train_real', 'train_sim'), default='train_real')
    parser.add_argument('--episode', help='Original sequence UUID/name; default: first sequence in sorted order')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--draws', type=int, default=4, help='Flow-loss noise/time/prefix draws per batch')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-chunks', type=int, default=0, help='0 evaluates all valid chunks')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--grad-norm', action='store_true', help='Compute gradient norm on the first augmented batch only')
    args = parser.parse_args()
    if min(args.batch_size, args.draws, args.stride) < 1 or min(args.num_workers, args.max_chunks) < 0:
        parser.error('batch-size, draws and stride must be positive; workers and max-chunks must be nonnegative')
    import matplotlib  # Fail before loading the checkpoint if the plotting dependency is missing.

    started = time.perf_counter()
    device = torch.device(f'cuda:{args.gpu_id}')
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    saved = checkpoint['train_config']
    config, flow = DiTConfig(**saved['model']), FlowConfig(**saved['flow'])
    prompt, clip = PromptConfig(**saved['prompt']), ClipConfig(**saved['clip'])
    if config.state_dim != 14 or config.action_dim != 14:
        raise ValueError('This report expects the 14D dual-arm joint policy')
    stats = parse_norm_stats(checkpoint['norm_stats'])
    cache_root = args.cache_root or Path(saved['cache_root'])
    task = ('sim_' if args.split == 'train_sim' else '') + 'load_the_plates_into_the_dish_rack'
    labels = load_operator_label_maps(prompt.operator_label_map_path) if prompt.use_operator_id_as_prompt else {}
    dataset = EpisodeDataset(cache_root / args.split, stats, False, task, flow.mask_state_ratio,
                             config, prompt_config=prompt, operator_label_maps=labels)
    sequence = SequenceViews(dataset, args.episode, args.stride, args.max_chunks)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(checkpoint=str(args.checkpoint.resolve()), global_step=checkpoint['global_step'],
                      cache_root=str(cache_root.resolve()), split=args.split, episode=sequence.episode,
                      segments=[str(r[0]) for r in dataset.episodes], chunks=len(sequence),
                      seed=args.seed, flow=vars(flow), dino_bf16=saved['dino_bf16'],
                      checkpoint_learning_rates=[g['lr'] for g in checkpoint['optimizer']['param_groups']],
                      sequence_ik=[m.get('sequence_ik') for m in sequence.metadata])
    (args.output / 'input.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: provenance[k] for k in ('checkpoint', 'global_step', 'episode', 'chunks')}, indent=2), flush=True)
    model = DiTPolicy(config)
    load_pretrained(model, args.checkpoint)
    model.img_backbone.set_bfloat16(saved['dino_bf16'])
    model.to(device)
    del checkpoint
    embedder = CLIPTextEmbedder(clip, device=device)
    fk = EEFPoseFK(stats['actions'], device)
    loader = DataLoader(sequence, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=collate_views, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed), multiprocessing_context='spawn'
                        if args.num_workers else None)
    flow_settings = {key: getattr(flow, key) for key in
                     ('max_action_prefix', 'prefix_conditioning_prob', 'prefix_noise_scale')}
    chunk_rows, batch_rows, predictions, targets, states = [], [], [], [], []
    gradient_norm = None
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (clean, augmented, locations) in enumerate(loader):
        clean = batch_to_device(clean, device, embedder, config.camera_keys)
        augmented = batch_to_device(augmented, device, embedder, config.camera_keys)
        train_losses, val_losses = [], []
        model.train()
        for draw in range(args.draws):
            backward = args.grad_norm and batch_index == 0 and draw == 0
            with torch.set_grad_enabled(backward):
                loss = model(augmented, **flow_settings)
                train_losses.append(loss.item())
                if backward:
                    loss.backward()
                    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')))
                    if not np.isfinite(gradient_norm):
                        raise ValueError('Nonfinite gradient norm in the first training batch')
                    model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            for _ in range(args.draws):
                val_losses.append(model(clean, max_action_prefix=0, prefix_conditioning_prob=0.).item())
            prediction = model.sample_actions(clean, num_steps=flow.num_diffusion_steps)
            metrics = reconstruction_metrics(prediction, clean['actions'], stats['actions'], fk)
        for collected, values, key in ((predictions, prediction, 'actions'),
                                       (targets, clean['actions'], 'actions'),
                                       (states, clean['state'], 'state')):
            collected.append(values.detach().cpu().float().numpy() * (stats[key]['std'] + 1e-6)
                             + stats[key]['mean'])
        row = dict(batch=batch_index, chunk_index=len(chunk_rows), chunks=len(locations),
                   train_flow_loss=float(np.mean(train_losses)), val_flow_loss=float(np.mean(val_losses)),
                   train_flow_draw_std=float(np.std(train_losses)), val_flow_draw_std=float(np.std(val_losses)),
                   masked_states=int(augmented['state_is_masked'].sum()))
        batch_rows.append(row)
        for i, (segment, frame, source_frame) in enumerate(locations):
            chunk_rows.append(dict(segment=int(segment), frame=int(frame), source_frame=int(source_frame),
                                   **{key: float(value[i]) for key, value in metrics.items()}))
        if not all(np.isfinite(v) for v in (*train_losses, *val_losses)) or not torch.isfinite(prediction).all():
            raise ValueError(f'Nonfinite loss/prediction in batch {batch_index}, frames {locations}')
        print(f'{len(chunk_rows)}/{len(sequence)} chunks: train_loss={row["train_flow_loss"]:.6f} '
              f'val_loss={row["val_flow_loss"]:.6f} recon={metrics["recon_mse_normalized"].mean():.6f}', flush=True)
    aggregate = {}
    for key in metrics:
        values = np.array([r[key] for r in chunk_rows])
        aggregate[key] = float(values.max() if '_max_' in key else
                               np.sqrt(np.mean(values**2)) if '_rmse' in key else values.mean())
    summary = dict(**provenance, metrics=aggregate,
                   train_flow_loss=float(np.mean([r['train_flow_loss'] for r in batch_rows])),
                   val_flow_loss=float(np.mean([r['val_flow_loss'] for r in batch_rows])),
                   first_batch_gradient_norm=gradient_norm, peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30,
                   seconds=time.perf_counter()-started,
                   notes=['Training flow loss uses training augmentation/state masking and random prefixes.',
                          f'Validation-style metrics use clean inputs, no prefix and {flow.num_diffusion_steps} sampling steps.',
                          'Losses are evaluated at fixed checkpoint weights; these are not historical training logs.',
                          'Flow losses average batch means. Reconstruction metrics weight every chunk equally.',
                          'Overlapping chunks are counted separately, as in training; segment gaps are never crossed.',
                          'EEF scalar errors use the training validation FK; joint deltas are unwrapped radians.',
                          'Gripper error is in opening fraction units; only arm joints use radians.'])
    write_csv(args.output / 'chunks.csv', chunk_rows)
    write_csv(args.output / 'batches.csv', batch_rows)
    np.savez_compressed(args.output / 'predictions.npz', predicted_actions=np.concatenate(predictions),
                        target_actions=np.concatenate(targets), states=np.concatenate(states),
                        source_frames=[r['source_frame'] for r in chunk_rows],
                        segments=[r['segment'] for r in chunk_rows], normalized=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    plot_metrics(args.output, chunk_rows, batch_rows)
    print(json.dumps(summary['metrics'], indent=2))
    print(f'Saved to {args.output}; {summary["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
