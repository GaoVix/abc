"""Compute train-only rotvec normalization and initialize two fresh 12->12 MLPs.

Position and joint normalization are retained from the reference adapters.
Only the 3D rotation mean/std are newly estimated. No dataset is rewritten and
neither MLP is pretrained: they will learn jointly with the original ABC DiT.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from abc_minimal.eef_adapters import EEFKinematics
from scripts.train_eef_adapters import EEFAdapters, JOINT_STATE_INDICES, load_adapters


class Moments:
    def __init__(self):
        self.count = 0
        self.mean = np.zeros(12)
        self.m2 = np.zeros(12)

    def update(self, values):
        count = len(values)
        mean = values.mean(0)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += ((values - mean) ** 2).sum(0) + delta ** 2 * self.count * count / total
        self.mean += delta * count / total
        self.count = total

    def std(self):
        if self.count < 2:
            raise ValueError("Need at least two training frames for rotation normalization")
        return np.sqrt(self.m2 / (self.count - 1))


def prepare_adapters(cache_root, reference_path, output, batch_size=4096, seed=123):
    cache_root, output = Path(cache_root), Path(output)
    stats_path = output.with_suffix(".json")
    if output.exists() or stats_path.exists() or output.is_symlink() or stats_path.is_symlink():
        raise FileExistsError(f"Choose a new output path; refusing to replace {output} or {stats_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    episodes = []
    for split in ("train_real", "train_sim"):
        paths = sorted((cache_root / split).glob("*/states_actions.bin"))
        if not paths:
            raise ValueError(f"No training episodes found under {cache_root / split}")
        episodes.extend(paths)
    reference_checkpoint = torch.load(reference_path, map_location="cpu", weights_only=False, mmap=True)
    reference = load_adapters(reference_checkpoint)
    fk = EEFKinematics(reference_checkpoint["config"]["camera_transform"], rotation_dim=3)
    moments, jump_count = Moments(), 0
    started = time.monotonic()
    for index, path in enumerate(episodes, 1):
        if path.stat().st_size == 0 or path.stat().st_size % (28 * 8):
            raise ValueError(f"Expected nonempty float64 joint state14 + action14 rows: {path}")
        rows = np.memmap(path, dtype=np.float64, mode="r").reshape(-1, 28)
        previous = None
        for offset in range(0, len(rows), batch_size):
            joints = rows[offset:offset + batch_size, 14:][:, JOINT_STATE_INDICES]
            if not np.isfinite(joints).all():
                raise ValueError(f"Nonfinite joint actions: {path} offset={offset}")
            poses = fk.fk(joints).astype(np.float64)
            if not np.isfinite(poses).all():
                raise ValueError(f"Nonfinite FK poses: {path} offset={offset}")
            moments.update(poses)
            rotations = poses.reshape(-1, 2, 6)[..., 3:]
            sequence = rotations if previous is None else np.concatenate((previous, rotations))
            jump_count += int((np.linalg.norm(np.diff(sequence, axis=0), axis=-1) > np.pi).sum())
            previous = rotations[-1:]
        if index == 1 or index % 25 == 0 or index == len(episodes):
            print(f"[{index}/{len(episodes)}] action_frames={moments.count} elapsed={time.monotonic() - started:.1f}s",
                  flush=True)

    mean, std = moments.mean.reshape(2, 6), moments.std().reshape(2, 6)
    mean[:, :3] = reference.pose_mean.reshape(2, -1)[:, :3].numpy()
    std[:, :3] = reference.pose_std.reshape(2, -1)[:, :3].numpy()
    torch.manual_seed(seed)
    model = EEFAdapters(reference.state_mean, reference.state_scale, mean.ravel(), std.ravel())
    config = {"camera_transform": fk.camera_transform, "rotation_dim": 3,
              "initialization": "fresh_random_mlps", "seed": seed}
    report = {"mean": mean.ravel().tolist(), "std": std.ravel().tolist(),
              "training_action_frames": moments.count, "rotation_statistics": "train_real + train_sim, ddof=1",
              "position_and_joint_statistics": str(reference_path), "rotation_representation": "principal rotvec, radians",
              "rotvec_coordinate_jumps_gt_pi": jump_count, "config": config}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "step": 0}, output)
    stats_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved fresh MLP_in 12->12 and MLP_out 12->12: {output}", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--reference-adapters", type=Path, required=True,
                        help="Original adapters supplying camera calibration and unchanged position/joint statistics")
    parser.add_argument("--output", type=Path, required=True, help="New initial MLP checkpoint, e.g. output/eef_rotvec/init.pt")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    torch.set_num_threads(2)
    prepare_adapters(args.cache_root, args.reference_adapters, args.output, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
