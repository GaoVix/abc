"""Compare rotation3D/6D adapters on identical held-out physical joint samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from abc_minimal.preprocess import parse_norm_stats
from scripts.train_eef_adapters import CAMERA_TRANSFORM, REPO, PairGenerator, load_adapters, validate


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--dit-checkpoint", type=Path, default=REPO / "cache/abc_dit_xl_200k_model.pt")
    parser.add_argument("--camera-transform", type=Path, default=CAMERA_TRANSFORM)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the shared mjwarp FK sample generator")
    stats = parse_norm_stats(torch.load(args.dit_checkpoint, map_location="cpu", mmap=True, weights_only=False)["norm_stats"])
    generator = PairGenerator(stats, args.samples, "cuda", args.camera_transform)
    generator.rng = np.random.default_rng(args.seed)
    joints, _ = generator()
    report = {"num_samples": args.samples, "seed": args.seed,
              "distribution": "Official state per-joint independent Gaussian with coordinate rejection at joint limits",
              "shared_sample_results": {}}
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = load_adapters(checkpoint, "cuda")
        generator.rotation_dim = model.arm_pose_dim - 3
        target = generator.fk(joints)
        metrics = {"step": checkpoint.get("step"), "eef_dim_with_grippers": model.pose_dim + 2,
                   **validate(model, joints, target, generator.fk)}
        report["shared_sample_results"][str(path)] = metrics
        print(f"{path}: {json.dumps(metrics)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved shared-sample comparison: {args.output}", flush=True)


if __name__ == "__main__":
    main()
