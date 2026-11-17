"""Compute a shared wrist pose with arithmetic translation and geodesic rotation means."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from abc_minimal.camera_parameters import average_camera_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--real-frames", required=True, type=int)
    parser.add_argument("--sim-frames", required=True, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    start = perf_counter()
    print(f"Sampling real={args.real_frames}, sim={args.sim_frames} frames; seed={args.seed}", flush=True)
    result = average_camera_pose(args.data_dirs, real_frames=args.real_frames,
                                 sim_frames=args.sim_frames, seed=args.seed)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(f"Geodesic average: {result['num_poses']} wrist poses, "
          f"{result['num_episodes']} episodes, {perf_counter() - start:.2f}s; saved {args.output}")


if __name__ == "__main__":
    main()
