"""Precompute camera poses and fixed directions beside states_actions.bin.

Run after state conversion/alignment, before training. RGB is not decoded or
rewritten. Calibration contributes fixed mounts, bases, K and distortion D.
Distortion inversion and direction-table resize/letterbox run offline only.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

from abc_minimal.camera_geometry import CameraGeometry
from abc_minimal.camera_parameters import prepare_camera_parameters

GEOMETRY = None


def initialize(calibration_path, model_dir):
    global GEOMETRY
    # Each worker handles offline inversions; avoid CPU thread oversubscription.
    import torch
    torch.set_num_threads(1)
    GEOMETRY = CameraGeometry(calibration_path, model_dir)


def prepare(job):
    episode, overwrite = job
    written = prepare_camera_parameters(episode, GEOMETRY, overwrite=overwrite)
    return f"{'written' if written else 'unchanged'}: {episode}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dirs", nargs="+", type=Path,
                        help="Processed episode, split, or dataset roots; searched recursively")
    parser.add_argument("--calibration-path", type=Path,
                        help="Complete accepted three-camera calibrations.json.gz; omit for sim-only data")
    parser.add_argument("--model-dir", type=Path, help="Original robot XMLs for unretargeted real states; mesh assets are not needed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate after changing calibration, bases or states")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for root in args.data_dirs:
        if not root.is_dir():
            parser.error(f"Data directory does not exist: {root}")
    # Calibrated subsets can link entire episodes to the official export.
    episodes = sorted({Path(directory).resolve() for root in args.data_dirs
                       for directory, _, files in os.walk(root, followlinks=True)
                       if "states_actions.bin" in files})
    if not episodes:
        parser.error("No processed states_actions.bin found in the input directories")
    jobs = [(episode, args.overwrite) for episode in episodes]
    if args.workers == 1:
        initialize(args.calibration_path, args.model_dir)
        for job in jobs:
            print(prepare(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(args.calibration_path, args.model_dir),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for result in pool.map(prepare, jobs, chunksize=1):
                print(result, flush=True)


if __name__ == "__main__":
    main()
