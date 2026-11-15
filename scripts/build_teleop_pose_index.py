"""Cache real teleop joints, base-local FK poses, and an EEF nearest-neighbor index."""

import argparse
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import scipy

if __package__:
    from .score_grid_teleop import ROOT, TeleopPoseIndex, arm_model, fk_poses, real_joints
else:
    from score_grid_teleop import ROOT, TeleopPoseIndex, arm_model, fk_poses, real_joints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--real-data', type=Path, nargs='+')
    source.add_argument('--from-index', type=Path, help='Reuse saved joints and FK poses')
    source.add_argument('--source-list', type=Path, help='Reuse episode paths from an index sources.json')
    parser.add_argument('--side', choices=('left', 'right'), default='left')
    parser.add_argument('--stream', choices=('state', 'action', 'both'), default='both')
    parser.add_argument('--limit-margin-deg', type=float, default=10)
    parser.add_argument('--model', type=Path, default=ROOT / 'abc_sim/models/yam_dishrack_base.xml')
    parser.add_argument('--position-scale', type=float, default=0.01)
    parser.add_argument('--rotation-scale', type=float, default=0.1)
    parser.add_argument('--cell-size', type=float, default=0.01)
    parser.add_argument('--rotation-cell-size', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory')
    if min(args.position_scale, args.rotation_scale, args.cell_size, args.rotation_cell_size, args.workers) <= 0:
        parser.error('Scales, cell size and workers must be positive')
    if args.from_index:
        started = time.perf_counter()
        metadata = json.loads((args.from_index / 'metadata.json').read_text())
        expected = dict(side=args.side, stream=args.stream, limit_margin_deg=args.limit_margin_deg,
                        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())
        if metadata['status'] != 'complete' or any(metadata[name] != value for name, value in expected.items()):
            parser.error('Saved joint/FK cache settings differ from requested settings')
        joints = np.load(args.from_index / 'joints.npy', mmap_mode='r', allow_pickle=False)
        poses = np.load(args.from_index / 'poses.npy', mmap_mode='r', allow_pickle=False)
        print(f'Reusing {len(joints):,} saved joint/FK pairs; building 6D pose cells', flush=True)
        index = TeleopPoseIndex(joints, poses, args.position_scale, args.rotation_scale,
                               args.cell_size, args.rotation_cell_size)
        metadata.update(reused_from=str(args.from_index.resolve()),
                        index_seconds=time.perf_counter() - started)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'sources.json').write_text((args.from_index / 'sources.json').read_text())
        index.save(args.output, metadata)
        print(f'Complete: {args.output}, {len(joints):,} samples, {time.perf_counter() - started:.1f}s', flush=True)
        return
    paths = []
    for path in args.real_data or ():
        files = [path] if path.is_file() else sorted(path.rglob('states_actions.bin'))
        if not files:
            files = sorted(path.glob('*.npz'))
        if not files:
            files = sorted(path.rglob('episode.mcap'))
        paths.extend(files)
    if args.source_list:
        paths = [Path(item['path']) for item in json.loads(args.source_list.read_text())]
    paths = sorted(set(path.resolve() for path in paths))
    if not paths:
        parser.error('No real episode files found')
    started = time.perf_counter()
    print(f'Reading {len(paths)} sources: {args.side}, {args.stream}', flush=True)
    arrays, counts = [], []
    read = partial(real_joints, side=args.side, stream=args.stream)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for number, values in enumerate(pool.map(read, paths, chunksize=4), 1):
            arrays.append(values)
            counts.append(len(values))
            if number % 100 == 0 or number == len(paths):
                print(f'Read {number}/{len(paths)}, elapsed={time.perf_counter() - started:.1f}s', flush=True)
    joints = np.concatenate(arrays)
    del arrays
    if joints.shape[1] != 6 or not np.isfinite(joints).all():
        parser.error('Real joints must be a finite (N,6) array')
    read_seconds = time.perf_counter() - started
    model = arm_model(args.model, args.side)
    margin = np.deg2rad(args.limit_margin_deg)
    lower, upper = model.jnt_range[:, 0] - margin, model.jnt_range[:, 1] + margin
    clipped = int(((joints < lower) | (joints > upper)).sum())
    np.clip(joints, lower, upper, out=joints)
    print(f'Samples={len(joints)}, clipped joint values={clipped}; starting FK', flush=True)
    tick = time.perf_counter()
    poses = fk_poses(model, joints, f'{args.side}_tcp_site')
    fk_seconds = time.perf_counter() - tick
    print(f'FK complete in {fk_seconds:.1f}s; building index', flush=True)
    tick = time.perf_counter()
    index = TeleopPoseIndex(joints, poses, args.position_scale, args.rotation_scale,
                           args.cell_size, args.rotation_cell_size)
    index_seconds = time.perf_counter() - tick
    args.output.mkdir(parents=True, exist_ok=True)
    offsets = np.r_[0, np.cumsum(counts)]
    sources = [dict(path=str(path), start=int(begin), end=int(end))
               for path, begin, end in zip(paths, offsets[:-1], offsets[1:])]
    (args.output / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    metadata = dict(side=args.side, stream=args.stream, source_count=len(paths),
                    model=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    limit_margin_deg=args.limit_margin_deg, reference_limits_rad=np.column_stack((lower, upper)).tolist(),
                    clipped_real_joint_values=clipped, scipy_version=scipy.__version__,
                    sampling='native recorded messages for MCAP; existing samples for exported/NumPy data',
                    pose_metric='sqrt((position/position_scale)^2 + (2*sin(rotation_angle/2)/rotation_scale)^2)',
                    joint_metric='Euclidean joint radians; clipped to expanded limits before FK; no normalization',
                    read_seconds=read_seconds, fk_seconds=fk_seconds, index_seconds=index_seconds)
    print('Saving joints, poses and cell index', flush=True)
    index.save(args.output, metadata)
    print(f'Complete: {args.output}, {len(joints):,} samples, {time.perf_counter() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
