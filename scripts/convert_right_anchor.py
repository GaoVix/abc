"""Retarget real left arms to the sim base spacing, preserving original right joints.

Run the existing top-100 DP -> Ceres -> equality IK, then the existing filtered
export. Targets are physical EEF poses in each sequence's original right base;
source joints are never clipped before FK. Sim binaries/RGB are linked into
independent episode directories so camera sidecars cannot overwrite other runs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from abc_minimal.camera_geometry import load_calibrations, pose, state_fk
from scripts.export_sequence_ik import save_json as write_json


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'abc_sim/models/yam_dishrack_base.xml'
CALIBRATIONS = MODEL_DIR = WORK = BASES = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target_bases():
    """World here is the right base; preserve the sim's full relative transform."""
    fk = state_fk(str(MODEL))
    fk.update(np.zeros(14))
    right_from_left = np.linalg.inv(pose(fk.data.body('right_arm'))) @ pose(fk.data.body('left_arm'))
    return np.stack((right_from_left, np.eye(4)))


def initialize(calibration, model_dir, grid, grid_index, work):
    from abc_minimal.eef_pose_grid_gpu import GPUEEFPoseGrid
    from scripts import batch_ceres_ik as dp

    global CALIBRATIONS, MODEL_DIR, WORK, BASES
    CALIBRATIONS = load_calibrations(str(calibration))
    MODEL_DIR, WORK, BASES = Path(model_dir), Path(work), target_bases()
    dp.GRID = GPUEEFPoseGrid(grid, index_source=grid_index)


def original_targets(record, values, model_dir):
    """FK of clean physical joints, expressed in the original calibrated right base."""
    fk = state_fk(str(Path(model_dir) / record['model_name']))
    bases = [record['base_poses'][side] for side in ('left', 'right')]
    fk.update(values[0], bases)
    right_from_world = np.linalg.inv(pose(fk.data.body('right_arm')))
    camera = record['cameras']['top']
    world_from_top = pose(fk.data.body(camera['reference_body'])) @ np.asarray(camera['T_reference_from_camera'])
    right_from_top = right_from_world @ world_from_top
    targets = np.empty((len(values), 2, 4, 4))
    for index, state in enumerate(values):
        fk.update(state, bases)
        for arm, side in enumerate(('left', 'right')):
            targets[index, arm] = right_from_world @ pose(fk.data.site(side + '_tcp_site'))
    return targets, right_from_top


def source_rows(job):
    if 'binary' in job:
        return np.memmap(job['binary'], mode='r', dtype=np.float64).reshape(-1, 28)
    with np.load(job['states']) as states, np.load(job['actions']) as actions:
        np.testing.assert_array_equal(states['frame_indices'], np.arange(len(states['states'])))
        return np.concatenate((states['states'], actions['actions']), axis=-1)


def prepare_episode(job):
    from scripts import batch_ceres_ik as dp
    from scripts.optimize_dp_sequences import save_arrays

    episode, split = job['episode'], job['split']
    rows = source_rows(job)
    if not len(rows) or not np.isfinite(rows).all():
        raise ValueError(f'Non-finite or empty original state/action rows: {episode}')
    record = CALIBRATIONS[episode]
    for stream, offset in (('states', 0), ('actions', 14)):
        directory = WORK / stream / 'dp/real' / ('real_' + episode)
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / 'result.json'
        if report_path.is_file():
            continue
        values = np.array(rows[:, offset:offset + 14])
        targets, right_from_top = original_targets(record, values, MODEL_DIR)
        local_left_targets = np.linalg.inv(BASES[0]) @ targets[:, 0]
        options = [dp.candidates('left', target.tobytes()) for target in local_left_targets]
        left = dp.select_available_path(options)
        initial = values[:, dp.COLUMNS].copy()
        initial[:, 0] = left['initial']
        valid = np.ones((len(values), 2), dtype=bool)
        valid[:, 0] = left['valid_mask']
        fk = state_fk(str(MODEL))
        fk.update(values[0], BASES)
        right_errors = dp.ceres_sequence_ik.pose_errors(fk.model, 'right', targets[:, 1], initial[:, 1])
        if np.max(right_errors) > 1e-6:
            raise ValueError(f'Original/target right-arm kinematics differ: {episode}: {right_errors.max(axis=0)}')
        errors = np.stack((left['dp_errors_mm_deg'], right_errors), axis=1)
        fields = dict(
            initial=initial, reference=initial.copy(), valid_mask=valid,
            dp_errors_mm_deg=errors, **{stream: values}, input_stream=stream,
            targets_world=targets, base_poses_world=BASES,
            targets_camera=np.linalg.inv(right_from_top) @ targets,
            camera_from_base=np.linalg.inv(right_from_top) @ BASES,
            world_from_camera=right_from_top, base_mode='right_anchor',
            retarget_sides=np.array(['left']), anchor_frame='right_arm',
            frame_indices=np.arange(len(values)), source_path=str(job.get('binary', job.get(stream))),
            source_split=split, timing_mode='official_causal_no_interpolation_no_timeoffset',
        )
        save_arrays(directory / 'input.npz', **fields)
        write_json(report_path, dict(status='prepared' if valid.all() else 'prepared_partial',
                   frames=len(values), input_sha256=digest(directory / 'input.npz')))
        print(f'DP {stream}: {episode}, left candidates {valid[:, 0].sum()}/{len(values)}', flush=True)
    return episode


def solver_config(work, stream, stage):
    from scripts.optimize_dp_sequences import WEIGHTS

    output = work / stream / stage
    source = work / stream / ('dp' if stage == 'ceres' else 'ceres')
    modules = ['scripts/ceres_sequence_ik.py', 'scripts/scipy_ik.py']
    modules += (['scripts/optimize_dp_sequences.py'] if stage == 'ceres' else
                ['scripts/refine_ceres_sequences.py', 'scripts/trust_constr_sequence_ik.py'])
    config = dict(source=str(source), output=str(output), input_stream=stream,
                  model=str(MODEL), model_sha256=digest(MODEL),
                  source_hashes={str(ROOT / name): digest(ROOT / name) for name in modules},
                  weights=WEIGHTS, maxiter=500, first_joint_radius_rad=.2,
                  joint_limit_margin_deg=15., base_mode='right_anchor')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'active').mkdir(exist_ok=True)
    path = output / 'config.json'
    if path.is_file() and json.loads(path.read_text()) != config:
        raise ValueError(f'Solver configuration changed; use a new work directory: {output}')
    write_json(path, config)
    return output


def run(command):
    print('Running:', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, cwd=ROOT)


def link_sim(source_root, output):
    """Link files, never episode directories: each run owns its camera sidecars."""
    count = 0
    for split in ('train_sim', 'val_sim'):
        for source in sorted((source_root / split).glob('episode_*')):
            if not (source / 'states_actions.bin').is_file():
                continue
            metadata = json.loads((source / 'episode_metadata.json').read_text())
            if metadata.get('sequence_ik') or metadata.get('joint_conversion'):
                raise ValueError(f'Sim must be the original unconverted data: {source}')
            destination = output / split / source.name
            if destination.is_symlink():
                raise ValueError(f'Sim episode must be an independent directory: {destination}')
            destination.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                if not path.is_file() or path.name.startswith('camera_'):
                    continue
                target = destination / path.name
                if path.name == 'episode_metadata.json':
                    if target.is_symlink():
                        raise ValueError(f'Sim metadata must belong to this dataset: {target}')
                    if target.is_file() and target.read_bytes() != path.read_bytes():
                        raise ValueError(f'Sim metadata changed: {target}')
                    shutil.copyfile(path, target)
                elif not target.exists():
                    target.symlink_to(path.resolve())
                elif not target.is_symlink() or target.resolve() != path.resolve():
                    raise ValueError(f'Sim link differs: {target}')
            count += 1
    return count


def export_dataset(args):
    settings = json.loads((args.work_dir / 'settings.json').read_text())
    run([sys.executable, '-m', 'scripts.export_sequence_ik', '--source-root', args.source_root,
         '--state-results', args.work_dir / 'states/equality_margin15',
         '--action-results', args.work_dir / 'actions/equality_margin15',
         '--output', args.output, '--base-mode', 'right_anchor', '--workers', args.workers])
    sim_count = 0 if settings['pilot'] else link_sim(args.source_root, args.output)
    summary = json.loads((args.output / 'summary.json').read_text())
    write_json(args.output / 'right_anchor.json', dict(status='complete', pilot=settings['pilot'],
               real_sequences=summary['selected'], sim_sequences=sim_count, anchor_frame='right_arm',
               target_right_from_left=settings['target_right_from_left'],
               calibration_sha256=settings['calibration_sha256'], retarget_sides=['left'],
               sim_handling='original', work_dir=str(args.work_dir)))
    print('Ready for camera preparation:', args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Original official RGB/state export')
    parser.add_argument('--state-sources', type=Path, help='Existing original state sources: real/sources/*.npz')
    parser.add_argument('--action-sources', type=Path, help='Matching original action sources: real/sources/*.npz')
    parser.add_argument('--calibration-path', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--grid', type=Path, help='Same FK grid and per-arm teleop scores as the old three-stage IK')
    parser.add_argument('--grid-index', type=Path, help='Override the old index_source path after moving the index to T7')
    parser.add_argument('--work-dir', type=Path, required=True, help='Retain the full state results here for noise statistics')
    parser.add_argument('--output', type=Path, help='New dataset root, separate from all existing runs')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--equality-workers', type=int, default=12)
    parser.add_argument('--episodes', nargs='+', help='Optional real episode IDs for a pilot; sim is not exported in pilot mode')
    parser.add_argument('--solve-only', action='store_true', help='Local IK only; export against the server RGB later')
    parser.add_argument('--export-only', action='store_true', help='Export transferred IK results; no FK grid needed')
    args = parser.parse_args()
    for name in ('source_root', 'state_sources', 'action_sources', 'calibration_path', 'model_dir', 'grid', 'grid_index', 'work_dir', 'output'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.solve_only and args.export_only:
        parser.error('Choose solve-only or export-only')
    if not args.solve_only and (args.source_root is None or args.output is None):
        parser.error('Export requires --source-root and --output')
    if args.output is not None and args.source_root is not None and (
            args.output == args.source_root or args.source_root in args.output.parents):
        raise ValueError('Choose a separate output dataset, outside the original data')
    if args.workers < 1 or args.equality_workers < 1:
        raise ValueError('Worker counts must be positive')
    if args.export_only:
        export_dataset(args)
        return
    # The server only exports finished results; the FK grid is for local IK.
    from abc_minimal.eef_pose_grid_gpu import GPUEEFPoseGrid

    if any(getattr(args, name) is None for name in ('calibration_path', 'model_dir', 'grid')):
        parser.error('IK requires --calibration-path, --model-dir and --grid')
    grid = GPUEEFPoseGrid(args.grid, index_source=args.grid_index)
    if not grid.cells_path.is_dir():
        raise ValueError(f'FK grid index is missing: {grid.cells_path}; keep the original index_source or relocate it explicitly')
    calibrations = load_calibrations(str(args.calibration_path))
    jobs = []
    if args.state_sources is not None or args.action_sources is not None:
        if args.state_sources is None or args.action_sources is None:
            parser.error('Supply both --state-sources and --action-sources')
        for episode, record in sorted(calibrations.items()):
            if args.episodes and not any(part in episode for part in args.episodes):
                continue
            job = dict(episode=episode, split=record['split'] + '_real')
            for stream, root in (('states', args.state_sources), ('actions', args.action_sources)):
                path = root / 'real/sources' / ('real_' + episode + '.npz')
                if not path.is_file():
                    raise ValueError(f'Original {stream} source missing: {path}')
                job[stream] = str(path)
            jobs.append(job)
    elif args.source_root is not None:
        for split in ('train_real', 'val_real'):
            for episode in sorted((args.source_root / split).glob('episode_*')):
                if args.episodes and not any(part in episode.name for part in args.episodes):
                    continue
                if episode.name not in calibrations:
                    raise ValueError(f'Missing accepted calibration: {episode.name}')
                metadata = json.loads((episode / 'episode_metadata.json').read_text())
                if metadata.get('sequence_ik') or metadata.get('joint_conversion'):
                    raise ValueError(f'Use original, unconverted data: {episode}')
                jobs.append(dict(split=split, episode=episode.name, binary=str(episode / 'states_actions.bin')))
    if not jobs:
        raise ValueError('No original real episodes found')
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = dict(source_root=str(args.source_root), pilot=bool(args.episodes),
                        calibration_sha256=digest(args.calibration_path), grid=str(args.grid),
                        source_model_hashes={name: digest(args.model_dir / name)
                                             for name in sorted({r['model_name'] for r in calibrations.values()})},
                        grid_metadata_sha256=digest(args.grid / 'metadata.json'),
                        grid_index=str(grid.cells_path.parent),
                        target_right_from_left=target_bases()[0].tolist(),
                        source_joints='original_unclipped', retarget_sides=['left'],
                        inputs=[dict(job, hashes={name: digest(job[name]) for name in ('binary', 'states', 'actions')
                                                 if name in job}) for job in jobs],
                        source_hashes={str(p): digest(p) for p in [Path(__file__), MODEL,
                            ROOT / 'abc_minimal/camera_geometry.py', ROOT / 'scripts/batch_ceres_ik.py',
                            ROOT / 'abc_minimal/grid_dp.py', ROOT / 'abc_minimal/eef_pose_grid_gpu.py',
                            ROOT / 'scripts/optimize_dp_sequences.py', ROOT / 'scripts/refine_ceres_sequences.py',
                            ROOT / 'scripts/ceres_sequence_ik.py', ROOT / 'scripts/trust_constr_sequence_ik.py',
                            ROOT / 'scripts/scipy_ik.py']})
        settings_path = args.work_dir / 'settings.json'
        # JSON converts input tuples to lists; compare the serialized form.
        settings = json.loads(json.dumps(settings))
        if settings_path.is_file() and json.loads(settings_path.read_text()) != settings:
            raise ValueError('Inputs/settings changed; use a new work directory')
        write_json(settings_path, settings)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize,
                                 initargs=(args.calibration_path, args.model_dir, args.grid, args.grid_index, args.work_dir)) as pool:
            for episode in pool.map(prepare_episode, jobs):
                print('Prepared:', episode, flush=True)
        for stream in ('states', 'actions'):
            for stage, module in (('ceres', 'scripts.optimize_dp_sequences'),
                                  ('equality_margin15', 'scripts.refine_ceres_sequences')):
                output = solver_config(args.work_dir, stream, stage)
                command = [sys.executable, '-m', module, '--output', output]
                if stage == 'ceres':
                    command += ['--source', args.work_dir / stream / 'dp', '--workers', args.workers]
                else:
                    previous = args.work_dir / stream / 'ceres'
                    reports = json.loads((previous / 'summary.json').read_text())['episodes']
                    write_json(output / 'jobs.json', [dict(kind=r['kind'], key=r['episode'], frames=r['frames'],
                        input=str(previous / r['kind'] / r['episode'] / 'trajectory.npz'),
                        input_sha256=digest(previous / r['kind'] / r['episode'] / 'trajectory.npz')) for r in reports])
                    command += ['--workers', args.equality_workers]
                run(command)
                summary = json.loads((output / 'summary.json').read_text())
                if summary['status'] != 'complete' or summary.get('failed', 0) or summary['completed'] != len(jobs):
                    raise RuntimeError(f'IK stage is incomplete: {output / "summary.json"}')
        if not args.solve_only:
            export_dataset(args)
        else:
            print('IK complete; retain states/actions equality_margin15 and settings.json for server export.', flush=True)


if __name__ == '__main__':
    main()
