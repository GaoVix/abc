"""Clip source joints and select distance-filtered DP paths, optionally running Ceres."""

import argparse
from contextlib import redirect_stdout
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import time
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from abc_minimal.eef_pose_grid_gpu import GPUEEFPoseGrid
from abc_minimal.grid_dp import select_path
from scripts import ceres_sequence_ik
from scripts.yam_reachability import STATUS, classify


ROOT = Path(__file__).resolve().parents[1]
SIDES = ('left', 'right')
COLUMNS = np.array([[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]])
GRID = CONFIG = OUTPUT = SOURCE_MODELS = SOURCE_DATA = WORLD_FROM_CAMERA = REFERENCE = LIMITS = None


def expanded_limits(model):
    joints = [[model.joint(f'{side}_joint{i}').id for i in range(1, 7)] for side in SIDES]
    return model.jnt_range[joints] + np.deg2rad(10.) * np.array([-1., 1.])


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def initialize(config, output):
    global CONFIG, OUTPUT, GRID, SOURCE_MODELS, SOURCE_DATA, WORLD_FROM_CAMERA, REFERENCE, LIMITS
    CONFIG, OUTPUT = config, Path(output)
    for name, expected in config['source_hashes'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source changed after launch: {name}')
    GRID = GPUEEFPoseGrid(config['grid'])
    SOURCE_MODELS = {kind: mujoco.MjModel.from_xml_path(str(ROOT / 'abc_sim/models' / filename))
                     for kind, filename in (('real', 'yam_bimanual_empty.xml'), ('sim', 'yam_dishrack_base.xml'))}
    SOURCE_DATA = {kind: mujoco.MjData(model) for kind, model in SOURCE_MODELS.items()}
    mujoco.mj_forward(SOURCE_MODELS['sim'], SOURCE_DATA['sim'])
    camera = SOURCE_DATA['sim'].camera('top')
    WORLD_FROM_CAMERA = np.eye(4)
    WORLD_FROM_CAMERA[:3, :3] = camera.xmat.reshape(3, 3) @ np.diag([1., -1., -1.])
    WORLD_FROM_CAMERA[:3, 3] = camera.xpos
    REFERENCE = np.asarray(config.get('target_camera_from_base', config.get('median_camera_from_base')))
    LIMITS = expanded_limits(SOURCE_MODELS['sim'])


def active(record, stage, **extra):
    write_json(OUTPUT / 'active' / f"{record['key']}.json",
               dict(episode=record['key'], stage=stage, updated=time.time(), **extra))


def clipped_targets(record):
    with np.load(record['source']) as saved:
        original = saved[CONFIG.get('input_stream', 'actions')].astype(np.float64)
        targets = saved['targets_camera'].copy() if 'targets_camera' in saved else None
    model, data = SOURCE_MODELS[record['kind']], SOURCE_DATA[record['kind']]
    joints = np.array([[model.joint(f'{side}_joint{i}').id for i in range(1, 7)] for side in SIDES])
    limits = expanded_limits(model)
    actions = original.copy()
    actions[:, COLUMNS] = np.clip(actions[:, COLUMNS], limits[..., 0], limits[..., 1])
    changed = actions[:, COLUMNS] != original[:, COLUMNS]
    if record['kind'] == 'real':
        calibration = record['calibration']
        for side in SIDES:
            transform = np.asarray(calibration['base_poses'][side])
            model.body(f'{side}_arm').pos[:] = transform[:3, 3]
            model.body(f'{side}_arm').quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        camera_from_world = np.asarray(calibration['left_base_to_camera'])
    else:
        camera_from_world = np.linalg.inv(WORLD_FROM_CAMERA)
    frames = np.flatnonzero(changed.any(axis=(1, 2))) if targets is not None else range(len(actions))
    if targets is None:
        targets = np.empty((len(actions), 2, 4, 4))
    for frame in frames:
        data.qpos[model.jnt_qposadr[joints]] = actions[frame, COLUMNS]
        mujoco.mj_kinematics(model, data)
        for arm, side in enumerate(SIDES):
            pose = np.eye(4)
            site = data.site(f'{side}_tcp_site')
            pose[:3, :3], pose[:3, 3] = site.xmat.reshape(3, 3), site.xpos
            targets[frame, arm] = camera_from_world @ pose
    return actions, targets, changed.sum(axis=0)


@lru_cache(maxsize=4096)
def candidates(side, pose_bytes):
    target = np.frombuffer(pose_bytes, dtype=np.float64).reshape(4, 4)
    for radius in (1, 2, 3, 4):
        found = GRID.query(target, k=100, neighbors=radius, side=side, distance_limit=.5)
        if len(found['rows']):
            cost = (found['position_error_m'] / .01) ** 2 + (found['rotation_error_rad'] / .1) ** 2
            errors = np.column_stack((found['position_error_m'] * 1000, np.rad2deg(found['rotation_error_rad'])))
            return found['joints'].astype(np.float64), cost, found['rows'], found['teleop_distance'], radius, errors
    return None


def select_available_path(options):
    count = len(options)
    counts = np.array([0 if item is None else len(item[0]) for item in options])
    valid = counts > 0
    segments = np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2)
    q = np.full((count, 6), np.nan)
    rows = np.full(count, -1, dtype=np.int64)
    scores, errors = np.full(count, np.nan, dtype=np.float32), np.full((count, 2), np.nan)
    radii = np.full(count, 4, dtype=np.int8)
    for begin, end in segments:
        available = options[begin:end]
        choice = select_path([item[0] for item in available], [item[1] for item in available], acceleration_weight=1.)
        for frame, item, index in zip(range(begin, end), available, choice):
            q[frame], rows[frame], scores[frame] = item[0][index], item[2][index], item[3][index]
            radii[frame], errors[frame] = item[4], item[5][index]
    return dict(initial=q, selected_rows=rows, selected_teleop_distance_rad=scores,
                dp_errors_mm_deg=errors, candidate_counts=counts, query_radii=radii,
                valid_mask=valid, segment_bounds=segments)


def prepare(record, directory):
    started = time.monotonic()
    active(record, 'clip_and_fk')
    actions, targets, clipped = clipped_targets(record)
    local_targets = np.linalg.inv(REFERENCE)[None] @ targets
    active(record, 'reachability')
    reachability = np.array([[classify(target, limits) for target, limits in zip(frame, LIMITS)]
                             for frame in local_targets], dtype=np.uint8)
    np.savez_compressed(directory / 'reachability.npz', status=reachability)
    write_json(directory / 'reachability.json',
               dict(criterion='target XML limits expanded outward by 10 degrees; clipped source actions',
                    limits_rad=LIMITS.tolist(), frames=len(actions),
                    counts={name: int((reachability == code).sum()) for code, name in enumerate(STATUS)},
                    arms=[dict(side=side, counts={name: int((reachability[:, arm] == code).sum())
                                                  for code, name in enumerate(STATUS)})
                          for arm, side in enumerate(SIDES)]))
    clip_fk_reachability_seconds = time.monotonic() - started
    arms, statistics = [], []
    for arm, side in enumerate(SIDES):
        cache = directory / f'dp_{side}.npz'
        if cache.exists():
            with np.load(cache) as saved:
                selected = {key: saved[key] for key in saved.files}
            query_seconds = dp_seconds = 0.
        else:
            before_query = time.monotonic()
            options = []
            for frame, target in enumerate(local_targets[:, arm]):
                if frame % 100 == 0:
                    active(record, 'grid', side=side, frame=frame, frames=len(targets))
                options.append(candidates(side, target.tobytes()))
            active(record, 'dp', side=side, frames=len(targets))
            before_dp = time.monotonic()
            selected = select_available_path(options)
            query_seconds, dp_seconds = before_dp - before_query, time.monotonic() - before_dp
            temporary = directory / f'dp_{side}.partial.npz'
            np.savez_compressed(temporary, **selected)
            temporary.replace(cache)
        arms.append(selected)
        counts = selected['candidate_counts']
        statistics.append(dict(side=side, candidate_count_min_median_max=[int(counts.min()), float(np.median(counts)), int(counts.max())],
                               missing_frames=np.flatnonzero(~selected['valid_mask']).tolist(),
                               query_seconds=query_seconds, dp_seconds=dp_seconds))
    fields = ('initial', 'selected_rows', 'selected_teleop_distance_rad', 'dp_errors_mm_deg',
              'candidate_counts', 'query_radii', 'valid_mask')
    selected = {key: np.stack([arm[key] for arm in arms], axis=1) for key in fields}
    temporary = directory / 'input.partial.npz'
    stream = CONFIG.get('input_stream', 'actions')
    np.savez_compressed(temporary, **selected, reference=selected['initial'].copy(),
                        **{stream: actions}, input_stream=stream,
                        targets_world=WORLD_FROM_CAMERA @ targets, base_poses_world=WORLD_FROM_CAMERA @ REFERENCE,
                        targets_camera=targets, camera_from_base=REFERENCE, world_from_camera=WORLD_FROM_CAMERA,
                        base_mode=CONFIG.get('base_mode', 'median'), episode=record['key'], source_path=record['source'],
                        segment_bounds_left=arms[0]['segment_bounds'], segment_bounds_right=arms[1]['segment_bounds'],
                        frame_indices=np.arange(len(actions)), timing_mode='official_causal_no_interpolation_no_timeoffset',
                        clipped_joint_counts=clipped)
    temporary.replace(directory / 'input.npz')
    write_json(directory / 'missing_candidates.json',
               dict(criterion='No grid candidate with distance < 0.5 within neighborhoods 1..4',
                    frames=len(actions), arms={side: statistics[arm]['missing_frames'] for arm, side in enumerate(SIDES)},
                    valid_arm_targets=int(selected['valid_mask'].sum())))
    write_json(directory / 'dp_summary.json', dict(frames=len(actions), clipped_joint_counts=clipped.tolist(),
               clip_fk_reachability_seconds=clip_fk_reachability_seconds,
               seconds=time.monotonic() - started, arms=statistics))


def run_episode(record):
    directory = OUTPUT / record['kind'] / record['key']
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'result.json'
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous['status'] == 'complete' or (CONFIG['prepare_only'] and previous['status'] in ('prepared', 'prepared_partial')):
            return previous
    started = time.monotonic()
    try:
        if not ((directory / 'input.npz').exists() and (directory / 'dp_summary.json').exists()):
            prepare(record, directory)
        if CONFIG['prepare_only']:
            path, error_key, q_key = directory / 'input.npz', 'dp_errors_mm_deg', 'initial'
            extra = dict(status='prepared', stage='dp_preprocessing', error_source='grid_fk_float32')
        else:
            with np.load(directory / 'input.npz') as saved:
                if 'valid_mask' in saved and not saved['valid_mask'].all():
                    raise ValueError('Input contains missing grid candidates; optimize its valid segments separately')
            active(record, 'ceres')
            with (directory / 'solver.log').open('w') as stream, redirect_stdout(stream):
                ceres_sequence_ik.main(['--input', str(directory / 'input.npz'), '--output', str(directory),
                                       '--first-joint-radius-rad', '.2', '--position-weight', '1', '--rotation-weight', '4'])
            path, error_key, q_key = directory / 'trajectory.npz', 'errors', 'q'
            with np.load(path) as result:
                extra = dict(status='complete', stage='optimized', error_source='mujoco_fk',
                             optimizer_usable=bool(result['ceres_usable'].all()),
                             converged=bool(result['solver_success'].all()))
        with np.load(path) as saved:
            errors, q = saved[error_key], saved[q_key]
            valid = saved['valid_mask'] if 'valid_mask' in saved else np.ones(q.shape[:2], dtype=bool)
            steps = np.rad2deg(np.abs(np.diff(q, axis=0)))[valid[:-1] & valid[1:]]
            valid_errors = errors[valid]
            if CONFIG['prepare_only'] and not valid.all():
                extra['status'] = 'prepared_partial'
            report = dict(episode=record['key'], kind=record['kind'], frames=len(q),
                          arm_targets=int(valid.size), valid_arm_targets=int(valid.sum()),
                          missing_candidate_arm_targets=int((~valid).sum()),
                          passed=int(np.all(valid_errors <= .1, axis=-1).sum()),
                          error_sum_mm_deg=valid_errors.sum(axis=0).tolist(),
                          error_max_mm_deg=valid_errors.max(axis=0, initial=0.).tolist(),
                          max_joint_step_deg=float(steps.max(initial=0.)),
                          jump_arm_frame_pairs=int((steps.max(axis=-1) > 30).sum()),
                          clipped_joint_values=int(saved['clipped_joint_counts'].sum()),
                          input_sha256=hashlib.sha256((directory / 'input.npz').read_bytes()).hexdigest(),
                          seconds=time.monotonic() - started, **extra)
    except Exception as error:
        report = dict(episode=record['key'], kind=record['kind'], status='failed', error=str(error),
                      traceback=traceback.format_exc(), seconds=time.monotonic() - started)
    reachability_path = directory / 'reachability.json'
    if reachability_path.exists():
        report['reachability'] = json.loads(reachability_path.read_text())['counts']
    write_json(report_path, report)
    (OUTPUT / 'active' / f"{record['key']}.json").unlink(missing_ok=True)
    return report


def records(source):
    with gzip.open(ROOT / 'scripts/dishrack_joint_retarget_calibrations.json.gz', 'rt') as stream:
        calibrations = {r['episode']: r['calibration'] for r in json.load(stream)['records']
                        if r['calibration_quality'] == 'accepted'}
    result = []
    for kind in ('real', 'sim'):
        for path in sorted((source / kind / 'sources').glob('*.npz')):
            episode = path.stem.removeprefix(kind + '_')
            result.append(dict(key=path.stem, kind=kind, source=str(path.resolve()),
                               calibration=calibrations[episode] if kind == 'real' else None))
    return result


def publish(output, reports, total, started, status):
    complete = [r for r in reports if r['status'] in ('complete', 'prepared', 'prepared_partial')]
    targets = sum(r['arm_targets'] for r in complete)
    valid_targets = sum(r.get('valid_arm_targets', r['arm_targets']) for r in complete)
    sums = np.sum([r['error_sum_mm_deg'] for r in complete], axis=0) if complete else np.zeros(2)
    progress = dict(status=status, selected=total, completed=len(complete), failed=len(reports) - len(complete),
                    seconds=time.monotonic() - started, frames=sum(r['frames'] for r in complete),
                    arm_targets=targets, valid_arm_targets=valid_targets, missing_candidate_arm_targets=targets-valid_targets,
                    partial_sequences=sum(r['status'] == 'prepared_partial' for r in complete),
                    passed=sum(r['passed'] for r in complete),
                    error_mean_mm_deg=(sums / max(1, valid_targets)).tolist(),
                    error_max_mm_deg=np.max([r['error_max_mm_deg'] for r in complete], axis=0).tolist() if complete else [0., 0.],
                    jump_arm_frame_pairs=sum(r['jump_arm_frame_pairs'] for r in complete),
                    prepared=sum(r['status'] in ('prepared', 'prepared_partial') for r in complete),
                    optimized=sum(r['status'] == 'complete' for r in complete),
                    nonconverged=sum(not r.get('converged', True) for r in complete))
    measured = [r['reachability'] for r in reports if 'reachability' in r]
    progress['reachability'] = dict(classified_episodes=len(measured),
        arm_targets={name: sum(r[name] for r in measured) for name in STATUS},
        episodes={name: sum(r[name] > 0 for r in measured) for name in STATUS})
    write_json(output / 'progress.json', progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/median_base_grid_fallback_bounded_ik_20261026')
    parser.add_argument('--grid', type=Path, default=Path('/media/ps/T7/Datasets/ABC-130k/eef_pose_grid_margin10_gpu_20261029'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episodes', nargs='*')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--stream', choices=('actions', 'states'), default='actions')
    parser.add_argument('--base-mode', choices=('median', 'xml'), default='median')
    parser.add_argument('--kind', choices=('real', 'sim', 'both'), default='both')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'active').mkdir(exist_ok=True)
    jobs = records(args.source)
    files = ['scripts/batch_ceres_ik.py', 'scripts/ceres_sequence_ik.py', 'scripts/scipy_ik.py',
             'abc_minimal/grid_dp.py', 'abc_minimal/eef_pose_grid_gpu.py', 'scripts/yam_reachability.py',
             'abc_sim/models/yam_dishrack_base.xml', 'abc_sim/models/yam_bimanual_empty.xml',
             'scripts/dishrack_joint_retarget_calibrations.json.gz', 'outputs/base_pose_median_20261026/median_base_poses.json']
    median = json.loads((ROOT / files[-1]).read_text())['poses']
    config = dict(grid=str(args.grid.resolve()),
                  source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
                  median_camera_from_base=[median[side]['T_camera_from_base'] for side in SIDES],
                  source_inventory_sha256=hashlib.sha256(json.dumps([
                      (r['key'], Path(r['source']).stat().st_size, Path(r['source']).stat().st_mtime_ns)
                      for r in jobs]).encode()).hexdigest(),
                  grid_metadata_sha256=hashlib.sha256((args.grid / 'metadata.json').read_bytes()).hexdigest(),
                  prepare_only=args.prepare_only,
                  pipeline='action clip XML+10deg -> source FK -> median-base distance<0.5 DP' + ('' if args.prepare_only else ' -> Ceres'),
                  action_clip_margin_deg=10., reachability_limit_margin_deg=10.,
                  reachability_filter=False,
                  candidate_k=100, neighborhoods=[1, 2, 3, 4],
                  missing_candidate_policy='mark_per_arm_and_split_valid_segments',
                  dp_position_scale_m=.01, dp_rotation_scale_rad=.1, dp_joint_scale_rad=.1,
                  dp_first_order_weight=1., dp_second_order_weight=1., first_joint_radius_rad=.2,
                  position_weight=1, rotation_weight=4, scale_bounds_by_dp_error=True, check_gradients=False)
    if args.stream != 'actions':
        config.update(input_stream=args.stream,
                      pipeline='state clip XML+10deg -> source FK -> median-base distance<0.5 DP')
    if args.base_mode == 'xml':
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'abc_sim/models/yam_dishrack_base.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = data.camera('top')
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = camera.xmat.reshape(3, 3) @ np.diag([1., -1., -1.])
        camera_pose[:3, 3] = camera.xpos
        bases = []
        for side in SIDES:
            body = data.body(f'{side}_arm')
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = body.xmat.reshape(3, 3), body.xpos
            bases.append((np.linalg.inv(camera_pose) @ pose).tolist())
        config.pop('median_camera_from_base')
        config.update(base_mode='xml', target_camera_from_base=bases,
                      pipeline=config['pipeline'].replace('median-base', 'xml-default-base'))
    if args.kind != 'both':
        config['selection_kind'] = args.kind
    if (output / 'config.json').exists():
        if json.loads((output / 'config.json').read_text()) != config:
            raise ValueError('Output configuration/source signature differs; use a new output directory')
    else:
        write_json(output / 'config.json', config)
    for name in files:
        snapshot = output / 'source_snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copy2(ROOT / name, snapshot)
    if args.episodes:
        jobs = [r for r in jobs if any(prefix in r['key'] for prefix in args.episodes)]
    if args.kind != 'both':
        jobs = [r for r in jobs if r['kind'] == args.kind]
    previous_failures = {path.parent.name for kind in ('real', 'sim') for path in (output / kind).glob('*/result.json')
                         if json.loads(path.read_text())['status'] == 'failed'}
    jobs.sort(key=lambda record: record['key'] not in previous_failures)
    write_json(output / 'episodes.json', jobs)
    started, reports = time.monotonic(), []
    publish(output, reports, len(jobs), started, 'running')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(config, str(output))) as pool:
        futures = [pool.submit(run_episode, record) for record in jobs]
        for future in as_completed(futures):
            result = future.result()
            reports.append(result)
            with (output / 'results.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            progress = publish(output, reports, len(jobs), started, 'running')
            print(json.dumps(dict(completed=progress['completed'], failed=progress['failed'], selected=len(jobs),
                                  episode=result['episode'], status=result['status'], seconds=result['seconds'])), flush=True)
    status = ('complete_with_failures' if any(r['status'] == 'failed' for r in reports) else
              'complete_with_missing_candidates' if any(r['status'] == 'prepared_partial' for r in reports) else 'complete')
    progress = publish(output, reports, len(jobs), started, status)
    write_json(output / 'summary.json', {**progress, 'episodes': reports})


if __name__ == '__main__':
    main()
