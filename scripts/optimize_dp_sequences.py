"""Refine saved DP trajectories with Ceres, retaining independent valid arm segments."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts import ceres_sequence_ik as ceres


SIDES = ('left', 'right')
COLUMNS = np.array([[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]])
WEIGHTS = dict(position_scale_mm=.1, rotation_scale_deg=.1, position_weight=1., rotation_weight=4.,
               reference_weight=1., first_order_weight=1., first_order_window=1, second_order_weight=1.)
CONFIG = OUTPUT = None


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_arrays(path, **values):
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def records(source):
    jobs = []
    for kind in ('real', 'sim'):
        for path in sorted((source / kind).glob('*/result.json')):
            report = json.loads(path.read_text())
            if report['status'] in ('prepared', 'prepared_partial'):
                jobs.append(dict(kind=kind, key=path.parent.name, input=str(path.parent / 'input.npz'),
                                 input_sha256=report['input_sha256'], frames=report['frames'],
                                 partial=report['status'] == 'prepared_partial'))
    return jobs


def initialize(config, output):
    global CONFIG, OUTPUT
    CONFIG, OUTPUT = config, Path(output)
    for path, expected in config['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Solver source changed: {path}')
    if hashlib.sha256(Path(config['model']).read_bytes()).hexdigest() != config['model_sha256']:
        raise ValueError('FK model changed')


def run_sequence(record):
    directory = OUTPUT / record['kind'] / record['key']
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['status'] == 'complete' and previous['input_sha256'] == record['input_sha256']:
            return previous
    started = time.monotonic()
    try:
        input_path = Path(record['input'])
        if hashlib.sha256(input_path.read_bytes()).hexdigest() != record['input_sha256']:
            raise ValueError('DP input changed')
        with np.load(input_path) as saved:
            source = {key: saved[key] for key in saved.files}
        model = mujoco.MjModel.from_xml_path(CONFIG['model'])
        for side, transform in zip(SIDES, source['base_poses_world']):
            body = model.body(side + '_arm')
            body.pos[:] = transform[:3, 3]
            body.quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        initial, reference = source['initial'], source['reference']
        valid = source.get('valid_mask', np.ones(initial.shape[:2], dtype=bool))
        q, errors = np.full_like(initial, np.nan), np.full((*valid.shape, 2), np.nan)
        half_width = np.full_like(initial, np.nan)
        usable, converged = np.zeros_like(valid), np.zeros_like(valid)
        segments, failures = [], []
        cache = directory / 'segments'
        cache.mkdir(exist_ok=True)
        for arm, side in enumerate(SIDES):
            radius = ceres.joint_radii(model, side, .2)
            intervals = np.flatnonzero(np.diff(np.r_[False, valid[:, arm], False])).reshape(-1, 2)
            for begin, end in intervals:
                name = f'{side}_{begin:06d}_{end:06d}'
                checkpoint = cache / (name + '.npz')
                write_json(OUTPUT / 'active' / (record['key'] + '.json'),
                           dict(episode=record['key'], side=side, begin=int(begin), end=int(end),
                                updated=time.time(), cached=checkpoint.exists()))
                try:
                    if checkpoint.exists():
                        with np.load(checkpoint) as saved:
                            values = {key: saved[key] for key in saved.files}
                        report = json.loads(str(values['summary']))
                    else:
                        before = time.monotonic()
                        targets = source['targets_world'][begin:end, arm]
                        seed, ref = initial[begin:end, arm], reference[begin:end, arm]
                        dp_error = ceres.pose_errors(model, side, targets, seed)
                        scale = np.sqrt(np.hypot(dp_error[:, 0] / 10., np.deg2rad(dp_error[:, 1]) / .1))
                        width = scale[:, None] * radius
                        solution, summary = ceres.solve_sequence(model, side, targets, seed, ref, width,
                                                               maxiter=500, check_gradients=False, **WEIGHTS)
                        residual = ceres.pose_errors(model, side, targets, solution)
                        if not (np.isfinite(solution).all() and np.isfinite(residual).all()):
                            raise ValueError('Non-finite Ceres result')
                        report = dict(side=side, begin=int(begin), end=int(end), seconds=time.monotonic()-before,
                                      usable=summary.IsSolutionUsable(),
                                      converged=summary.termination_type == ceres.pyceres.TerminationType.CONVERGENCE,
                                      solver_report=summary.BriefReport(), initial_cost=summary.initial_cost,
                                      final_cost=summary.final_cost,
                                      iterations=summary.num_successful_steps + summary.num_unsuccessful_steps,
                                      solver_seconds=summary.total_time_in_seconds,
                                      max_bound_violation_rad=float(np.maximum(np.abs(solution-seed)-width, 0.).max()))
                        values = dict(q=solution, errors=residual, half_width=width, summary=np.array(json.dumps(report)))
                        save_arrays(checkpoint, **values)
                    q[begin:end, arm], errors[begin:end, arm] = values['q'], values['errors']
                    half_width[begin:end, arm] = values['half_width']
                    usable[begin:end, arm], converged[begin:end, arm] = report['usable'], report['converged']
                    segments.append(report)
                except Exception as error:
                    failure = dict(side=side, begin=int(begin), end=int(end), error=str(error), traceback=traceback.format_exc())
                    failures.append(failure)
                    write_json(cache / (name + '_error.json'), failure)
        available = valid & np.isfinite(q).all(axis=-1)
        stream = str(source.get('input_stream', 'actions'))
        optimized_actions = source[stream].copy()
        optimized_actions[:, COLUMNS] = q
        source.update(q=q, errors=errors, valid_mask=valid, optimized_mask=available,
                      ceres_usable=usable, solver_success=converged,
                      **{f'optimized_{stream}': optimized_actions},
                      solver='ceres', model_path=CONFIG['model'], **WEIGHTS,
                      joint_limits_enabled=True, joint_lower_bounds_rad=initial-half_width,
                      joint_upper_bounds_rad=initial+half_width)
        save_arrays(directory / 'trajectory.npz', **source)
        observed = errors[available]
        steps = np.rad2deg(np.abs(np.diff(q, axis=0)))[available[:-1] & available[1:]]
        result = dict(episode=record['key'], kind=record['kind'], status='complete' if not failures else 'complete_with_failures',
                      input_sha256=record['input_sha256'], frames=len(q), arm_targets=int(valid.size),
                      valid_arm_targets=int(valid.sum()), optimized_arm_targets=int(available.sum()),
                      missing_candidate_arm_targets=int((~valid).sum()), usable_arm_targets=int(usable.sum()),
                      passed=int(np.all(observed <= .1, axis=-1).sum()),
                      error_sum_mm_deg=observed.sum(axis=0).tolist(),
                      error_max_mm_deg=observed.max(axis=0, initial=0.).tolist(),
                      max_joint_step_deg=float(steps.max(initial=0.)),
                      jump_arm_frame_pairs=int((steps.max(axis=-1) > 30).sum()),
                      segment_count=len(segments), segment_failures=len(failures),
                      frame_iterations=sum((s['end']-s['begin'])*s['iterations'] for s in segments),
                      nonconverged_segments=sum(not s['converged'] for s in segments),
                      max_bound_violation_rad=max((s['max_bound_violation_rad'] for s in segments), default=0.),
                      seconds=time.monotonic()-started)
        write_json(directory / 'summary.json', dict(**result, segments=segments, failures=failures))
    except Exception as error:
        result = dict(episode=record['key'], kind=record['kind'], status='failed', input_sha256=record['input_sha256'],
                      error=str(error), traceback=traceback.format_exc(), seconds=time.monotonic()-started)
    write_json(result_path, result)
    (OUTPUT / 'active' / (record['key'] + '.json')).unlink(missing_ok=True)
    return result


def publish(output, reports, total, started, status):
    complete = [r for r in reports if r['status'] in ('complete', 'complete_with_failures')]
    optimized = sum(r['optimized_arm_targets'] for r in complete)
    sums = np.sum([r['error_sum_mm_deg'] for r in complete], axis=0) if complete else np.zeros(2)
    progress = dict(status=status, selected=total, completed=len(complete),
                    failed=sum(r['status'] != 'complete' for r in reports), seconds=time.monotonic()-started,
                    frames=sum(r['frames'] for r in complete), optimized_arm_targets=optimized,
                    missing_candidate_arm_targets=sum(r['missing_candidate_arm_targets'] for r in complete),
                    passed=sum(r['passed'] for r in complete),
                    error_mean_mm_deg=(sums/max(1,optimized)).tolist(),
                    error_max_mm_deg=np.max([r['error_max_mm_deg'] for r in complete], axis=0).tolist() if complete else [0.,0.],
                    max_joint_step_deg=max((r['max_joint_step_deg'] for r in complete),default=0.),
                    jump_arm_frame_pairs=sum(r['jump_arm_frame_pairs'] for r in complete),
                    nonconverged_segments=sum(r['nonconverged_segments'] for r in complete))
    write_json(output / 'progress.json', progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    output = args.output.resolve()
    config = json.loads((output / 'config.json').read_text())
    jobs = records(args.source.resolve())
    jobs.sort(key=lambda record: record['frames'], reverse=True)
    reports, pending = [], []
    for record in jobs:
        path = output / record['kind'] / record['key'] / 'result.json'
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and previous['status'] == 'complete' and previous['input_sha256'] == record['input_sha256']:
            reports.append(previous)
        else:
            pending.append(record)
    write_json(output / 'runtime.json', dict(status='running', workers=args.workers, started=time.time()))
    started = time.monotonic()
    publish(output, reports, len(jobs), started, 'running')
    print(json.dumps(dict(selected=len(jobs), reused=len(reports), pending=len(pending), workers=args.workers)),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(config,str(output))) as pool:
        for future in as_completed([pool.submit(run_sequence, record) for record in pending]):
            result = future.result()
            reports.append(result)
            with (output / 'results.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            progress = publish(output, reports, len(jobs), started, 'running')
            print(json.dumps(dict(completed=progress['completed'], failed=progress['failed'],
                                  episode=result['episode'], seconds=result['seconds'], status=result['status'])),flush=True)
    progress = publish(output,reports,len(jobs),started,'complete_with_failures' if any(r['status']!='complete' for r in reports) else 'complete')
    write_json(output/'summary.json',dict(**progress,episodes=reports))
    write_json(output/'runtime.json',dict(status=progress['status'],workers=args.workers,seconds=time.monotonic()-started,finished=time.time()))


if __name__ == '__main__':
    main()
