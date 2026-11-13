"""Refine Ceres trajectories with exact EEF constraints and joint bounds."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts import trust_constr_sequence_ik as trust


SIDES = ('left', 'right')
COLUMNS = np.array([[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]])
CONFIG = OUTPUT = None


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_arrays(path, **values):
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


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
    started = time.monotonic()
    active_path = OUTPUT / 'active' / (record['key'] + '.json')
    try:
        path = Path(record['input'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['input_sha256']:
            raise ValueError('Ceres input changed')
        with np.load(path) as saved:
            source = {key: saved[key] for key in saved.files}
        reference = source['q']
        initial = reference.copy()
        valid = source['valid_mask']
        if not np.isfinite(initial[valid]).all():
            raise ValueError('Ceres initial values are non-finite on valid frames')
        model = mujoco.MjModel.from_xml_path(CONFIG['model'])
        for side, transform in zip(SIDES, source['base_poses_world']):
            body = model.body(side + '_arm')
            body.pos[:] = transform[:3, 3]
            body.quat[:] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        q, errors = np.full_like(initial, np.nan), np.full((*valid.shape, 2), np.nan)
        lower_bounds, upper_bounds = np.full_like(initial, np.nan), np.full_like(initial, np.nan)
        margin = CONFIG.get('joint_limit_margin_deg')
        unlimited = CONFIG.get('unlimited', False)
        if unlimited:
            lower_bounds[valid], upper_bounds[valid] = -np.inf, np.inf
        if margin is not None:
            ranges = np.array([[model.joint(f'{side}_joint{i}').range for i in range(1, 7)] for side in SIDES])
            lower_bounds[:] = ranges[..., 0] - np.deg2rad(margin)
            upper_bounds[:] = ranges[..., 1] + np.deg2rad(margin)
            initial = np.clip(reference, lower_bounds, upper_bounds)
            lower_bounds[~valid], upper_bounds[~valid] = np.nan, np.nan
        usable, converged = np.zeros_like(valid), np.zeros_like(valid)
        segments, failures = [], []
        cache = directory / 'segments'
        cache.mkdir(exist_ok=True)
        for arm, side in enumerate(SIDES):
            radius = trust.joint_radii(model, side, CONFIG['first_joint_radius_rad']) if margin is None and not unlimited else None
            intervals = np.flatnonzero(np.diff(np.r_[False, valid[:, arm], False])).reshape(-1, 2)
            for begin, end in intervals:
                name = f'{side}_{begin:06d}_{end:06d}'
                checkpoint = cache / (name + '.npz')
                try:
                    targets = source['targets_world'][begin:end, arm]
                    seed, ref = initial[begin:end, arm], reference[begin:end, arm]
                    if margin is None and not unlimited:
                        seed_error = trust.pose_errors(model, side, targets, seed)
                        scale = np.sqrt(np.hypot(seed_error[:, 0] / 10., np.deg2rad(seed_error[:, 1]) / .1))
                        width = scale[:, None] * radius
                        lower_bounds[begin:end, arm], upper_bounds[begin:end, arm] = seed-width, seed+width
                    lower, upper = lower_bounds[begin:end, arm], upper_bounds[begin:end, arm]
                    if checkpoint.exists():
                        with np.load(checkpoint) as saved:
                            values = {key: saved[key] for key in saved.files}
                        report = json.loads(str(values['summary']))
                        report.setdefault('hessian_method', 'finite_difference')
                    else:
                        before = time.monotonic()
                        active = dict(episode=record['key'], pid=os.getpid(), side=side,
                                      begin=int(begin), end=int(end), iteration=0, updated=time.time())
                        write_json(active_path, active)

                        def progress(x, state):
                            now = time.time()
                            if now - active['updated'] >= 5.:
                                active.update(iteration=int(state.nit), updated=now,
                                              constraint_violation=float(state.constr_violation))
                                write_json(active_path, active)
                            return False

                        solution = trust.solve_sequence(model, side, targets, seed, ref,
                                                        None if unlimited else lower, None if unlimited else upper,
                                                        maxiter=CONFIG['maxiter'], callback=progress, **CONFIG['weights'])
                        joints = solution.x.reshape(-1, 6)
                        residual = trust.pose_errors(model, side, targets, joints)
                        finite = bool(np.isfinite(joints).all() and np.isfinite(residual).all())
                        if not finite:
                            raise ValueError('Non-finite trust-constr result')
                        violation = 0. if unlimited else max(0., float((lower-joints).max()), float((joints-upper).max()))
                        if violation > 1e-8:
                            raise ValueError(f'Joint bound violation: {violation} rad')
                        report = dict(side=side, begin=int(begin), end=int(end), seconds=time.monotonic()-before,
                                      hessian_method='analytic',
                                      usable=finite, converged=bool(solution.success), status=int(solution.status),
                                      message=str(solution.message), iterations=int(solution.nit),
                                      constraint_jacobian_evaluations=int(solution.constr_njev[0]),
                                      constraint_hessian_evaluations=int(solution.constr_nhev[0]),
                                      constraint_violation=float(solution.constr_violation),
                                      equality_satisfied=bool(solution.constr_violation <= 1e-7),
                                      optimality=float(solution.optimality), joint_cost=float(solution.fun),
                                      max_bound_violation_rad=violation)
                        values = dict(q=joints, errors=residual, lower_bounds=lower, upper_bounds=upper,
                                      summary=np.array(json.dumps(report)))
                        save_arrays(checkpoint, **values)
                    q[begin:end, arm], errors[begin:end, arm] = values['q'], values['errors']
                    usable[begin:end, arm], converged[begin:end, arm] = report['usable'], report['converged']
                    segments.append(report)
                except Exception as error:
                    failure = dict(side=side, begin=int(begin), end=int(end), error=str(error), traceback=traceback.format_exc())
                    failures.append(failure)
                    write_json(cache / (name + '_error.json'), failure)
        available = valid & np.isfinite(q).all(axis=-1)
        stream = str(source.get('input_stream', 'actions'))
        actions = source[stream].copy()
        actions[:, COLUMNS] = q
        source.update(dp_initial=source['initial'], initial=initial, reference=reference,
                      ceres_errors=source['errors'], ceres_solver_success=source['solver_success'],
                      q=q, errors=errors, optimized_mask=available, **{f'optimized_{stream}': actions},
                      solver_usable=usable, solver_success=converged, solver='trust-constr',
                      model_path=CONFIG['model'], **CONFIG['weights'], recalculate_bounds=margin is None and not unlimited,
                      joint_limits_enabled=not unlimited, joint_lower_bounds_rad=lower_bounds,
                      joint_upper_bounds_rad=upper_bounds,
                      constraint_hessian_methods=np.array(sorted({s['hessian_method'] for s in segments})))
        if margin is not None:
            source['joint_limit_margin_deg'] = margin
        save_arrays(directory / 'trajectory.npz', **source)
        observed = errors[available]
        steps = np.rad2deg(np.abs(np.diff(q, axis=0)))[available[:-1] & available[1:]]
        result = dict(episode=record['key'], kind=record['kind'], status='complete' if not failures else 'complete_with_failures',
                      input_sha256=record['input_sha256'], frames=len(q), valid_arm_targets=int(valid.sum()),
                      optimized_arm_targets=int(available.sum()), missing_candidate_arm_targets=int((~valid).sum()),
                      passed=int(np.all(observed <= .1, axis=-1).sum()),
                      error_sum_mm_deg=observed.sum(axis=0).tolist(), error_max_mm_deg=observed.max(axis=0, initial=0.).tolist(),
                      max_joint_step_deg=float(steps.max(initial=0.)),
                      jump_arm_frame_pairs=int((steps.max(axis=-1) > 30).sum()),
                      adjacent_arm_frame_pairs=len(steps), segment_count=len(segments), segment_failures=len(failures),
                      frame_iterations=sum((s['end']-s['begin'])*s['iterations'] for s in segments),
                      frame_evaluations=sum((s['end']-s['begin'])*(s['constraint_jacobian_evaluations']
                                            + 12*s['constraint_hessian_evaluations']) for s in segments),
                      nonconverged_segments=sum(not s['converged'] for s in segments),
                      hessian_methods=sorted({s['hessian_method'] for s in segments}),
                      clipped_initial_joint_values=int(((initial != reference) & valid[..., None]).sum()),
                      clipped_initial_arm_frames=int(((initial != reference).any(axis=-1) & valid).sum()),
                      equality_failed_segments=sum(not s['equality_satisfied'] for s in segments),
                      max_bound_violation_rad=max((s['max_bound_violation_rad'] for s in segments), default=0.),
                      seconds=time.monotonic()-started)
        write_json(directory / 'summary.json', dict(**result, segments=segments, failures=failures))
    except Exception as error:
        result = dict(episode=record['key'], kind=record['kind'], status='failed', input_sha256=record['input_sha256'],
                      error=str(error), traceback=traceback.format_exc(), seconds=time.monotonic()-started)
    write_json(result_path, result)
    active_path.unlink(missing_ok=True)
    return result


def publish(reports, total, started, status):
    complete = [r for r in reports if r['status'] in ('complete', 'complete_with_failures')]
    targets = sum(r['optimized_arm_targets'] for r in complete)
    sums = np.sum([r['error_sum_mm_deg'] for r in complete], axis=0) if complete else np.zeros(2)
    progress = dict(status=status, selected=total, completed=len(complete),
                    failed=sum(r['status'] != 'complete' for r in reports), seconds=time.monotonic()-started,
                    frames=sum(r['frames'] for r in complete), optimized_arm_targets=targets,
                    missing_candidate_arm_targets=sum(r['missing_candidate_arm_targets'] for r in complete),
                    passed=sum(r['passed'] for r in complete), error_mean_mm_deg=(sums/max(1, targets)).tolist(),
                    error_max_mm_deg=np.max([r['error_max_mm_deg'] for r in complete], axis=0).tolist() if complete else [0., 0.],
                    max_joint_step_deg=max((r['max_joint_step_deg'] for r in complete), default=0.),
                    jump_arm_frame_pairs=sum(r['jump_arm_frame_pairs'] for r in complete),
                    nonconverged_segments=sum(r['nonconverged_segments'] for r in complete),
                    equality_failed_segments=sum(r['equality_failed_segments'] for r in complete))
    write_json(OUTPUT / 'progress.json', progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--selection', type=Path, help='Optional list of sequence keys for reusable benchmarks')
    args = parser.parse_args()
    config = json.loads((args.output / 'config.json').read_text())
    initialize(config, args.output.resolve())
    jobs = json.loads((OUTPUT / 'jobs.json').read_text())
    if args.selection:
        selected = set(json.loads(args.selection.read_text()))
        jobs = [record for record in jobs if record['key'] in selected]
    reports, pending = [], []
    for record in jobs:
        path = OUTPUT / record['kind'] / record['key'] / 'result.json'
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and previous['status'] == 'complete' and previous['input_sha256'] == record['input_sha256']:
            reports.append(previous)
        else:
            pending.append(record)
    started = time.monotonic()
    write_json(OUTPUT / 'runtime.json', dict(status='running', pid=os.getpid(), workers=args.workers, started=time.time()))
    publish(reports, len(jobs), started, 'running')
    print(json.dumps(dict(selected=len(jobs), reused=len(reports), pending=len(pending), workers=args.workers)), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(config, str(OUTPUT))) as pool:
        for future in as_completed([pool.submit(run_sequence, record) for record in pending]):
            result = future.result()
            reports.append(result)
            with (OUTPUT / 'results.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
            progress = publish(reports, len(jobs), started, 'running')
            print(json.dumps(dict(completed=progress['completed'], failed=progress['failed'],
                                  episode=result['episode'], seconds=result['seconds'], status=result['status'])), flush=True)
    status = 'complete_with_failures' if any(r['status'] != 'complete' for r in reports) else 'complete'
    progress = publish(reports, len(jobs), started, status)
    write_json(OUTPUT / 'summary.json', dict(**progress, episodes=reports))
    write_json(OUTPUT / 'runtime.json', dict(status=status, workers=args.workers, seconds=time.monotonic()-started, finished=time.time()))


if __name__ == '__main__':
    main()
