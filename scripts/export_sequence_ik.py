"""Export final state/action sequence IK results into the official training layout."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
from functools import partial
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from abc_minimal.export_mcap import FPS, TIMESCALE, TICKS_PER_FRAME, X264_STRICT_FFMPEG_ARGS


VIDEO = 'combined_camera-images-rgb.mp4'
ARM_COLUMNS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_solution(path, stream):
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key] for key in (
            f'optimized_{stream}', 'frame_indices', 'errors', 'valid_mask', 'optimized_mask',
            'camera_from_base')}
        if str(data['base_mode']) != 'median' or str(data['solver']) != 'trust-constr':
            raise ValueError(f'Expected final median-base equality IK: {path}')
    return values


def kept_intervals(keep, blocked_edges, minimum):
    if not len(keep):
        return []
    starts = np.flatnonzero(keep & np.r_[True, ~keep[:-1] | blocked_edges])
    ends = np.flatnonzero(keep & np.r_[~keep[1:] | blocked_edges, True]) + 1
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= minimum]


def select_frames(state, action, args):
    indices = state['frame_indices']
    np.testing.assert_array_equal(indices, action['frame_indices'])
    np.testing.assert_allclose(state['camera_from_base'], action['camera_from_base'], atol=1e-12, rtol=0)
    solutions = (state, action)
    rows = np.concatenate((state['optimized_states'], action['optimized_actions']), axis=-1)
    available = np.stack([x['valid_mask'] & x['optimized_mask']
                          & np.isfinite(x['errors']).all(axis=-1) for x in solutions], axis=1)
    available &= np.isfinite(rows[:, [*ARM_COLUMNS, *[14 + i for i in ARM_COLUMNS]]]).reshape(-1, 2, 2, 6).all(axis=-1)
    keep = available.all(axis=(1, 2)) & np.isfinite(rows).all(axis=-1)
    pose_bad = np.stack([(x['errors'] > [args.position_tolerance_mm, args.rotation_tolerance_deg]).any(axis=-1)
                         for x in solutions], axis=1)
    keep &= ~pose_bad.any(axis=(1, 2))
    gaps = np.diff(indices) != 1
    joint_rows = rows[:, [*ARM_COLUMNS, *[14 + i for i in ARM_COLUMNS]]]
    jumps = np.max(np.abs(np.diff(joint_rows, axis=0)), axis=-1) > np.deg2rad(args.jump_threshold_deg)
    blocked = gaps | (jumps if args.jump_threshold_deg > 0 else False)
    intervals = kept_intervals(keep, blocked, args.chunk_length)
    return rows, intervals, dict(
        frames=len(rows), unavailable_frames=int((~available.all(axis=(1, 2))).sum()),
        rejected_frames=int((~keep).sum()), jump_boundaries=int((jumps & ~gaps).sum()) if args.jump_threshold_deg > 0 else 0,
        retained_frames=sum(b - a for a, b in intervals),
        original_chunks=max(0, len(rows) - args.chunk_length + 1),
        usable_chunks=sum(b - a - args.chunk_length + 1 for a, b in intervals))


def probe_video(path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=nb_frames,r_frame_rate,time_base,start_pts', '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(command, text=True))['streams'][0]


def export_video(source, target, begin, end, total, info):
    strict = info['time_base'] == f'1/{TIMESCALE}' and int(info.get('start_pts', 0)) == 0
    if begin == 0 and end == total and strict:
        target.symlink_to(source.resolve())
        return
    filters = f'trim=start_frame={begin}:end_frame={end},settb=1/{TIMESCALE},setpts=N*{TICKS_PER_FRAME}'
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '1', '-i', str(source),
               '-map', '0:v:0', '-vf', filters, '-frames:v', str(end - begin), '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', *X264_STRICT_FFMPEG_ARGS, str(target)]
    subprocess.run(command, check=True, capture_output=True)
    if int(probe_video(target)['nb_frames']) != end - begin:
        raise ValueError(f'Video frame count mismatch: {target}')


def segment_metadata(metadata, begin, end, name, conversion):
    result = dict(metadata, episode_id=name, num_steps=end - begin, sequence_ik=conversion,
                  source_frame_range=[begin, end])
    for key in ('scene_xml_file', 'scene_qpos_file', 'initial_qpos_file', 'initial_qpos_dim'):
        result.pop(key, None)
    if 't0_ns' in result:
        result['t0_ns'] += begin * result['tick_ns']
    if 'top_camera_timestamps_ns' in result:
        result['top_camera_timestamps_ns'] = result['top_camera_timestamps_ns'][begin:end]
    timeline = result.get('prompt_timeline')
    if timeline:
        active = [entry for entry in timeline if entry['frame'] <= begin]
        first = dict(active[-1] if active else timeline[0], frame=0)
        result['prompt_timeline'] = [first] + [dict(e, frame=e['frame'] - begin)
                                             for e in timeline if begin < e['frame'] < end]
    return result


def export_episode(job, args, settings):
    key, split, source, action_path, state_path = job
    report_path = args.output / 'reports' / f'{key}.json'
    work = args.output / '.work' / key
    try:
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
        for p in (action_path, state_path, source / 'episode_metadata.json'):
            digest.update(p.read_bytes())
        for p in sorted(source.iterdir()):
            stat = p.stat()
            digest.update(str((p.name, stat.st_size, stat.st_mtime_ns)).encode())
        signature = digest.hexdigest()
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if previous['signature'] != signature:
                raise ValueError('Existing export uses different inputs or settings; choose another output')
            for begin, end in previous['intervals']:
                folder = args.output / split / f'{source.name}_frames_{begin:06d}_{end:06d}'
                if not (folder / VIDEO).exists() or (folder / 'states_actions.bin').stat().st_size != (end - begin) * 28 * 8:
                    raise ValueError(f'Existing exported segment is incomplete: {folder}')
            return dict(previous, resumed=True)
        metadata = json.loads((source / 'episode_metadata.json').read_text())
        if 'joint_conversion' in metadata or 'sequence_ik' in metadata:
            raise ValueError('Use the original official dataset as --source-root')
        state, action = load_solution(state_path, 'states'), load_solution(action_path, 'actions')
        indices = state['frame_indices']
        original = np.memmap(source / 'states_actions.bin', dtype=np.float64, mode='r').reshape(-1, 28)
        if len(indices) != len(original) or not np.array_equal(indices, np.arange(len(original))):
            raise ValueError('Result frames do not match the original official episode')
        np.testing.assert_array_equal(state['optimized_states'][:, [6, 13]], original[:, [6, 13]])
        np.testing.assert_array_equal(action['optimized_actions'][:, [6, 13]], original[:, [20, 27]])
        rows, intervals, counts = select_frames(state, action, args)
        conversion = dict(signature=signature, source_episode=source.name, base_mode='median',
                          camera_from_base=state['camera_from_base'].tolist(),
                          action_result=str(action_path), state_result=str(state_path),
                          timing='unchanged official frame indices; no interpolation or timeoffset',
                          normalized=False, **settings['filters'])
        result = dict(episode=key, split=split, signature=signature, status='complete',
                      segments=len(intervals), intervals=intervals, **counts)
        if args.dry_run:
            return result
        info = probe_video(source / VIDEO)
        if int(info['nb_frames']) != len(original) or info['r_frame_rate'] != f'{FPS}/1':
            raise ValueError('Original video does not match the official 30 Hz state/action rows')
        if work.exists():
            shutil.rmtree(work)
        subtasks = json.loads((source / 'subtasks.json').read_text()) if (source / 'subtasks.json').exists() else {}
        for begin, end in intervals:
            name = f'{source.name}_frames_{begin:06d}_{end:06d}'
            destination = args.output / split / name
            if destination.exists():
                saved = json.loads((destination / 'episode_metadata.json').read_text())
                if saved.get('sequence_ik', {}).get('signature') != signature:
                    raise ValueError(f'Existing segment uses different inputs: {destination}')
                if not (destination / VIDEO).exists() or (destination / 'states_actions.bin').stat().st_size != (end - begin) * 28 * 8:
                    raise ValueError(f'Existing exported segment is incomplete: {destination}')
                continue
            segment = work / name
            segment.mkdir(parents=True)
            np.asarray(rows[begin:end], dtype=np.float64).tofile(segment / 'states_actions.bin')
            export_video(source / VIDEO, segment / VIDEO, begin, end, len(original), info)
            save_json(segment / 'episode_metadata.json', segment_metadata(metadata, begin, end, name, conversion))
            if subtasks:
                save_json(segment / 'subtasks.json', {str(i - begin): value for frame, value in subtasks.items()
                                                    if begin <= (i := int(frame)) < end})
            for filename in ('operator.json', 'randomization.json'):
                if (source / filename).exists():
                    shutil.copy2(source / filename, segment / filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            segment.rename(destination)
        if work.exists():
            shutil.rmtree(work)
        save_json(report_path, result)
        return result
    except Exception as error:
        return dict(episode=key, split=split, status='failed', error=f'{type(error).__name__}: {error}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Original train/val real/sim directories')
    parser.add_argument('--action-results', type=Path, required=True)
    parser.add_argument('--state-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--position-tolerance-mm', type=float, default=.1)
    parser.add_argument('--rotation-tolerance-deg', type=float, default=.1)
    parser.add_argument('--jump-threshold-deg', type=float, default=30., help='Split at larger raw joint steps; 0 disables')
    parser.add_argument('--chunk-length', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episodes', nargs='+', help='Optional episode IDs or prefixes for a pilot')
    parser.add_argument('--dry-run', action='store_true', help='Count segments/chunks without writing episodes or videos')
    args = parser.parse_args()
    for name in ('source_root', 'action_results', 'state_results', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    source = {f'{kind}_{p.name}': (f'{split}_{kind}', p) for kind in ('real', 'sim') for split in ('train', 'val')
              for p in (args.source_root / f'{split}_{kind}').glob('episode_*') if (p / 'states_actions.bin').exists()}
    results = [{f'{kind}_{p.parent.name.removeprefix(kind + "_")}': p for kind in ('real', 'sim')
                for p in (root / kind).glob('*/trajectory.npz')} for root in (args.action_results, args.state_results)]
    keys = sorted(results[0])
    if args.episodes:
        keys = [key for key in keys if any(name in key for name in args.episodes)]
    missing = [key for key in keys if key not in source or key not in results[1]]
    if not keys or missing:
        raise ValueError(f'No matching episodes or missing original/state results: {missing[:5]} ({len(missing)} missing)')
    filters = {key: getattr(args, key) for key in (
        'position_tolerance_mm', 'rotation_tolerance_deg', 'jump_threshold_deg', 'chunk_length')}
    settings = dict(source_root=str(args.source_root), action_results=str(args.action_results),
                    state_results=str(args.state_results), filters=filters,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'export.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = args.output / 'config.json'
        if config.exists() and json.loads(config.read_text()) != settings:
            raise ValueError('Output settings changed; choose a new output directory')
        save_json(config, settings)
        if not args.dry_run:
            for split in ('train_real', 'val_real', 'train_sim', 'val_sim'):
                (args.output / split).mkdir(exist_ok=True)
            stats = next((p for p in (args.source_root / 'norm_stats.json', args.source_root.parent / 'norm_stats.json')
                          if p.exists()), None)
            if stats is not None:
                shutil.copy2(stats, args.output / 'norm_stats.json')
        jobs = [(key, *source[key], results[0][key], results[1][key]) for key in keys]
        reports, started = [], time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(partial(export_episode, args=args, settings=settings), job) for job in jobs]
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(json.dumps(dict(completed=len(reports), selected=len(jobs), **report)), flush=True)
        failed = sum(r['status'] == 'failed' for r in reports)
        totals = {key: sum(r.get(key, 0) for r in reports) for key in (
            'frames', 'retained_frames', 'segments', 'original_chunks', 'usable_chunks')}
        totals['usable_chunk_percent'] = 100 * totals['usable_chunks'] / max(totals['original_chunks'], 1)
        summary = dict(status='complete' if not failed else 'failed', selected=len(jobs), failed=failed,
                       seconds=time.monotonic() - started, dry_run=args.dry_run,
                       totals=totals, episodes=reports)
        save_json(args.output / ('dry_run_summary.json' if args.dry_run else 'summary.json'), summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'episodes'}, indent=2), flush=True)
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
