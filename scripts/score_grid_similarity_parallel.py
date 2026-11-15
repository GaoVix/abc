"""Run contiguous GPU score shards and merge them in grid-row order."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .score_grid_similarity_gpu import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_MAX_PAIRS,
        DISK_RESERVE_BYTES,
        SCORE_BYTES,
        save_progress,
    )
else:
    from score_grid_similarity_gpu import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_MAX_PAIRS,
        DISK_RESERVE_BYTES,
        SCORE_BYTES,
        save_progress,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--real-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--max-pairs', type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--shard-directories', type=Path, nargs='+')
    parser.add_argument('--keep-shards', action='store_true', help='Keep score shards without merging')
    return parser.parse_args()


def build_parallel_config(args, total, grid_metadata):
    return dict(
        grid=str(args.grid.resolve()),
        real_index=str(args.real_index.resolve()),
        total_rows=total,
        workers=args.workers,
        compute_dtype='float32',
        algorithm='gpu-6d-voxel-median-v2',
        k=100,
        grid_config=grid_metadata['config'],
        reference_config=json.loads((args.real_index / 'metadata.json').read_text()),
    )


def load_checkpoint(output, config):
    metadata_path = output / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata['config'] != config:
            raise ValueError('Existing parallel score settings differ')
    else:
        metadata = dict(config=config, status='running', merged_rows=0)
        save_progress(metadata_path, metadata)
    return metadata


def check_merge_storage(output, ranges, paths, total):
    existing = sum(
        (path / 'distance_median.bin').stat().st_size
        for path in paths if (path / 'distance_median.bin').exists()
    )
    final_path = output / 'distance_median.bin'
    existing += final_path.stat().st_size if final_path.exists() else 0
    largest_shard_bytes = max(end - start for start, end in ranges) * SCORE_BYTES
    required = max(0, total * SCORE_BYTES - existing) + largest_shard_bytes + DISK_RESERVE_BYTES
    if shutil.disk_usage(output).free < required:
        raise OSError('Insufficient space for scores and one shard of merge workspace')


def check_shard_storage(ranges, paths):
    volumes = {}
    for (start, end), path in zip(ranges, paths):
        path.mkdir(parents=True, exist_ok=True)
        score = path / 'distance_median.bin'
        existing = score.stat().st_size if score.exists() else 0
        device = path.stat().st_dev
        volume = volumes.setdefault(device, [path, 0])
        volume[1] += max(0, (end - start) * SCORE_BYTES - existing)
    for path, required in volumes.values():
        if shutil.disk_usage(path).free < required + DISK_RESERVE_BYTES:
            raise OSError(f'Insufficient space for score shards on {path}')


def worker_command(args, start, end, output):
    return [
        sys.executable,
        str(Path(__file__).with_name('score_grid_similarity_gpu.py')),
        '--grid', str(args.grid),
        '--real-index', str(args.real_index),
        '--output', str(output),
        '--start-row', str(start),
        '--count', str(end - start),
        '--batch-size', str(args.batch_size),
        '--max-pairs', str(args.max_pairs),
    ]


@contextmanager
def worker_processes(args, ranges, paths):
    """Own worker processes and log handles, including cleanup after failures."""
    processes, logs = [], []
    try:
        for rank, ((start, end), path) in enumerate(zip(ranges, paths)):
            path.mkdir(parents=True, exist_ok=True)
            log = (path / 'worker.log').open('a', buffering=1)
            logs.append(log)
            process = subprocess.Popen(
                worker_command(args, start, end, path), stdout=log, stderr=subprocess.STDOUT
            )
            processes.append(process)
            print(f'worker {rank}: PID={process.pid}, rows=[{start:,}, {end:,})', flush=True)
        yield processes
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for log in logs:
            log.close()


def wait_for_workers(processes, paths, progress_path, total, started):
    initial_completed = None
    while True:
        completed = 0
        shards = []
        for process, path in zip(processes, paths):
            code = process.poll()
            if code not in (None, 0):
                raise RuntimeError(f'Worker PID {process.pid} exited with {code}; see {path / "worker.log"}')
            metadata_path = path / 'metadata.json'
            progress = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            completed += progress.get('completed', 0)
            shards.append(dict(
                pid=process.pid,
                completed=progress.get('completed', 0),
                rows_per_second=progress.get('rows_per_second', 0),
            ))
        if initial_completed is None:
            initial_completed = completed
        elapsed = time.perf_counter() - started
        rate = (completed - initial_completed) / elapsed
        save_progress(progress_path, dict(
            status='running', completed=completed, total_rows=total,
            elapsed_seconds=elapsed, rows_per_second=rate, shards=shards,
        ))
        print(f'total {completed:,}/{total:,}, {rate:,.0f} rows/s', flush=True)
        if all(process.poll() == 0 for process in processes):
            return
        time.sleep(30)


def validate_shards(ranges, paths):
    for (start, end), path in zip(ranges, paths):
        progress = json.loads((path / 'metadata.json').read_text())
        if progress['status'] != 'complete' or progress['completed'] != end - start:
            raise ValueError(f'Incomplete shard: {path}')
        if (path / 'distance_median.bin').stat().st_size != (end - start) * SCORE_BYTES:
            raise ValueError(f'Incorrect shard size: {path}')


def merge_shards(output, ranges, paths, metadata):
    """Reuse the first shard, then append and release each remaining shard."""
    final_path = output / 'distance_median.bin'
    for (start, end), path in zip(ranges, paths):
        source = path / 'distance_median.bin'
        if end <= metadata['merged_rows']:
            source.unlink(missing_ok=True)
            continue
        print(f'merging rows [{start:,}, {end:,})', flush=True)
        if start == 0 and not final_path.exists():
            source.replace(final_path)
        elif start == 0 and not source.exists() and final_path.stat().st_size == end * SCORE_BYTES:
            pass
        else:
            with final_path.open('r+b') as destination, source.open('rb') as stream:
                destination.truncate(start * SCORE_BYTES)
                destination.seek(start * SCORE_BYTES)
                shutil.copyfileobj(stream, destination, length=64 * 2**20)
                destination.flush()
                os.fsync(destination.fileno())
        metadata['merged_rows'] = end
        save_progress(output / 'metadata.json', metadata)
        source.unlink(missing_ok=True)
    return final_path


def mark_complete(args, grid_metadata, metadata, final_path, total):
    if final_path.stat().st_size != total * SCORE_BYTES:
        raise ValueError('Incorrect merged score size')
    metadata.update(status='complete', dtype='float32', total_rows=total, bytes=total * SCORE_BYTES)
    save_progress(args.output / 'metadata.json', metadata)
    save_progress(args.output / 'progress.json', dict(status='complete', completed=total, total_rows=total))
    register_similarity(args, grid_metadata, total, dict(path=str(final_path.resolve())))
    print(f'Complete: {final_path}, {total:,} scores', flush=True)


def register_similarity(args, grid_metadata, total, storage):
    if total != grid_metadata['total_rows']:
        return
    reference = json.loads((args.real_index / 'metadata.json').read_text())
    side = reference['side']
    descriptor = dict(
        **storage, dtype='float32', k=100, statistic='median_joint_l2_rad', rows=total,
        reference_side=side, reference_streams=['state', 'action'] if reference['stream'] == 'both'
        else [reference['stream']], reference_index=str(args.real_index.resolve()),
    )
    by_side = grid_metadata.setdefault('similarity_by_side', {})
    if 'similarity' in grid_metadata:
        previous = grid_metadata['similarity']
        by_side.setdefault(previous.get('reference_side', 'left'), previous)
    by_side[side] = descriptor
    if side == 'left':
        grid_metadata['similarity'] = descriptor
    save_progress(args.grid / 'metadata.json', grid_metadata)


def main():
    args = parse_args()
    grid_metadata = json.loads((args.grid / 'metadata.json').read_text())
    total = min(args.count, grid_metadata['total_rows']) if args.count else grid_metadata['total_rows']
    ranges = [(total * rank // args.workers, total * (rank + 1) // args.workers) for rank in range(args.workers)]
    paths = [args.output / 'shards' / f'part_{rank:02d}' for rank in range(args.workers)]
    if args.shard_directories:
        if not args.keep_shards or len(args.shard_directories) != args.workers:
            raise ValueError('Custom shard directories require --keep-shards and one directory per worker')
        paths = [path.resolve() for path in args.shard_directories]
    args.output.mkdir(parents=True, exist_ok=True)
    config = build_parallel_config(args, total, grid_metadata)
    if args.keep_shards:
        config.update(keep_shards=True, shard_directories=[str(path.resolve()) for path in paths])
    metadata = load_checkpoint(args.output, config)
    if metadata['status'] == 'complete':
        print('Already complete', flush=True)
        return
    if args.keep_shards:
        check_shard_storage(ranges, paths)
    else:
        check_merge_storage(args.output, ranges, paths, total)

    if metadata['merged_rows'] == 0 and metadata['status'] != 'merging':
        started = time.perf_counter()
        with worker_processes(args, ranges, paths) as processes:
            wait_for_workers(processes, paths, args.output / 'progress.json', total, started)
            validate_shards(ranges, paths)
        if args.keep_shards:
            shards = [dict(start_row=start, end_row=end, path=str((path / 'distance_median.bin').resolve()))
                      for (start, end), path in zip(ranges, paths)]
            metadata.update(status='complete', dtype='float32', total_rows=total,
                            bytes=total * SCORE_BYTES, shards=shards)
            save_progress(args.output / 'metadata.json', metadata)
            save_progress(args.output / 'progress.json', dict(status='complete', completed=total, total_rows=total))
            register_similarity(args, grid_metadata, total, dict(shards=shards))
            print(f'Complete: {total:,} scores in {len(shards)} shards', flush=True)
            return
        metadata['status'] = 'merging'
        save_progress(args.output / 'metadata.json', metadata)

    final_path = merge_shards(args.output, ranges, paths, metadata)
    mark_complete(args, grid_metadata, metadata, final_path, total)


if __name__ == '__main__':
    main()
