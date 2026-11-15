"""Score grid joints against real teleop using batched GPU 6D-cell queries."""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from abc_minimal.eef_pose_grid_gpu import GPUEEFPoseGrid

if __package__:
    from .score_grid_teleop import TeleopPoseIndex, pose_features
else:
    from score_grid_teleop import TeleopPoseIndex, pose_features


DEFAULT_BATCH_SIZE = 1_048_576
DEFAULT_MAX_PAIRS = 64_000_000
SCORE_BYTES = 4
DISK_RESERVE_BYTES = 2 * 2**30
CELL_BITS = 9
POSE_DIMENSIONS = 6
CELL_BIAS = np.array([256, 256, 256, 64, 64, 64])


@wp.func
def relation(centers: wp.array3d(dtype=wp.int32), query: int, radius: int, node: int,
             lower: wp.array2d(dtype=wp.int16), sizes: wp.array(dtype=wp.int32)):
    position_inside = True
    for axis in range(3):
        a = wp.int32(lower[node, axis])
        b = a + sizes[node] - 1
        c = centers[query, 0, axis]
        if b < c - radius or a > c + radius:
            return 0
        position_inside = position_inside and a >= c - radius and b <= c + radius
    intersects = False
    contains = False
    for representation in range(3):
        hit = True
        full = True
        for axis in range(3, 6):
            a = wp.int32(lower[node, axis])
            b = a + sizes[node] - 1
            c = centers[query, representation, axis]
            hit = hit and b >= c - radius and a <= c + radius
            full = full and a >= c - radius and b <= c + radius
        intersects = intersects or hit
        contains = contains or full
    if not intersects:
        return 0
    if position_inside and contains:
        return 2
    return 1


@wp.func
def count_region(centers: wp.array3d(dtype=wp.int32), query: int, radius: int,
                 lower: wp.array2d(dtype=wp.int16), sizes: wp.array(dtype=wp.int32),
                 escape: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                 ends: wp.array(dtype=wp.int32), limit: int):
    node = int(0)
    count = int(0)
    while node < sizes.shape[0] and count < limit:
        hit = relation(centers, query, radius, node, lower, sizes)
        if hit == 2:
            count += ends[node] - starts[node]
            node = escape[node]
        elif hit == 0:
            node = escape[node]
        else:
            node += 1
    return count


@wp.kernel
def find_regions(centers: wp.array3d(dtype=wp.int32), k: int,
                 lower: wp.array2d(dtype=wp.int16), sizes: wp.array(dtype=wp.int32),
                 escape: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                 ends: wp.array(dtype=wp.int32), radii: wp.array(dtype=wp.int32),
                 counts: wp.array(dtype=wp.int32)):
    query = wp.tid()
    radius = int(1)
    while count_region(centers, query, radius, lower, sizes, escape, starts, ends, k) < k:
        radius *= 2
    first = radius // 2
    while radius - first > 1:
        middle = (first + radius) // 2
        if count_region(centers, query, middle, lower, sizes, escape, starts, ends, k) >= k:
            radius = middle
        else:
            first = middle
    radii[query] = radius
    counts[query] = count_region(centers, query, radius, lower, sizes, escape, starts, ends, 2147483647)


@wp.kernel
def gather_candidates(centers: wp.array3d(dtype=wp.int32), selected: wp.array(dtype=wp.int32),
                      radii: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int64),
                      lower: wp.array2d(dtype=wp.int16), sizes: wp.array(dtype=wp.int32),
                      escape: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                      ends: wp.array(dtype=wp.int32), real_rows: wp.array(dtype=wp.int32),
                      candidates: wp.array(dtype=wp.int32)):
    local_query = wp.tid()
    query = selected[local_query]
    radius = radii[query]
    destination = offsets[local_query]
    node = int(0)
    while node < sizes.shape[0]:
        hit = relation(centers, query, radius, node, lower, sizes)
        if hit == 2:
            for row in range(starts[node], ends[node]):
                candidates[destination] = real_rows[row]
                destination += wp.int64(1)
            node = escape[node]
        elif hit == 0:
            node = escape[node]
        else:
            node += 1


@wp.kernel
def candidate_distances(selected: wp.array(dtype=wp.int32), owners: wp.array(dtype=wp.int32),
                        offsets: wp.array(dtype=wp.int64), candidates: wp.array(dtype=wp.int32),
                        real_features: wp.array2d(dtype=wp.float32),
                        query_features: wp.array2d(dtype=wp.float32),
                        distances: wp.array2d(dtype=wp.float32)):
    pair = wp.tid()
    local_query = owners[pair]
    query = selected[local_query]
    real = candidates[pair]
    value = wp.float32(0.0)
    for axis in range(12):
        difference = real_features[real, axis] - query_features[query, axis]
        value += difference * difference
    distances[local_query, wp.int64(pair) - offsets[local_query]] = value


def voxel_tree(index):
    """Arrange occupied 6D cells into contiguous Morton-code subtrees."""
    cells = np.column_stack(np.unravel_index(index.keys, index.shape)) + index.origin
    cells += CELL_BIAS
    if np.any(cells < 0) or np.any(cells >= 1 << CELL_BITS):
        raise ValueError('Reference cells must fit the 9-bit spatial hierarchy')

    codes = np.zeros(len(cells), dtype=np.uint64)
    # Interleave the six coordinates so cells sharing a prefix stay together.
    for bit in range(CELL_BITS):
        for axis in range(POSE_DIMENSIONS):
            shift = POSE_DIMENSIONS * bit + POSE_DIMENSIONS - 1 - axis
            codes |= ((cells[:, axis].astype(np.uint64) >> bit) & 1) << shift
    order = np.argsort(codes)
    codes = codes[order]
    sample_offsets = np.r_[0, np.cumsum(np.diff(index.offsets)[order])]
    real_rows = index.rows[index._expand_ranges(index.offsets[order], index.offsets[order + 1])]

    prefixes, depths = [], []
    for depth in range(CELL_BITS + 1):
        shift = POSE_DIMENSIONS * (CELL_BITS - depth)
        values = np.unique(codes >> shift) << shift
        prefixes.append(values)
        depths.append(np.full(len(values), depth, dtype=np.int32))
    prefixes = np.concatenate(prefixes)
    depths = np.concatenate(depths)
    order = np.lexsort((depths, prefixes))
    prefixes, depths = prefixes[order], depths[order]
    upper_codes = prefixes + (
        np.uint64(1) << (POSE_DIMENSIONS * (CELL_BITS - depths)).astype(np.uint64)
    )

    lower = np.zeros((len(prefixes), POSE_DIMENSIONS), dtype=np.int16)
    for bit in range(CELL_BITS):
        for axis in range(POSE_DIMENSIONS):
            shift = POSE_DIMENSIONS * bit + POSE_DIMENSIONS - 1 - axis
            lower[:, axis] |= (((prefixes >> shift) & 1) << bit).astype(np.int16)
    return dict(
        lower=lower,
        sizes=(1 << (CELL_BITS - depths)).astype(np.int32),
        escape=np.searchsorted(prefixes, upper_codes).astype(np.int32),
        starts=sample_offsets[np.searchsorted(codes, prefixes)].astype(np.int32),
        ends=sample_offsets[np.searchsorted(codes, upper_codes)].astype(np.int32),
        real_rows=real_rows.astype(np.int32),
    )


class GPUScorer:
    """Find EEF neighbors, then take the median of their joint distances."""

    def __init__(self, index, device='cuda:0', max_pairs=DEFAULT_MAX_PAIRS):
        self.index = index
        self.device = device
        self.max_pairs = max_pairs
        wp.init()
        wp.set_device(device)
        started = time.perf_counter()
        tree = voxel_tree(index)
        fields = ('lower', 'sizes', 'escape', 'starts', 'ends')
        self.tree = [wp.array(tree[name], device=device) for name in fields]
        self.real_rows = wp.array(tree['real_rows'], device=device)
        features = pose_features(index.poses, index.position_scale, index.rotation_scale)
        self.real_features = wp.array(features, dtype=wp.float32, device=device)
        self.real_joints = torch.as_tensor(np.asarray(index.joints).astype(np.float32), device=device)
        elapsed = time.perf_counter() - started
        print(f'GPU voxel hierarchy: {len(tree["sizes"]):,} nodes, {elapsed:.2f}s', flush=True)

    def centers(self, poses):
        """Quantize position and three equivalent rotation-vector representations."""
        vector = Rotation.from_rotvec(poses[:, 3:]).as_rotvec()
        norm = np.linalg.norm(vector, axis=1)
        ratio = np.divide(2 * np.pi, norm, out=np.zeros_like(norm), where=norm > 1e-12)
        vectors = np.stack((vector, vector * (1 + ratio[:, None]), vector * (1 - ratio[:, None])), axis=1)
        centers = np.empty((len(poses), 3, POSE_DIMENSIONS), dtype=np.int32)
        centers[:, :, :3] = np.floor(poses[:, None, :3] / self.index.cell_size).astype(np.int32) + CELL_BIAS[:3]
        centers[:, :, 3:] = np.floor(vectors / self.index.rotation_cell_size).astype(np.int32) + CELL_BIAS[3:]
        return centers

    def _find_regions(self, centers, k):
        """Find the smallest cell radius containing at least k real samples."""
        radii = torch.empty(centers.shape[0], dtype=torch.int32, device=self.device)
        counts = torch.empty_like(radii)
        wp.launch(
            find_regions,
            dim=centers.shape[0],
            inputs=[centers, k, *self.tree, wp.from_torch(radii), wp.from_torch(counts)],
        )
        wp.synchronize_device()
        counts_cpu = counts.cpu().numpy()
        if np.any(counts_cpu < k):
            raise ValueError('GPU cell search returned fewer than k samples')
        return radii, counts, counts_cpu

    def _candidate_groups(self, counts_cpu):
        """Group similar candidate counts without exceeding the padded pair budget."""
        order = np.argsort(counts_cpu)
        begin = 0
        while begin < len(order):
            end = begin + 1
            while end < len(order) and (end - begin + 1) * int(counts_cpu[order[end]]) <= self.max_pairs:
                end += 1
            yield order[begin:end]
            begin = end

    def _nearest_neighbors(self, centers, radii, counts, query_features, selected, width, k):
        """Gather region candidates and select the k smallest EEF distances."""
        group_counts = counts[selected.to(torch.int64)]
        offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=self.device), group_counts.cumsum(0)))
        pairs = int(offsets[-1])
        candidates = torch.empty(pairs, dtype=torch.int32, device=self.device)
        wp.launch(
            gather_candidates,
            dim=len(selected),
            inputs=[centers, wp.from_torch(selected), wp.from_torch(radii), wp.from_torch(offsets),
                    *self.tree, self.real_rows, wp.from_torch(candidates)],
        )
        wp.synchronize_device()
        owners = torch.repeat_interleave(
            torch.arange(len(selected), device=self.device, dtype=torch.int32), group_counts
        )
        distances = torch.full((len(selected), width), torch.inf, dtype=torch.float32, device=self.device)
        wp.launch(
            candidate_distances,
            dim=pairs,
            inputs=[wp.from_torch(selected), wp.from_torch(owners), wp.from_torch(offsets),
                    wp.from_torch(candidates), self.real_features, query_features, wp.from_torch(distances)],
        )
        wp.synchronize_device()
        nearest = torch.topk(distances, k, dim=1, largest=False, sorted=False).indices
        return candidates[(offsets[:-1, None] + nearest).ravel()].reshape(-1, k).to(torch.int64)

    def score(self, joints, poses, k=100, details=False):
        poses = np.asarray(poses, dtype=np.float64)
        centers = wp.array(self.centers(poses), device=self.device)
        radii, counts, counts_cpu = self._find_regions(centers, k)
        features = pose_features(poses, self.index.position_scale, self.index.rotation_scale)
        query_features = wp.array(features, dtype=wp.float32, device=self.device)
        queries = torch.as_tensor(joints, dtype=torch.float32, device=self.device)
        scores = torch.empty(len(poses), dtype=torch.float32, device=self.device)
        neighbors = torch.empty((len(poses), k), dtype=torch.int64, device=self.device) if details else None

        for selected_cpu in self._candidate_groups(counts_cpu):
            selected = torch.as_tensor(selected_cpu.astype(np.int32), device=self.device)
            width = int(counts_cpu[selected_cpu].max())
            real = self._nearest_neighbors(centers, radii, counts, query_features, selected, width, k)
            difference = self.real_joints[real] - queries[selected.to(torch.int64), None, :]
            joint_distance = difference.square().sum(-1).sqrt().sort(dim=1).values
            middle = k // 2
            if k % 2:
                median = joint_distance[:, middle]
            else:
                median = joint_distance[:, middle - 1:middle + 1].mean(dim=1)
            scores[selected.to(torch.int64)] = median
            if details:
                neighbors[selected.to(torch.int64)] = real

        result = scores.cpu().numpy()
        if details:
            return result, dict(radii=radii.cpu().numpy(), counts=counts_cpu, neighbors=neighbors.cpu().numpy())
        return result


def save_progress(path, metadata):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--real-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--max-pairs', type=int, default=DEFAULT_MAX_PAIRS)
    parser.add_argument('--checkpoint-every', type=float, default=30)
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--start-row', type=int, default=0)
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def build_score_config(args, grid):
    reference = json.loads((args.real_index / 'metadata.json').read_text())
    grid_metadata = json.loads((args.grid / 'metadata.json').read_text())
    if grid_metadata['config']['model_sha256'] != reference['model_sha256']:
        raise ValueError('Reference model differs from grid')
    np.testing.assert_allclose(
        grid_metadata['config']['joint_ranges_rad'], reference['reference_limits_rad'], rtol=0, atol=0
    )
    if not 0 <= args.start_row < grid.total_rows:
        raise ValueError('Start row is outside the grid')
    remaining = grid.total_rows - args.start_row
    total = min(args.count, remaining) if args.count else remaining
    return dict(
        grid=str(args.grid.resolve()),
        real_index=str(args.real_index.resolve()),
        total_rows=total,
        k=args.k,
        start_row=args.start_row,
        metric='median joint L2 radians among EEF top-k in the first 6D radius with k real samples',
        grid_config=grid_metadata['config'],
        reference_config=reference,
        algorithm='gpu-6d-voxel-median-v2',
        compute_dtype='float32',
        dtype='float32',
        reference_files={
            name: dict(bytes=(args.real_index / name).stat().st_size,
                       mtime_ns=(args.real_index / name).stat().st_mtime_ns)
            for name in ('joints.npy', 'poses.npy', 'cell_rows.npy', 'cell_keys.npy', 'cell_offsets.npy')
        },
    )


def load_checkpoint(output, config):
    metadata_path = output / 'metadata.json'
    if not metadata_path.exists():
        return dict(config=config, status='running', completed=0)
    metadata = json.loads(metadata_path.read_text())
    if metadata['config'] != config:
        raise ValueError('Existing score settings differ')
    return metadata


def check_score_storage(output, total, completed):
    score_path = output / 'distance_median.bin'
    if score_path.exists() and score_path.stat().st_size < completed * SCORE_BYTES:
        raise ValueError('Score file is shorter than the committed checkpoint')
    required = (total - completed) * SCORE_BYTES + DISK_RESERVE_BYTES
    if shutil.disk_usage(output).free < required:
        raise OSError('Insufficient space for one float32 score per remaining grid row')


def save_score_checkpoint(stream, metadata_path, metadata, completed, initial_completed, started):
    """Commit score bytes before publishing the corresponding row count."""
    stream.flush()
    os.fsync(stream.fileno())
    now = time.perf_counter()
    total = metadata['config']['total_rows']
    metadata.update(
        completed=completed,
        status='complete' if completed == total else 'running',
        elapsed_seconds=now - started,
        rows_per_second=(completed - initial_completed) / (now - started),
        committed_bytes=completed * SCORE_BYTES,
    )
    save_progress(metadata_path, metadata)
    print(f'scored {completed:,}/{total:,}, {metadata["rows_per_second"]:,.0f} rows/s', flush=True)
    return now


def score_batches(args, grid, scorer, metadata):
    metadata_path = args.output / 'metadata.json'
    score_path = args.output / 'distance_median.bin'
    total = metadata['config']['total_rows']
    completed = metadata['completed']
    save_progress(metadata_path, metadata)
    started = time.perf_counter()
    last_checkpoint = started

    with score_path.open('r+b' if score_path.exists() else 'w+b') as stream:
        stream.truncate(completed * SCORE_BYTES)
        stream.seek(completed * SCORE_BYTES)
        for offset in range(completed, total, args.batch_size):
            start = args.start_row + offset
            stop = args.start_row + min(offset + args.batch_size, total)
            rows = np.arange(start, stop, dtype=np.int64)
            poses = np.asarray(grid.poses[start:stop], dtype=np.float64)
            values = scorer.score(grid.joints(rows), poses, args.k)
            if not np.isfinite(values).all():
                raise ValueError('Non-finite similarity score')
            stream.write(values.astype('<f4').tobytes())
            now = time.perf_counter()
            batch_completed = offset + len(rows)
            if now - last_checkpoint >= args.checkpoint_every or batch_completed == total:
                last_checkpoint = save_score_checkpoint(
                    stream, metadata_path, metadata, batch_completed, completed, started
                )


def main():
    args = parse_args()
    grid = GPUEEFPoseGrid(args.grid)
    config = build_score_config(args, grid)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = load_checkpoint(args.output, config)
    if metadata['completed'] == config['total_rows']:
        print('Already complete', flush=True)
        return
    check_score_storage(args.output, config['total_rows'], metadata['completed'])
    index = TeleopPoseIndex.load(args.real_index)
    scorer = GPUScorer(index, args.device, args.max_pairs)
    score_batches(args, grid, scorer, metadata)


if __name__ == '__main__':
    main()
