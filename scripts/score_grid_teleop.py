"""Find teleop neighbors in base-frame EEF space, then compare joint angles."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from scipy.spatial.transform import Rotation

from abc_minimal.eef_pose_grid_gpu import GPUEEFPoseGrid
from abc_minimal.episode_io import load_episode

if __package__:
    from .build_eef_workspace import arm_model
else:
    from build_eef_workspace import arm_model


ROOT = Path(__file__).resolve().parents[1]


def real_joints(path, side, stream='both'):
    files = [path] if path.is_file() else sorted(path.rglob('states_actions.bin'))
    if not files:
        files = sorted(path.glob('*.npz'))
    if not files:
        files = sorted(path.rglob('episode.mcap'))
    columns = slice(0, 6) if side == 'left' else slice(7, 13)
    streams = ('state', 'action') if stream == 'both' else (stream,)
    arrays = []
    for source in files:
        if source.suffix == '.npy':
            values = np.load(source, allow_pickle=False)
        elif source.suffix == '.npz':
            with np.load(source, allow_pickle=False) as saved:
                values = np.concatenate([saved[name + 's'] for name in streams])
        elif source.suffix == '.mcap':
            topics = {f'/{side}-arm-{name}': [] for name in streams}
            with source.open('rb') as handle:
                reader = make_reader(handle, decoder_factories=[DecoderFactory()])
                for _, channel, _, message in reader.iter_decoded_messages(
                    topics=list(topics), log_time_order=False
                ):
                    topics[channel.topic].append(list(message.position))
            values = np.concatenate([np.stack(rows) for rows in topics.values()])
        else:
            metadata, states, actions = load_episode(source.parent)
            if 'joint_conversion' in metadata:
                raise ValueError(f'{source}: use original real data, not retargeted joints')
            values = np.concatenate([states if name == 'state' else actions for name in streams])
        arrays.append(values if values.shape[1] == 6 else values[:, columns])
    return np.concatenate(arrays).astype(np.float64)


def fk_poses(model, joints, site_name):
    data = mujoco.MjData(model)
    site = model.site(site_name).id
    result = np.empty((len(joints), 6))
    for index, q in enumerate(joints):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        result[index, :3] = data.site_xpos[site]
        result[index, 3:] = Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_rotvec()
        if (index + 1) % 100000 == 0:
            print(f'real FK: {index + 1}/{len(joints)}', flush=True)
    return result


def pose_features(poses, position_scale, rotation_scale):
    rotations = Rotation.from_rotvec(poses[:, 3:]).as_matrix().reshape(-1, 9)
    return np.column_stack((poses[:, :3] / position_scale,
                            rotations / (np.sqrt(2) * rotation_scale)))


class TeleopPoseIndex:
    def __init__(self, joints, poses, position_scale=0.01, rotation_scale=0.1,
                 cell_size=0.01, rotation_cell_size=0.1):
        self.joints, self.poses = joints, poses
        self.position_scale, self.rotation_scale = position_scale, rotation_scale
        self.cell_size, self.rotation_cell_size = cell_size, rotation_cell_size
        cells = np.floor(poses / self.cell_steps).astype(np.int64)
        self.origin = cells.min(axis=0)
        self.shape = cells.max(axis=0) - self.origin + 1
        keys = np.ravel_multi_index((cells - self.origin).T, self.shape)
        order = np.argsort(keys, kind='stable')
        self.rows = order.astype(np.uint32 if len(joints) < 2 ** 32 else np.uint64)
        self.keys, starts = np.unique(keys[order], return_index=True)
        self.offsets = np.r_[starts, len(joints)]
        self._prepare_cells()

    @property
    def cell_steps(self):
        return np.array([self.cell_size] * 3 + [self.rotation_cell_size] * 3)

    def _prepare_cells(self):
        self.rotation_cells_per_position = int(np.prod(self.shape[3:]))
        rotations = np.column_stack(np.unravel_index(
            self.keys % self.rotation_cells_per_position, self.shape[3:]))
        self.rotation_cells = (rotations + self.origin[3:]).astype(np.int32)

    @staticmethod
    def _expand_ranges(begin, end):
        counts = end - begin
        starts = np.r_[0, np.cumsum(counts)]
        return np.repeat(begin - starts[:-1], counts) + np.arange(starts[-1])

    def _cells_in_radius(self, center, rotations, radius):
        begin = np.clip(center - radius, 0, self.shape[:3])
        end = np.clip(center + radius + 1, 0, self.shape[:3])
        if np.any(begin >= end):
            return np.empty(0, dtype=np.int64)
        x, y = np.meshgrid(np.arange(begin[0], end[0]), np.arange(begin[1], end[1]), indexing='ij')
        base = (x.ravel() * self.shape[1] + y.ravel()) * self.shape[2]
        first = np.searchsorted(self.keys, (base + begin[2]) * self.rotation_cells_per_position)
        last = np.searchsorted(self.keys, (base + end[2]) * self.rotation_cells_per_position)
        cells = self._expand_ranges(first, last)
        inside = np.zeros(len(cells), dtype=bool)
        for rotation in rotations:
            inside |= (np.abs(self.rotation_cells[cells] - rotation) <= radius).all(axis=1)
        return cells[inside]

    def _query_centers(self, pose):
        center = np.floor(pose[:3] / self.cell_size).astype(np.int64) - self.origin[:3]
        vector = Rotation.from_rotvec(pose[3:]).as_rotvec()
        angle = np.linalg.norm(vector)
        representations = [vector]
        if angle > 1e-12:
            representations.extend((vector * (1 + 2 * np.pi / angle),
                                    vector * (1 - 2 * np.pi / angle)))
        rotations = np.floor(np.asarray(representations) / self.rotation_cell_size).astype(np.int64)
        return center, rotations

    def candidate_indices(self, pose, k):
        center, rotations = self._query_centers(pose)
        radius = 1
        cells = self._cells_in_radius(center, rotations, radius)
        while (self.offsets[cells + 1] - self.offsets[cells]).sum() < k:
            radius *= 2
            cells = self._cells_in_radius(center, rotations, radius)
        lower = radius // 2
        while radius - lower > 1:
            middle = (radius + lower) // 2
            selected = self._cells_in_radius(center, rotations, middle)
            if (self.offsets[selected + 1] - self.offsets[selected]).sum() >= k:
                radius = middle
                cells = selected
            else:
                lower = middle
        positions = self._expand_ranges(self.offsets[cells], self.offsets[cells + 1])
        return self.rows[positions], radius

    def _nearest(self, pose, k):
        rows, _ = self.candidate_indices(pose, k)
        features = pose_features(self.poses[rows], self.position_scale, self.rotation_scale)
        target = pose_features(pose[None], self.position_scale, self.rotation_scale)[0]
        squared = np.sum((features - target) ** 2, axis=1)
        selected = np.argpartition(squared, k - 1)[:k]
        selected = selected[np.argsort(squared[selected], kind='stable')]
        return rows[selected], np.sqrt(squared[selected])

    def save(self, path, metadata=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / 'joints.npy', self.joints, allow_pickle=False)
        np.save(path / 'poses.npy', self.poses, allow_pickle=False)
        np.save(path / 'cell_rows.npy', self.rows, allow_pickle=False)
        np.save(path / 'cell_offsets.npy', self.offsets, allow_pickle=False)
        np.save(path / 'cell_keys.npy', self.keys, allow_pickle=False)
        metadata = dict(metadata or {})
        metadata.update(status='complete', samples=len(self.joints), index_type='pose_6d_voxels',
                        cell_size_m=self.cell_size, rotation_cell_size_rad=self.rotation_cell_size,
                        cell_origin=self.origin.tolist(), cell_shape=self.shape.tolist(),
                        occupied_cells=len(self.keys),
                        search='Start at radius 1 in all six dimensions; expand to first radius with k samples',
                        rotation_boundary='Query principal rotvec and equivalent +/- 2*pi representations; deduplicate cells',
                        global_topk=False,
                        position_scale=self.position_scale, rotation_scale=self.rotation_scale,
                        frame='arm-base local TCP',
                        pose_columns=['x_m', 'y_m', 'z_m', 'rotvec_x_rad', 'rotvec_y_rad', 'rotvec_z_rad'])
        (path / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    @classmethod
    def load(cls, path):
        """Load a saved fixed-cell index without recomputing FK."""
        path = Path(path)
        metadata = json.loads((path / 'metadata.json').read_text())
        result = cls.__new__(cls)
        result.joints = np.load(path / 'joints.npy', mmap_mode='r', allow_pickle=False)
        result.poses = np.load(path / 'poses.npy', mmap_mode='r', allow_pickle=False)
        result.position_scale = metadata['position_scale']
        result.rotation_scale = metadata['rotation_scale']
        result.cell_size = metadata['cell_size_m']
        result.rotation_cell_size = metadata['rotation_cell_size_rad']
        result.origin = np.asarray(metadata['cell_origin'])
        result.shape = np.asarray(metadata['cell_shape'])
        result.rows = np.load(path / 'cell_rows.npy', mmap_mode='r', allow_pickle=False)
        result.offsets = np.load(path / 'cell_offsets.npy', mmap_mode='r', allow_pickle=False)
        result.keys = np.load(path / 'cell_keys.npy', mmap_mode='r', allow_pickle=False)
        result._prepare_cells()
        return result

    def query(self, joints, poses, k=10, workers=1):
        if not 1 <= k <= len(self.joints):
            raise ValueError('k must be between 1 and the number of real samples')
        if workers == 1:
            results = [self._nearest(pose, k) for pose in poses]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda pose: self._nearest(pose, k), poses))
        indices = np.stack([item[0] for item in results])
        distance = np.stack([item[1] for item in results])
        neighbors = self.joints[indices]
        rotations = Rotation.from_rotvec(poses[:, 3:])[np.repeat(np.arange(len(poses)), k)]
        neighbor_rotations = Rotation.from_rotvec(self.poses[indices.ravel(), 3:])
        rotation_error = (rotations.inv() * neighbor_rotations).magnitude()
        return {
            'real_indices': indices,
            'real_joints': neighbors,
            'pose_distance': distance,
            'position_error_m': np.linalg.norm(poses[:, None, :3] - self.poses[indices, :3], axis=-1),
            'rotation_error_rad': rotation_error.reshape(-1, k),
            'joint_distance_rad': np.linalg.norm(joints[:, None, :] - neighbors, axis=-1),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument('--real-data', type=Path, nargs='+',
                        help='Original exported/raw episode directory, MCAP, NPZ states/actions, or NPY joints in radians')
    reference.add_argument('--real-index', type=Path, help='Saved local TeleopPoseIndex directory')
    parser.add_argument('--side', choices=('left', 'right'), default='left')
    parser.add_argument('--stream', choices=('state', 'action', 'both'), default='both')
    parser.add_argument('--limit-margin-deg', type=float, default=10.0)
    parser.add_argument('--model', type=Path, default=ROOT / 'abc_sim/models/yam_dishrack_base.xml')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--position-scale', type=float, default=0.01)
    parser.add_argument('--rotation-scale', type=float, default=0.1)
    parser.add_argument('--cell-size', type=float, default=0.01, help='Position voxel edge length in meters')
    parser.add_argument('--rotation-cell-size', type=float, default=0.1, help='Rotvec voxel edge length in radians')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=10000, help='Number of grid rows; 0 means all remaining rows')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.grid / 'metadata.json').read_text())
    config = metadata['config']
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != config['model_sha256']:
        parser.error('The FK model must match the model used to build the grid')
    grid = GPUEEFPoseGrid(args.grid)
    model = arm_model(args.model, config['site'].split('_', 1)[0])
    margin = np.deg2rad(args.limit_margin_deg)
    lower, upper = model.jnt_range[:, 0] - margin, model.jnt_range[:, 1] + margin
    if args.real_index:
        saved = json.loads((args.real_index / 'metadata.json').read_text())
        expected = dict(model_sha256=config['model_sha256'], side=args.side, stream=args.stream,
                        limit_margin_deg=args.limit_margin_deg,
                        position_scale=args.position_scale, rotation_scale=args.rotation_scale,
                        index_type='pose_6d_voxels', cell_size_m=args.cell_size,
                        rotation_cell_size_rad=args.rotation_cell_size)
        if saved['status'] != 'complete' or any(saved.get(name) != value for name, value in expected.items()):
            parser.error('Saved real index settings differ from the requested reference settings')
        index = TeleopPoseIndex.load(args.real_index)
        joints, clipped_values = index.joints, saved['clipped_real_joint_values']
    else:
        joints = np.concatenate([real_joints(path, args.side, args.stream) for path in args.real_data])
        if joints.ndim != 2 or joints.shape[1] != 6 or not np.isfinite(joints).all():
            parser.error('Real joints must be a finite (N,6) array in radians')
        clipped_values = int(((joints < lower) | (joints > upper)).sum())
        joints = np.clip(joints, lower, upper)
        index = TeleopPoseIndex(joints, fk_poses(model, joints, config['site']),
                               args.position_scale, args.rotation_scale, args.cell_size, args.rotation_cell_size)
    if not 1 <= args.k <= len(joints):
        parser.error('k must be between 1 and the number of real samples')
    end = min(args.start + args.count, grid.total_rows) if args.count else grid.total_rows
    if not 0 <= args.start < end or min(args.batch_size, args.position_scale, args.rotation_scale) <= 0:
        parser.error('Use a nonempty grid range and positive batch size/scales')
    print(f'real samples={len(joints)}, clipped joint values={clipped_values}, '
          f'grid rows={args.start}:{end}, k={args.k}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {name: [str(path) for path in value] if isinstance(value, list)
              else str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    report.update(real_samples=len(joints), end=end, completed=0, frame='arm base, TCP site',
                  index_type='pose_6d_voxels', global_topk=False,
                  reference_limits_rad=np.column_stack((lower, upper)).tolist(),
                  clipped_real_joint_values=clipped_values,
                  real_sampling='native recorded messages for MCAP; existing samples for exported/NumPy data',
                  pose_metric='sqrt((position_distance/position_scale)^2 + (2*sin(rotation_angle/2)/rotation_scale)^2)',
                  joint_metric='Euclidean joint radians; real joints clipped to expanded limits before FK; no normalization')
    for start in range(args.start, end, args.batch_size):
        rows = np.arange(start, min(start + args.batch_size, end), dtype=np.int64)
        q, poses = grid.joints(rows), np.asarray(grid.poses[rows], dtype=np.float64)
        result = index.query(q, poses, args.k, args.workers)
        np.savez_compressed(args.output / f'{start:012d}.npz', grid_rows=rows,
                            grid_joints=q, grid_poses=poses, **result)
        report['completed'] += len(rows)
        (args.output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'scored {report["completed"]}/{end - args.start}', flush=True)


if __name__ == '__main__':
    main()
