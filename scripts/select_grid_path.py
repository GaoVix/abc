import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from abc_minimal.eef_pose_grid_gpu import GPUEEFPoseGrid


def select_path(candidates, pose_costs, acceleration_weight=1.):
    scale = .1 ** 2
    parents = []
    if not acceleration_weight or len(candidates) < 3:
        cost = pose_costs[0].copy()
        for frame, (previous, current) in enumerate(zip(candidates, candidates[1:]), 1):
            transition = cdist(previous, current, "sqeuclidean") / scale + cost[:, None]
            parent = np.argmin(transition, axis=0)
            parents.append(parent)
            cost = transition[parent, np.arange(len(current))] + pose_costs[frame]
        choice = np.empty(len(candidates), dtype=np.int32)
        choice[-1] = np.argmin(cost)
        for frame in range(len(candidates) - 1, 0, -1):
            choice[frame - 1] = parents[frame - 1][choice[frame]]
        return choice

    candidates = [np.asarray(q, dtype=np.float64) for q in candidates]
    pose_costs = [np.asarray(cost, dtype=np.float64) for cost in pose_costs]
    cost = (pose_costs[0][:, None] + pose_costs[1][None, :]
            + cdist(candidates[0], candidates[1], "sqeuclidean") / scale)
    for frame in range(2, len(candidates)):
        before, previous, current = candidates[frame - 2:frame + 1]
        next_cost = pose_costs[frame][None, :] + cdist(previous, current, "sqeuclidean") / scale
        parent = np.empty(next_cost.shape, dtype=np.int32)
        acceleration = np.empty((len(before), len(current)), dtype=np.float64)
        columns = np.arange(len(current))
        for index, q in enumerate(previous):
            cdist(2 * q - before, current, "sqeuclidean", out=acceleration)
            acceleration *= acceleration_weight / scale
            acceleration += cost[:, index, None]
            parent[index] = np.argmin(acceleration, axis=0)
            next_cost[index] += acceleration[parent[index], columns]
        parents.append(parent)
        cost = next_cost
    choice = np.empty(len(candidates), dtype=np.int32)
    choice[-2:] = np.unravel_index(np.argmin(cost), cost.shape)
    for frame in range(len(candidates) - 1, 1, -1):
        choice[frame - 2] = parents[frame - 2][choice[frame - 1], choice[frame]]
    return choice


def query_path(grid, targets, side, k=200, distance_limit=.5, acceleration_weight=1.):
    joints, costs, distances = [], [], []
    for frame, target in enumerate(targets):
        for radius in range(1, 5):
            found = grid.query(target, k=grid.total_rows, neighbors=radius)
            distance = grid.teleop_distance(found['rows'], side)
            keep = np.flatnonzero(distance < distance_limit)[:k]
            if len(keep):
                break
        if not len(keep):
            raise ValueError(f'No {side} candidates with distance < {distance_limit:g} at frame {frame}')
        joints.append(found['joints'][keep])
        costs.append((found['position_error_m'][keep] / .01) ** 2 + (found['rotation_error_rad'][keep] / .1) ** 2)
        distances.append(distance[keep])
    choice = select_path(joints, costs, acceleration_weight)
    selected_joints = np.stack([q[index] for q, index in zip(joints, choice)]).astype(np.float64)
    selected_distances = np.array([d[index] for d, index in zip(distances, choice)])
    return selected_joints, selected_distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=Path, required=True, help='NPZ with targets_camera shaped (frames, 2, 4, 4), left then right; OpenCV frame, meters')
    parser.add_argument('--grid', type=Path, required=True)
    parser.add_argument('--bases', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/base_pose_median_20261026/median_base_poses.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, default=200)
    parser.add_argument('--distance-limit', type=float, default=.5)
    parser.add_argument('--dp-acceleration-weight', type=float, default=1.)
    args = parser.parse_args()

    with np.load(args.targets, allow_pickle=False) as saved:
        targets_camera = saved['targets_camera']
    bases = json.loads(args.bases.read_text())['poses']
    base_poses = np.asarray([bases[side]['T_camera_from_base'] for side in ('left', 'right')])
    targets_base = np.linalg.inv(base_poses)[None] @ targets_camera
    grid = GPUEEFPoseGrid(args.grid)
    paths = [query_path(grid, targets_base[:, arm], side, args.k, args.distance_limit, args.dp_acceleration_weight) for arm, side in enumerate(('left', 'right'))]
    joints = np.stack([path[0] for path in paths], axis=1)
    distance = np.stack([path[1] for path in paths], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, joints=joints, teleop_distance_rad=distance)
    print(f'Saved {len(joints)} frames for both arms to {args.output}')


if __name__ == '__main__':
    main()
