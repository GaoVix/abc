"""Choose one joint candidate per frame using exact first- or second-order DP.

The objective adds pose costs, squared joint steps, and optionally squared
second differences. Joint differences are measured in radians and scaled by 0.1.
"""

import numpy as np
from scipy.spatial.distance import cdist


JOINT_SCALE_RAD = 0.1


def select_path(candidates, pose_costs, acceleration_weight=1.0):
    """Return one candidate index per frame; weight 0 selects first-order DP.

    candidates[t] has shape (K_t, 6); pose_costs[t] has shape (K_t,).
    """
    if acceleration_weight and len(candidates) > 2:
        return select_second_order_path(candidates, pose_costs, acceleration_weight)
    return _select_first_order_path(candidates, pose_costs)


def _select_first_order_path(candidates, pose_costs):
    parents = []
    cost = pose_costs[0].copy()
    for frame, (previous, current) in enumerate(zip(candidates, candidates[1:]), 1):
        transition = cdist(previous, current, "sqeuclidean") / JOINT_SCALE_RAD ** 2 + cost[:, None]
        parent = np.argmin(transition, axis=0)
        parents.append(parent)
        cost = transition[parent, np.arange(len(current))] + pose_costs[frame]
    choice = np.empty(len(candidates), dtype=np.int32)
    choice[-1] = np.argmin(cost)
    for frame in range(len(candidates) - 1, 0, -1):
        choice[frame - 1] = parents[frame - 1][choice[frame]]
    return choice


def select_second_order_path(candidates, pose_costs, acceleration_weight):
    """Use a pair of adjacent candidates as the state, then backtrack the best path."""
    candidates = [np.asarray(q, dtype=np.float64) for q in candidates]
    pose_costs = [np.asarray(cost, dtype=np.float64) for cost in pose_costs]
    scale = JOINT_SCALE_RAD ** 2
    cost = (pose_costs[0][:, None] + pose_costs[1][None, :]
            + cdist(candidates[0], candidates[1], "sqeuclidean") / scale)
    parents = []
    for frame in range(2, len(candidates)):
        before, previous, current = candidates[frame - 2:frame + 1]
        next_cost = (pose_costs[frame][None, :]
                     + cdist(previous, current, "sqeuclidean") / scale)
        parent = np.empty(next_cost.shape, dtype=np.int32)
        points = (2 * previous[:, None, :] - before[None, :, :]).reshape(-1, 6)
        acceleration = cdist(points, current, "sqeuclidean").reshape(len(previous), len(before), len(current))
        acceleration *= acceleration_weight / scale
        acceleration += cost.T[:, :, None]
        parent[:] = np.argmin(acceleration, axis=1)
        next_cost += np.take_along_axis(acceleration, parent[:, None, :], axis=1)[:, 0, :]
        parents.append(parent)
        cost = next_cost
    choice = np.empty(len(candidates), dtype=np.int32)
    choice[-2:] = np.unravel_index(np.argmin(cost), cost.shape)
    for frame in range(len(candidates) - 1, 1, -1):
        choice[frame - 2] = parents[frame - 2][choice[frame - 1], choice[frame]]
    return choice
