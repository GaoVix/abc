"""Refresh fixed geometry after per-world model poses change."""

import warp as wp
from mujoco_warp._src import math


@wp.kernel
def sync_static_geom_poses(
    geom_ids: wp.array[int],
    body_parentid: wp.array[int],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
    geom_bodyid: wp.array[int],
    geom_pos: wp.array2d[wp.vec3],
    geom_quat: wp.array2d[wp.quat],
    geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
):
    world, index = wp.tid()
    geom = geom_ids[index]
    position = geom_pos[world % geom_pos.shape[0], geom]
    rotation = geom_quat[world % geom_quat.shape[0], geom]
    body = geom_bodyid[geom]
    # These bodies have no joint or mocap ancestor, so model transforms alone
    # determine their world poses. MJWarp forward skips these cached geoms.
    while body > 0:
        parent_rotation = body_quat[world % body_quat.shape[0], body]
        position = math.rot_vec_quat(position, parent_rotation) + body_pos[world % body_pos.shape[0], body]
        rotation = math.mul_quat(parent_rotation, rotation)
        body = body_parentid[body]
    geom_xpos[world, geom] = position
    geom_xmat[world, geom] = math.quat_to_mat(rotation)
