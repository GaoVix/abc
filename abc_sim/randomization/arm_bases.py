"""Sample calibrated bases while keeping the top camera fixed."""

from functools import lru_cache
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np

from .core import _quat_from_yaw, _quat_mul, _yaw_from_quat


@lru_cache(maxsize=1)
def training_base_pool():
    path = Path(__file__).resolve().parents[1] / "models/dishrack_training_base_poses.json.gz"
    with gzip.open(path, "rt") as stream:
        pool = json.load(stream)
    if pool["split"] != "train" or pool["calibration_quality"] != "accepted":
        raise ValueError("The base pose pool must contain accepted training episodes")
    if not pool["records"]:
        raise ValueError("The training base pose pool is empty")
    return pool["records"]


def sample_training_arm_bases(model, data, seed, *, full_pose=False):
    """Sample one training pair without added noise or camera movement.

    By default only world x/y/yaw change, retaining XML z/roll/pitch.
    full_pose applies the complete calibrated 6D transforms instead.
    """
    records = training_base_pool()
    rng = np.random.default_rng(None if seed is None else [seed, 1])
    record = records[int(rng.integers(len(records)))]
    poses = _set_arm_bases(model, data, record["camera_from_base"], full_pose=full_pose)
    for pose in poses.values():
        pose["source_episode"] = record["episode"]
    return poses


def set_median_arm_bases(model, data):
    """Place both arms at the training median's full 6D poses; keep the camera fixed."""
    path = Path(__file__).resolve().parents[2] / "abc_minimal/dishrack_median_base_poses.json"
    median = json.loads(path.read_text())["poses"]
    targets = {side: median[side]["T_camera_from_base"] for side in ("left", "right")}
    poses = _set_arm_bases(model, data, targets, full_pose=True)
    for pose in poses.values():
        pose["source"] = "median"
    return poses


def _set_arm_bases(model, data, camera_from_base, *, full_pose):
    mujoco.mj_kinematics(model, data)
    camera = data.camera("top")
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = camera.xmat.reshape(3, 3) @ np.diag([1., -1., -1.])
    world_from_camera[:3, 3] = camera.xpos
    poses = {}
    for side in ("left", "right"):
        name = f"{side}_arm"
        body = model.body(name)
        if model.body_parentid[body.id] != 0:
            raise ValueError(f"Expected {name} to be attached to the world body")
        world_from_base = world_from_camera @ np.asarray(camera_from_base[side])
        if full_pose:
            body.pos[:] = world_from_base[:3, 3]
            mujoco.mju_mat2Quat(body.quat, world_from_base[:3, :3].ravel())
        else:
            yaw = np.arctan2(world_from_base[1, 0], world_from_base[0, 0])
            body.pos[:2] = world_from_base[:2, 3]
            body.quat[:] = _quat_mul(
                _quat_from_yaw(yaw - _yaw_from_quat(body.quat)), body.quat,
            )
        poses[name] = {
            "pos": body.pos.tolist(), "quat": body.quat.tolist(),
            "projection": "full_pose" if full_pose else "xy_yaw",
        }
    mujoco.mj_forward(model, data)
    return poses


def disable_arm_collisions_except_eef(model):
    """Keep link_6 and the fingers, but disable the arm bases, links 1-5 and cameras."""
    roots = {model.body(f"{side}_arm").id for side in ("left", "right")}
    eef = {model.body(f"{side}_link_6").id for side in ("left", "right")}
    cameras = {model.body(f"{side}_camera_d405").id for side in ("left", "right")}
    disabled = np.zeros(model.nbody, dtype=bool)
    for body in range(1, model.nbody):
        disabled[body] = (body in roots or body in cameras or disabled[model.body_parentid[body]]) and body not in eef
    geoms = disabled[model.geom_bodyid]
    model.geom_contype[geoms] = 0
    model.geom_conaffinity[geoms] = 0
    model.body_contype[disabled] = 0
    model.body_conaffinity[disabled] = 0
