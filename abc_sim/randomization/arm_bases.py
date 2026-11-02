"""Sample paired training-base x/y/yaw while retaining XML height and tilt."""

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


def sample_training_arm_bases(model, data, seed):
    """Project one training pair onto the existing mounts, without added noise.

    The randomizer reloads the XML model before sampling. Only world x/y/yaw
    change; XML z/roll/pitch stay intact. Metadata stores the applied poses so
    replay and base compensation use the physical placements.
    """
    records = training_base_pool()
    rng = np.random.default_rng(None if seed is None else [seed, 1])
    record = records[int(rng.integers(len(records)))]
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
        world_from_base = world_from_camera @ np.asarray(record["camera_from_base"][side])
        yaw = np.arctan2(world_from_base[1, 0], world_from_base[0, 0])
        body.pos[:2] = world_from_base[:2, 3]
        body.quat[:] = _quat_mul(
            _quat_from_yaw(yaw - _yaw_from_quat(body.quat)), body.quat,
        )
        poses[name] = {
            "pos": body.pos.tolist(), "quat": body.quat.tolist(),
            "source_episode": record["episode"],
            "projection": "xy_yaw",
        }
    mujoco.mj_forward(model, data)
    return poses
