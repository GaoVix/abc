"""Sample calibrated bases on the existing physical mounts."""

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


def _pose(entity):
    transform = np.eye(4)
    transform[:3, :3] = entity.xmat.reshape(3, 3)
    transform[:3, 3] = entity.xpos
    return transform


def top_camera_pose(model, data):
    """Record both the camera-body placement and its OpenCV world transform."""
    body = model.body("top_camera_d405")
    transform = _pose(data.camera("top"))
    transform[:3, :3] @= np.diag([1., -1., -1.])
    return {"body_name": body.name, "body_pos": body.pos.tolist(),
            "body_quat": body.quat.tolist(), "world_from_camera": transform.tolist()}


def sample_training_camera_arm_bases(model, data, seed):
    """Keep the left mount fixed, match its calibrated camera pose exactly.

    The right mount takes calibrated x/y/yaw relative to the left, retaining
    its nominal relative height and tilt. Only the top camera housing moves;
    the gate, operating table, objects and wrist camera mounts stay intact.
    """
    records = training_base_pool()
    rng = np.random.default_rng(None if seed is None else [seed, 1])
    record = records[int(rng.integers(len(records)))]
    poses = _align_camera_arm_bases(model, data, record["camera_from_base"])
    for pose in poses.values():
        pose["source_episode"] = record["episode"]
    return poses


def set_median_camera_arm_bases(model, data):
    """Use the training median, with an exact left pose and planar right mount."""
    path = Path(__file__).resolve().parents[2] / "abc_minimal/dishrack_median_base_poses.json"
    with path.open() as stream:
        median = json.load(stream)["poses"]
    targets = {side: median[side]["T_camera_from_base"] for side in ("left", "right")}
    poses = _align_camera_arm_bases(model, data, targets)
    for pose in poses.values():
        pose["source"] = "median"
    return poses


def _align_camera_arm_bases(model, data, camera_from_base):
    """Move the top camera and right x/y/yaw, retaining both physical mounts."""
    for side in ("left", "right"):
        if model.body_parentid[model.body(f"{side}_arm").id] != 0:
            raise ValueError("Expected arm bases attached to the world body")
    mujoco.mj_forward(model, data)
    left_world = _pose(data.body("left_arm"))
    nominal_relative = np.linalg.inv(left_world) @ _pose(data.body("right_arm"))
    camera_left = np.asarray(camera_from_base["left"])
    calibrated_relative = np.linalg.inv(camera_left) @ camera_from_base["right"]
    relative = nominal_relative.copy()
    relative[:2, 3] = calibrated_relative[:2, 3]
    yaw = np.arctan2(calibrated_relative[1, 0], calibrated_relative[0, 0])
    nominal_yaw = np.arctan2(nominal_relative[1, 0], nominal_relative[0, 0])
    cosine, sine = np.cos(yaw - nominal_yaw), np.sin(yaw - nominal_yaw)
    relative[:3, :3] = np.array([[cosine, -sine, 0.], [sine, cosine, 0.],
                                [0., 0., 1.]]) @ nominal_relative[:3, :3]
    right_world = left_world @ relative
    right = model.body("right_arm")
    right.pos[:] = right_world[:3, 3]
    mujoco.mju_mat2Quat(right.quat, right_world[:3, :3].ravel())

    # Account for the camera's fixed local transform and OpenCV axis convention.
    camera_body = model.body("top_camera_d405")
    old_camera = np.asarray(top_camera_pose(model, data)["world_from_camera"])
    body_from_camera = np.linalg.inv(_pose(data.body(camera_body.id))) @ old_camera
    desired_camera = left_world @ np.linalg.inv(camera_left)
    parent_world = _pose(data.body(int(model.body_parentid[camera_body.id])))
    body_local = np.linalg.inv(parent_world) @ desired_camera @ np.linalg.inv(body_from_camera)
    camera_body.pos[:] = body_local[:3, 3]
    mujoco.mju_mat2Quat(camera_body.quat, body_local[:3, :3].ravel())
    mujoco.mj_forward(model, data)
    return {f"{side}_arm": {
        "pos": model.body(f"{side}_arm").pos.tolist(),
        "quat": model.body(f"{side}_arm").quat.tolist(),
        "projection": "left_camera_right_xy_yaw",
    } for side in ("left", "right")}
