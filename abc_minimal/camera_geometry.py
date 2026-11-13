"""Offline fixed camera calibration + FK of the processed dataset state.

Calibration contains camera-to-link mounts. State/image alignment belongs
to the dataset; this module does not read timestamps or apply time offsets.
"""

import copy
import gzip
import json
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def pose(entity):
    transform = np.eye(4)
    transform[:3, :3] = entity.xmat.reshape(3, 3)
    transform[:3, 3] = entity.xpos
    return transform


@lru_cache(maxsize=1)
def load_calibrations(path):
    """Read fixed calibration; convert real OpenCV K to snippet pixel centers."""
    with gzip.open(path, "rt") as stream:
        records = json.load(stream)["records"]
    result = {}
    for record in records:
        if record.get("calibration_quality") != "accepted":
            continue
        cameras = {}
        for name, camera in record["cameras"].items():
            intrinsics = dict(camera["recorded_intrinsics"])
            k = np.array(intrinsics["K"], dtype=np.float64).reshape(3, 3).copy()
            # Real calibration indexes pixel centers at (0,0); the official
            # snippet indexes them at (.5,.5). Shift K before any resizing.
            k[:2, 2] += .5
            intrinsics["K"] = k.tolist()
            intrinsics["pixel_center"] = .5
            cameras[name] = {"reference_body": camera["reference_body"],
                             "T_reference_from_camera": camera["T_reference_from_camera"],
                             "recorded_intrinsics": intrinsics}
        result[record["episode"]] = {
            "model_name": record["model_name"],
            "base_poses": record["base_pose"]["base_poses"],
            "cameras": cameras,
        }
    return result


def arm_model_xml(model_path):
    """Copy the official arm/camera tree, without scene objects or mesh assets."""
    source = ET.parse(model_path).getroot()
    root = ET.Element("mujoco", {"model": "yam_fk"})
    compiler = source.find("compiler")
    attributes = {key: compiler.get(key) for key in ("angle", "eulerseq")
                  if compiler is not None and compiler.get(key) is not None}
    # FK does not use inertia; these bounds support links whose mass came from geoms.
    ET.SubElement(root, "compiler", dict(attributes, boundmass="0.001", boundinertia="0.001"))
    for defaults in source.findall("default"):
        defaults = copy.deepcopy(defaults)
        for parent in defaults.iter():
            for child in list(parent):
                if child.tag not in ("default", "joint", "camera"):
                    parent.remove(child)
        root.append(defaults)
    needed = {"joint": {f"{side}_joint{i}" for side in ("left", "right") for i in range(1, 7)},
              "camera": {"top", "left", "right"}}

    def is_needed(element):
        return element.tag in needed and element.get("name") in needed[element.tag]

    def copy_tree(body):
        if not any(is_needed(element) for element in body.iter()):
            return None
        result = ET.Element(body.tag, body.attrib)
        for child in body:
            if child.tag == "body":
                branch = copy_tree(child)
                if branch is not None:
                    result.append(branch)
            elif child.tag == "inertial" or is_needed(child):
                result.append(copy.deepcopy(child))
        return result

    root.append(copy_tree(source.find("worldbody")))
    return ET.tostring(root, encoding="unicode")


class StateFK:
    """Small MuJoCo FK wrapper using the same 14D joint layout as the policy."""

    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_string(arm_model_xml(model_path))
        self.data = mujoco.MjData(self.model)
        self.default_pos = self.model.body_pos.copy()
        self.default_quat = self.model.body_quat.copy()
        self.joints = np.array([[int(self.model.joint(f"{side}_joint{i}").qposadr[0])
                                 for i in range(1, 7)] for side in ("left", "right")])

    def update(self, state, bases=None):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("CaPE FK requires the current finite, unnormalized 14D state")
        # Restore before applying per-episode bases: a worker reuses this model.
        self.model.body_pos[:] = self.default_pos
        self.model.body_quat[:] = self.default_quat
        self.data.qpos[self.joints] = state.reshape(2, 7)[:, :6]
        mujoco.mj_kinematics(self.model, self.data)
        if bases is not None:
            for side, base in zip(("left", "right"), bases):
                base = np.asarray(base)
                body = self.model.body(f"{side}_arm").id
                parent = self.model.body_parentid[body]
                rotation = self.data.xmat[parent].reshape(3, 3)
                self.model.body_pos[body] = rotation.T @ (base[:3, 3] - self.data.xpos[parent])
                mujoco.mju_mat2Quat(self.model.body_quat[body], (rotation.T @ base[:3, :3]).ravel())
            mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)


@lru_cache(maxsize=4)
def state_fk(model_path):
    return StateFK(model_path)


class CameraGeometry:
    """Offline camera parameters from fixed calibration and current-state FK."""

    def __init__(self, calibration_path=None, model_dir=None):
        self.calibrations = load_calibrations(str(Path(calibration_path).expanduser().resolve())) if calibration_path else {}
        self.model_dir = Path(model_dir).expanduser() if model_dir else ROOT / "abc_sim/models"
        self.target_model = ROOT / "abc_sim/models/yam_dishrack_base.xml"
        target = state_fk(str(self.target_model))
        target.update(np.zeros(14))
        self.world_from_top = pose(target.data.camera("top"))
        self.world_from_top[:3, :3] @= np.diag([1., -1., -1.])

    def camera_poses(self, episode, state, metadata):
        original = metadata.get("sequence_ik", {}).get("source_episode", episode.name)
        record = self.calibrations.get(original)
        if record is None:
            raise ValueError(f"CaPE calibration missing for {original}; pass the complete three-camera --calibration-path")
        conversion = metadata.get("sequence_ik", {})
        retargeted = bool(conversion or metadata.get("joint_conversion"))
        if retargeted:
            fk = state_fk(str(self.target_model))
            camera_from_base = conversion.get("camera_from_base")
            bases = self.world_from_top @ np.asarray(camera_from_base) if camera_from_base is not None else None
            fk.update(state, bases)
            world_from_reference = np.eye(4)
        else:
            fk = state_fk(str(self.model_dir / record["model_name"]))
            fk.update(state, [record["base_poses"][side] for side in ("left", "right")])
            top_mount = np.asarray(record["cameras"]["top"]["T_reference_from_camera"])
            top_reference = pose(fk.data.body(record["cameras"]["top"]["reference_body"]))
            world_from_reference = self.world_from_top @ np.linalg.inv(top_reference @ top_mount)
        poses = {"top": self.world_from_top}
        for cam, view in (("left", "left-wrist"), ("right", "right-wrist")):
            camera = record["cameras"][view]
            mount = np.asarray(camera["T_reference_from_camera"])
            poses[cam] = world_from_reference @ pose(fk.data.body(camera["reference_body"])) @ mount
        return record, poses

    def camera_parameters(self, episode, state, metadata):
        """Return fixed intrinsics and current camera-to-world poses, without rays."""
        original = metadata.get("sequence_ik", {}).get("source_episode", episode.name)
        if original not in self.calibrations:
            is_sim = (metadata.get("render_backend") is not None
                      or str(metadata.get("task_name", "")).startswith(("sim_", "sim ")))
            if not is_sim:
                raise ValueError(f"CaPE calibration missing for {original}; pass the complete three-camera --calibration-path")
            return self.sim_camera_parameters(episode, state, metadata)
        record, poses = self.camera_poses(episode, state, metadata)
        cameras = {cam: record["cameras"][view]["recorded_intrinsics"] for cam, view in (
            ("top", "top"), ("left", "left-wrist"), ("right", "right-wrist"))}
        return cameras, poses

    def sim_camera_parameters(self, episode, state, metadata):
        """Fixed ABC arm/camera configuration plus FK; no episode scene is needed."""
        fk = state_fk(str(self.target_model))
        camera_from_base = metadata.get("sequence_ik", {}).get("camera_from_base")
        bases = self.world_from_top @ np.asarray(camera_from_base) if camera_from_base is not None else None
        fk.update(state, bases)
        cameras, poses = {}, {}
        for cam in ("top", "left", "right"):
            resolution = metadata.get("source_camera_resolutions", {}).get(cam)
            if resolution is None and "image_width" in metadata and "image_height" in metadata:
                # Published sim episodes record their camera image grid here
                # (e.g. 224x168); there is no export-time square letterbox.
                resolution = (metadata["image_width"], metadata["image_height"])
            if resolution is None:
                raise ValueError(f"CaPE sim source image size missing for {cam}: {episode}; "
                                 "provide source_camera_resolutions or image_width/image_height")
            if len(resolution) != 2 or any(int(v) != v or v <= 0 for v in resolution):
                raise ValueError(f"Invalid CaPE source image size for {cam}: {resolution}")
            width, height = map(int, resolution)
            index = fk.model.camera(cam).id
            focal = height / (2 * np.tan(np.deg2rad(fk.model.cam_fovy[index]) / 2))
            cameras[cam] = {"width": width, "height": height,
                            "K": [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                            "distortion_model": "none", "D": [0.] * 5, "pixel_center": .5}
            transform = pose(fk.data.camera(cam))
            transform[:3, :3] @= np.diag([1., -1., -1.])
            poses[cam] = transform
        return cameras, poses
