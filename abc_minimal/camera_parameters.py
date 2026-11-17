"""Per-frame camera parameters and fixed directions beside states_actions.bin.

camera_parameters.json v4 stores calibration, camera order and reference_frame.
camera_parameters.npy stores camera-to-reference matrices and the exact clean
dataset state in each row. The NPY is memory-mapped, so sampling reads only that frame.
camera_directions.npy stores fixed camera-space unit directions after the
RGB exporter's resize/letterbox.
"""

from collections import OrderedDict
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
import torch

from abc_minimal.cape import CAMERA_REFERENCE_FRAME, validate_camera_matrices
from abc_minimal.camera_rays import (
    cached_camera_directions,
    camera_calibration,
    plucker_from_directions,
)


_DIRECTION_TABLES = OrderedDict()


def require_camera_parameters(episode):
    for name in ("camera_parameters.json", "camera_parameters.npy", "camera_directions.npy"):
        path = Path(episode) / name
        if not path.is_file():
            raise ValueError(f"CaPE camera parameters missing: {path}; run scripts/prepare_camera_parameters.py --overwrite first")


@lru_cache(maxsize=16)
def load_camera_parameters(episode):
    require_camera_parameters(episode)
    episode = Path(episode)
    metadata = json.loads((episode / "camera_parameters.json").read_text())
    if metadata.get("version") != 4 or metadata.get("convention") != "opencv":
        raise ValueError(f"Unsupported CaPE camera parameter format: {episode}; regenerate with --overwrite")
    if metadata.get("reference_frame") not in ("left_arm", "top"):
        raise ValueError(f"Unsupported CaPE camera reference frame: {episode}; regenerate with --overwrite")
    frames = np.load(episode / "camera_parameters.npy", mmap_mode="r", allow_pickle=False)
    expected = np.dtype([("state", "<f8", (14,)),
                         ("camera_to_world", "<f4", (len(metadata["camera_keys"]), 4, 4))])
    if frames.ndim != 1 or frames.dtype != expected:
        raise ValueError(f"Invalid CaPE camera parameter shape/dtype: {episode}")
    return metadata, frames


@lru_cache(maxsize=16)
def load_camera_directions(episode):
    """Read the small fixed table once per worker/episode, never per frame."""
    metadata, _ = load_camera_parameters(episode)
    # Random frame sampling often alternates between many episodes with the
    # same physical camera calibration. Share their tables to avoid repeatedly
    # reading 1.72 MiB just because the 16-episode metadata LRU was evicted.
    key = json.dumps({"grid": metadata["ray_image_size"],
                      "cameras": [metadata["cameras"][cam] for cam in metadata["camera_keys"]]},
                     sort_keys=True, allow_nan=False)
    if key in _DIRECTION_TABLES:
        _DIRECTION_TABLES.move_to_end(key)
        return _DIRECTION_TABLES[key]
    directions = np.load(Path(episode) / "camera_directions.npy", allow_pickle=False)
    expected = (len(metadata["camera_keys"]), 3, *metadata["ray_image_size"])
    if directions.dtype != np.float32 or directions.shape != expected or not np.isfinite(directions).all():
        raise ValueError(f"Invalid CaPE camera direction table: {episode}; regenerate with --overwrite")
    table = torch.from_numpy(directions)
    _DIRECTION_TABLES[key] = table
    if len(_DIRECTION_TABLES) > 32:
        _DIRECTION_TABLES.popitem(last=False)
    return table


def clear_camera_parameter_caches():
    load_camera_parameters.cache_clear()
    load_camera_directions.cache_clear()
    _DIRECTION_TABLES.clear()


def geodesic_rotation_mean(matrices, tol=1e-10, max_iter=100):
    """SO(3) Karcher mean using the supplied Log / average / Exp iteration.

    https://chatgpt.com/share/6afc93af-c654-83e8-a046-c7ec5c605280
    scipy Rotation.mean() is chordal and must not replace this calculation.
    """
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_matrix(matrices)
    n = len(rotations)
    if not n:
        raise ValueError("Geodesic mean requires at least one rotation")
    weights = np.ones(n) / n
    mean = rotations[0]
    for _ in range(max_iter):
        # Follow the shared example: relative rotation, Log, weighted mean, Exp.
        relative = mean.inv() * rotations
        phi = relative.as_rotvec()
        delta = np.sum(weights[:, None] * phi, axis=0)
        if np.linalg.norm(delta) < tol:
            return mean.as_matrix()
        mean = mean * Rotation.from_rotvec(delta)
    raise ValueError(f"Geodesic rotation mean did not converge in {max_iter} iterations")


def average_camera_pose(data_dirs, *, real_frames, sim_frames, seed=0):
    """Sample training frames uniformly, pairing both wrists in each top frame.

    Positions use the arithmetic mean; rotations use the SO(3) geodesic mean.
    Only the sampled poses are loaded, independently of the full dataset size.
    """
    roots = {Path(root).name: Path(root) for root in data_dirs}
    if len(roots) != 2 or set(roots) != {"train_real", "train_sim"}:
        raise ValueError("Average pose requires train_real and train_sim directories")
    rng = np.random.default_rng(seed)
    sampled_poses, sampled_frames, available_frames, sampled_episodes = [], {}, {}, 0
    for split, requested in (("train_real", real_frames), ("train_sim", sim_frames)):
        root = roots[split]
        if not root.is_dir() or requested <= 0:
            raise ValueError(f"Average pose requires an existing split and positive sample count: {root}")
        episodes = sorted({Path(directory).resolve() for directory, _, files in os.walk(root, followlinks=True)
                           if "states_actions.bin" in files})
        lengths = []
        for episode in episodes:
            metadata, frames = load_camera_parameters(str(episode))
            if metadata["reference_frame"] != "top":
                raise ValueError(f"Average pose requires top-relative camera parameters: {episode}")
            lengths.append(len(frames))
        ends = np.cumsum(lengths)
        total = int(ends[-1]) if len(ends) else 0
        if not total:
            raise ValueError(f"No training camera poses found: {root}")
        count = min(requested, total)
        # Global frame indices give each frame equal probability, including long episodes.
        selected = np.sort(rng.choice(total, size=count, replace=False))
        start = 0
        for episode, end in zip(episodes, ends):
            chosen = selected[np.searchsorted(selected, start):np.searchsorted(selected, end)] - start
            start = int(end)
            if not len(chosen):
                continue
            metadata, frames = load_camera_parameters(str(episode))
            indices = [metadata["camera_keys"].index(cam) for cam in ("left", "right")]
            sampled_poses.append(frames["camera_to_world"][chosen][:, indices].reshape(-1, 4, 4))
            sampled_episodes += 1
        sampled_frames[split], available_frames[split] = count, total
    poses = np.concatenate(sampled_poses).astype(np.float64)
    average = np.eye(4)
    average[:3, :3] = geodesic_rotation_mean(poses[:, :3, :3])
    average[:3, 3] = poses[:, :3, 3].mean(axis=0)
    validate_camera_matrices(np.eye(3), average)
    return {"reference_frame": "top", "camera_keys": ["left", "right"],
            "top_from_average": average.tolist(), "num_poses": len(poses), "num_episodes": sampled_episodes,
            "weighting": "equal_per_sampled_camera_frame", "rotation_mean": "geodesic",
            "sample_seed": seed, "sampled_frames": sampled_frames, "available_frames": available_frames,
            "training_splits": [str(roots[split].resolve()) for split in ("train_real", "train_sim")]}


def load_camera_average(path):
    """Load the one training-set reference reused by every split and inference."""
    metadata = json.loads(Path(path).read_text())
    if metadata.get("reference_frame") != "top" or metadata.get("camera_keys") != ["left", "right"]:
        raise ValueError(f"Expected a shared top-relative left/right wrist average: {path}")
    if metadata.get("rotation_mean") != "geodesic":
        raise ValueError(f"CaPE requires a geodesic rotation average: {path}")
    pose = np.asarray(metadata["top_from_average"], dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected one 4x4 average camera pose: {path}")
    validate_camera_matrices(np.eye(3), pose)
    return pose


def exported_intrinsics(camera, height, width):
    """Match the RGB exporter's aspect-preserving resize and letterbox."""
    raw_h, raw_w = int(camera["height"]), int(camera["width"])
    scale = min(height / raw_h, width / raw_w)
    new_h, new_w = round(raw_h * scale), round(raw_w * scale)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    k = np.diag([new_w / raw_w, new_h / raw_h, 1.]) @ np.asarray(camera["K"]).reshape(3, 3)
    k[:2, 2] += [left, top]
    valid = torch.zeros(height, width)
    valid[top:top + new_h, left:left + new_w] = 1
    return k, valid


def episode_plucker(episode, frame_index, state, num_frames, images, *,
                    reference_frame=CAMERA_REFERENCE_FRAME, reference_from_top=None):
    """Read the same frame as RGB/state, check its source state, then make rays."""
    metadata, frames = load_camera_parameters(str(Path(episode).resolve()))
    source_frame = "top" if reference_frame == "top_average" else reference_frame
    if metadata["reference_frame"] != source_frame:
        raise ValueError(f"CaPE camera reference frame differs from model ({reference_frame}): {episode}; regenerate with --overwrite")
    if len(frames) != num_frames:
        raise ValueError(f"CaPE frame count differs from states_actions.bin: {episode}")
    frame = frames[frame_index]
    if not np.array_equal(frame["state"], state):
        raise ValueError(f"CaPE cached state differs at frame {frame_index}: {episode}; regenerate with --overwrite")
    directions = load_camera_directions(str(Path(episode).resolve()))
    for cam in images:
        if cam not in metadata["camera_keys"]:
            raise ValueError(f"CaPE camera {cam!r} missing from {episode}")
    indices = [metadata["camera_keys"].index(cam) for cam in images]
    # Compute only the views that actually receive CaPE.
    poses = frame["camera_to_world"][indices].copy()
    if reference_frame == "top_average":
        if reference_from_top is None:
            raise ValueError("CaPE requires the fixed training average pose")
        poses = reference_from_top @ poses
    rays = plucker_from_directions(directions[indices], poses)
    result = {}
    for index, (cam, image) in enumerate(images.items()):
        if tuple(image.shape[-2:]) != tuple(metadata["ray_image_size"]):
            raise ValueError(f"CaPE {cam} image size does not match offline direction table: {episode}")
        result[cam] = rays[index]
    return result


def prepare_camera_parameters(episode, geometry, camera_keys=("top", "left", "right"), *, overwrite=False):
    """Solve fixed distortion once and FK per state; save beside that state file.

    ``geometry`` is an offline CameraGeometry instance. Neither this function
    nor its caller selects states by timestamps or applies time offsets.
    Return True when written, False when matching output already exists.
    """
    episode = Path(episode).resolve()
    source = episode / "states_actions.bin"
    metadata_path = episode / "episode_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    retargeted = bool(metadata.get("sequence_ik") or metadata.get("joint_conversion"))
    row_bytes = 28 * np.dtype(np.float64).itemsize
    if not source.stat().st_size or source.stat().st_size % row_bytes:
        raise ValueError(f"Expected nonempty (T,28) float64 states/actions: {source}")
    rows = np.memmap(source, dtype=np.float64, mode="r").reshape(-1, 28)
    states = rows[:, :14]
    if retargeted and "source_frame_range" in metadata:
        begin, end = metadata["source_frame_range"]
        if end - begin != len(states):
            raise ValueError(f"Converted state length differs from original RGB frame interval: {episode}")
    output = episode / "camera_parameters.npy"
    manifest = episode / "camera_parameters.json"
    direction_file = episode / "camera_directions.npy"
    if output.is_file() and manifest.is_file() and not overwrite:
        saved, frames = load_camera_parameters(str(episode))
        if saved["reference_frame"] != geometry.reference_frame:
            raise ValueError(f"CaPE camera reference frame changed: {episode}; regenerate with --overwrite")
        if retargeted and geometry.reference_frame == "top" and saved.get("pose_source") != "original_rgb_state":
            raise ValueError(f"Top-relative CaPE must use original RGB-aligned states: {episode}; regenerate with --overwrite")
        if saved["camera_keys"] != list(camera_keys) or not np.array_equal(frames["state"], states):
            raise ValueError(f"CaPE camera order or source state changed: {episode}; regenerate with --overwrite")
        load_camera_directions(str(episode))
        return False
    cameras, poses = geometry.camera_parameters(episode, states[0], metadata)
    fixed = {cam: camera_calibration(cameras[cam]) for cam in camera_keys}
    for cam in camera_keys:
        source_size = metadata.get("source_camera_resolutions", {}).get(cam)
        calibration_size = (fixed[cam]["width"], fixed[cam]["height"])
        if source_size is not None and tuple(source_size) != calibration_size:
            raise ValueError(f"CaPE {cam} source RGB size {source_size} differs from calibration "
                             f"size {calibration_size}: {episode}; use matching camera calibration")
    default_export_size = (metadata.get("image_width", 224), metadata.get("image_height", 224))
    sizes = {tuple(metadata.get("camera_resolutions", {}).get(cam, default_export_size)) for cam in camera_keys}
    if len(sizes) != 1:
        raise ValueError(f"CaPE exported camera images must share a grid: {episode}")
    width, height = next(iter(sizes))
    directions = torch.stack([cached_camera_directions(fixed[cam], height, width) for cam in camera_keys])
    description = {"version": 4, "convention": "opencv", "camera_keys": list(camera_keys),
                   "reference_frame": geometry.reference_frame,
                   "pose_source": "original_rgb_state" if retargeted and geometry.reference_frame == "top" else "state",
                   "cameras": fixed, "ray_image_size": [height, width]}
    dtype = np.dtype([("state", "<f8", (14,)), ("camera_to_world", "<f4", (len(camera_keys), 4, 4))])
    temporary = output.with_suffix(".npy.tmp")
    temporary_manifest = manifest.with_suffix(".json.tmp")
    temporary_directions = direction_file.with_suffix(".npy.tmp")
    try:
        frames = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=(len(states),))
        frames["state"] = states
        for index, state in enumerate(states):
            if index:
                _, poses = geometry.camera_parameters(episode, state, metadata, frame_index=index)
            frames["camera_to_world"][index] = np.stack([poses[cam] for cam in camera_keys])
        for index, cam in enumerate(camera_keys):
            validate_camera_matrices(fixed[cam]["K"], frames["camera_to_world"][:, index])
        frames.flush()
        del frames
        with temporary_directions.open("wb") as stream:
            np.save(stream, directions.numpy(), allow_pickle=False)
        temporary_manifest.write_text(json.dumps(description, indent=2, allow_nan=False) + "\n")
        # The manifest marks completion; interrupted replacement must be retried.
        manifest.unlink(missing_ok=True)
        temporary.replace(output)
        temporary_directions.replace(direction_file)
        temporary_manifest.replace(manifest)
        clear_camera_parameter_caches()
    finally:
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        temporary_directions.unlink(missing_ok=True)
    return True
