"""Per-frame camera parameters and fixed directions beside states_actions.bin.

camera_parameters.json v4 stores fixed OpenCV calibration and camera order.
camera_parameters.npy stores camera-to-world matrices and the exact source
state in each row. The NPY is memory-mapped, so sampling reads only that frame.
camera_directions.npy stores fixed camera-space unit directions after the
RGB exporter's resize/letterbox.
"""

from collections import OrderedDict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from abc_minimal.cape import validate_camera_matrices
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


def episode_plucker(episode, frame_index, state, num_frames, images):
    """Read the same frame as RGB/state, check its source state, then make rays."""
    metadata, frames = load_camera_parameters(str(Path(episode).resolve()))
    if len(frames) != num_frames:
        raise ValueError(f"CaPE frame count differs from states_actions.bin: {episode}")
    frame = frames[frame_index]
    if not np.array_equal(frame["state"], state):
        raise ValueError(f"CaPE cached state differs at frame {frame_index}: {episode}; regenerate with --overwrite")
    directions = load_camera_directions(str(Path(episode).resolve()))
    # Update the fixed direction table with this frame's camera poses.
    rays = plucker_from_directions(directions, frame["camera_to_world"].copy())
    result = {}
    for cam, image in images.items():
        if cam not in metadata["camera_keys"]:
            raise ValueError(f"CaPE camera {cam!r} missing from {episode}")
        index = metadata["camera_keys"].index(cam)
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
    row_bytes = 28 * np.dtype(np.float64).itemsize
    if not source.stat().st_size or source.stat().st_size % row_bytes:
        raise ValueError(f"Expected nonempty (T,28) float64 states/actions: {source}")
    rows = np.memmap(source, dtype=np.float64, mode="r").reshape(-1, 28)
    states = rows[:, :14]
    output = episode / "camera_parameters.npy"
    manifest = episode / "camera_parameters.json"
    direction_file = episode / "camera_directions.npy"
    if output.is_file() and manifest.is_file() and not overwrite:
        saved, frames = load_camera_parameters(str(episode))
        if saved["camera_keys"] != list(camera_keys) or not np.array_equal(frames["state"], states):
            raise ValueError(f"CaPE camera order or source state changed: {episode}; regenerate with --overwrite")
        load_camera_directions(str(episode))
        return False
    metadata_path = episode / "episode_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
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
                _, poses = geometry.camera_parameters(episode, state, metadata)
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
