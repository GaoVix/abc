# Distortion inversion adapted from librealsense (Apache-2.0).
# Copyright (c) 2024 RealSense, Inc. All Rights Reserved.
# See assets/third_party/librealsense/LICENSE.
"""Fixed camera direction tables; distortion is solved before training.

Brown/Inverse-Brown inversion follows librealsense's CPU implementation:
https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp
No RealSense runtime dependency is required. RGB pixels are never modified.
"""

from functools import lru_cache
import json

import numpy as np
import torch
from torch.nn import functional as F


def camera_calibration(camera):
    """Canonical, serializable calibration for a fixed pixel-to-ray mapping."""
    height, width = int(camera["height"]), int(camera["width"])
    k = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    if min(height, width) <= 0 or not np.isfinite(k).all():
        raise ValueError("CaPE requires positive image sizes and finite intrinsics")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("CaPE requires positive focal lengths and homogeneous intrinsics")
    model = str(camera.get("distortion_model", "none")).lower()
    model = {"": "none", "plumb_bob": "brown_conrady"}.get(model, model)
    if model not in ("none", "brown_conrady", "inverse_brown_conrady"):
        raise ValueError(f"Unsupported CaPE distortion model: {model}")
    if model != "none" and "D" not in camera:
        raise ValueError(f"CaPE {model} requires five distortion coefficients D")
    coefficients = np.asarray(camera.get("D", [0.] * 5), dtype=np.float64)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("CaPE distortion D must contain five finite coefficients")
    if model == "none" and coefficients.any():
        raise ValueError("Nonzero CaPE distortion coefficients require a distortion_model")
    if model != "none" and (k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError("CaPE Brown distortion requires zero-skew intrinsics")
    # ABC converts recorded K to CamPose's half-pixel convention on loading.
    # Native SDK K can instead be used explicitly with pixel_center=0.
    center = float(camera.get("pixel_center", .5))
    if center not in (0., .5):
        raise ValueError("CaPE pixel_center must be 0 or 0.5")
    return {"height": height, "width": width, "K": k.tolist(),
            "distortion_model": model, "D": coefficients.tolist(),
            "pixel_center": center}


def camera_direction_map(camera):
    """Return unit directions (3,H,W) in camera coordinates, before letterbox."""
    camera = camera_calibration(camera)
    k = torch.tensor(camera["K"], dtype=torch.float32)
    vv, uu = torch.meshgrid(
        torch.arange(camera["height"], dtype=torch.float32) + camera["pixel_center"],
        torch.arange(camera["width"], dtype=torch.float32) + camera["pixel_center"],
        indexing="ij",
    )
    if camera["distortion_model"] == "none" or not any(camera["D"]):
        pixels = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1)
        return F.normalize(pixels @ torch.linalg.inv(k).T, dim=-1).movedim(-1, 0)
    xo, yo = (uu - k[0, 2]) / k[0, 0], (vv - k[1, 2]) / k[1, 1]
    x, y = xo, yo
    k1, k2, p1, p2, k3 = camera["D"]
    inverse = camera["distortion_model"] == "inverse_brown_conrady"
    # This is the SDK's fixed-point update, including its distinct tangential
    # convention for INVERSE_BROWN_CONRADY (xq/yq rather than x/y).
    for _ in range(10):
        r2 = x.square() + y.square()
        radial = 1 + ((k3 * r2 + k2) * r2 + k1) * r2
        xq, yq = (x * radial, y * radial) if inverse else (x, y)
        dx = 2 * p1 * xq * yq + p2 * (r2 + 2 * xq.square())
        dy = 2 * p2 * xq * yq + p1 * (r2 + 2 * yq.square())
        x, y = (xo - dx) / radial, (yo - dy) / radial
    # Reject divergent/poor inversions offline instead of saving invalid rays.
    r2 = x.square() + y.square()
    radial = 1 + ((k3 * r2 + k2) * r2 + k1) * r2
    xf, yf = x * radial, y * radial
    tx, ty = (xf, yf) if inverse else (x, y)
    px = xf + 2 * p1 * tx * ty + p2 * (r2 + 2 * tx.square())
    py = yf + 2 * p2 * tx * ty + p1 * (r2 + 2 * ty.square())
    error = torch.hypot((px - xo) * k[0, 0], (py - yo) * k[1, 1])
    if not torch.isfinite(error).all() or error.max().item() > .01:
        raise ValueError("CaPE distortion inversion did not converge within 0.01 pixels")
    return F.normalize(torch.stack((x, y, torch.ones_like(x))), dim=0)


@lru_cache(maxsize=64)
def _cached_camera_directions(calibration_json, height, width, device):
    directions = camera_direction_map(json.loads(calibration_json))[None]
    raw_h, raw_w = directions.shape[-2:]
    if (raw_h, raw_w) != (height, width):
        scale = min(height / raw_h, width / raw_w)
        new_h, new_w = max(1, round(raw_h * scale)), max(1, round(raw_w * scale))
        # RGB export uses bicubic; antialiasing also matches Pillow's resize.
        directions = F.interpolate(directions, (new_h, new_w), mode="bicubic",
                                   align_corners=False, antialias=True)
        top, left = (height - new_h) // 2, (width - new_w) // 2
        directions = F.pad(directions, (left, width - new_w - left,
                                        top, height - new_h - top))
    return F.normalize(directions[0], dim=0).to(device=device).contiguous()


def cached_camera_directions(camera, height, width, device="cpu"):
    """Cache by calibration/grid/device, never by a frame's changing pose.

    Resize to the supplied RGB grid with bicubic, then zero-pad and normalize.
    Offline preprocessing saves this fixed tensor beside the episode. Online
    inference can reuse it in memory, including directly on the target device.
    Callers must treat the returned cached tensor as read-only.
    """
    if min(height, width) <= 0:
        raise ValueError("CaPE direction grid must have positive dimensions")
    key = json.dumps(camera_calibration(camera), sort_keys=True, allow_nan=False)
    return _cached_camera_directions(key, int(height), int(width), str(device))


def plucker_from_directions(directions, camera_to_world):
    """Only current-frame work: rotate fixed directions and compute o cross d."""
    directions = torch.as_tensor(directions)
    pose = torch.as_tensor(camera_to_world, device=directions.device, dtype=directions.dtype)
    if directions.ndim < 3 or directions.shape[-3] != 3 or pose.shape[-2:] != (4, 4):
        raise ValueError("CaPE needs directions (...,3,H,W) and poses (...,4,4)")
    world = pose[..., :3, :3] @ directions.flatten(-2)
    origin = pose[..., :3, 3, None].expand_as(world)
    moment = torch.cross(origin, world, dim=-2)
    return torch.cat((moment, world), dim=-2).unflatten(-1, directions.shape[-2:])
