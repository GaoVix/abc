"""Plücker camera conditioning, following CamPose's ACT late fusion.

Reference: https://github.com/ripl/CamPoseOpensource
Architecture adapted from CamPose (MIT, Copyright (c) 2025 RIPL);
see assets/third_party/campose/LICENSE.
Camera matrices use OpenCV axes (right, down, forward), pixel centers at
(u + .5, v + .5), and metric camera-to-world poses. Ray channels are [m, d],
where d is a unit direction and m = camera_center cross d, matching the
released ACT implementation. Geometric features are fused after DINO/CLIP;
the image encoder, CLS token, pooling, and camera ID embeddings are retained.
"""


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def validate_camera_matrices(intrinsics, camera_to_world):
    """Validate calibration at the data boundary, outside compiled forwards."""
    k = torch.as_tensor(intrinsics)
    pose = torch.as_tensor(camera_to_world)
    if k.shape[-2:] != (3, 3) or pose.shape[-2:] != (4, 4):
        raise ValueError("CaPE requires intrinsics (...,3,3) and camera_to_world (...,4,4)")
    if not torch.isfinite(k).all() or not torch.isfinite(pose).all():
        raise ValueError("CaPE camera matrices must be finite")
    if not ((k[..., 0, 0] > 0) & (k[..., 1, 1] > 0)).all():
        raise ValueError("CaPE focal lengths must be positive")
    if not torch.allclose(k[..., 2, :], k.new_tensor([0, 0, 1]).expand_as(k[..., 2, :])):
        raise ValueError("CaPE intrinsics must have homogeneous row [0,0,1]")
    if not torch.allclose(pose[..., 3, :], pose.new_tensor([0, 0, 0, 1]).expand_as(pose[..., 3, :])):
        raise ValueError("CaPE camera_to_world must have homogeneous row [0,0,0,1]")
    rotation = pose[..., :3, :3].float()
    identity = torch.eye(3, device=rotation.device).expand_as(rotation)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4, rtol=1e-4):
        raise ValueError("CaPE camera rotation must be orthonormal")
    if not torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=1e-4):
        raise ValueError("CaPE camera rotation must be right-handed")


# Copied from the CamPose README's official OpenCV "Plücker Snippet".
def get_plucker_raymap(K, c2w, height, width):
    """intrinsics (3,3), cam2world (4,4), height int, width int"""
    vv, uu = torch.meshgrid(
        torch.arange(height, device=K.device, dtype=K.dtype) + 0.5,
        torch.arange(width, device=K.device, dtype=K.dtype) + 0.5,
        indexing="ij",
    )
    rays = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
    d_world = torch.nn.functional.normalize(
        (rays @ torch.linalg.inv(K).T) @ c2w[:3, :3].T,
        dim=-1,
        eps=1e-9,
    )
    o = c2w[:3, 3].view(1, 1, 3)
    m = torch.cross(o, d_world, dim=-1)
    return torch.cat([d_world, m], dim=-1)


def plucker_raymap(intrinsics, camera_to_world, height, width):
    """Batch adapter for the official function; return (...,6,H,W) in [m,d] order."""
    k = torch.as_tensor(intrinsics).float()
    pose = torch.as_tensor(camera_to_world, device=k.device).float()
    batch_shape = torch.broadcast_shapes(k.shape[:-2], pose.shape[:-2])
    if batch_shape:
        k = k.expand(*batch_shape, 3, 3).reshape(-1, 3, 3)
        pose = pose.expand(*batch_shape, 4, 4).reshape(-1, 4, 4)
        rays = torch.vmap(lambda ki, pi: get_plucker_raymap(ki, pi, height, width))(k, pose)
        rays = rays.reshape(*batch_shape, height, width, 6)
    else:
        rays = get_plucker_raymap(k, pose, height, width)
    return torch.cat((rays[..., 3:], rays[..., :3]), dim=-1).movedim(-1, -3)


def normalize_plucker(rays):
    """Renormalize both line coordinates after spatial resampling; padding stays 0."""
    length = torch.linalg.vector_norm(rays[..., 3:, :, :], dim=-3, keepdim=True)
    return rays / length.clamp_min(1e-8)


def resize_pad_plucker(rays, target_h=224, target_w=224):
    """Apply the same bilinear/antialias resize and zero padding as RGB inference."""
    height, width = rays.shape[-2:]
    if (height, width) != (target_h, target_w):
        ratio = max(width / target_w, height / target_h)
        new_h, new_w = max(1, round(height / ratio)), max(1, round(width / ratio))
        rays = F.interpolate(rays, (new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
        rays = F.pad(rays, (left, target_w - new_w - left, top, target_h - new_h - top))
    return normalize_plucker(rays)


def observation_plucker(obs, camera_keys, device):
    """Make rays; optional camera_distortion uses cached calibrated directions.

    camera_distortion[cam] supplies distortion_model/D (and optionally the
    original height/width and pixel_center). First match the supplied RGB grid,
    then apply the model's shared RGB/ray resize and padding.
    """
    if "camera_intrinsics" not in obs or "camera_to_world" not in obs:
        raise ValueError("CaPE inference requires per-camera camera_intrinsics and camera_to_world in the observation")
    batch_size = 1 if np.ndim(obs["state"]) == 1 else len(obs["state"])
    result = {}
    for cam in camera_keys:
        k = torch.as_tensor(obs["camera_intrinsics"][cam], device=device, dtype=torch.float32)
        pose = torch.as_tensor(obs["camera_to_world"][cam], device=device, dtype=torch.float32)
        validate_camera_matrices(k, pose)
        height, width = obs["images"][cam].shape[-2:]
        distortion = obs.get("camera_distortion", {}).get(cam)
        if distortion is not None:
            from abc_minimal.camera_rays import cached_camera_directions, plucker_from_directions

            calibrations = k.reshape(-1, 3, 3).cpu().tolist()
            directions = torch.stack([
                cached_camera_directions({"K": ki, "height": height, "width": width,
                                          **distortion}, height, width, device)
                for ki in calibrations
            ])
            rays = plucker_from_directions(directions, pose).reshape(-1, 6, height, width)
        else:
            rays = plucker_raymap(k, pose, height, width).reshape(-1, 6, height, width)
        if len(rays) == 1:
            rays = rays.expand(batch_size, -1, -1, -1)
        if len(rays) != batch_size:
            raise ValueError(f"CaPE {cam}: camera batch does not match state batch")
        result[cam] = resize_pad_plucker(rays)
    return result


def mujoco_camera_geometry(model, data, camera_keys, height, width):
    """Read poses refreshed before RGB by MuJoCoYAMEnv.get_obs, in OpenCV axes."""
    intrinsics, poses = {}, {}
    for cam in camera_keys:
        index = model.camera(cam).id
        focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[index]) / 2))
        intrinsics[cam] = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = data.cam_xmat[index].reshape(3, 3) @ np.diag([1, -1, -1])
        pose[:3, 3] = data.cam_xpos[index]
        poses[cam] = pose
    return {"camera_intrinsics": intrinsics, "camera_to_world": poses}


# Copied from CamPose's ACT models/backbone.py, including checkpoint loading.
class FrozenBatchNorm2d(torch.nn.Module):

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PluckerLateFusion(nn.Module):
    """Shared ACT-style CNN matching ABC's patch grid and embedding width."""

    def __init__(self, embed_dim):
        super().__init__()
        # CamPose ACT's five-layer layout, with final stride 1 and output
        # embed_dim (768 by default) to match ABC's ViT-B/16 features.
        # At 224x224: 112 -> 56 -> 28 -> 14 -> 14, matching the RGB patch grid.
        self.plucker_encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FrozenBatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            FrozenBatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            FrozenBatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            FrozenBatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            FrozenBatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # ACT uses a 1x1 convolution; Linear is equivalent on patch tokens.
        self.fusion = nn.Linear(2 * embed_dim, embed_dim)
        # Preserve the parent's RGB features at initialization. Unlike the
        # SmolVLA variant, no extra LayerNorm is inserted into the RGB stream.
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, :embed_dim].copy_(torch.eye(embed_dim))
            self.fusion.bias.zero_()

    def forward(self, tokens, rays, grid_size):
        if rays.ndim != 4 or rays.shape[1] != 6 or rays.shape[0] != tokens.shape[0]:
            raise ValueError("CaPE rays must have shape (B,6,H,W), aligned with image tokens")
        if tokens.shape[1] != 1 + grid_size[0] * grid_size[1]:
            raise ValueError("CaPE expects one CLS token followed by a 2D patch grid")
        features = self.plucker_encoder(rays.to(dtype=self.plucker_encoder[0].weight.dtype))
        if features.shape[-2:] != grid_size:
            raise ValueError(f"CaPE geometry grid {features.shape[-2:]} does not match image patch grid {grid_size}")
        features = features.flatten(2).transpose(1, 2).to(tokens.dtype)
        patches = self.fusion(torch.cat((tokens[:, 1:], features), dim=-1))
        return torch.cat((tokens[:, :1], patches), dim=1)
