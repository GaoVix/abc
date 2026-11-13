"""Shared state/action normalization and image preprocessing."""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Image normalization stats, keyed by preset: DINOv3 uses ImageNet stats,
# CLIP its own. (mean, std) as (3, 1, 1) tensors.
NORM_PRESETS = {
    "imagenet": (
        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
        torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
    ),
    "clip": (
        torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1),
        torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1),
    ),
}


def preset_for_backbone(vision_backbone):
    """Map a vision backbone name to its normalization preset."""
    return "clip" if vision_backbone == "clip" else "imagenet"


def parse_norm_stats(raw):
    stats = raw.get("norm_stats", raw)
    if "state" not in stats and "actions" not in stats:
        key = "xdof" if "xdof" in stats else next(iter(stats))
        stats = stats[key]
    return {
        key: {k: np.asarray(v, dtype=np.float32) for k, v in stats[key].items()}
        for key in ("state", "actions")
    }


def load_norm_stats(path):
    return parse_norm_stats(json.loads(Path(path).read_text()))


def normalize(x, stats):
    return (x - stats["mean"]) / (stats["std"] + 1e-6)


def unnormalize(x, stats):
    return x * (stats["std"] + 1e-6) + stats["mean"]


def resize_with_pad(img_hwc, target_h=224, target_w=224):
    h, w, _ = img_hwc.shape
    if (h, w) == (target_h, target_w):
        return img_hwc
    ratio = max(w / target_w, h / target_h)
    new_h = max(1, round(h / ratio))
    new_w = max(1, round(w / ratio))
    resized = F.interpolate(
        img_hwc.permute(2, 0, 1).unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    pad_h0 = (target_h - new_h) // 2
    pad_h1 = target_h - new_h - pad_h0
    pad_w0 = (target_w - new_w) // 2
    pad_w1 = target_w - new_w - pad_w0
    padded = F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), value=0)
    return padded.permute(1, 2, 0)


def normalize_image(img_chw, preset="imagenet"):
    """Normalize a CHW image with the given preset ("imagenet" or "clip")."""
    mean, std = NORM_PRESETS[preset]
    mean = mean.to(device=img_chw.device, dtype=img_chw.dtype)
    std = std.to(device=img_chw.device, dtype=img_chw.dtype)
    return (img_chw - mean) / (std + 1e-6)


def resize_pad_raw(img_chw, target_h=224, target_w=224):
    """Aspect-preserving resize/pad of a CHW image into raw [0, 1].

    Where the VLA image path stops: SigLIP owns mean/std. DiT continues
    through resize_pad_normalize.
    """
    x = torch.as_tensor(img_chw).float()
    if x.max() > 1.0:
        x = x / 255.0
    return resize_with_pad(x.permute(1, 2, 0), target_h, target_w).permute(2, 0, 1)


def resize_pad_normalize(img_chw, target_h=224, target_w=224, preset="imagenet"):
    x = resize_pad_raw(img_chw, target_h, target_w)
    return normalize_image(x, preset=preset)


def resize_pad_normalize_batch(img_bchw, target_h=224, target_w=224, preset="imagenet"):
    """Resize/pad (B, 3, H, W) uint8 [0,255] or floating [0,1] images.

    ``preset=None`` stops at raw [0, 1], like ``resize_pad_raw``.
    """
    x = torch.as_tensor(img_bchw)
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    elif torch.is_floating_point(x):
        if not ((x >= 0) & (x <= 1)).all():
            raise ValueError("Float images must be finite and in [0, 1]")
    else:
        raise TypeError("Images must be uint8 or floating point")
    _, _, h, w = x.shape
    if (h, w) != (target_h, target_w):
        ratio = max(w / target_w, h / target_h)
        new_h = max(1, int(round(h / ratio)))
        new_w = max(1, int(round(w / ratio)))
        x = F.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True
        )
        pad_h0 = (target_h - new_h) // 2
        pad_w0 = (target_w - new_w) // 2
        x = F.pad(x, (pad_w0, target_w - new_w - pad_w0, pad_h0, target_h - new_h - pad_h0), value=0)
    return x if preset is None else normalize_image(x, preset=preset)


def _rotate(img_hwc, angle_deg):
    """Rotate (H,W,C) with reflection padding."""
    if abs(angle_deg) < 0.1:
        return img_hwc
    H, W, _ = img_hwc.shape
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )
    grid = torch.stack([gx * cos_a - gy * sin_a, gx * sin_a + gy * cos_a], dim=-1)
    out = F.grid_sample(
        img_hwc.permute(2, 0, 1).unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    return out.squeeze(0).permute(1, 2, 0)


def augment_and_normalize(images, train, norm_preset="imagenet", image_size=224, *, plucker_maps=None):
    """Apply augmentations, optionally followed by backbone normalization.

    ``norm_preset=None`` is the VLA path: Gemma/SigLIP owns normalization and
    therefore receives resized images in raw [0, 1] space.
    With ``plucker_maps``, return (images, maps) after identical spatial
    augmentation; color augmentation and normalization apply only to RGB.
    """
    out = {}
    out_plucker = {}
    for cam, img in images.items():
        x = img.permute(1, 2, 0)
        if plucker_maps is not None:
            rays = plucker_maps[cam]
            if rays.shape != (6, *img.shape[-2:]):
                raise ValueError(f"{cam}: Plücker map must align with the unaugmented RGB image")
            # One sampling grid for RGB and rays, including rotation, crop,
            # resize, and padding. Photometric augmentation stays RGB-only.
            x = torch.cat((x, rays.permute(1, 2, 0)), dim=-1)
        if train and "top" in cam:
            angle = (torch.rand(1) * 10 - 5).item()
            x = _rotate(x, angle)
            H, W, _ = x.shape
            ch, cw = int(H * 0.95), int(W * 0.95)
            if H - ch > 0 and W - cw > 0:
                sh = torch.randint(0, H - ch + 1, (1,)).item()
                sw = torch.randint(0, W - cw + 1, (1,)).item()
                x = x[sh : sh + ch, sw : sw + cw, :]
                x = F.interpolate(
                    x.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
        x = resize_with_pad(x, image_size, image_size)
        if plucker_maps is not None:
            from abc_minimal.cape import normalize_plucker

            out_plucker[cam] = normalize_plucker(x[..., 3:].permute(2, 0, 1))
            x = x[..., :3]
        if train:
            b = 0.7 + torch.rand(1).item() * 0.6
            x = x * b
            c = 0.6 + torch.rand(1).item() * 0.8
            mean = x.mean()
            x = (x - mean) * c + mean
            s = 0.5 + torch.rand(1).item() * 1.0
            gray = x.mean(dim=-1, keepdim=True)
            x = gray + (x - gray) * s
            x = torch.clamp(x, 0, 1)
        x = x.permute(2, 0, 1)
        out[cam] = x if norm_preset is None else normalize_image(x, preset=norm_preset)
    return (out, out_plucker) if plucker_maps is not None else out
