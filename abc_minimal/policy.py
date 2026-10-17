"""Reusable ABC-DiT and VLA inference policies for sim and real deployment."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from abc_minimal.checkpointing import model_state_dict
from abc_minimal.config import (
    FlowConfig,
    VLAModelConfig,
    validate_vla_checkpoint_config,
)
from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained
from abc_minimal.eef_adapters import EEF_GRIPPER_INDICES, EEF_POSE_INDICES, EEFInferenceAdapter
from abc_minimal.preprocess import (
    normalize,
    parse_norm_stats,
    preset_for_backbone,
    resize_pad_normalize,
    resize_pad_normalize_batch,
    unnormalize,
)
from abc_minimal.vla import VLAPolicy, inference_model_config, stack_camera_batch


def resolve_norm_stats(ckpt: dict[str, Any], override: str | None) -> dict[str, Any]:
    if override:
        raw = json.loads(Path(override).expanduser().read_text())
    elif ckpt.get("norm_stats") is not None:
        raw = ckpt["norm_stats"]
    else:
        raise ValueError("No norm_stats in checkpoint; pass --norm-stats-path")
    return parse_norm_stats(raw)


def resolve_trained_max_prefix(ckpt: dict[str, Any]) -> int:
    """The checkpoint's max_action_prefix training bound.

    The bound is EXCLUSIVE: the trainer samples prefix lengths from
    randint(0, max_action_prefix), so the longest prefix actually seen in
    training is max_action_prefix - 1. Production checkpoints store a flat
    train_config dict with max_action_prefix; this repo's trainer saves the
    TrainConfig asdict, which nests it under flow. Fall back to the local
    FlowConfig default when the checkpoint predates either convention.
    """
    train_config = ckpt.get("train_config")
    raw = None
    if isinstance(train_config, dict):
        raw = train_config.get("max_action_prefix")
        if raw is None and isinstance(train_config.get("flow"), dict):
            raw = train_config["flow"].get("max_action_prefix")
    elif train_config is not None:
        # Tolerate non-dict train_configs (dataclass, Namespace, OmegaConf).
        raw = getattr(train_config, "max_action_prefix", None)
        if raw is None:
            flow = getattr(train_config, "flow", None)
            if flow is not None:
                raw = getattr(flow, "max_action_prefix", None)
    if raw is not None:
        return int(raw)
    return FlowConfig().max_action_prefix


class InferencePolicy:
    """Prefix conditioning and RTC warmup shared by the DiT and VLA policies."""

    fast_rtc_warmup_replays = 1
    fast_inference_replay_warmups = 1
    _warming_up = False

    def reset(self) -> None:
        if getattr(self, "eef_adapter", None) is not None:
            self.eef_adapter.reset()

    def enable_fast_inference(
        self,
        compile_mode: str = "max-autotune",
        replay_warmups: int | None = None,
        warmup_obs: dict[str, Any] | None = None,
        warmup_noise: np.ndarray | None = None,
        rtc_prefix_length: int | None = None,
    ) -> None:
        """Compile the model's hot path and warm it so rollouts skip the JIT cost.

        The compile step differs per policy (see ``_compile_for_fast_inference``);
        the CUDA/TF32 setup and the warmup that forces compilation are shared.
        """
        if self._fast_inference_enabled:
            return
        if self.device.type != "cuda":
            raise RuntimeError("fast inference requires a CUDA device")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self._compile_for_fast_inference(compile_mode)
        self._fast_inference_enabled = True

        # torch.compile is lazy: without this warmup the first real inference
        # of a rollout pays the compilation cost.
        if replay_warmups is None:
            replay_warmups = self.fast_inference_replay_warmups
        m = self.model_config
        if warmup_obs is None:
            warmup_obs = {
                "state": np.zeros(m.state_dim, dtype=np.float32),
                "images": {
                    cam: np.zeros(
                        (3, self.config.camera_height, self.config.camera_width),
                        dtype=np.uint8,
                    )
                    for cam in m.camera_keys
                },
                "prompt": self.config.prompt,
            }
        if warmup_noise is None:
            warmup_noise = np.zeros((m.chunk_length, m.action_dim), dtype=np.float32)
        self._warming_up = True
        for _ in range(max(1, replay_warmups)):
            self.infer(warmup_obs, noise=warmup_noise)
        self._warming_up = False
        torch.cuda.synchronize(self.device)
        if rtc_prefix_length is not None:
            self.warmup_rtc(warmup_obs, warmup_noise, rtc_prefix_length)

    def _compile_for_fast_inference(self, compile_mode: str) -> None:
        """Compile the policy-specific hot path. Implemented by each subclass."""
        raise NotImplementedError

    def normalized_action_prefix(
        self,
        action_prefix: np.ndarray,
        prefix_length: int,
        stats=None,
    ) -> np.ndarray:
        """Pad a prefix to a full chunk and normalize it (leading batch dims ok)."""
        stats = self.norm_stats["actions"] if stats is None else stats
        chunk_length, action_dim = self.chunk_length, len(stats["mean"])
        prefix = np.asarray(action_prefix, dtype=np.float32)
        if prefix.shape[-2:] == (prefix_length, action_dim):
            full_prefix = np.zeros(
                (*prefix.shape[:-2], chunk_length, action_dim), dtype=np.float32
            )
            full_prefix[..., :prefix_length, :] = prefix
            prefix = full_prefix
        if prefix.shape[-2:] != (chunk_length, action_dim):
            raise ValueError(
                f"action_prefix must end in shape {(chunk_length, action_dim)} "
                f"or {(prefix_length, action_dim)}, got {prefix.shape}"
            )
        return normalize(prefix, stats).astype(
            np.float32, copy=False
        )

    def warmup_rtc(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None,
        prefix_length: int,
    ) -> None:
        """Warm the prefix-conditioned sampler that RTC rollouts replay."""
        leading = np.shape(obs["state"])[:-1]
        action_prefix = np.zeros(
            (*leading, self.chunk_length, self.action_dim), dtype=np.float32
        )
        replays = self.fast_rtc_warmup_replays if self._fast_inference_enabled else 1
        self._warming_up = True
        for _ in range(replays):
            self.infer(
                obs,
                noise=noise,
                action_prefix=action_prefix,
                prefix_length=prefix_length,
            )
        self._warming_up = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class DiTInferencePolicy(InferencePolicy):
    """Checkpoint-backed DiT inference shared by sim and deploy adapters.

    ``infer`` takes one observation (state ``(S,)``, images ``(3, H, W)``) or a
    batch of worlds (state ``(B, S)``, images ``(B, 3, H, W)`` as numpy arrays
    or CUDA tensors, one prompt or one per world) and returns one action chunk
    per world.
    """

    fast_rtc_warmup_replays = 8  # compiled samplers are graph-replayed
    fast_inference_replay_warmups = 24

    def __init__(self, checkpoint: Path, config: Any, device: str, model_config: Any = None):
        self.config = config
        self.model_config = model_config if model_config is not None else config.model
        self.device = torch.device(device)
        self.diffusion_steps = config.diffusion_steps
        self.chunk_length = self.model_config.chunk_length
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.model_config.action_dim = model_state_dict(ckpt)["y_embedder.weight"].shape[1]
        self.action_dim = 14 if self.model_config.action_dim == 20 else self.model_config.action_dim
        self.model = DiTPolicy(self.model_config).to(self.device)
        load_pretrained(self.model, ckpt)
        self.model.eval()
        self.norm_preset = preset_for_backbone(self.model_config.vision_backbone)
        self.norm_stats = resolve_norm_stats(ckpt, config.norm_stats_path)
        self.trained_max_prefix = resolve_trained_max_prefix(ckpt)
        self.embedder = CLIPTextEmbedder(config.clip, device=self.device)
        self._prompt = config.prompt
        self.task_vec = self.embedder.encode([self._prompt]).to(self.device)
        self._fast_inference_enabled = False
        adapter_checkpoint = getattr(config, "eef_adapters", None) or ckpt.get("eef_adapters")
        self.eef_adapter = (
            EEFInferenceAdapter(adapter_checkpoint, self.device, config.execute_chunk_dim, self.model_config.action_dim)
            if adapter_checkpoint else None
        )
        self.action_norm_stats = self.norm_stats["actions"]
        if self.model_config.action_dim == 20:
            self.action_norm_stats = {
                key: np.empty(20, dtype=np.float32)
                for key in ("mean", "std")
            }
            for key, values in self.action_norm_stats.items():
                values[EEF_POSE_INDICES] = getattr(self.eef_adapter.mlps, f"pose_{key}").cpu().numpy()
                values[EEF_GRIPPER_INDICES] = self.norm_stats["actions"][key][[6, 13]]

    def set_prompt(self, prompt: str) -> None:
        if prompt == self._prompt:
            return
        self._prompt = prompt
        self.task_vec = self.embedder.encode([prompt]).to(
            device=self.device, dtype=self.model.x_embedder.weight.dtype
        )

    def _compile_for_fast_inference(self, compile_mode: str) -> None:
        """Cast to bf16 and compile the default graph-break-free samplers."""
        self.model.to(torch.bfloat16)
        self.model.img_backbone.set_bfloat16(True)
        self.task_vec = self.task_vec.to(device=self.device, dtype=torch.bfloat16)

        compile_kwargs: dict[str, Any] = {"dynamic": False, "fullgraph": True}
        if compile_mode:
            compile_kwargs["mode"] = compile_mode
        self.model.sample_actions = torch.compile(
            self.model.sample_actions, **compile_kwargs
        )
        self.model.sample_actions_rtc = torch.compile(
            self.model.sample_actions_rtc, **compile_kwargs
        )

    @torch.no_grad()
    def infer(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = 0,
    ) -> np.ndarray:
        if action_prefix is not None and prefix_length is None:
            prefix_length = np.shape(action_prefix)[-2]
        prefix_length = int(prefix_length or 0)
        m = self.model_config
        state = np.asarray(obs["state"], dtype=np.float32)
        raw_state = state
        if self.eef_adapter is not None:
            state = self.eef_adapter.state(state)
        batched = state.ndim == 2
        prompt = obs.get("prompt", self._prompt)
        if batched:
            prompts = (
                [prompt] * len(state) if isinstance(prompt, str) else list(prompt)
            )
            task_vec = self.embedder.encode(prompts).to(
                device=self.device, dtype=self.task_vec.dtype
            )
            images = {
                cam: resize_pad_normalize_batch(
                    torch.as_tensor(obs["images"][cam]).to(self.device),
                    preset=self.norm_preset,
                )
                for cam in m.camera_keys
            }
        else:
            self.set_prompt(str(prompt))
            task_vec = self.task_vec
            images = {
                cam: resize_pad_normalize(obs["images"][cam], preset=self.norm_preset)
                .unsqueeze(0)
                .to(self.device)
                for cam in m.camera_keys
            }
        state = normalize(state, self.norm_stats["state"]).reshape(-1, m.state_dim)
        num_worlds = len(state)
        batch = {
            "state": torch.from_numpy(state).to(self.device),
            "actions": torch.zeros(
                num_worlds, m.chunk_length, m.action_dim, device=self.device
            ),
            "images": images,
            "task_vec_clip": task_vec,
        }
        noise_t = None
        if noise is not None:
            noise_arr = np.asarray(noise, dtype=np.float32).reshape(
                num_worlds, m.chunk_length, m.action_dim
            )
            noise_t = torch.from_numpy(noise_arr).to(self.device)
        if action_prefix is None:
            actions = self.model.sample_actions(
                batch, num_steps=self.diffusion_steps, noise=noise_t
            )
        else:
            prefix_for_policy = action_prefix
            if self.eef_adapter is not None and not self._warming_up:
                prefix_for_policy = self.eef_adapter.action_prefix(action_prefix, prefix_length)
            elif m.action_dim == 20:
                prefix_for_policy = np.zeros((*np.shape(action_prefix)[:-1], 20), dtype=np.float32)
            prefix = self.normalized_action_prefix(
                prefix_for_policy, prefix_length, self.action_norm_stats
            ).reshape(num_worlds, m.chunk_length, m.action_dim)
            prefix_t = torch.from_numpy(prefix).to(
                device=self.device, dtype=batch["state"].dtype
            )
            actions = self.model.sample_actions_rtc(
                batch,
                prefix_t,
                prefix_length=prefix_length,
                num_steps=self.diffusion_steps,
                noise=noise_t,
            )
        actions_np = actions.float().detach().cpu().numpy()
        actions_np = unnormalize(actions_np, self.action_norm_stats).astype(
            np.float32
        )
        if self.eef_adapter is not None and not self._warming_up:
            actions_np = self.eef_adapter.actions(actions_np, raw_state, action_prefix, prefix_length)
        return actions_np if batched else actions_np[0]


@dataclass
class InferenceConfig:
    """Everything an inference policy reads except its architecture."""

    checkpoint_path: str = ""
    norm_stats_path: str | None = None
    """Only needed for checkpoints without embedded stats."""
    prompt: str = "throw plastic bottles in bin"
    diffusion_steps: int = 10
    device: str = "auto"
    deterministic: bool = False
    fast_inference: bool = False
    fast_compile_mode: str = "max-autotune"
    rtc_prefix_length: int | None = None
    camera_height: int = 480
    camera_width: int = 640


def shared_inference_fields(config: Any) -> dict[str, Any]:
    """Pull the InferenceConfig fields off any config carrying them."""
    return {f.name: getattr(config, f.name) for f in fields(InferenceConfig)}


@dataclass
class VLAPolicyConfig(InferenceConfig):
    """The config the deploy adapter hands VLAInferencePolicy.

    Sim eval and the viewer pass SimEvalConfig directly with ``model_config``.
    """

    model: VLAModelConfig = field(default_factory=VLAModelConfig)


class VLAInferencePolicy(InferencePolicy):
    """Checkpoint-backed VLA inference shared by sim and deploy adapters.

    Same contract as DiTInferencePolicy, different conditioning: raw-string
    prompts (no CLIP embedding, so no set_prompt), raw [0, 1] images stacked
    into one tensor, and one sample_actions taking the RTC prefix as kwargs.
    """

    def __init__(self, checkpoint: Path, config: Any, device: str, model_config: Any = None):
        self.config = config
        self.model_config = model_config if model_config is not None else config.model
        self.device = torch.device(device)
        self.diffusion_steps = config.diffusion_steps
        self.camera_keys = tuple(self.model_config.camera_keys)
        self.chunk_length = self.model_config.chunk_length
        self.action_dim = self.model_config.action_dim

        checkpoint = Path(checkpoint).expanduser().resolve()
        ckpt = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        # Check the architecture before building 4B parameters of it.
        if not validate_vla_checkpoint_config(
            self.model_config, ckpt, source=str(checkpoint)
        ):
            warnings.warn(
                "VLA checkpoint has no architecture metadata; only tensor keys "
                "and shapes can be validated",
                RuntimeWarning,
                stacklevel=2,
            )
        self.model = VLAPolicy(
            inference_model_config(self.model_config),
            backbone_dtype=torch.bfloat16,
            backbone_autocast=False,
        ).to(self.device)
        self.model.load_state_dict(model_state_dict(ckpt), strict=True)
        self.model.eval()
        self.norm_stats = resolve_norm_stats(ckpt, config.norm_stats_path)
        self.trained_max_prefix = resolve_trained_max_prefix(ckpt)
        self._fast_inference_enabled = False

    fast_rtc_warmup_replays = 4
    fast_inference_replay_warmups = 4

    def _compile_for_fast_inference(self, compile_mode: str) -> None:
        """Compile SigLIP, the Gemma decoder, and the head's velocity network.

        Every shape is static (fixed_seq_len tokens, chunk_length actions), so
        graph replay removes the ~3k kernel launches that dominate at batch 1.
        """
        kwargs: dict[str, Any] = {"dynamic": False, "mode": compile_mode or None}
        gemma = self.model.vla.gemma_model
        gemma.siglip_vision_model.forward = torch.compile(  # type: ignore[method-assign]
            gemma.siglip_vision_model.forward, **kwargs
        )
        gemma.model.forward = torch.compile(gemma.model.forward, **kwargs)  # type: ignore[method-assign]
        head = self.model.diffusion_head
        head.predict_velocity = torch.compile(head.predict_velocity, **kwargs)  # type: ignore[method-assign]

    @torch.no_grad()
    def infer(
        self,
        obs: dict[str, Any],
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = 0,
    ) -> np.ndarray:
        if action_prefix is not None and prefix_length is None:
            prefix_length = np.shape(action_prefix)[-2]
        prefix_length = int(prefix_length or 0)
        state = normalize(
            np.asarray(obs["state"], dtype=np.float32), self.norm_stats["state"]
        )
        batched = state.ndim == 2
        state = state.reshape(-1, state.shape[-1])
        num_worlds = len(state)
        prompt = obs.get("prompt", self.config.prompt)
        if batched and not isinstance(prompt, str):
            prompts = [str(item) for item in prompt]
        else:
            prompts = [str(prompt)] * num_worlds
        image_size = self.model_config.backbone.image_size
        images = {}
        for cam in self.camera_keys:
            image = torch.as_tensor(obs["images"][cam]).to(self.device)
            # resize_pad_normalize_batch rescales integer dtypes only, so a float
            # source must already be in [0, 1]. Every producer hands over uint8.
            images[cam] = resize_pad_normalize_batch(
                image if batched else image[None], image_size, image_size, preset=None
            )
        batch = stack_camera_batch(
            {
                "state": torch.from_numpy(state).to(self.device),
                "images": images,
                "prompt": prompts,
            },
            self.camera_keys,
        )
        noise_t = None
        if noise is not None:
            noise_arr = np.asarray(noise, dtype=np.float32).reshape(
                num_worlds, self.chunk_length, self.action_dim
            )
            noise_t = torch.from_numpy(noise_arr).to(self.device)
        prefix_kwargs: dict[str, Any] = {}
        if action_prefix is not None:
            prefix = self.normalized_action_prefix(action_prefix, prefix_length).reshape(
                num_worlds, self.chunk_length, self.action_dim
            )
            prefix_kwargs["action_prefix"] = torch.from_numpy(prefix).to(self.device)
            prefix_kwargs["prefix_length"] = prefix_length
        actions = self.model.sample_actions(
            batch, num_steps=self.diffusion_steps, noise=noise_t, **prefix_kwargs
        )
        actions_np = actions.float().detach().cpu().numpy()
        actions_np = unnormalize(actions_np, self.norm_stats["actions"]).astype(np.float32)
        return actions_np if batched else actions_np[0]
