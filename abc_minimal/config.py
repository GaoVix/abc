"""Training and model configuration dataclasses."""

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = REPO_ROOT / "cache"


def default_cache_root() -> Path:
    return Path(os.environ.get("ABC_CACHE", str(DEFAULT_CACHE_ROOT))).expanduser()


@dataclass
class OptimConfig:
    """AdamW with a linear-warmup-then-constant LR schedule."""
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1000
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    vision_lr_scale: float = 1.0
    # VLA policy only: LR multiplier for the Gemma/SigLIP backbone param group.
    backbone_lr_scale: float = 1.0


@dataclass
class FlowConfig:
    """Rectified-flow matching + action-prefix conditioning."""
    mask_state_ratio: float = 0.1
    max_action_prefix: int = 8
    prefix_conditioning_prob: float = 1.0
    prefix_noise_scale: float = 0.0
    num_diffusion_steps: int = 10
    # VLA policy only: independent noise/timestep draws per sample per step.
    num_diffusion_draws: int = 1


@dataclass
class PromptConfig:
    """Task/subtask/operator prompt composition.
    """
    # Condition on per-frame subtask labels (episodes' subtasks.json sidecars).
    use_subtask_as_prompt: bool = False
    # "replace" means the subtask label replaces the task prompt.
    # "append" formats it according to append format below
    subtask_mode: Literal["replace", "append"] = "replace"
    subtask_append_format: str = "{prompt}. {subtask}"
    # Probability of dropping the subtask label (reverting to the task prompt) during training
    subtask_dropout_prob: float = 0.2

    # "text_indexed" -> "operator N"; "text_name" -> an English first name from a fixed pool (overflow falls back to "operator N").
    use_operator_id_as_prompt: bool = False
    operator_prompting_mode: Literal["text_indexed", "text_name"] = "text_indexed"
    operator_append_format: str = "{prompt}. {operator}"
    # Per-task label-map manifest; required when operator prompting is on.
    # Build a priori with scripts/build_operator_label_map.py.
    operator_label_map_path: str = ""
    # Probability of dropping the operator label
    operator_dropout_prob: float = 0.2


@dataclass
class ClipConfig:
    """CLIP asset cache: ViT-B/32 text encoder + ViT-B/16 vision weights."""
    cache_dir: str = field(default_factory=lambda: str(Path.home() / ".cache" / "clip"))
    model_url: str = (
        "https://openaipublic.azureedge.net/clip/models/"
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    )
    bpe_url: str = (
        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
    )
    model_name: str = "ViT-B-32.pt"
    bpe_name: str = "bpe_simple_vocab_16e6.txt.gz"
    vision_model_url: str = (
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
    )
    vision_model_name: str = "ViT-B-16.pt"


@dataclass
class MixtureComponent:
    """One source in the train/val mixture."""
    train_dir: str
    val_dir: str
    weight: float
    task_name: str


@dataclass
class DiTConfig:
    """ABC-DiT architecture defaults."""
    hidden_size: int = 1536
    depth: int = 32
    num_heads: int = 24
    mlp_ratio: float = 4.0
    state_dim: int = 14
    action_dim: int = 14
    chunk_length: int = 30
    camera_keys: tuple[str, ...] = ("top", "left", "right")
    task_embed_dim: int = 512

    # Vision backbone: "dinov3" or "clip"
    vision_backbone: Literal["dinov3", "clip"] = "dinov3"

    vit_embed_dim: int = 768
    vit_depth: int = 12
    vit_num_heads: int = 12
    vision_pool_num_queries: int = 12
    vision_pool_num_heads: int = 8
    vision_pool_mlp_ratio: int = 4


MIXTURE_PRESETS: dict[str, list[MixtureComponent]] = {
    "bottles": [
        MixtureComponent("train_real", "val_real", 0.8172, "throw_plastic_bottles_in_bin"),
        MixtureComponent("train_sim", "val_sim", 0.1828, "sim_put_the_plastic_bottles_in_the_bin"),
    ],
    "tshirt": [
        MixtureComponent("train_real", "val_real", 1.0, "folding_tshirt_pile_and_stacking"),
    ],
    "load_the_plates_into_the_dish_rack": [
        MixtureComponent("train_real", "val_real", 0.8, "load_the_plates_into_the_dish_rack"),
        MixtureComponent("train_sim", "val_sim", 0.2, "sim_load_the_plates_into_the_dish_rack"),
    ],
    # Single-task sim finetuning: point --cache-root at a cache holding exactly
    # one task's episodes (prepare.py --sim-data <task> into a fresh root). The
    # component reads everything under train_sim/, and each episode's own
    # metadata task_name drives the training prompt, so one preset serves any
    # sim task.
    "sim_task": [
        MixtureComponent("train_sim", "val_sim", 1.0, ""),
    ],
}


@dataclass
class TrainConfig:
    """Minimal ABC training: the CLIP/DINOv3 ABC-DiT policy (default) or the
    ABC-VLA, selected with ``--policy {dit,vla}``. Both share this
    single entry point (``train.py``), training loop, optimizer/scheduler,
    validation, and checkpoint format; ``--policy vla`` swaps in the Gemma
    backbone (``--vla-model.*``) and enables the VLA-only optim/flow knobs."""
    policy: Literal["dit", "vla"] = "dit"

    cache_root: str = field(
        default_factory=lambda: str(default_cache_root())
    )
    # Where checkpoints are written. Defaults per policy when unset:
    # dit -> cache/finetune_checkpoints, vla -> cache/vla_checkpoints.
    output_dir: str | None = None
    seed: int = 123
    batch_size: int = 90
    num_workers: int = 16
    train_steps: int = 75_000

    mixture_preset: Literal[
        "bottles", "tshirt", "sim_task", "load_the_plates_into_the_dish_rack"
    ] = "bottles"
    mixture: list[MixtureComponent] = field(default_factory=list)

    load_pretrained: bool = False
    pretrained_ckpt_name: str = "abc_dit_xl_200k_model.pt"
    inherit_ckpt_norm_stats: bool = True  # scale inputs with the checkpoint's own norm_stats, as during pretraining
    resume_from: str | None = None
    dino_bf16: bool = True
    # Backbone compute only; parameters and pool/head stay FP32.
    vla_bf16_autocast: bool = True
    # VLA only: shard parameters, gradients, and Adam state across the ranks (FSDP2)
    # instead of replicating them (DDP). Needs torchrun with more than one process.
    # Multi-node automatically uses HSDP (shard within node over NVLink, replicate
    # across nodes); single node uses flat full FSDP. See train_loop._shard_vla.
    fsdp: bool = False
    compile: bool = True
    # VLA policy only: torch.compile the SigLIP tower (the Gemma stack is not
    # compiled). The DiT policy uses `compile` above.
    compile_siglip: bool = False

    log_every: int = 20
    val_every: int = 2500
    val_batches: int = 4
    ckpt_every: int = 5000
    log_wandb: bool = False
    wandb_project: str = "minimal-abc"

    optim: OptimConfig = field(default_factory=OptimConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    # DiT policy architecture (used when policy="dit").
    model: DiTConfig = field(default_factory=DiTConfig)
    # VLA policy architecture (used when policy="vla").
    vla_model: "VLAModelConfig" = field(default_factory=lambda: VLAModelConfig())

    def resolve_mixture(self) -> list[MixtureComponent]:
        return self.mixture if self.mixture else MIXTURE_PRESETS[self.mixture_preset]


@dataclass
class GemmaVLAConfig:
    """Gemma 3 4B/SigLIP backbone settings for ABC-VLA."""
    checkpoint: str | None = None
    load_base_checkpoint: bool = True
    image_size: int = 224
    fixed_seq_len: int = 256
    feature_layer: int = -1
    train_backbone: bool = True
    train_siglip: bool = True
    activation_checkpointing: bool = True
    proprio_token: bool = True
    # The 262k-row token table only gets gradient on the prompt's few tokens;
    # training it costs a dense 671M-parameter gradient, all-reduce, and Adam state.
    train_token_embedding: bool = False


@dataclass
class VLADiTConfig:
    """AdaLN diffusion action-head architecture for the VLA."""
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    state_dim: int = 14
    action_dim: int = 14
    chunk_length: int = 30
    num_pool_tokens: int = 8
    pool_num_heads: int = 8
    pool_qk_norm: bool = True
    direct_state_conditioning: bool = True


@dataclass
class VLAModelConfig:
    """VLA model architecture.

    Prompt composition (task/subtask/operator) is shared with the DiT policy
    and configured via ``config.prompt`` (see ``PromptConfig``)."""
    camera_keys: tuple[str, ...] = ("top", "left", "right")
    backbone: GemmaVLAConfig = field(default_factory=GemmaVLAConfig)
    dit: VLADiTConfig = field(default_factory=VLADiTConfig)

    @property
    def state_dim(self) -> int:
        return self.dit.state_dim

    @property
    def action_dim(self) -> int:
        return self.dit.action_dim

    @property
    def chunk_length(self) -> int:
        return self.dit.chunk_length


@dataclass
class SimEvalConfig:
    """MuJoCo-Warp sim evaluation, defaulting to the put-bottles task."""
    checkpoint: str
    # "auto" identifies DiT vs VLA from checkpoint tensor keys.
    policy: Literal["auto", "dit", "vla"] = "auto"
    task: str = "put_plastic_bottles_in_bin"  # any abc_sim task name, alias, or prompt
    norm_stats_path: str | None = None
    output_dir: str | None = None  # None resolves to $REPO/outputs/sim_eval_<task>.
    num_worlds: int = 5
    seed: int = 20260511
    num_chunks: int = 236  # x15 actions: the production dashboard horizon; shorter under-reports long tasks
    execute_chunk_dim: int = 15
    prefix_length: int | None = None # ignored when doing rtc
    diffusion_steps: int = 10
    policy_seed: int = 0
    camera_height: int = 168
    camera_width: int = 224
    device: str = "auto"
    gpu_id: int | None = None
    camera_backend: Literal["mjwarp", "mujoco", "blender"] = "mjwarp"  # "mujoco" is the CPU/macOS fallback
    parallel_worlds: int = 0  # >0: step this many worlds together in MJWarp physics; 0: one CPU MuJoCo world at a time
    randomization: str | None = None  # JSON reset request for the task randomizer, applied to every world
    fast_inference: bool = True
    fast_compile_mode: str = "max-autotune"
    vanilla_physics: bool = False
    rtc: bool = True  # condition each inference on the next rtc_prefix_length unexecuted actions (prefix == lead)
    rtc_prefix_length: int = 4
    rtc_inference_lead_steps: int = 4
    log_every_chunk: bool = False
    save_video: bool = False
    video_fps: int = 30
    video_every_n_actions: int = 1
    # None resolves to "sim " + the task's prompt from the abc_sim spec.
    prompt: str | None = None

    clip: ClipConfig = field(default_factory=ClipConfig)
    model: DiTConfig = field(default_factory=DiTConfig)
    vla_model: VLAModelConfig = field(default_factory=VLAModelConfig)


@dataclass
class VizSimEvalConfig(SimEvalConfig):
    """Single-world sim config defaults for the live Viser viewer."""
    checkpoint: str = ""  # Empty resolves the task's cached recommended checkpoint.
    num_chunks: int = 200
    # Synchronous viewer rollouts are unprefixed; RTC uses its own future prefix.
    prefix_length: int | None = 0


@dataclass
class VizPolicyConfig:
    """Live viser viewer over a single ABC DiT or VLA sim rollout."""
    sim: VizSimEvalConfig
    port: int = 8080
    fast_inference: bool = True
    fast_compile_mode: str = "max-autotune"


@dataclass
class VizEpisodeConfig:
    """Viser playback of downloaded dataset episodes (no policy, no torch)."""

    # Play just this one episode directory.
    episode_dir: Path | None = None
    # Episode pool to browse, grouped by task ($ABC_CACHE/train_sim by default).
    root: Path | None = None
    # Start on this task (default: first task found in the pool).
    task: str = ""
    # Viser server port.
    port: int = 8080
    # pose: posed exactly from the recording — the whole scene when the episode
    # ships scene_qpos.npy, else the 14 arm dofs with objects at their start
    # pose. physics: recorded actions stepped open loop from the initial state.
    mode: str = "pose"
    # Playback speed multiplier over the 30 Hz data clock.
    speed: float = 1.0
    # Show the recorded combined camera video beside the 3D scene.
    video_panel: bool = True
    # Per-task episode dropdown cap; a full task pool holds thousands.
    max_episodes: int = 500


def validate_model_config(model: DiTConfig) -> list[str]:
    model_dims = [
        model.hidden_size, model.depth, model.num_heads, model.mlp_ratio,
        model.state_dim, model.action_dim, model.chunk_length, model.task_embed_dim,
        model.vit_embed_dim, model.vit_depth, model.vit_num_heads,
        model.vision_pool_num_queries, model.vision_pool_num_heads,
        model.vision_pool_mlp_ratio,
    ]
    errors = []
    if min(model_dims) <= 0 or not model.camera_keys:
        errors.append("model dimensions and camera_keys must be positive/non-empty")
    if (
        model.hidden_size % model.num_heads
        or model.vit_embed_dim % model.vit_num_heads
        or model.vit_embed_dim % model.vision_pool_num_heads
        or model.hidden_size % 2
        or (model.vit_embed_dim // model.vit_num_heads) % 4
    ):
        errors.append("attention dimensions must be compatible with their head counts")
    return errors


def validate_train_config(
    config: TrainConfig, cache_root: Path, checkpoint_path: Path
) -> list[MixtureComponent]:
    components = config.resolve_mixture()
    weights = [c.weight for c in components]
    errors = []

    if config.load_pretrained and config.resume_from:
        errors.append("--load-pretrained and --resume-from are mutually exclusive")
    if config.fsdp and (config.policy != "vla" or config.compile_siglip):
        errors.append("--fsdp is VLA-only and excludes --compile-siglip")

    if (
        min(config.batch_size, config.train_steps, config.log_every, config.val_every,
            config.val_batches, config.ckpt_every, config.flow.num_diffusion_steps,
            config.flow.num_diffusion_draws) <= 0
        or config.num_workers < 0
        or config.flow.max_action_prefix < 0
    ):
        errors.append(
            "batch size, step intervals, val_batches, and diffusion steps/draws must be "
            "positive; num_workers and max_action_prefix must be non-negative"
        )
    if (
        not 0 <= config.flow.mask_state_ratio <= 1
        or not 0 <= config.flow.prefix_conditioning_prob <= 1
        or config.flow.prefix_noise_scale < 0
    ):
        errors.append(
            "flow probabilities must be in [0, 1] and prefix_noise_scale must be non-negative"
        )
    if config.policy == "vla":
        errors.extend(validate_vla_model_config(config.vla_model))
    else:
        errors.extend(validate_model_config(config.model))
    if (
        not 0 <= config.prompt.subtask_dropout_prob <= 1
        or not 0 <= config.prompt.operator_dropout_prob <= 1
    ):
        errors.append("prompt dropout probabilities must be in [0, 1]")
    for name in ("subtask_append_format", "operator_append_format"):
        try:
            getattr(config.prompt, name).format(prompt="p", subtask="s", operator="o")
        except (KeyError, IndexError, ValueError) as e:
            errors.append(f"prompt.{name} is not renderable: {e!r}")
    if config.prompt.use_operator_id_as_prompt and not config.prompt.operator_label_map_path:
        errors.append(
            "operator prompting requires --prompt.operator-label-map-path; build the "
            "manifest a priori with scripts/build_operator_label_map.py"
        )
    if (
        not components
        or any(not math.isfinite(w) or w <= 0 for w in weights)
        or not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        errors.append(
            f"mixture weights must be positive and sum to 1.0, "
            f"got {sum(weights) if weights else 0:.8g}"
        )

    required = [cache_root / p for c in components for p in (c.train_dir, c.val_dir)]
    if config.load_pretrained:
        required.append(checkpoint_path)
    if config.prompt.use_operator_id_as_prompt and config.prompt.operator_label_map_path:
        required.append(Path(config.prompt.operator_label_map_path).expanduser())
    # norm_stats.json is only required when we are NOT inheriting stats embedded
    # in a checkpoint (pretrained parent or resume checkpoint; see train_loop.main)
    if not (config.inherit_ckpt_norm_stats and (config.load_pretrained or config.resume_from)):
        required.append(cache_root / "norm_stats.json")
    if config.resume_from:
        required.append(Path(config.resume_from).expanduser())
    backbone = config.vla_model.backbone
    if config.policy == "vla" and not (
        backbone.load_base_checkpoint or config.load_pretrained or config.resume_from
    ):
        errors.append(
            "VLA training requires starting weights: enable base checkpoint loading "
            "and set --vla-model.backbone.checkpoint, or use --load-pretrained "
            "or --resume-from; random backbone initialization is not supported"
        )
    if config.policy == "vla" and backbone.load_base_checkpoint and not (
        config.resume_from or config.load_pretrained
    ):
        if backbone.checkpoint:
            required.append(Path(backbone.checkpoint).expanduser())
        else:
            errors.append("vla_model.backbone.checkpoint is required when loading base weights")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        errors.append("missing required paths: " + ", ".join(missing))

    if errors:
        raise ValueError("Invalid training config:\n  - " + "\n  - ".join(errors))
    return components


def validate_vla_model_config(model: VLAModelConfig) -> list[str]:
    """Return a list of VLA model-config errors (empty if valid)."""
    backbone = model.backbone
    dit = model.dit
    errors = []
    dims = (
        dit.hidden_size,
        dit.depth,
        dit.num_heads,
        dit.mlp_ratio,
        dit.state_dim,
        dit.action_dim,
        dit.chunk_length,
        dit.num_pool_tokens,
        dit.pool_num_heads,
        backbone.image_size,
        backbone.fixed_seq_len,
    )
    if min(dims) <= 0 or not model.camera_keys:
        errors.append("VLA dimensions and camera_keys must be positive/non-empty")
    if (
        dit.hidden_size % dit.num_heads
        or dit.hidden_size % dit.pool_num_heads
        or dit.hidden_size % 2
    ):
        errors.append("VLA hidden_size must be divisible by DiT and pool head counts")
    if backbone.image_size % 14:
        errors.append("Gemma SigLIP image_size must be divisible by patch size 14")
    if not -34 <= backbone.feature_layer < 34:
        errors.append("feature_layer must resolve to one of the 34 Gemma 3 4B blocks")
    return errors


_MISSING = object()


def _nested_value(mapping: Mapping, *path: str):
    value = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _vla_model_values(model: VLAModelConfig) -> dict[str, object]:
    """Architecture values whose mismatch can change checkpoint behavior."""
    return {
        "camera_keys": tuple(model.camera_keys),
        "backbone.image_size": model.backbone.image_size,
        "backbone.fixed_seq_len": model.backbone.fixed_seq_len,
        # -1 and 33 name the same layer in the 34-block Gemma 3 4B model.
        "backbone.feature_layer": model.backbone.feature_layer % 34,
        "backbone.proprio_token": model.backbone.proprio_token,
        "dit.hidden_size": model.dit.hidden_size,
        "dit.depth": model.dit.depth,
        "dit.num_heads": model.dit.num_heads,
        "dit.mlp_ratio": model.dit.mlp_ratio,
        "dit.state_dim": model.dit.state_dim,
        "dit.action_dim": model.dit.action_dim,
        "dit.chunk_length": model.dit.chunk_length,
        "dit.num_pool_tokens": model.dit.num_pool_tokens,
        "dit.pool_num_heads": model.dit.pool_num_heads,
        "dit.pool_qk_norm": model.dit.pool_qk_norm,
        "dit.direct_state_conditioning": model.dit.direct_state_conditioning,
    }


def _nested_vla_checkpoint_values(
    raw: Mapping, names: Iterable[str]
) -> tuple[dict[str, object], list[str]]:
    values = {}
    missing = []
    for name in names:
        value = _nested_value(raw, *name.split("."))
        if value is _MISSING:
            missing.append(name)
        else:
            values[name] = value
    return values, missing


def _legacy_vla_checkpoint_values(raw: Mapping) -> tuple[dict[str, object], list[str]]:
    """Translate the original ABC VLA's flat training config to local names."""
    mappings = {
        "backbone.image_size": "siglip_image_size",
        "backbone.fixed_seq_len": "max_seq_len",
        "dit.hidden_size": "diffusion_hidden_size",
        "dit.depth": "diffusion_depth",
        "dit.num_heads": "diffusion_num_heads",
        "dit.mlp_ratio": "diffusion_mlp_ratio",
        "dit.state_dim": "state_dim",
        "dit.action_dim": "action_dim",
        "dit.chunk_length": "chunk_length",
        "dit.num_pool_tokens": "num_obs_pool_tokens",
        "dit.pool_qk_norm": "obs_pool_qk_norm",
        "dit.direct_state_conditioning": "direct_state_conditioning",
    }
    values = {
        # Camera ordering and pool heads were constants in the released model.
        "camera_keys": tuple(raw.get("camera_keys", ("top", "left", "right"))),
        "dit.pool_num_heads": raw.get(
            "obs_pool_num_heads", raw.get("diffusion_num_heads", 8)
        ),
    }
    missing = []
    for local_name, legacy_name in mappings.items():
        if legacy_name not in raw:
            missing.append(local_name)
        else:
            values[local_name] = raw[legacy_name]

    layers = raw.get("obs_encoding_layers", _MISSING)
    if isinstance(layers, (tuple, list)) and len(layers) == 1:
        values["backbone.feature_layer"] = layers[0]
    else:
        missing.append("backbone.feature_layer (single obs_encoding_layers entry)")
    if "exclude_state_from_vlm" in raw:
        values["backbone.proprio_token"] = not raw["exclude_state_from_vlm"]
    else:
        missing.append("backbone.proprio_token (exclude_state_from_vlm)")
    return values, missing


def validate_vla_checkpoint_config(
    model: VLAModelConfig, checkpoint: Mapping, *, source: str = "checkpoint"
) -> bool:
    """Fail if checkpoint metadata disagrees with the CLI VLA architecture.

    The CLI remains authoritative, matching the DiT path.  This check only
    catches semantic, shape-compatible mistakes (for example camera order or
    feature layer) before strict state-dict loading catches tensor-shape
    mistakes.  It reads both the current nested metadata and the original
    released VLA's flat ``config`` metadata.  Returns ``False`` only when no
    architecture metadata is present.
    """
    if not isinstance(checkpoint, Mapping):
        return False

    raw = checkpoint.get("model_config")
    metadata_name = "model_config"
    if not isinstance(raw, Mapping):
        train_config = checkpoint.get("train_config")
        candidate = (
            train_config.get("vla_model") if isinstance(train_config, Mapping) else None
        )
        if isinstance(candidate, Mapping):
            raw = candidate
            metadata_name = "train_config.vla_model"

    legacy = False
    if isinstance(raw, Mapping):
        saved, missing = _nested_vla_checkpoint_values(raw, _vla_model_values(model))
    else:
        raw = checkpoint.get("config")
        if not isinstance(raw, Mapping) or not any(
            key in raw for key in ("diffusion_hidden_size", "gemma_variant")
        ):
            return False
        legacy = True
        metadata_name = "config"
        saved, missing = _legacy_vla_checkpoint_values(raw)

    expected = _vla_model_values(model)
    if "camera_keys" in saved:
        saved["camera_keys"] = tuple(saved["camera_keys"])
    if "backbone.feature_layer" in saved:
        feature_layer = saved["backbone.feature_layer"]
        if isinstance(feature_layer, int):
            saved["backbone.feature_layer"] = feature_layer % 34

    problems = [f"{name}: missing from {metadata_name}" for name in missing]
    for name, cli_value in expected.items():
        if name in saved and saved[name] != cli_value:
            problems.append(
                f"{name}: checkpoint={saved[name]!r}, CLI={cli_value!r}"
            )

    if legacy:
        unsupported = {
            "gemma_variant": "4b",
            "keep_original_resolution": False,
            "obs_encoding_use_layer_mix": False,
            "use_cross_attn_conditioning": False,
            "num_register_tokens": 0,
            "use_backbone_kv": False,
            "mode": "diffusion_only",
        }
        for name, supported in unsupported.items():
            if name in raw and raw[name] != supported:
                problems.append(
                    f"{name}: checkpoint={raw[name]!r}, supported={supported!r}"
                )

    if problems:
        raise ValueError(
            f"{source} VLA architecture metadata is incompatible with the CLI config:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )
    return True


