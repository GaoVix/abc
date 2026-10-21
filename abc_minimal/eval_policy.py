"""Run MuJoCo-Warp sim eval for ABC-DiT checkpoints.

Builds the scene, executes policy rollouts, and writes JSON/video outputs.
Every task runs through the abc_sim catalogue via abc_minimal/sim_env.py.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from abc_minimal.config import (
    SimEvalConfig,
    validate_model_config,
    validate_vla_model_config,
)
from abc_minimal.policy import (
    DiTInferencePolicy,
    VLAInferencePolicy,
)
from abc_minimal.policy import DiTInferencePolicy as SimPolicy
from abc_sim.randomization.core import RandomizationSamplingError
from deploy.policy.selector import sniff_policy_kind

if TYPE_CHECKING:
    from abc_minimal.sim_env import SimTaskEnv

torch.set_float32_matmul_precision("high")


# Config.

ROOT = Path(__file__).resolve().parents[1]


def require_mjwarp() -> None:
    try:
        import mujoco_warp  # noqa: F401
        import warp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "eval_policy.py uses MJWarp only. Install/import `mujoco_warp` and `warp` "
            "in the eval environment before running sim eval."
        ) from exc


class RTCManager:
    def __init__(
        self,
        policy: SimPolicy,
        *,
        prefix_length: int,
        inference_lead_steps: int,
        execute_chunk_dim: int,
    ):
        self.policy = policy
        self.prefix_length = prefix_length
        self.inference_lead_steps = inference_lead_steps
        self.execute_chunk_dim = execute_chunk_dim
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._pending: concurrent.futures.Future[tuple[np.ndarray, float]] | None = None

    def _action_prefix(self, actions: np.ndarray) -> np.ndarray:
        executed = np.asarray(actions[..., : self.execute_chunk_dim, :], dtype=np.float32)
        prefix = np.zeros(
            (*executed.shape[:-2], self.policy.chunk_length, self.policy.action_dim),
            dtype=np.float32,
        )
        prefix[..., : self.prefix_length, :] = executed[..., -self.prefix_length :, :]
        return prefix

    def start(
        self,
        obs: dict[str, Any],
        current_actions: np.ndarray,
        noise: np.ndarray | None,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("RTC inference is already pending")
        action_prefix = self._action_prefix(current_actions)

        def _run() -> tuple[np.ndarray, float]:
            t0 = time.perf_counter()
            actions = self.policy.infer(
                obs,
                noise=noise,
                action_prefix=action_prefix,
                prefix_length=self.prefix_length,
            )
            return actions[..., self.prefix_length :, :], time.perf_counter() - t0

        self._pending = self._executor.submit(_run)

    def ready(self) -> bool:
        return self._pending is not None and self._pending.done()

    def get(self) -> tuple[np.ndarray, float, bool]:
        if self._pending is None:
            raise RuntimeError("No RTC inference is pending")
        ready = self._pending.done()
        actions, infer_s = self._pending.result()
        self._pending = None
        return actions, infer_s, ready

    def close(self) -> None:
        if self._pending is not None:
            self._pending.result()
            self._pending = None
        self._executor.shutdown(wait=True)


# Rollout.


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if hasattr(x, "__dataclass_fields__"):
        return jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


# Chunk-metric fields that only exist for a task whose evaluator counts bottles.
CHUNK_COUNT_KEYS = ("bottles", "max_bottles")


def without_missing_counts(metric: dict[str, Any]) -> dict[str, Any]:
    """A chunk metric without the bottle counters the task does not have.

    build_summary leaves ``mean_max_bottles_in_bin`` out entirely rather than
    reporting null for a task with no such evaluator, and the per-chunk counters
    follow it: a consumer can ask ``"bottles" in metric`` instead of having to
    know that the key is present but null on every other task.
    """
    return {
        key: value
        for key, value in metric.items()
        if value is not None or key not in CHUNK_COUNT_KEYS
    }


def resolve_prefix_length(
    config: SimEvalConfig, trained_max_prefix: int, model_config: Any
) -> int:
    """Effective synchronous prefix length for the sequential loop (0 = off).

    Needs the checkpoint's trained max, so this runs after loading rather than
    in the static config validation.
    """
    # max_action_prefix is an exclusive sampling bound (randint(0, max)), so
    # the longest prefix training ever produced is max - 1.
    longest_trained = max(0, trained_max_prefix - 1)
    if config.rtc:
        # RTC builds its own prefixes; cap them to the checkpoint's trained
        # range here for the same post-load reason (a checkpoint trained with
        # a short max_action_prefix would otherwise reject the default).
        if config.rtc_prefix_length > longest_trained:
            capped = max(0, longest_trained)
            print(
                f"[prefix] rtc_prefix_length {config.rtc_prefix_length} -> "
                f"{capped} (checkpoint trained max_action_prefix="
                f"{trained_max_prefix}, exclusive bound)"
            )
            config.rtc_prefix_length = capped
        # Each RTC re-inference yields chunk_length - rtc_prefix_length actions
        # (RTCManager._run strips the prefix); the loops execute execute_chunk_dim of them.
        if config.rtc_prefix_length + config.execute_chunk_dim > model_config.chunk_length:
            raise ValueError(
                f"rtc_prefix_length + execute_chunk_dim "
                f"({config.rtc_prefix_length} + {config.execute_chunk_dim}) "
                f"must be <= chunk_length ({model_config.chunk_length})"
            )
        return 0
    length = config.prefix_length
    if length is None:
        # Unprefixed is the synchronous default: the sync loop's prefix feeds
        # already-executed actions into a committed-future interface, which
        # correctly-labeled (v3) checkpoints take literally. Pass an explicit
        # --prefix-length (production used 5) to reproduce historical
        # synchronous numbers.
        length = 0
    if length == 0:
        return 0
    errors = []
    if length < 0:
        errors.append(f"prefix_length must be >= 0, got {length}")
    if length > longest_trained:
        errors.append(
            f"prefix_length ({length}) exceeds the longest trained prefix "
            f"({longest_trained}; the checkpoint's max_action_prefix bound of "
            f"{trained_max_prefix} is exclusive)"
        )
    if length > config.execute_chunk_dim:
        errors.append(
            f"prefix_length ({length}) must be <= execute_chunk_dim "
            f"({config.execute_chunk_dim}) so executed actions can seed the next prefix"
        )
    if length + config.execute_chunk_dim > model_config.chunk_length:
        errors.append(
            f"prefix_length + execute_chunk_dim ({length} + {config.execute_chunk_dim}) "
            f"must be <= chunk_length ({model_config.chunk_length})"
        )
    if errors:
        raise ValueError("Invalid prefix config:\n  - " + "\n  - ".join(errors))
    return length


def rollout_over(task_eval: dict[str, Any]) -> bool:
    """Whether a rollout has nothing left to do and can stop before its budget.

    For most tasks that is the first success: once the bottles are in the bin,
    stepping on only burns wall clock. A maintenance task is the other way round
    -- ball_tray_balancing reports ``ever_failed`` and defines ``ever_success``
    as "has not dropped the ball yet", so it is already succeeding at reset and
    is over only once it fails. Stopping such a task on ``ever_success`` ends it
    on step one and scores a 15-second balance on a single simulator step, so
    ``ever_failed`` is the signal when the evaluator reports one.
    """
    if "ever_failed" in task_eval:
        return bool(task_eval["ever_failed"])
    return bool(task_eval["ever_success"])


def progress_text(task_eval: dict[str, Any], bottles_key: str) -> str:
    """Progress for one rollout: the bottles counter, or reward for other tasks."""
    count = task_eval.get(bottles_key)
    if count is None:
        return f"reward={float(task_eval.get('reward', 0.0)):.2f}"
    total = task_eval.get("num_active_bottles", task_eval.get("success_count"))
    return f"bottles={count}/{total}"


def video_frame(images: dict[str, np.ndarray], camera_keys: tuple[str, ...]) -> np.ndarray:
    frames = []
    for name in camera_keys:
        frame = images[name].transpose(1, 2, 0)
        frames.append(np.ascontiguousarray(frame))
    return np.concatenate(frames, axis=1)


def resolve_device(device: str, gpu_id: int | None = None) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # The renderer pins itself to cuda:{gpu_id}; a bare "cuda"
    # policy device means cuda:0, so --gpu-id N != 0 split policy and renderer
    # across devices and crashed mid-rollout. Pin the policy alongside.
    if gpu_id is not None and device == "cuda":
        return f"cuda:{gpu_id}"
    return device


def validate_rtc_config(config: SimEvalConfig) -> list[str]:
    if not config.rtc:
        return []
    errors = []
    # The upper bound is the checkpoint's trained prefix range, which is
    # only known after loading; resolve_prefix_length enforces it there.
    if config.rtc_prefix_length < 1:
        errors.append("rtc_prefix_length must be >= 1")
    if config.prefix_length:
        errors.append(
            "--prefix-length is ignored under --rtc, which builds its own "
            "prefixes; pass --prefix-length 0 or drop one of the flags"
        )
    if config.rtc_prefix_length != config.rtc_inference_lead_steps:
        errors.append("rtc_prefix_length must equal rtc_inference_lead_steps")
    if config.rtc_inference_lead_steps > config.execute_chunk_dim:
        errors.append("rtc_inference_lead_steps must be <= execute_chunk_dim")
    if config.rtc and config.save_video:
        # Rollouts are unaffected (rtc.get() joins blockingly; the executed
        # action sequence never depends on wall time), but per-action rendering
        # inflates steps_s, so the overlap telemetry in chunk_metrics
        # (infer_s vs steps_s, rtc_ready) stops being meaningful.
        print(
            "[warn] --save-video under --rtc: scores are unaffected, but the "
            "chunk timing/overlap telemetry includes render time"
        )
    return errors


def validate_batched_config(config: SimEvalConfig) -> list[str]:
    if config.parallel_worlds < 0:
        return ["parallel_worlds must be >= 0"]
    if config.parallel_worlds == 0:
        return []
    errors = []
    if config.num_worlds % config.parallel_worlds:
        errors.append(
            f"num_worlds ({config.num_worlds}) must be a multiple of "
            f"parallel_worlds ({config.parallel_worlds})"
        )
    if config.camera_backend != "mjwarp":
        errors.append("--parallel-worlds renders with MJWarp; drop --camera-backend mujoco")
    if config.vanilla_physics:
        errors.append("--parallel-worlds steps MJWarp physics; drop --vanilla-physics")
    return errors


def reset_options(config: SimEvalConfig) -> dict[str, Any] | None:
    """Env reset options carrying the --randomization request, or None."""
    if config.randomization is None:
        return None
    return {"randomization": json.loads(config.randomization)}


def checkpoint_sim_prompt(config: SimEvalConfig) -> str | None:
    """The prompt this checkpoint trained ``config.task`` under, or None.

    Published checkpoints ship a ``<stem>.json`` metadata sidecar whose
    ``sim_prompt_map`` records, per task, the prompt its episodes actually
    carried in the training mixture. That is not always the prompt the abc_sim
    spec generates: the mixtures remap their sim task names, and prompts are
    derived from the name after the remap. Prompting a flow policy
    off-distribution costs success silently, with no error to trace it to, so
    the map is read back rather than left to the docs.
    """
    sidecar = Path(config.checkpoint).with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        prompt_map = json.loads(sidecar.read_text())["sim_prompt_map"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"[prompt] ignoring unusable sidecar {sidecar.name}: {exc!r}")
        return None
    from abc_sim import maybe_get_task_spec

    spec = maybe_get_task_spec(config.task)
    return prompt_map.get(spec.name) if spec else None


def resolve_prompt(config: SimEvalConfig) -> str:
    """Prompt to condition on: --prompt if given, else the prompt the checkpoint
    trained this task under, else the task's sim prompt."""
    if config.prompt is not None:
        return config.prompt
    # Deliberately ahead of the task-spec fallback: the 200k policy's sidecar
    # maps put_plastic_bottles_in_bin to the throw prompt it actually trained
    # that scene under, and corrects it here. The throw scene (--task bottles)
    # is absent from that map, having been left out of the mixture, so it falls
    # through to its own spec prompt, already correct.
    trained = checkpoint_sim_prompt(config)
    if trained is not None:
        print(f"[prompt] {trained!r} (the prompt this checkpoint trained the task under)")
        return trained
    from abc_minimal.sim_env import task_prompt

    return task_prompt(config.task)


def resolve_output_dir(config: SimEvalConfig) -> str:
    """Output directory: --output-dir if given, else one named after the task."""
    if config.output_dir is not None:
        return config.output_dir
    return str(ROOT / "outputs" / f"sim_eval_{config.task}")


def _make_env(config: SimEvalConfig, camera_keys: tuple[str, ...]) -> SimTaskEnv:
    """Build the rollout env for the configured task (abc_sim catalogue)."""
    from abc_minimal.sim_env import SimTaskEnv

    return SimTaskEnv(
        task=config.task,
        height=config.camera_height,
        width=config.camera_width,
        camera_keys=camera_keys,
        prompt=config.prompt,
        camera_backend=config.camera_backend,
        gpu_id=config.gpu_id,
    )


def resolved_physics(env: Any) -> dict[str, Any] | None:
    """The physics the env actually steps with, read off the live objects.

    ``summary["config"]`` echoes the *requested* physics, but scene tasks can
    override physics_dt/control_decimation at env construction
    (task_registry physics_defaults) — the 181.8 Hz pour eval bug shipped
    summaries claiming 29.4 Hz.
    """
    for obj in (env, getattr(env, "env", None)):
        if obj is None:
            continue
        dec = getattr(obj, "control_decimation", None)
        if dec is None:
            dec = getattr(obj, "_control_decimation", None)
        model = getattr(obj, "model", None)
        dt = getattr(getattr(model, "opt", None), "timestep", None)
        if dt is None:
            dt = getattr(obj, "_physics_dt", None)
        if dec is not None and dt is not None:
            dt = float(dt)
            dec = int(dec)
            return {
                "physics_dt": dt,
                "control_decimation": dec,
                "control_hz": 1.0 / (dt * dec),
            }
    return None


def build_summary(
    *,
    config: SimEvalConfig,
    ckpt_path: Path,
    device: str,
    worlds: list[dict[str, Any]],
    out_dir: Path,
    physics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-world records, write summary.json, and print the tail lines."""
    success = np.asarray([w["success"] for w in worlds], dtype=bool)
    rewards = np.asarray([w["reward"] for w in worlds], dtype=np.float32)
    per_world_max_bottles = [
        w["final_task_eval"].get("max_bottles_in_bin_so_far") for w in worlds
    ]
    has_bottles_metric = all(value is not None for value in per_world_max_bottles)
    per_world_max_reward = [w.get("max_reward") for w in worlds]
    has_max_reward = all(value is not None for value in per_world_max_reward)
    summary = {
        "format": "abc_minimal_sim_eval/v1",
        "checkpoint": str(ckpt_path),
        "prompt": config.prompt,
        "config": asdict(config),
        "resolved_device": device,
        "success_rate": float(success.mean()) if success.size else None,
        "num_success": int(success.sum()),
        "num_worlds": len(worlds),
        "mean_reward": float(rewards.mean()) if rewards.size else None,
        "worlds": worlds,
    }
    summary["task"] = config.task
    if physics is not None:
        summary["resolved_physics"] = physics
    if has_max_reward:
        # Mean over worlds of the best instantaneous progress fraction — the
        # statistic the production dishrack eval (and the paper's "sim
        # progress" plots) aggregated. mean_reward stays the final-step value.
        max_rewards = np.asarray(per_world_max_reward, dtype=np.float32)
        summary["mean_max_progress"] = (
            float(max_rewards.mean()) if max_rewards.size else None
        )
    if has_bottles_metric:
        max_bottles = np.asarray(per_world_max_bottles, dtype=np.float32)
        summary["mean_max_bottles_in_bin"] = (
            float(max_bottles.mean()) if max_bottles.size else None
        )
    (out_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    progress_text_part = (
        f" mean_max_progress={summary['mean_max_progress']}" if has_max_reward else ""
    )
    bottles_text = (
        f" mean_max_bottles={summary['mean_max_bottles_in_bin']}"
        if has_bottles_metric
        else ""
    )
    print(
        f"summary: success_rate={summary['success_rate']} "
        f"num_success={summary['num_success']}/{summary['num_worlds']} "
        f"mean_reward={summary['mean_reward']}{progress_text_part}{bottles_text}",
        flush=True,
    )
    print(f"wrote {out_dir / 'summary.json'}", flush=True)
    return summary


def run_eval(config: SimEvalConfig) -> dict[str, Any]:
    ckpt_path = Path(config.checkpoint).expanduser().resolve()
    policy_kind = (
        sniff_policy_kind(str(ckpt_path))
        if config.policy == "auto"
        else config.policy
    )
    model_config = config.vla_model if policy_kind == "vla" else config.model
    model_errors = (
        validate_vla_model_config(config.vla_model)
        if policy_kind == "vla"
        else validate_model_config(config.model)
    )
    config_errors = (
        model_errors + validate_rtc_config(config) + validate_batched_config(config)
    )
    if config_errors:
        raise ValueError("Invalid sim eval config:\n  - " + "\n  - ".join(config_errors))

    require_mjwarp()
    options = reset_options(config)
    config = replace(
        config,
        policy=policy_kind,
        prompt=resolve_prompt(config),
        output_dir=resolve_output_dir(config),
    )
    device = resolve_device(config.device, config.gpu_id)
    policy_cls = VLAInferencePolicy if policy_kind == "vla" else DiTInferencePolicy
    policy = policy_cls(ckpt_path, config, device, model_config=model_config)
    prefix_length = resolve_prefix_length(
        config, policy.trained_max_prefix, model_config
    )
    config = replace(config, prefix_length=prefix_length)
    prefix_text = (
        f"conditioning on the last {prefix_length} executed actions"
        if prefix_length
        else "unprefixed (off-distribution for prefix-trained checkpoints)"
    )
    print(
        f"prefix conditioning: {prefix_text} "
        f"(checkpoint max_action_prefix={policy.trained_max_prefix})",
        flush=True,
    )
    out_dir = Path(config.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.parallel_worlds:
        from abc_minimal.batched_eval import make_batched_env, rollout_batched_worlds

        env = make_batched_env(config)
        rollout = rollout_batched_worlds
    else:
        env = _make_env(config, tuple(model_config.camera_keys))
        rollout = rollout_worlds
    physics = resolved_physics(env)
    if physics is not None:
        print(
            f"[physics] dt={physics['physics_dt']} "
            f"decimation={physics['control_decimation']} "
            f"control={physics['control_hz']:.2f}Hz",
            flush=True,
        )
    worlds = rollout(config, policy, env, prefix_length, options, out_dir, model_config)
    return build_summary(
        config=config,
        ckpt_path=ckpt_path,
        device=device,
        worlds=worlds,
        out_dir=out_dir,
        physics=physics,
    )


def rollout_worlds(
    config: SimEvalConfig,
    policy: SimPolicy,
    env: SimTaskEnv,
    prefix_length: int,
    options: dict[str, Any] | None,
    out_dir: Path,
    model_config: Any,
) -> list[dict[str, Any]]:
    """Roll out the worlds one at a time in CPU MuJoCo; returns their records."""
    rng = np.random.default_rng(config.policy_seed)
    worlds = []
    fast_inference_ready = False
    rtc_warmup_ready = False
    action_shape = (model_config.chunk_length, model_config.action_dim)

    def sample_noise(generator: np.random.Generator) -> np.ndarray:
        return generator.standard_normal(action_shape, dtype=np.float32)

    try:
        for world_index in range(config.num_worlds):
            video = None
            video_path = None
            t0 = time.perf_counter()
            seed = int(config.seed + world_index)
            try:
                obs = env.reset(seed=seed, options=options)
            except RandomizationSamplingError:
                # The carried arm pose from the previous episode can block a
                # seed that places fine from init_q; retry from init_q before
                # resampling so the run keeps its world count.
                env.forget_arm_state()
                try:
                    obs = env.reset(seed=seed, options=options)
                    print(f"world={world_index:03d} unplaceable from the carried "
                          "arm pose, placed from init_q")
                except RandomizationSamplingError:
                    seed = int(config.seed + world_index + 100_000)
                    print(f"world={world_index:03d} unplaceable seed, resampled -> {seed}")
                    obs = env.reset(seed=seed, options=options)
            if config.fast_inference and not fast_inference_ready:
                warmup_rng = np.random.default_rng(config.policy_seed)
                warmup_noise = sample_noise(warmup_rng)
                t_fast = time.perf_counter()
                policy.enable_fast_inference(
                    compile_mode=config.fast_compile_mode,
                    warmup_obs=obs,
                    warmup_noise=warmup_noise,
                )
                torch.cuda.synchronize()
                print(
                    f"fast inference ready in {time.perf_counter() - t_fast:.1f}s",
                    flush=True,
                )
                if prefix_length:
                    # Same prefix-conditioned graph RTC uses, replayed
                    # synchronously here. The plain graph captured above will
                    # never be replayed on this run (every infer carries a
                    # prefix), but warmup_rtc requires it as its capture gate.
                    t_prefix = time.perf_counter()
                    policy.warmup_rtc(obs, warmup_noise, prefix_length)
                    print(
                        f"prefix-conditioned inference (length {prefix_length}) "
                        f"ready in {time.perf_counter() - t_prefix:.1f}s",
                        flush=True,
                    )
                fast_inference_ready = True
            if config.save_video:
                import imageio.v2 as imageio

                video_path = out_dir / f"world_{world_index:03d}.mp4"
                video = imageio.get_writer(str(video_path), fps=config.video_fps, macro_block_size=1)
                video.append_data(video_frame(obs["images"], model_config.camera_keys))
            final_eval = env.evaluate_vanilla() if config.vanilla_physics else env.evaluate()
            # Best instantaneous progress fraction over the episode; the
            # production dishrack eval aggregated this, not the final state.
            max_reward = float(final_eval.get("reward", 0.0))
            steps = 0
            chunk_metrics = []
            rtc = None
            try:
                obs_fn = env.obs_vanilla_state if config.vanilla_physics else env.obs
                eval_fn = env.evaluate_vanilla if config.vanilla_physics else env.evaluate
                step_fn = env.step_one_vanilla if config.vanilla_physics else env.step_one
                render_fn = (
                    env.render_cameras_vanilla_state
                    if config.vanilla_physics
                    else env.render_cameras
                )
                action_prefix = None
                if prefix_length:
                    # No actions executed yet: condition the first chunk on the
                    # current state tiled, the production harness convention.
                    action_prefix = np.tile(
                        np.asarray(obs["state"], dtype=np.float32)[None, :],
                        (prefix_length, 1),
                    )
                noise = sample_noise(rng)
                t_infer = time.perf_counter()
                actions = policy.infer(
                    obs,
                    noise=noise,
                    action_prefix=action_prefix,
                    prefix_length=prefix_length,
                )
                current_infer_s = time.perf_counter() - t_infer
                if config.rtc:
                    if not rtc_warmup_ready:
                        rtc_warmup_rng = np.random.default_rng(config.policy_seed)
                        warmup_noise = sample_noise(rtc_warmup_rng)
                        t_rtc_warm = time.perf_counter()
                        policy.warmup_rtc(obs, warmup_noise, config.rtc_prefix_length)
                        print(
                            f"rtc inference ready in {time.perf_counter() - t_rtc_warm:.1f}s",
                            flush=True,
                        )
                        rtc_warmup_ready = True
                    rtc = RTCManager(
                        policy,
                        prefix_length=config.rtc_prefix_length,
                        inference_lead_steps=config.rtc_inference_lead_steps,
                        execute_chunk_dim=config.execute_chunk_dim,
                    )
                for chunk in range(config.num_chunks):
                    t_chunk = time.perf_counter()
                    chunk_infer_s = current_infer_s
                    t_steps = time.perf_counter()
                    rtc_started = False
                    rtc_ready = None
                    rtc_infer_s = None
                    rtc_obs_s = 0.0
                    lead_index = config.execute_chunk_dim - config.rtc_inference_lead_steps
                    # The first prefix_length positions of a prefixed chunk
                    # reconstruct actions that were already executed; skip them.
                    executed = actions[
                        prefix_length : prefix_length + config.execute_chunk_dim
                    ]
                    for action_index, action in enumerate(executed):
                        if (
                            rtc is not None
                            and chunk + 1 < config.num_chunks
                            and action_index == lead_index
                        ):
                            t_obs = time.perf_counter()
                            next_obs = obs_fn()
                            rtc_obs_s = time.perf_counter() - t_obs
                            next_noise = sample_noise(rng)
                            rtc.start(next_obs, actions, next_noise)
                            rtc_started = True
                        step_fn(action)
                        final_eval = eval_fn()
                        max_reward = max(max_reward, float(final_eval.get("reward", 0.0)))
                        steps += 1
                        if video is not None and steps % config.video_every_n_actions == 0:
                            video.append_data(video_frame(render_fn(), model_config.camera_keys))
                        if rollout_over(final_eval):
                            break
                    steps_s = time.perf_counter() - t_steps
                    if rollout_over(final_eval):
                        break
                    if rtc is None and chunk + 1 < config.num_chunks:
                        t_obs = time.perf_counter()
                        obs = obs_fn()
                        rtc_obs_s = time.perf_counter() - t_obs
                        if prefix_length:
                            action_prefix = np.asarray(
                                executed[-prefix_length:], dtype=np.float32
                            )
                        noise = sample_noise(rng)
                        t_infer = time.perf_counter()
                        actions = policy.infer(
                            obs,
                            noise=noise,
                            action_prefix=action_prefix,
                            prefix_length=prefix_length,
                        )
                        current_infer_s = time.perf_counter() - t_infer
                    elif rtc_started:
                        actions, rtc_infer_s, rtc_ready = rtc.get()
                        current_infer_s = rtc_infer_s
                    logged_infer_s = rtc_infer_s if rtc_infer_s is not None else chunk_infer_s
                    metric = {
                        "chunk": chunk,
                        "infer_s": float(logged_infer_s),
                        "current_chunk_infer_s": float(chunk_infer_s),
                        "rtc_next_infer_s": (
                            float(rtc_infer_s) if rtc_infer_s is not None else None
                        ),
                        "steps_s": float(steps_s),
                        "obs_render_s": float(rtc_obs_s),
                        "wall_s": float(time.perf_counter() - t_chunk),
                        "rtc_ready_at_chunk_end": rtc_ready,
                        "reward": float(final_eval.get("reward", 0.0)),
                        "bottles": _optional_int(final_eval.get("num_bottles_in_bin")),
                        "max_bottles": _optional_int(
                            final_eval.get("max_bottles_in_bin_so_far")
                        ),
                    }
                    chunk_metrics.append(without_missing_counts(metric))
                    if config.log_every_chunk:
                        rtc_text = (
                            f" rtc_ready_at_chunk_end={rtc_ready}"
                            if config.rtc
                            else ""
                        )
                        print(
                            f"world={world_index:03d} chunk={chunk:02d} "
                            f"infer={metric['infer_s'] * 1000:.0f}ms "
                            f"steps={metric['steps_s'] * 1000:.0f}ms "
                            f"render={metric['obs_render_s'] * 1000:.0f}ms "
                            f"{progress_text(final_eval, 'num_bottles_in_bin')} "
                            f"done={rollout_over(final_eval)}{rtc_text}",
                            flush=True,
                        )
            finally:
                if rtc is not None:
                    rtc.close()
                if video is not None:
                    video.close()

            world = {
                "world_index": world_index,
                "world_seed": seed,
                "success": bool(final_eval["ever_success"]),
                "final_success": bool(final_eval["success"]),
                "reward": float(final_eval["reward"]),
                "max_reward": float(max_reward),
                "steps": steps,
                "wall_s": time.perf_counter() - t0,
                "chunk_metrics": chunk_metrics,
                "randomization": env.randomization,
                "final_task_eval": final_eval,
                "video_path": str(video_path) if video_path is not None else None,
            }
            worlds.append(world)
            print(
                f"world={world_index:03d} done success={world['success']} "
                f"{progress_text(final_eval, 'max_bottles_in_bin_so_far')} "
                f"steps={steps}",
                flush=True,
            )
    finally:
        env.close()
    return worlds


def main(config: SimEvalConfig) -> None:
    run_eval(config)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(SimEvalConfig))
