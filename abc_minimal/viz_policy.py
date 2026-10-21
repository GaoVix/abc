"""Live Viser viewer for an ABC DiT or VLA sim rollout.

Visualise a policy on any task from the vendored abc_sim catalogue
(``--sim.task``); every task streams through ``run_sim_task_viewer``.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import viser
from mjviser import ViserMujocoScene

from abc_minimal.config import (
    VizPolicyConfig,
    VizSimEvalConfig,
    default_cache_root,
    validate_model_config,
    validate_vla_model_config,
)
from abc_minimal.eval_policy import (
    RTCManager,
    checkpoint_sim_prompt,
    progress_text,
    require_mjwarp,
    resolve_device,
    resolve_prefix_length,
    validate_rtc_config,
)
from abc_minimal.policy import (
    DiTInferencePolicy,
    VLAInferencePolicy,
)
from deploy.policy.selector import sniff_policy_kind

# An unedited --sim.prompt default means "resolve from the task"; reading it
# off the class keeps this true even if the default is ever re-pinned.
VIZ_PROMPT_DEFAULT = VizSimEvalConfig.prompt
FINETUNED_MANIFEST_NAME = "finetuned_sim_manifest.json"


def resolve_viz_prompt(
    task: str, prompt: str | None, release_prompt: str | None = None
) -> str:
    """Prompt the viewer conditions the policy on.

    A ``--sim.prompt`` the user actually chose always wins. An unedited default
    is treated as unset and resolved from the task, the same prompt
    ``eval_policy.resolve_prompt`` would pick, so watching a task in the viewer
    and evaluating it in ``eval_policy.py`` condition on the same text.
    """
    if prompt is not None and prompt != VIZ_PROMPT_DEFAULT:
        return prompt
    if release_prompt:
        return release_prompt
    from abc_minimal.sim_env import task_prompt

    return task_prompt(task)


def resolve_release_checkpoint(task: str) -> tuple[Path, str | None]:
    """Resolve a sim task to its downloaded recommended release checkpoint."""
    cache = default_cache_root().expanduser().resolve()
    manifest_path = cache / FINETUNED_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No finetuned checkpoint catalogue found at {manifest_path}. "
            f"Run: uv run prepare.py --sim-bundle {task}"
        )
    try:
        tasks = json.loads(manifest_path.read_text())["tasks"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid checkpoint catalogue {manifest_path}: {exc}") from exc

    candidates = [task]
    try:
        from abc_sim.task_specs import get_task_spec

        spec = get_task_spec(task)
        candidates.extend((spec.name, spec.env_task, *spec.aliases))
    except KeyError:
        pass
    # One old release prefix predates the dataset task's current spelling.
    if "sim_pouring_beads" in candidates:
        candidates.append("pour")

    release_name = next((name for name in candidates if name in tasks), None)
    if release_name is None:
        available = ", ".join(sorted(tasks))
        raise ValueError(
            f"No published finetuned checkpoint matches task {task!r}. "
            f"Available checkpoint tasks: {available}"
        )
    release = tasks[release_name]
    step = str(release["recommended_step"])
    checkpoint = cache / "finetuned_sim" / release_name / f"{step}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Recommended checkpoint is not downloaded: {checkpoint}\n"
            f"Run: uv run prepare.py --sim-bundle {task}"
        )
    return checkpoint, release.get("prompt")


def scene_is_stale(scene_model: mujoco.MjModel, env_model: mujoco.MjModel) -> bool:
    """Whether the viser scene wraps a model the env has since replaced.

    Most abc_sim task randomizers recompile the MjModel on reset (they swap in
    different object meshes), which leaves the scene's geometry handles bound to
    a model nothing steps any more; the scene has to be rebuilt around the new
    one.
    """
    return scene_model is not env_model


def viz_step_period_s(timestep: float, decimation: int | None) -> float:
    """Wall-clock seconds one policy action should occupy, for realtime playback.

    Uses the model timestep and control decimation rather than a fixed rate.
    """
    if decimation is None or timestep <= 0.0 or decimation <= 0:
        return 1.0 / 30.0
    return timestep * decimation


def viz_status_text(
    task: str, world_seed: int, chunk: int, progress: str, success: bool
) -> str:
    """The one-line GUI status shown during a sim-task rollout."""
    text = f"{task} seed={world_seed} chunk={chunk} {progress}"
    return f"{text} success" if success else text


def run_sim_task_viewer(cfg: VizPolicyConfig) -> None:
    """Viewer for any abc_sim task: stream its own model/data to the browser.

    Tasks come from the vendored catalogue through ``SimTaskEnv``, which steps
    CPU MuJoCo and renders the policy's cameras with ``--sim.camera-backend``.
    Rollouts run the eval loop — infer a chunk, execute
    ``execute_chunk_dim`` actions paced at the task's own control rate, evaluate,
    loop on success or when the chunk budget runs out — and reset with a fresh
    world seed each time.
    """
    torch.set_float32_matmul_precision("high")

    # The DiT policy embeds config.prompt into a CLIP vector in __init__, so the
    # checkpoint and prompt have to be resolved before the policy (and env) are built.
    release_prompt = None
    if cfg.sim.checkpoint:
        checkpoint = Path(cfg.sim.checkpoint).expanduser().resolve()
    else:
        checkpoint, release_prompt = resolve_release_checkpoint(cfg.sim.task)
        print(f"[checkpoint] {checkpoint} (recommended for {cfg.sim.task})", flush=True)
    checkpoint_config = replace(cfg.sim, checkpoint=str(checkpoint))
    sidecar_prompt = checkpoint_sim_prompt(checkpoint_config)
    trained_prompt = sidecar_prompt or release_prompt
    if cfg.sim.prompt is None and sidecar_prompt:
        print(
            f"[prompt] {sidecar_prompt!r} "
            "(the prompt this checkpoint trained the task under)",
            flush=True,
        )
    sim = replace(
        checkpoint_config,
        prompt=resolve_viz_prompt(cfg.sim.task, cfg.sim.prompt, trained_prompt),
    )
    policy_kind = (
        sniff_policy_kind(str(checkpoint)) if sim.policy == "auto" else sim.policy
    )
    model_config = sim.vla_model if policy_kind == "vla" else sim.model
    device = resolve_device(sim.device)
    model_errors = (
        validate_vla_model_config(sim.vla_model)
        if policy_kind == "vla"
        else validate_model_config(sim.model)
    )
    errors = model_errors + validate_rtc_config(sim)
    if cfg.fast_inference and not device.startswith("cuda"):
        errors.append(
            f"--fast-inference needs a CUDA device, resolved device is {device!r}; "
            "pass --no-fast-inference to run the viewer on CPU"
        )
    if errors:
        raise ValueError("Invalid sim eval config:\n  - " + "\n  - ".join(errors))
    if sim.camera_backend == "mjwarp":
        # abc_sim always steps physics in CPU MuJoCo; only its renderer is MJWarp.
        require_mjwarp()

    from abc_minimal.sim_env import SimTaskEnv

    policy_cls = VLAInferencePolicy if policy_kind == "vla" else DiTInferencePolicy
    policy = policy_cls(checkpoint, sim, device, model_config=model_config)
    resolve_prefix_length(sim, policy.trained_max_prefix, model_config)
    env = SimTaskEnv(
        task=sim.task,
        height=sim.camera_height,
        width=sim.camera_width,
        camera_keys=model_config.camera_keys,
        prompt=sim.prompt,
        camera_backend=sim.camera_backend,
        gpu_id=sim.gpu_id,
    )
    # The gym env underneath owns the MjModel/MjData viser streams; the object
    # itself outlives a model swap, only its .model/.data are replaced.
    gym_env = env.env
    seed = [sim.seed]
    reset_requested = threading.Event()
    step_period_s = viz_step_period_s(
        float(gym_env.model.opt.timestep),
        getattr(gym_env, "_control_decimation", None),
    )
    action_shape = (model_config.chunk_length, model_config.action_dim)
    print(
        f"task={sim.task} prompt={sim.prompt!r} control={1.0 / step_period_s:.0f}Hz",
        flush=True,
    )

    setup_obs = env.reset(seed[0])
    if cfg.fast_inference:
        t0 = time.perf_counter()
        warmup_noise = np.random.default_rng(0).standard_normal(
            action_shape, dtype=np.float32
        )
        policy.enable_fast_inference(
            compile_mode=cfg.fast_compile_mode,
            warmup_obs=setup_obs,
            warmup_noise=warmup_noise,
            rtc_prefix_length=sim.rtc_prefix_length if sim.rtc else None,
        )
        torch.cuda.synchronize()
        print(f"fast inference ready in {time.perf_counter() - t0:.1f}s", flush=True)

    server = viser.ViserServer(host="0.0.0.0", port=cfg.port)
    actual_port = server.get_port()

    default_camera_position = np.array([-0.42, 0.0, 1.66], dtype=np.float64)
    default_camera_look_at = np.array([0.45, 0.0, 0.87], dtype=np.float64)

    def apply_default_view(client: viser.ClientHandle) -> None:
        client.camera.position = default_camera_position
        client.camera.look_at = default_camera_look_at
        client.camera.up_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        client.camera.fov = math.radians(50.0)

    @server.on_client_connect
    async def _(client: viser.ClientHandle) -> None:
        await asyncio.sleep(0.1)
        apply_default_view(client)

    status = server.gui.add_text("status", initial_value="idle", disabled=True)
    btn = server.gui.add_button("Reset & rollout")
    view_btn = server.gui.add_button("Default view")

    @view_btn.on_click
    def _(_) -> None:
        for client in server.get_clients().values():
            apply_default_view(client)

    def build_scene() -> ViserMujocoScene:
        """Rebuild the scene over the model the env is stepping right now.

        ViserMujocoScene scales the model's visualization defaults in place, so
        each MjModel may only be wrapped once — safe here because a rebuild only
        happens for a model object we have not seen before. scene.reset() drops
        the geometry of the model being replaced; it leaves the GUI and each
        client's camera alone, so the user's viewpoint survives a reset.
        """
        server.scene.reset()
        scene = ViserMujocoScene(server, gym_env.model, num_envs=1)
        scene.camera_tracking_enabled = False
        # mjviser renders every MuJoCo plane as an infinite grid, which for the
        # table surface would stretch a grid across the world at table height.
        server.scene.remove_by_name("/fixed_bodies/world/table_plane")
        scene.update_from_mjdata(gym_env.data)
        return scene

    scene = build_scene()
    scene_model = gym_env.model

    def rollout(initial_obs: dict[str, Any] | None = None) -> None:
        nonlocal scene, scene_model
        world_seed = seed[0]
        print(f"[rollout] start task={sim.task} seed={world_seed}", flush=True)
        if initial_obs is not None:
            obs = initial_obs
        else:
            obs = env.reset(world_seed)
            if scene_is_stale(scene_model, gym_env.model):
                scene = build_scene()
                scene_model = gym_env.model
            else:
                scene.update_from_mjdata(gym_env.data)
        if obs["prompt"] != sim.prompt:
            # Prompt-varying tasks (put_relative, count_into_opaque_box) rewrite
            # the prompt per episode; the policy still conditions on sim.prompt.
            print(f"[rollout] scene asks for {obs['prompt']!r}", flush=True)
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(action_shape, dtype=np.float32)
        t_inf = time.perf_counter()
        actions = policy.infer(obs, noise=noise)
        current_infer_s = time.perf_counter() - t_inf
        rtc = (
            RTCManager(
                policy,
                prefix_length=sim.rtc_prefix_length,
                inference_lead_steps=sim.rtc_inference_lead_steps,
                execute_chunk_dim=sim.execute_chunk_dim,
            )
            if sim.rtc
            else None
        )
        task_eval = env.evaluate()
        cancelled = False
        for chunk in range(sim.num_chunks):
            if reset_requested.is_set():
                cancelled = True
                break
            chunk_infer_s = current_infer_s
            obs_render_s = 0.0
            t_steps = time.perf_counter()
            lead_index = sim.execute_chunk_dim - sim.rtc_inference_lead_steps
            for action_index, action in enumerate(actions[: sim.execute_chunk_dim]):
                if reset_requested.is_set():
                    cancelled = True
                    break
                if (
                    rtc is not None
                    and chunk + 1 < sim.num_chunks
                    and action_index == lead_index
                ):
                    t_obs = time.perf_counter()
                    obs = env.obs()
                    obs_render_s = time.perf_counter() - t_obs
                    noise = rng.standard_normal(action_shape, dtype=np.float32)
                    rtc.start(obs, actions, noise)
                t_step = time.perf_counter()
                env.step_one(action)
                scene.update_from_mjdata(gym_env.data)
                task_eval = env.evaluate()
                status.value = viz_status_text(
                    sim.task,
                    world_seed,
                    chunk,
                    progress_text(task_eval, "num_bottles_in_bin"),
                    bool(task_eval["ever_success"]),
                )
                if task_eval["ever_success"]:
                    break
                # Realtime playback: one action per step_period_s of wall-clock,
                # or as fast as the machine manages when it cannot keep up.
                sleep_s = step_period_s - (time.perf_counter() - t_step)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            steps_s = time.perf_counter() - t_steps
            if cancelled or task_eval["ever_success"]:
                break
            if chunk + 1 < sim.num_chunks:
                if rtc is not None:
                    actions, current_infer_s, _ = rtc.get()
                else:
                    t_obs = time.perf_counter()
                    obs = env.obs()
                    obs_render_s = time.perf_counter() - t_obs
                    noise = rng.standard_normal(action_shape, dtype=np.float32)
                    t_inf = time.perf_counter()
                    actions = policy.infer(obs, noise=noise)
                    current_infer_s = time.perf_counter() - t_inf
            print(
                f"chunk={chunk:2d} infer={chunk_infer_s * 1000:.0f}ms "
                f"steps={steps_s * 1000:.0f}ms render={obs_render_s * 1000:.0f}ms "
                f"{progress_text(task_eval, 'num_bottles_in_bin')}",
                flush=True,
            )
        if rtc is not None:
            rtc.close()
        final_eval = env.evaluate()
        progress = progress_text(final_eval, "max_bottles_in_bin_so_far")
        success = bool(final_eval["ever_success"])
        if cancelled:
            print(f"[rollout] reset seed={world_seed} {progress}", flush=True)
            status.value = f"{sim.task} seed={world_seed} reset"
        else:
            print(
                f"[rollout] done seed={world_seed} {progress} success={success}",
                flush=True,
            )
            status.value = f"{sim.task} seed={world_seed} done {progress}"
            if success and not reset_requested.is_set():
                deadline = time.perf_counter() + 3.0
                while not reset_requested.is_set() and time.perf_counter() < deadline:
                    time.sleep(0.05)
        if reset_requested.is_set():
            reset_requested.clear()
        else:
            seed[0] += 1
            print(f"[rollout] auto reset seed={seed[0]}", flush=True)
            status.value = f"resetting to seed={seed[0]}"
        threading.Thread(target=rollout, daemon=True).start()

    @btn.on_click
    def _(_) -> None:
        seed[0] += 1
        print(f"[click] reset seed={seed[0]}", flush=True)
        status.value = f"resetting to seed={seed[0]}"
        reset_requested.set()

    threading.Thread(target=rollout, args=(setup_obs,), daemon=True).start()
    print(f"Viser ready on port {actual_port}", flush=True)
    while True:
        time.sleep(60)


def main(cfg: VizPolicyConfig) -> None:
    """Open the viewer for the configured task."""
    run_sim_task_viewer(cfg)
