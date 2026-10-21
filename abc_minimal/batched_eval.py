"""Batched sim eval: ``parallel_worlds`` abc_sim worlds stepped together in MJWarp.

The rollout is the sequential loop of ``eval_policy.rollout_worlds`` with a
world axis: one policy call produces a chunk per world, one env step advances
every world, and the RTC schedule (inference on the next ``rtc_prefix_length``
unexecuted actions, overlapped with their execution) is unchanged. A world that
is over stops being recorded but keeps stepping until its batch finishes, so its
record matches the sequential loop's early break.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import abc_sim
from abc_minimal.config import SimEvalConfig
from abc_minimal.eval_policy import (
    RTCManager,
    SimPolicy,
    _optional_int,
    jsonable,
    progress_text,
    video_frame,
    without_missing_counts,
)
from abc_sim.batched_env import BatchedWarpYAMEnv
from abc_sim.task_eval import TaskEvalResult

EMPTY_EVAL = {"reward": 0.0, "success": False, "ever_success": False}


def make_batched_env(config: SimEvalConfig) -> BatchedWarpYAMEnv:
    from abc_minimal.sim_env import task_spec

    spec = task_spec(config.task)
    if spec.evaluator_name is None:
        print(
            f"warning: sim task '{spec.name}' has no success evaluator; "
            "reward stays 0.0, success/ever_success stay False",
            flush=True,
        )
    return abc_sim.make_batched_env(
        spec.name,
        num_worlds=config.parallel_worlds,
        prompt=config.prompt,
        camera_height=config.camera_height,
        camera_width=config.camera_width,
        camera_gpu_id=config.gpu_id,
    )


def world_eval(result: TaskEvalResult | None, index: int, per_world_keys: set[str]) -> dict[str, Any]:
    """One world's evaluation, in the sequential loop's single-world schema."""
    if result is None:
        return dict(EMPTY_EVAL)
    per_world = per_world_keys | {"reward", "success"}
    info = {"reward": result.reward, "success": result.success, **result.metrics}
    return {key: jsonable(value[index] if key in per_world else value) for key, value in info.items()}


def rewards(result: TaskEvalResult | None, num_worlds: int) -> np.ndarray:
    if result is None:
        return np.zeros(num_worlds, dtype=np.float32)
    return result.reward


def over(result: TaskEvalResult | None, num_worlds: int) -> np.ndarray:
    """Per-world rollout_over: ever_failed for maintenance tasks, else ever_success."""
    if result is None:
        return np.zeros(num_worlds, dtype=bool)
    flags = result.metrics.get("ever_failed", result.metrics.get("ever_success"))
    return np.asarray(flags, dtype=bool)


def counts(result: TaskEvalResult | None, key: str, index: int) -> int | None:
    values = None if result is None else result.metrics.get(key)
    return None if values is None else _optional_int(values[index])


def frames_to_numpy(images: dict[str, Any], camera_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: images[name].cpu().numpy() for name in camera_keys}


def rollout_batched_worlds(
    config: SimEvalConfig,
    policy: SimPolicy,
    env: BatchedWarpYAMEnv,
    prefix_length: int,
    options: dict[str, Any] | None,
    out_dir: Path,
    model_config: Any,
) -> list[dict[str, Any]]:
    """Roll out the worlds ``parallel_worlds`` at a time; returns their records."""
    n = config.parallel_worlds
    camera_keys = model_config.camera_keys
    action_shape = (model_config.chunk_length, model_config.action_dim)
    # One noise stream per world, so a world's rollout does not depend on the batch width.
    rngs = [np.random.default_rng([config.policy_seed, i]) for i in range(config.num_worlds)]

    def sample_noise(batch: range) -> np.ndarray:
        return np.stack([rngs[i].standard_normal(action_shape, dtype=np.float32) for i in batch])

    worlds = []
    warmed_up = False
    try:
        for batch_start in range(0, config.num_worlds, n):
            batch = range(batch_start, batch_start + n)
            t0 = time.perf_counter()
            obs = env.reset([int(config.seed + i) for i in batch], options)
            missing = [name for name in camera_keys if name not in obs["images"]]
            if missing:
                raise ValueError(
                    f"Cameras missing from the {config.task} scene: {', '.join(missing)} "
                    f"(scene has {', '.join(obs['images'])})"
                )
            if not warmed_up:
                warmup_noise = np.random.default_rng(config.policy_seed).standard_normal(
                    (n, *action_shape), dtype=np.float32
                )
                t_warm = time.perf_counter()
                if config.fast_inference:
                    policy.enable_fast_inference(
                        compile_mode=config.fast_compile_mode,
                        warmup_obs=obs,
                        warmup_noise=warmup_noise,
                    )
                if prefix_length:
                    policy.warmup_rtc(obs, warmup_noise, prefix_length)
                if config.rtc:
                    policy.warmup_rtc(obs, warmup_noise, config.rtc_prefix_length)
                torch.cuda.synchronize()
                print(
                    f"inference ready in {time.perf_counter() - t_warm:.1f}s ({n} worlds)",
                    flush=True,
                )
                warmed_up = True

            videos = [None] * n
            if config.save_video:
                import imageio.v2 as imageio

                videos = [
                    imageio.get_writer(
                        str(out_dir / f"world_{i:03d}.mp4"),
                        fps=config.video_fps,
                        macro_block_size=1,
                    )
                    for i in batch
                ]
                frames = frames_to_numpy(obs["images"], camera_keys)
                for index, video in enumerate(videos):
                    video.append_data(
                        video_frame({k: frames[k][index] for k in camera_keys}, camera_keys)
                    )
            keys = env.per_world_metric_keys
            result = env.evaluate()
            # The result each world was last scored with while still running.
            last_result = [result] * n
            max_reward = rewards(result, n).astype(np.float32)
            steps = np.zeros(n, dtype=np.int64)
            active = np.ones(n, dtype=bool)
            diverged = np.zeros(n, dtype=bool)
            chunk_metrics: list[list[dict[str, Any]]] = [[] for _ in range(n)]
            batch_steps = 0
            rtc = None
            try:
                action_prefix = None
                if prefix_length:
                    action_prefix = np.repeat(obs["state"][:, None, :], prefix_length, axis=1)
                noise = sample_noise(batch)
                t_infer = time.perf_counter()
                actions = policy.infer(
                    obs,
                    noise=noise,
                    action_prefix=action_prefix,
                    prefix_length=prefix_length,
                )
                current_infer_s = time.perf_counter() - t_infer
                if config.rtc:
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
                    executed = actions[:, prefix_length : prefix_length + config.execute_chunk_dim]
                    for action_index in range(config.execute_chunk_dim):
                        if (
                            rtc is not None
                            and chunk + 1 < config.num_chunks
                            and action_index == lead_index
                        ):
                            t_obs = time.perf_counter()
                            next_obs = env.get_obs()
                            rtc_obs_s = time.perf_counter() - t_obs
                            rtc.start(next_obs, actions, sample_noise(batch))
                            rtc_started = True
                        result = env.step(executed[:, action_index])
                        batch_steps += 1
                        for index in np.flatnonzero(active):
                            last_result[index] = result
                        max_reward = np.where(
                            active, np.maximum(max_reward, rewards(result, n)), max_reward
                        )
                        steps += active
                        newly_diverged = active & ~np.isfinite(env.qpos).all(axis=1)
                        for index in np.flatnonzero(newly_diverged):
                            print(
                                f"world={batch_start + index:03d} physics diverged "
                                f"at step {steps[index]}; recorded as failed",
                                flush=True,
                            )
                        diverged |= newly_diverged
                        if config.save_video and batch_steps % config.video_every_n_actions == 0:
                            frames = frames_to_numpy(env.render_cameras(), camera_keys)
                            for index in np.flatnonzero(active):
                                videos[index].append_data(
                                    video_frame({k: frames[k][index] for k in camera_keys}, camera_keys)
                                )
                        active &= ~diverged & ~over(result, n)
                        if not active.any():
                            break
                    steps_s = time.perf_counter() - t_steps
                    if not active.any():
                        break
                    if rtc is None and chunk + 1 < config.num_chunks:
                        t_obs = time.perf_counter()
                        obs = env.get_obs()
                        rtc_obs_s = time.perf_counter() - t_obs
                        if prefix_length:
                            action_prefix = np.asarray(executed[:, -prefix_length:], dtype=np.float32)
                        noise = sample_noise(batch)
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
                    wall_s = time.perf_counter() - t_chunk
                    reward = rewards(result, n)
                    for index in np.flatnonzero(active):
                        metric = {
                            "chunk": chunk,
                            "infer_s": float(logged_infer_s),
                            "current_chunk_infer_s": float(chunk_infer_s),
                            "rtc_next_infer_s": (
                                float(rtc_infer_s) if rtc_infer_s is not None else None
                            ),
                            "steps_s": float(steps_s),
                            "obs_render_s": float(rtc_obs_s),
                            "wall_s": float(wall_s),
                            "rtc_ready_at_chunk_end": rtc_ready,
                            "reward": float(reward[index]),
                            "bottles": counts(result, "num_bottles_in_bin", index),
                            "max_bottles": counts(result, "max_bottles_in_bin_so_far", index),
                        }
                        chunk_metrics[index].append(without_missing_counts(metric))
                    if config.log_every_chunk:
                        rtc_text = (
                            f" rtc_ready_at_chunk_end={rtc_ready}" if config.rtc else ""
                        )
                        print(
                            f"batch={batch_start:03d} chunk={chunk:02d} "
                            f"active={int(active.sum())}/{n} "
                            f"infer={logged_infer_s * 1000:.0f}ms "
                            f"steps={steps_s * 1000:.0f}ms "
                            f"render={rtc_obs_s * 1000:.0f}ms{rtc_text}",
                            flush=True,
                        )
            finally:
                if rtc is not None:
                    rtc.close()
                for video in videos:
                    if video is not None:
                        video.close()

            wall_s = time.perf_counter() - t0
            for index, world_index in enumerate(batch):
                final_eval = world_eval(last_result[index], index, keys)
                world = {
                    "world_index": world_index,
                    "world_seed": env.seeds[index],
                    "success": bool(final_eval["ever_success"]),
                    "final_success": bool(final_eval["success"]),
                    "reward": float(final_eval["reward"]),
                    "max_reward": float(max_reward[index]),
                    "steps": int(steps[index]),
                    "wall_s": wall_s,
                    "chunk_metrics": chunk_metrics[index],
                    "randomization": env.randomization[index],
                    "final_task_eval": final_eval,
                    "video_path": (
                        str(out_dir / f"world_{world_index:03d}.mp4") if config.save_video else None
                    ),
                }
                if diverged[index]:
                    world["diverged"] = True
                worlds.append(world)
                print(
                    f"world={world_index:03d} done success={world['success']} "
                    f"{progress_text(final_eval, 'max_bottles_in_bin_so_far')} "
                    f"steps={world['steps']}",
                    flush=True,
                )
    finally:
        env.close()
    return worlds
