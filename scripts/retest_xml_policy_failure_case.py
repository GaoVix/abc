"""Retest and film one original XML world with deterministic inference.

With --checkpoint, no historical failure is required. Otherwise a matched
old-success/new-failure report supplies the protocol. The production rollout
selects the original world index, preserving its environment and noise seeds.
Video is rendered from recorded integration states only after rollout ends.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import copy
import json
from dataclasses import fields
from pathlib import Path

from abc_minimal.config import ClipConfig, DiTConfig, SimEvalConfig
from scripts.find_xml_policy_failure_case import build_report


def prepare_case(report, output, gpu_id, disable_arm_control_limits):
    if report.get("status") != "found" or not report.get("selected"):
        raise ValueError("No matched bounded old-success/new-failure case was found")
    selected = report["selected"]
    saved = report["selected_configs"]["new"]
    if not selected["old"]["success"] or selected["new"]["success"]:
        raise ValueError("Selection is not old-success/new-failure")
    for side in ("old", "new"):
        if report["selected_configs"][side].get("disable_arm_control_limits", False):
            raise ValueError("Historical selection must have arm control limits enabled")
    if (saved.get("parallel_worlds") != 0 or saved.get("fast_inference") is not False
            or saved.get("deterministic_inference") is not True
            or saved.get("compensate_arm_bases") is not False
            or saved.get("camera_backend") != "mjwarp"
            or saved.get("policy") != "dit"):
        raise ValueError("Expected the historical strict deterministic direct XML DiT protocol")
    request = saved.get("randomization") or {}
    request = json.loads(request) if isinstance(request, str) else request
    if request.get("arm_base_source") != "xml" or request.get("randomize_arm_bases") is not False:
        raise ValueError("Historical run must use fixed XML bases")
    keys = {item.name for item in fields(SimEvalConfig)} - {"model", "clip", "vla_model"}
    values = {key: value for key, value in saved.items() if key in keys}
    values.update(checkpoint=selected["new"]["checkpoint"],
                  world_indices=(selected["world_index"],), gpu_id=gpu_id, output_dir=str(output),
                  log_every_chunk=True, save_video=False,
                  disable_arm_control_limits=disable_arm_control_limits)
    config = SimEvalConfig(**values, model=DiTConfig(**saved.get("model", {})),
                           clip=ClipConfig(**saved.get("clip", {})))
    record = report["selected_world_records"]["new"]
    if record["world_index"] != selected["world_index"] or record["world_seed"] != selected["world_seed"]:
        raise ValueError("Selected world identity does not match its historical record")
    if selected["world_seed"] not in (config.seed + selected["world_index"],
                                       config.seed + selected["world_index"] + 100_000):
        raise ValueError("Historical environment seed does not match the original world index")
    if record["randomization"] != report["selected_world_records"]["old"]["randomization"]:
        raise ValueError("Old and new historical scenes differ")
    return config, record


def direct_config(args, output):
    if args.world_index is None or args.policy_seed is None:
        raise ValueError("--checkpoint requires explicit --world-index and --policy-seed")
    request = dict(plate_count=3, plate_variant="plate_0", dish_rack_variant="dish_rack_0",
                   randomize_variants=False, randomize_scales=False,
                   randomize_arm_bases=False, arm_base_source="xml", arm_collision_mode="eef_only")
    return SimEvalConfig(
        checkpoint=str(args.checkpoint), policy="dit", task="dishrack",
        num_worlds=args.num_worlds, world_indices=(args.world_index,), seed=args.seed,
        policy_seed=args.policy_seed, gpu_id=args.gpu_id, parallel_worlds=0,
        camera_backend="mjwarp", camera_height=168, camera_width=224,
        deterministic_inference=True, fast_inference=False, num_chunks=236,
        diffusion_steps=10, execute_chunk_dim=15, rtc=True,
        rtc_prefix_length=4, rtc_inference_lead_steps=4, compensate_arm_bases=False,
        disable_arm_control_limits=args.disable_arm_control_limits,
        randomization=json.dumps(request), log_every_chunk=True, save_video=False,
        output_dir=str(output))


class CheckedScene:
    """Observe the normal CPU rollout without rendering extra policy images."""
    def __init__(self, env, expected_randomization=None):
        self.env = env
        self.expected = expected_randomization
        self.integration, self.actions, self.times = [], [], []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed, options=None):
        from abc_minimal.eval_policy import jsonable
        obs = self.env.reset(seed=seed, options=options)
        if self.expected is not None and jsonable(self.env.randomization) != self.expected:
            raise ValueError("Current reset differs from the exact historical initial scene")
        self.snapshot()
        return obs

    def snapshot(self):
        import mujoco
        import numpy as np
        native = self.env.env
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(native.model, spec))
        mujoco.mj_getState(native.model, native.data, state, spec)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite MuJoCo integration state")
        self.integration.append(state)
        self.times.append(float(native.data.time))

    def step_one(self, action):
        import numpy as np
        self.env.step_one(action)
        self.actions.append(np.array(action, copy=True))
        self.snapshot()

    step_one_vanilla = step_one

    def close(self):
        pass  # Keep the renderer alive for offline video export; main closes it.


class RecordedPolicy:
    """Save the exact production noise and predictions; never substitute noise."""
    def __init__(self, policy):
        self.policy = policy
        self.noise, self.predictions = [], []

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def infer(self, obs, noise=None, **kwargs):
        import numpy as np
        result = self.policy.infer(obs, noise=noise, **kwargs)
        self.noise.append(np.array(noise, copy=True))
        self.predictions.append(np.array(result, copy=True))
        return result


def render_top(env, path, stride):
    import imageio.v2 as imageio
    import mujoco
    native = env.env.env
    indices = list(range(0, len(env.integration), stride))
    if indices[-1] != len(env.integration) - 1:
        indices.append(len(env.integration) - 1)
    dt = native._physics_dt * native._control_decimation
    native._reset_camera_backend()
    with imageio.get_writer(str(path), fps=1 / (dt * stride), codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for count, index in enumerate(indices, 1):
            mujoco.mj_setState(native.model, native.data, env.integration[index],
                              mujoco.mjtState.mjSTATE_INTEGRATION)
            native.cur_step = index
            mujoco.mj_forward(native.model, native.data)
            writer.append_data(native.get_obs()["images"]["top"].transpose(1, 2, 0))
            if count % 100 == 0 or count == len(indices):
                print(f"Video frames: {count}/{len(indices)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/afs/L202500703/abc_storage/output"))
    parser.add_argument("--selection", type=Path, help="Optional existing matched selection.json")
    parser.add_argument("--checkpoint", type=Path, help="Explicit policy to test; no prior new-model failure required")
    parser.add_argument("--seed", type=int, default=20260511, help="Original 100-world base environment seed")
    parser.add_argument("--num-worlds", type=int, default=100, help="Original evaluation scope, not the number rerun")
    parser.add_argument("--policy-seed", type=int, help="Original policy seed; required with --checkpoint")
    parser.add_argument("--world-index", type=int, help="Only retest this original world; never choose a different world")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--disable-arm-control-limits", action="store_true",
                        help="Change only arm PD target clamping; physical joint/force limits remain")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-stride", type=int, default=2)
    args = parser.parse_args()
    if args.render_stride < 1:
        parser.error("render-stride must be positive")
    if args.checkpoint and args.selection:
        parser.error("Choose --checkpoint or --selection, not both")
    output = args.output.resolve()
    if args.checkpoint:
        config, record = direct_config(args, output), None
        report = dict(status="explicit_retest", historical_new_success=None,
                      world_index=args.world_index, policy_seed=args.policy_seed)
    else:
        report = json.loads(args.selection.read_text()) if args.selection else build_report(
            args.root, policy_seed=args.policy_seed, world_index=args.world_index)
        config, record = prepare_case(report, output, args.gpu_id, args.disable_arm_control_limits)
    if report.get("selected"):
        if args.world_index is not None and report["selected"]["world_index"] != args.world_index:
            raise ValueError("Selection does not match --world-index")
        if args.policy_seed is not None and report["selected"]["policy_seed"] != args.policy_seed:
            raise ValueError("Selection does not match --policy-seed")
    from abc_minimal.eval_policy import validate_batched_config
    errors = validate_batched_config(config)
    if config.policy_seed < 0 or config.seed < 0:
        errors.append("Seeds must be nonnegative")
    if errors:
        raise ValueError("Invalid config: " + "; ".join(errors))
    checkpoint = Path(config.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    selected = report.get("selected")
    index = config.world_indices[0]
    print(json.dumps(dict(new_checkpoint=config.checkpoint,
        original_num_worlds=config.num_worlds, worlds_to_run=list(config.world_indices),
        source_world_index=index, environment_seed=config.seed + index,
        policy_noise_stream=[config.policy_seed, index],
        deterministic_inference=config.deterministic_inference,
        previous_old_success=selected["old"]["success"] if selected else None,
        previous_new_success=selected["new"]["success"] if selected else None,
        disable_arm_control_limits=config.disable_arm_control_limits), indent=2), flush=True)

    import torch
    import numpy as np
    from abc_minimal.determinism import configure_determinism
    from abc_minimal.eval_policy import (
        _make_env, build_summary, resolve_prefix_length, resolve_prompt, resolved_physics, rollout_worlds,
    )
    from abc_minimal.policy import DiTInferencePolicy

    configure_determinism(config.policy_seed)
    torch.cuda.set_device(args.gpu_id)
    config.prompt = resolve_prompt(config)
    policy = DiTInferencePolicy(checkpoint, config, f"cuda:{args.gpu_id}")
    policy = RecordedPolicy(policy)
    config.prefix_length = resolve_prefix_length(config, policy.trained_max_prefix, config.model)
    env = CheckedScene(_make_env(config, tuple(config.model.camera_keys)),
                       record["randomization"] if record else None)
    physics = resolved_physics(env)
    try:
        if selected and physics != report["selected_physics"]["new"]:
            raise ValueError("Current physics differs from the historical evaluation")
        request = config.randomization
        options = {"randomization": json.loads(request) if isinstance(request, str) else request}
        worlds = rollout_worlds(config, policy, env, config.prefix_length, options, output, config.model)
        np.savez_compressed(output / "trajectory.npz", integration=np.array(env.integration),
                            time=np.array(env.times), actions=np.array(env.actions),
                            diffusion_noise=np.array(policy.noise), predictions=np.array(policy.predictions))
        build_summary(config=config, ckpt_path=checkpoint, device=f"cuda:{args.gpu_id}",
                      worlds=worlds, out_dir=output, physics=physics)
        comparison = dict(status="rendering", source_world_index=index,
            world_seed=worlds[0]["world_seed"], policy_seed=config.policy_seed,
            initial_randomization_verified_equal=True if record else None,
            historical_old=copy.deepcopy(selected["old"]) if selected else None,
            historical_new_with_control_limits=copy.deepcopy(selected["new"]) if selected else None,
            retest_new=dict(success=worlds[0]["success"], max_reward=worlds[0]["max_reward"],
                           steps=worlds[0]["steps"], disable_arm_control_limits=config.disable_arm_control_limits))
        (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
        del policy
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        video = output / f"world_{index:03d}_top.mp4"
        render_top(env, video, args.render_stride)
        worlds[0]["video_path"] = str(video)
        build_summary(config=config, ckpt_path=checkpoint, device=f"cuda:{args.gpu_id}",
                      worlds=worlds, out_dir=output, physics=physics)
        comparison.update(status="complete", video=str(video))
        (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
        print(f"VIDEO={video}", flush=True)
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
