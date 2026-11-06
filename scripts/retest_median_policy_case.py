"""Replay a saved median-policy case with its original low-resolution cameras.

Clone the saved configuration and restore the saved scene, then optionally
disable arm PD-target clamping. Verify the historical initial arrays and first
policy observation before inference. Record physics first and render afterward.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
import copy
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from abc_minimal.config import ClipConfig, DiTConfig, SimEvalConfig
from abc_minimal.sim_env import SimTaskEnv, task_spec
from scripts.search_eval_failure_case import TraceScene
from scripts.replay_eval_failure_case import WorldNoise, digest
from scripts.retest_xml_policy_failure_case import render_top

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ADAPTER_SHA256 = "2653fcdf115990d18272c1209ff202a1ee254c5f1d5a875016129ef4bf620e8e"
# These changed only in the new control/subset entry points, the camera adapter
# restored below, or diagnostic entry points. Numerical policy/render/base
# implementations must still match the historical manifest byte for byte.
ALLOWED_CHANGED_SOURCES = {
    "abc_minimal/config.py", "abc_minimal/eval_policy.py", "abc_minimal/sim_env.py",
    "abc_sim/env.py", "scripts/search_eval_failure_case.py",
    "scripts/replay_eval_failure_case.py", "scripts/find_eval_failure_case.py",
}


def write_json(path, value):
    from abc_minimal.eval_policy import jsonable
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2))


def load_case(source, output, gpu_id, disabled):
    summary = json.loads((source / "new/summary.json").read_text())
    manifest = json.loads((source.parent / "manifest.json").read_text())
    scene = json.loads((source / "scene.json").read_text())
    saved = summary["config"]
    if summary["checkpoint"] != manifest["config"]["median_checkpoint"]:
        raise ValueError("Saved checkpoint is not the manifest's original median model")
    request = saved["randomization"]
    request = json.loads(request) if isinstance(request, str) else request
    if (saved.get("policy") != "dit" or saved.get("task") != "dishrack"
            or saved.get("deterministic_inference") is not True
            or saved.get("fast_inference") is not False or saved.get("parallel_worlds") != 0
            or saved.get("compensate_arm_bases") is not False
            or saved.get("disable_arm_control_limits", False)
            or saved.get("num_worlds") != 1 or saved.get("camera_backend") != "mjwarp"
            or request.get("arm_base_source") != "median"
            or request.get("randomize_arm_bases") is not False):
        raise ValueError("Expected the original deterministic direct median-base single-world case")
    historical_sources = manifest["source_hashes"]
    if historical_sources["abc_minimal/sim_env.py"] != HISTORICAL_ADAPTER_SHA256:
        raise ValueError("Historical camera adapter is not the verified original 224x168 implementation")
    changed = [name for name, value in historical_sources.items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value]
    unexpected = sorted(set(changed) - ALLOWED_CHANGED_SOURCES)
    if unexpected:
        raise ValueError(f"Policy/render/base sources differ from history: {unexpected}")
    config_keys = {item.name for item in fields(SimEvalConfig)} - {"model", "clip", "vla_model"}
    values = {key: copy.deepcopy(value) for key, value in saved.items() if key in config_keys}
    values.update(checkpoint=summary["checkpoint"], gpu_id=gpu_id, output_dir=str(output),
                  log_every_chunk=True, save_video=False, world_indices=None,
                  disable_arm_control_limits=disabled)
    config = SimEvalConfig(**values, model=DiTConfig(**saved["model"]), clip=ClipConfig(**saved["clip"]))
    world = summary["worlds"][0]
    index = world["world_index"]
    if (world["world_seed"] != config.seed
            or config.policy_seed != manifest["config"]["policy_seed"]
            or config.seed != manifest["config"]["seed"] + index):
        raise ValueError("Saved world identity or noise stream is inconsistent")
    return config, summary, scene, changed


class HistoricalCameraEnv(SimTaskEnv):
    """The pre-high-resolution adapter; native pixels go directly to policy."""
    def __init__(self, config):
        import abc_sim
        self.spec = task_spec(config.task)
        self.height, self.width = config.camera_height, config.camera_width
        self.camera_keys = tuple(config.model.camera_keys)
        self.prompt, self.randomization = config.prompt, None
        self.env = abc_sim.make_env(
            task=self.spec.name, prompt=self.prompt, render_cameras=True,
            camera_backend=config.camera_backend, camera_gpu_id=config.gpu_id,
            camera_height=self.height, camera_width=self.width,
            disable_arm_control_limits=config.disable_arm_control_limits)

    def _camera_images(self, images):
        missing = set(self.camera_keys) - images.keys()
        if missing:
            raise ValueError(f"Missing policy cameras: {sorted(missing)}")
        return {name: images[name] for name in self.camera_keys}


class VerifiedTrace(TraceScene):
    def __init__(self, env, scene, source, summary):
        super().__init__(env, scene, "median", "eef_only")
        self.source, self.summary = source, summary

    def reset(self, seed, options=None):
        from abc_minimal.eval_policy import jsonable
        obs = super().reset(seed, options)
        with np.load(self.source / "new/initial.npz", allow_pickle=False) as original:
            if set(original.files) != self.initial_arrays.keys():
                raise ValueError("Initial-state fields differ from history")
            for name in original.files:
                np.testing.assert_array_equal(self.initial_arrays[name], original[name], err_msg=name)
        if jsonable(self.randomization) != self.summary["worlds"][0]["randomization"]:
            raise ValueError("Restored median bases/objects/metadata differ from history")
        original_obs = json.loads((self.source / "new/observations.json").read_text())[0]
        current_obs = dict(state=digest(obs["state"]), images={
            name: digest(value) for name, value in obs["images"].items()})
        if current_obs != original_obs:
            raise ValueError("Initial policy state or camera pixels differ from history")
        native = self.env.env
        arm = [native.model.actuator(f"{side}_joint{joint}").id
               for side in ("left", "right") for joint in range(1, 7)]
        expected = not native._disable_arm_control_limits
        if not np.all(native.model.actuator_ctrllimited[arm] == expected):
            raise ValueError("Arm PD control-limit switch was lost during reset/model reload")
        print("Verified: historical initial state, objects, median bases, camera pixels and policy state match exactly.", flush=True)
        return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--disable-arm-control-limits", action="store_true")
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.render_stride < 1:
        parser.error("render-stride must be positive")
    source, output = args.source_case.resolve(), args.output.resolve()
    config, history, scene, changed = load_case(source, output, args.gpu_id, args.disable_arm_control_limits)
    checkpoint = Path(config.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    write_json(output / "historical_summary.json", history)
    write_json(output / "replay_config.json", asdict(config))
    import torch
    from abc_minimal.determinism import configure_determinism
    from abc_minimal.eval_policy import build_summary, resolved_physics, rollout_worlds
    from abc_minimal.policy import DiTInferencePolicy

    original = history["worlds"][0]
    index = original["world_index"]
    print(json.dumps(dict(checkpoint=config.checkpoint, base="median", world_index=index,
        environment_seed=config.seed, policy_noise_stream=[config.policy_seed, index],
        deterministic_inference=True, native_camera_size=[config.camera_width, config.camera_height],
        disable_arm_control_limits=config.disable_arm_control_limits), indent=2), flush=True)
    configure_determinism(config.policy_seed)
    torch.cuda.set_device(args.gpu_id)
    policy = DiTInferencePolicy(checkpoint, config, f"cuda:{args.gpu_id}")
    # Match the original search's per-world reseed after model loading.
    torch.manual_seed(config.policy_seed)
    torch.cuda.manual_seed_all(config.policy_seed)
    policy = WorldNoise(policy, config.policy_seed, index)
    env = VerifiedTrace(HistoricalCameraEnv(config), scene, source, history)
    try:
        physics = resolved_physics(env.env)
        if physics is None or physics != history["resolved_physics"]:
            raise ValueError(f"Physics settings differ from history: current={physics}, historical={history['resolved_physics']}")
        print("[physics verified] " + json.dumps(physics), flush=True)
        worlds = rollout_worlds(config, policy, env, config.prefix_length, None, output, config.model)
        worlds[0]["world_index"] = index
        with np.load(source / "new/trajectory.npz", allow_pickle=False) as old:
            noise = np.asarray(policy.noise)
            np.testing.assert_array_equal(noise, old["diffusion_noise"][:len(noise)])
        np.savez_compressed(output / "trajectory.npz", integration=np.asarray(env.integration),
                            qpos=np.asarray(env.qpos), qvel=np.asarray(env.qvel), time=np.asarray(env.times),
                            actions=np.asarray(env.actions), predictions=np.asarray(policy.predictions),
                            diffusion_noise=noise)
        np.savez(output / "initial.npz", **env.initial_arrays)
        write_json(output / "observations.json", policy.observations)
        build_summary(config=config, ckpt_path=checkpoint, device=f"cuda:{args.gpu_id}",
                      worlds=worlds, out_dir=output, physics=physics)
        result = dict(status="rendering", source_case=str(source), world_index=index,
            world_seed=config.seed, policy_seed=config.policy_seed,
            native_camera_size=[config.camera_width, config.camera_height],
            initial_arrays_and_policy_observation_match_history=True,
            diffusion_noise_matches_historical_prefix=True, changed_source_files=changed,
            historical=dict(success=original["success"], steps=original["steps"], max_reward=original["max_reward"]),
            retest=dict(success=worlds[0]["success"], steps=worlds[0]["steps"], max_reward=worlds[0]["max_reward"],
                        disable_arm_control_limits=config.disable_arm_control_limits))
        write_json(output / "comparison.json", result)
        print(json.dumps(result, indent=2), flush=True)
        del policy
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        video = output / f"world_{index:03d}_top.mp4"
        render_top(env, video, args.render_stride)
        worlds[0]["video_path"] = str(video)
        build_summary(config=config, ckpt_path=checkpoint, device=f"cuda:{args.gpu_id}",
                      worlds=worlds, out_dir=output, physics=physics)
        result.update(status="complete", video=str(video))
        write_json(output / "comparison.json", result)
        print(f"VIDEO={video}", flush=True)
    finally:
        env.env.close()


if __name__ == "__main__":
    main()
