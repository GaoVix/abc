"""Search for XML-retargeted policy success versus median-retargeted failure.

CPU MuJoCo physics, default MJWarp rendering, and deterministic FP32 inference.
Each model/world runs once. Model weights are loaded once and reused; every
world starts from an independent state. Matching cases are rendered afterward.
"""
import os

# Set before importing PyTorch or MuJoCo.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import fcntl
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch

import abc_sim
from abc_minimal.config import SimEvalConfig
from abc_minimal.determinism import configure_determinism
from abc_minimal.eval_policy import (
    _make_env, build_summary, jsonable, resolve_prefix_length, resolve_prompt,
    resolved_physics, rollout_worlds,
)
from abc_minimal.policy import DiTInferencePolicy
from abc_sim.randomization.core import RandomizationSamplingError
from scripts.replay_eval_failure_case import (
    RecordedScene, WorldNoise, comparison_video, render_trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
SERVER_OUTPUT = Path("/mnt/afs/L202500703/abc_storage/output")
SOURCES = (
    "scripts/search_eval_failure_case.py", "scripts/replay_eval_failure_case.py",
    "scripts/find_eval_failure_case.py", "abc_minimal/eval_policy.py",
    "abc_minimal/config.py", "abc_minimal/determinism.py", "abc_minimal/policy.py", "abc_minimal/dit.py",
    "abc_minimal/preprocess.py", "abc_minimal/sim_env.py", "abc_sim/env.py",
    "abc_sim/rendering/replay/renderer.py", "abc_sim/rendering/replay/static_geoms.py",
    "abc_sim/rendering/replay/camera_providers.py",
    "abc_sim/randomization/arm_bases.py", "abc_sim/randomization/tasks/dishrack.py",
    "abc_minimal/dishrack_median_base_poses.json",
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True))
    temporary.replace(path)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def array_hashes(arrays):
    result = {}
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        header = f"{array.dtype}:{array.shape}:".encode()
        result[name] = hashlib.sha256(header + array.tobytes()).hexdigest()
    return result


def deterministic_inference(seed, gpu_id):
    configure_determinism(seed)
    if not 0 <= gpu_id < torch.cuda.device_count():
        raise ValueError(f"GPU {gpu_id} is unavailable; visible GPUs: {torch.cuda.device_count()}")
    torch.cuda.set_device(gpu_id)


class TraceScene(RecordedScene):
    """Record full MuJoCo integration state in addition to joints/actions."""
    def __init__(self, *args):
        super().__init__(*args)
        self.integration = []

    def snapshot(self):
        super().snapshot()
        native = self.env.env
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(native.model, spec))
        mujoco.mj_getState(native.model, native.data, state, spec)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite MuJoCo integration state")
        self.integration.append(state)


def run_model(task, policies):
    folder = Path(task["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "result.json").exists():
        return json.loads((folder / "result.json").read_text())
    config = SimEvalConfig(**task["config"], output_dir=str(folder))
    scene = json.loads(Path(task["scene"]).read_text())
    env = None
    try:
        started = time.perf_counter()
        if task["side"] not in policies:
            policies[task["side"]] = DiTInferencePolicy(
                Path(config.checkpoint), config, f"cuda:{config.gpu_id}")
        torch.manual_seed(config.policy_seed)
        torch.cuda.manual_seed_all(config.policy_seed)
        policy = WorldNoise(policies[task["side"]], config.policy_seed, task["world_index"])
        prefix = resolve_prefix_length(config, policy.trained_max_prefix, config.model)
        config.prefix_length = prefix
        env = TraceScene(_make_env(config, tuple(config.model.camera_keys)), scene,
                         task["base"], task["collision_mode"])
        worlds = rollout_worlds(config, policy, env, prefix, None, folder, config.model)
        world = worlds[0]
        world["world_index"] = task["world_index"]
        build_summary(config=config, ckpt_path=Path(config.checkpoint), device=f"cuda:{config.gpu_id}",
                      worlds=worlds, out_dir=folder, physics=resolved_physics(env.env))
        arrays = dict(qpos=np.array(env.qpos), qvel=np.array(env.qvel), time=np.array(env.times),
                      integration=np.array(env.integration), actions=np.array(env.actions),
                      predictions=np.array(policy.predictions), diffusion_noise=np.array(policy.noise))
        np.savez_compressed(folder / "trajectory.npz", **arrays)
        np.savez(folder / "initial.npz", **env.initial_arrays)
        write_json(folder / "observations.json", policy.observations)
        result = dict(success=world["success"], max_reward=world["max_reward"], steps=world["steps"],
                      base=task["base"], collision_mode=task["collision_mode"],
                      seconds=time.perf_counter() - started, folder=str(folder),
                      fingerprints=dict(initial=array_hashes(env.initial_arrays), trace=array_hashes(arrays),
                                        observations=hashlib.sha256(json.dumps(policy.observations, sort_keys=True).encode()).hexdigest()),
                      torch=torch.__version__, mujoco=mujoco.__version__,
                      gpu=torch.cuda.get_device_name(config.gpu_id))
        write_json(folder / "result.json", result)
        print(f"world={task['world_index']:03d} {task['side']}: success={result['success']} "
              f"max_reward={result['max_reward']:.2f} steps={result['steps']}", flush=True)
        return result
    finally:
        if env is not None:
            env.env.close()


def eval_config(args, side, seed):
    base = "xml" if side == "old" else "median"
    collision_mode = args.xml_collisions if side == "old" else args.median_collisions
    request = dict(plate_count=3, plate_variant="plate_0", dish_rack_variant="dish_rack_0",
                   randomize_variants=False, randomize_scales=False, randomize_arm_bases=False,
                   arm_base_source=base, arm_collision_mode=collision_mode)
    config = dict(checkpoint=str(args.xml_checkpoint if side == "old" else args.median_checkpoint),
                  policy="dit", task="dishrack", num_worlds=1, parallel_worlds=0, seed=seed,
                  policy_seed=args.policy_seed, gpu_id=args.gpu_id, num_chunks=args.num_chunks,
                  diffusion_steps=10, execute_chunk_dim=15, rtc=True, rtc_prefix_length=4,
                  rtc_inference_lead_steps=4, camera_height=168, camera_width=224,
                  camera_backend=args.camera_backend, deterministic_inference=True,
                  fast_inference=False, compensate_arm_bases=False,
                  randomization=json.dumps(request), prompt=args.prompt, log_every_chunk=args.log_every_chunk)
    config["prompt"] = resolve_prompt(SimEvalConfig(**config))
    return config


def prepare_scene(path, seed, collision_mode="eef_only"):
    if path.exists():
        return
    env = abc_sim.make_env(task="dishrack", render_cameras=False)
    try:
        request = dict(plate_count=3, plate_variant="plate_0", dish_rack_variant="dish_rack_0",
                       randomize_variants=False, randomize_scales=False, randomize_arm_bases=False,
                       arm_base_source="xml", arm_collision_mode=collision_mode)
        env.forget_arm_state()
        _, info = env.reset(seed=seed, options={"randomization": request}, randomize=True)
        write_json(path, dict(randomization=info["randomization"]))
    finally:
        env.close()


def render_case(tasks, results, folder, stride):
    videos = {}
    for side in ("old", "new"):
        task = tasks[side]
        config = SimEvalConfig(**task["config"])
        scene = json.loads(Path(task["scene"]).read_text())
        output = Path(results[side]["folder"])
        env = RecordedScene(_make_env(config, tuple(config.model.camera_keys)), scene,
                            task["base"], task["collision_mode"])
        try:
            env.reset(seed=config.seed)
            with np.load(output / "trajectory.npz") as trace:
                env.qpos = list(trace["qpos"])
                env.qvel = list(trace["qvel"])
                env.times = list(trace["time"])
            video, fps = render_trajectory(env, output, stride, tuple(config.model.camera_keys))
            videos[side] = dict(results[side], video=str(video), fps=fps)
        finally:
            env.env.close()
    video = comparison_video(videos, folder, dict(old=224, new=224), dict(old=0, new=0))
    return dict(video=str(video), old=videos["old"]["video"], new=videos["new"]["video"])


def search(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_keys = ("xml_checkpoint", "median_checkpoint", "gpu_id", "seed", "policy_seed",
                   "start_world", "num_worlds", "num_chunks", "max_matches",
                   "camera_backend", "xml_collisions", "median_collisions", "prompt", "render_stride",
                   "recheck_case")
    config = {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path)
              else getattr(args, key) for key in config_keys}
    checkpoints = {str(path): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                   for path in (args.xml_checkpoint, args.median_checkpoint)}
    manifest = dict(config=config, checkpoints=checkpoints, source_hashes=source_hashes(),
                    gl_backend=os.environ.get("MUJOCO_GL"), physics="CPU MuJoCo",
                    inference="FP32 eager; TF32 off; math attention; deterministic algorithms",
                    runs_per_model_world=1, noise="[policy_seed, world_index]")
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Parameters, checkpoints, or source files differ from this output; use a new directory")
    write_json(manifest_path, manifest)
    summary = dict(status="running", checked_worlds=0, xml_successes=0, median_evaluated=0,
                   matches=[], skipped_unplaceable=[], worlds=[], manifest=str(manifest_path))
    deterministic_inference(args.policy_seed, args.gpu_id)
    policies = {}
    started = time.perf_counter()
    for index in range(args.start_world, args.start_world + args.num_worlds):
        seed = args.seed + index
        folder = root / f"world_{index:03d}_seed_{seed}"
        folder.mkdir(exist_ok=True)
        try:
            if source_hashes() != manifest["source_hashes"]:
                raise RuntimeError("Source files changed during the search")
            try:
                prepare_scene(folder / "scene.json", seed, args.xml_collisions)
            except RandomizationSamplingError as error:
                summary["skipped_unplaceable"].append(dict(world_index=index, seed=seed, error=str(error)))
                print(f"world={index:03d} seed={seed}: placement failed; skipped without changing seed", flush=True)
                write_json(root / "summary.json", summary)
                continue
            tasks = {}
            for side in ("old", "new"):
                config = eval_config(args, side, seed)
                tasks[side] = dict(side=side, world_index=index,
                                   base="xml" if side == "old" else "median", config=config,
                                   collision_mode=args.xml_collisions if side == "old" else args.median_collisions,
                                   scene=str(folder / "scene.json"), folder=str(folder / side),
                                   source_hashes=manifest["source_hashes"])
            if tasks["old"]["config"]["prompt"] != tasks["new"]["config"]["prompt"]:
                raise ValueError("Checkpoint prompts differ; pass --prompt explicitly for this comparison")
            print(f"world={index:03d} seed={seed}", flush=True)
            results = {"old": run_model(tasks["old"], policies)}
            summary["checked_worlds"] += 1
            if results["old"]["success"]:
                summary["xml_successes"] += 1
            if results["old"]["success"] or args.recheck_case:
                results["new"] = run_model(tasks["new"], policies)
                summary["median_evaluated"] += 1
            match = results["old"]["success"] and "new" in results and not results["new"]["success"]
            world = dict(world_index=index, world_seed=seed, matched=match, results=results)
            if match or args.recheck_case:
                world["videos"] = render_case(tasks, results, folder, args.render_stride)
            if match:
                summary["matches"].append(dict(world_index=index, seed=seed, **world["videos"]))
                print(f"FOUND: XML-retargeted policy succeeds, median policy fails; "
                      f"video={world['videos']['video']}", flush=True)
            if args.recheck_case:
                print(f"RECHECK: xml_success={results['old']['success']} "
                      f"median_success={results['new']['success']} original_pattern_holds={match}; "
                      f"video={world['videos']['video']}", flush=True)
            write_json(folder / "result.json", world)
            summary["worlds"].append(world)
            summary["seconds_this_invocation"] = time.perf_counter() - started
            print(f"progress: checked={summary['checked_worlds']} xml_success={summary['xml_successes']} "
                  f"median_tested={summary['median_evaluated']} matches={len(summary['matches'])}", flush=True)
            if match and args.max_matches and len(summary["matches"]) >= args.max_matches:
                summary["status"] = "found"
                write_json(root / "summary.json", summary)
                return summary
            write_json(root / "summary.json", summary)
        except Exception as error:
            summary.update(status="stopped_on_error", stopped_world=index, error=f"{type(error).__name__}: {error}")
            write_json(root / "summary.json", summary)
            raise
    summary["status"] = "complete"
    summary["seconds_this_invocation"] = time.perf_counter() - started
    write_json(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-checkpoint", "--baseline-checkpoint", dest="xml_checkpoint", type=Path,
                        default=SERVER_OUTPUT / "dit_joint_retargeted_official_timing_20261022_125331/100000.pt",
                        help="Policy trained on XML-base-retargeted data; baseline policies are rejected")
    parser.add_argument("--median-checkpoint", type=Path,
                        default=SERVER_OUTPUT / "dit_median_sequence_ik_20261101_183529/100000.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--start-world", type=int, default=0)
    parser.add_argument("--num-worlds", type=int, default=100)
    parser.add_argument("--num-chunks", type=int, default=236)
    parser.add_argument("--max-matches", type=int, default=1, help="Stop after this many cases; 0 scans all")
    parser.add_argument("--recheck-case", action="store_true",
                        help="Run both models and render one selected world regardless of success/failure")
    parser.add_argument("--camera-backend", choices=("mujoco", "mjwarp"), default="mjwarp")
    parser.add_argument("--xml-collisions", "--baseline-collisions", dest="xml_collisions",
                        choices=("full", "eef_only"), default="eef_only")
    parser.add_argument("--median-collisions", choices=("full", "eef_only"), default="eef_only")
    parser.add_argument("--prompt")
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--log-every-chunk", action="store_true")
    args = parser.parse_args()
    if min(args.num_worlds, args.num_chunks, args.render_stride) < 1:
        parser.error("num-worlds, num-chunks, and render-stride must be positive")
    if min(args.start_world, args.max_matches, args.policy_seed, args.seed) < 0:
        parser.error("world index, seeds, and max-matches must be nonnegative")
    if args.recheck_case and args.num_worlds != 1:
        parser.error("--recheck-case requires --num-worlds 1")
    for name in ("xml_checkpoint", "median_checkpoint"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            parser.error(f"Checkpoint not found: {path}")
        setattr(args, name, path)
    if args.xml_checkpoint.parent.name.startswith("dit_joint_baseline_"):
        parser.error("Use the XML-retargeted policy, not dit_joint_baseline, for this comparison")
    if args.xml_checkpoint.stem != args.median_checkpoint.stem:
        parser.error("XML and median checkpoints must use the same training step")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    args.output = args.output or ROOT / "outputs" / f"search_xml_retargeted_vs_median_{args.xml_checkpoint.stem}_{stamp}"
    print(f"XML-retargeted policy: {args.xml_checkpoint}; base=xml", flush=True)
    print(f"Median-retargeted policy: {args.median_checkpoint}; base=median", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another search is using {args.output}")
        result = search(args)
    print(json.dumps({key: value for key, value in result.items() if key != "worlds"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
