"""Find and film one old-XML success versus new-XML failure.

Both models run once with the saved protocol and original world-noise stream.
Physics is recorded first; lossless videos are rendered afterward.
"""
import os

os.environ["MUJOCO_GL"] = "disable"  # This diagnostic uses MJWarp rendering.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import gc
import hashlib
import json
from dataclasses import fields
from itertools import zip_longest
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from abc_minimal.config import ClipConfig, DiTConfig, SimEvalConfig
from abc_minimal.determinism import configure_determinism
from abc_minimal.eval_policy import (
    _make_env, build_summary, jsonable, resolve_prefix_length, resolve_prompt, resolved_physics,
    rollout_worlds,
)
from abc_minimal.policy import DiTInferencePolicy
from scripts.find_xml_policy_failure_case import build_report
from scripts.replay_eval_failure_case import RecordedScene, WorldNoise


MODEL_FIELDS = ("body_pos", "body_quat", "cam_pos", "cam_quat", "geom_contype", "geom_conaffinity")
VIDEO_OPTIONS = dict(codec="libx264rgb", pixelformat="rgb24", quality=None,
                     macro_block_size=1, output_params=["-crf", "0", "-preset", "fast"])


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def fingerprints(arrays):
    return {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for key, value in arrays.items()}


class XMLTrace(RecordedScene):
    """Use the normal production reset, checking the saved initial scene exactly."""
    def __init__(self, env, record, folder):
        super().__init__(env, record, "xml", "eef_only")
        self.folder = folder
        self.integration = []

    def reset(self, seed, options=None):
        obs = self.env.reset(seed=seed, options=options)
        if jsonable(self.env.randomization) != self.record["randomization"]:
            raise ValueError("Reset scene differs from the historical randomization")
        native = self.env.env
        self.initial_arrays = {key: np.array(getattr(native.model, key)) for key in MODEL_FIELDS}
        self.initial_arrays.update(qpos=native.data.qpos.copy(), qvel=native.data.qvel.copy(),
                                   ctrl=native.data.ctrl.copy(),
                                   top_camera_world=np.r_[native.data.camera("top").xpos,
                                                          native.data.camera("top").xmat])
        for key, pixels in obs["images"].items():
            Image.fromarray(pixels.transpose(1, 2, 0)).save(self.folder / f"policy_input_{key}.png")
        self.snapshot()
        self.initial_arrays["integration"] = self.integration[0].copy()
        return obs

    def snapshot(self):
        super().snapshot()
        native = self.env.env
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(native.model, spec))
        mujoco.mj_getState(native.model, native.data, state, spec)
        self.integration.append(state)

    step_one_vanilla = RecordedScene.step_one


def render_top(env, folder, stride):
    native = env.env.env
    dt = native._physics_dt * native._control_decimation
    fps = 1 / (dt * stride)
    indices = list(range(0, len(env.integration), stride))
    if indices[-1] != len(env.integration) - 1:
        indices.append(len(env.integration) - 1)
    path = folder / "top.mp4"
    with imageio.get_writer(str(path), fps=fps, **VIDEO_OPTIONS) as writer:
        for index in indices:
            mujoco.mj_setState(native.model, native.data, env.integration[index],
                              mujoco.mjtState.mjSTATE_INTEGRATION)
            native.cur_step = index
            mujoco.mj_forward(native.model, native.data)
            frame = native.get_obs()["images"]["top"].transpose(1, 2, 0).copy()
            writer.append_data(frame)
    return path, fps


def run_one(side, report, output, gpu_id, stride):
    saved = report["selected_configs"][side]
    selected = report["selected"]
    folder = output / side
    folder.mkdir()
    keys = {item.name for item in fields(SimEvalConfig)} - {"model", "clip", "vla_model"}
    config = SimEvalConfig(**{key: value for key, value in saved.items() if key in keys},
                           model=DiTConfig(**saved.get("model", {})),
                           clip=ClipConfig(**saved.get("clip", {})))
    if config.camera_backend != "mjwarp" or not config.deterministic_inference or config.fast_inference:
        raise ValueError("Expected the original deterministic MJWarp-camera protocol")
    config.checkpoint = selected[side]["checkpoint"]
    config.num_worlds = 1
    config.seed = selected["world_seed"]
    config.gpu_id = gpu_id
    config.output_dir = str(folder)
    config.log_every_chunk = True
    config.save_video = False  # No extra renders during policy inference.
    config.prompt = resolve_prompt(config)
    configure_determinism(config.policy_seed)
    torch.cuda.set_device(gpu_id)
    print(f"{side}: checkpoint={config.checkpoint}; policy_seed={config.policy_seed}; "
          f"source_world={selected['world_index']:03d}; env_seed={config.seed}", flush=True)
    policy = DiTInferencePolicy(Path(config.checkpoint), config, f"cuda:{gpu_id}")
    policy = WorldNoise(policy, config.policy_seed, selected["world_index"])
    prefix = resolve_prefix_length(config, policy.trained_max_prefix, config.model)
    config.prefix_length = prefix
    env = XMLTrace(_make_env(config, tuple(config.model.camera_keys)),
                   report["selected_world_records"][side], folder)
    try:
        if resolved_physics(env.env) != report["selected_physics"][side]:
            raise ValueError("Current physics settings differ from the historical evaluation")
        request = config.randomization
        options = {"randomization": json.loads(request) if isinstance(request, str) else request}
        worlds = rollout_worlds(config, policy, env, prefix, options, folder, config.model)
        world = worlds[0]
        world["world_index"] = selected["world_index"]
        np.savez_compressed(folder / "trajectory.npz", qpos=np.array(env.qpos),
                            qvel=np.array(env.qvel), time=np.array(env.times),
                            integration=np.array(env.integration), actions=np.array(env.actions),
                            diffusion_noise=np.array(policy.noise), predictions=np.array(policy.predictions))
        np.savez(folder / "initial.npz", **env.initial_arrays)
        write_json(folder / "observation_hashes.json", policy.observations)
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        video, fps = render_top(env, folder, stride)
        world["video_path"] = str(video)
        build_summary(config=config, ckpt_path=Path(config.checkpoint), device=f"cuda:{gpu_id}",
                      worlds=worlds, out_dir=folder, physics=resolved_physics(env.env))
        return dict(success=world["success"], historical_success=selected[side]["success"],
                    max_reward=world["max_reward"], steps=world["steps"],
                    historical_max_reward=selected[side].get("max_reward"),
                    historical_steps=selected[side].get("steps"),
                    initial_hashes=fingerprints(env.initial_arrays), video=str(video), fps=fps)
    finally:
        env.env.close()


def comparison_video(results, output):
    readers = {side: imageio.get_reader(results[side]["video"]) for side in ("old", "new")}
    path = output / "comparison_top.mp4"
    last = {}
    try:
        with imageio.get_writer(str(path), fps=results["old"]["fps"], **VIDEO_OPTIONS) as writer:
            for pair in zip_longest(readers["old"], readers["new"]):
                for side, frame in zip(("old", "new"), pair):
                    if frame is not None:
                        last[side] = frame
                height, width = last["old"].shape[:2]
                canvas = Image.new("RGB", (width * 2, height + 40), "#171b20")
                draw = ImageDraw.Draw(canvas)
                for column, side in enumerate(("old", "new")):
                    canvas.paste(Image.fromarray(last[side]), (column * width, 40))
                    outcome = "SUCCESS" if results[side]["success"] else "FAIL"
                    draw.text((column * width + 12, 12), f"{side.upper()} XML 100k | {outcome}",
                              fill="#8be6a0" if results[side]["success"] else "#ff9d89")
                writer.append_data(np.array(canvas))
    finally:
        for reader in readers.values():
            reader.close()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/afs/L202500703/abc_storage/output"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.render_stride < 1:
        parser.error("render-stride must be positive")
    report = json.loads(args.selection.read_text()) if args.selection else build_report(args.root)
    if report["status"] != "found":
        raise ValueError("No matching old-success/new-failure case in the saved summaries")
    if not 0 <= args.gpu_id < torch.cuda.device_count():
        raise ValueError(f"Unavailable GPU {args.gpu_id}; visible count={torch.cuda.device_count()}")
    for side in ("old", "new"):
        if not Path(report["selected"][side]["checkpoint"]).is_file():
            raise FileNotFoundError(report["selected"][side]["checkpoint"])
    if any((args.output / name).exists()
           for name in ("old", "new", "selection.json", "summary.json", "progress.json")):
        raise FileExistsError(f"Refusing to overwrite a previous diagnostic: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "selection.json", report)
    results = {}
    for side in ("old", "new"):
        results[side] = run_one(side, report, args.output, args.gpu_id, args.render_stride)
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.output / "progress.json", results)
    if results["old"]["initial_hashes"] != results["new"]["initial_hashes"]:
        raise ValueError("Old/new initial robot, objects, cameras or collision settings differ")
    reproduced = all(row["success"] == row["historical_success"] for row in results.values())
    result = dict(status="reproduced" if reproduced else "outcome_mismatch",
                  historical_outcomes_reproduced=reproduced,
                  policy_seed=report["selected"]["policy_seed"],
                  world_index=report["selected"]["world_index"],
                  world_seed=report["selected"]["world_seed"],
                  initial_states_verified_equal=True, results=results,
                  comparison_video=str(comparison_video(results, args.output)))
    write_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not reproduced:
        raise SystemExit("Replay outcomes differ from history; videos show the actual replay outcomes.")


if __name__ == "__main__":
    main()
