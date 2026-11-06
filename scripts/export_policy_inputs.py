"""Export PNGs of a saved policy observation and verify its recorded hashes.

No policy inference, video decoding/re-encoding for the images, or physics
stepping. Re-render the exact recorded observation step and require every raw
image and state hash to match before publishing the diagnostic preview.
"""
import os

os.environ.setdefault("MUJOCO_GL", "disable")

import argparse
from dataclasses import fields
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from abc_minimal.config import ClipConfig, DiTConfig, SimEvalConfig
from abc_minimal.eval_policy import _make_env
from abc_minimal.preprocess import preset_for_backbone, resize_pad_normalize, resize_pad_raw
from scripts.replay_eval_failure_case import RecordedScene, digest


MODEL_FIELDS = ("body_pos", "body_quat", "cam_pos", "cam_quat", "geom_contype", "geom_conaffinity")


def load_json(path):
    return json.loads(path.read_text())


def observation_steps(config, count):
    execute = config.execute_chunk_dim
    if config.rtc:
        lead = execute - config.rtc_inference_lead_steps
        return np.array([0] + [lead + execute * index for index in range(count - 1)])
    return np.arange(count) * execute


def input_sheet(images, network_images, side):
    names = list(images)
    width = max(224, max(image.shape[2] for image in images.values()))
    raw_height = max(image.shape[1] for image in images.values())
    canvas = Image.new("RGB", (len(names) * width, 64 + raw_height + 224), "#202020")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(names):
        x = column * width
        raw = images[name].transpose(1, 2, 0)
        draw.text((x + 4, 8), f"{side.upper()} / {name}: raw {raw.shape[1]}x{raw.shape[0]}", fill="white")
        canvas.paste(Image.fromarray(raw), (x, 32))
        draw.text((x + 4, 40 + raw_height), "Model image: 224x224 before normalization", fill="white")
        canvas.paste(Image.fromarray(network_images[name]), (x, 64 + raw_height))
    return canvas


def export_side(args, side, scene):
    folder = args.case_dir / side
    summary = load_json(folder / "summary.json")
    result = load_json(folder / "result.json")
    observations = load_json(folder / "observations.json")
    if not observations:
        raise ValueError(f"{side}: no saved policy observations")
    values = summary["config"].copy()
    values["model"], values["clip"] = DiTConfig(**values["model"]), ClipConfig(**values["clip"])
    allowed = {field.name for field in fields(SimEvalConfig)}
    config = SimEvalConfig(**{key: value for key, value in values.items() if key in allowed})
    config.gpu_id = args.gpu_id
    steps = observation_steps(config, len(observations))
    with np.load(folder / "trajectory.npz") as trace:
        if args.observation_index is None:
            video = args.case_dir / "comparison_top.mp4"
            if not video.exists():
                video = folder / "rollout.mp4"
            with imageio.get_reader(str(video)) as reader:
                fps = float(reader.get_meta_data()["fps"])
            manifest = args.case_dir.parent / "manifest.json"
            stride = load_json(manifest)["config"]["render_stride"] if manifest.exists() else 2
            video_frame = int(np.floor(args.video_time * fps))
            frames = list(range(0, len(trace["qpos"]), stride))
            if frames[-1] != len(trace["qpos"]) - 1:
                frames.append(len(trace["qpos"]) - 1)
            if video_frame >= len(frames):
                raise ValueError(f"{side}: this video time is after the rollout; the displayed image is a held final frame")
            target_step = frames[video_frame]
            index = int(np.searchsorted(steps, target_step, side="right") - 1)
        else:
            index = args.observation_index
        if not 0 <= index < len(observations):
            raise ValueError(f"{side}: observation index is outside the saved observations")
        step = int(steps[index])
        if not 0 <= step < len(trace["qpos"]):
            raise ValueError(f"{side}: inferred observation step is outside the trajectory")
        qpos, qvel, timestamp = trace["qpos"][step], trace["qvel"][step], float(trace["time"][step])
        integration = trace["integration"][step] if "integration" in trace else None
    env = RecordedScene(_make_env(config, tuple(config.model.camera_keys)), scene,
                        result["base"], result["collision_mode"])
    try:
        env.reset(seed=summary["worlds"][0]["world_seed"])
        native = env.env.env
        with np.load(folder / "initial.npz") as initial:
            for name in MODEL_FIELDS:
                np.testing.assert_allclose(getattr(native.model, name), initial[name], atol=1e-12, rtol=0)
        if integration is not None:
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            if len(integration) != mujoco.mj_stateSize(native.model, spec):
                raise ValueError("Saved integration state layout differs from this MuJoCo version")
            mujoco.mj_setState(native.model, native.data, integration, spec)
        else:
            native.data.qpos[:], native.data.qvel[:], native.data.time = qpos, qvel, timestamp
        native.cur_step = step
        mujoco.mj_forward(native.model, native.data)
        obs = env.env.obs()
        expected = observations[index]
        actual = dict(state=digest(obs["state"]), images={name: digest(value) for name, value in obs["images"].items()})
        if actual != expected:
            raise ValueError(f"{side}: re-rendered observation hash differs; it cannot be labeled the original policy input. "
                             f"observation={index}, step={step}, expected={expected}, actual={actual}")
        output = args.output / side
        output.mkdir(parents=True, exist_ok=True)
        tensors, previews = {}, {}
        for name, raw in obs["images"].items():
            pixels = raw.transpose(1, 2, 0)
            path = output / f"{name}_raw.png"
            Image.fromarray(pixels).save(path)
            np.testing.assert_array_equal(np.asarray(Image.open(path)), pixels)
            tensor = resize_pad_normalize(raw, preset=preset_for_backbone(config.model.vision_backbone)).numpy()
            preview = resize_pad_raw(raw).numpy().transpose(1, 2, 0)
            preview = np.rint(np.clip(preview, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(preview).save(output / f"{name}_model_224.png")
            tensors[name], previews[name] = tensor, preview
        np.savez_compressed(output / "model_image_tensors.npz", **tensors)
        sheet = input_sheet(obs["images"], previews, side)
        sheet.save(output / "policy_input.png")
        report = dict(side=side, base=result["base"], checkpoint=summary["checkpoint"],
                      observation_index_zero_based=index, physics_step=step,
                      simulation_time_s=timestamp, requested_video_time_s=args.video_time,
                      selection="Latest actual policy observation at or before the requested video time",
                      original_raw_state_and_image_hashes_verified=True, hashes=actual,
                      normalized_image_tensor_hashes={name: hashlib.sha256(value.tobytes()).hexdigest()
                                                     for name, value in tensors.items()},
                      sheet=str(output / "policy_input.png"))
        return report, sheet
    finally:
        env.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--video-time", type=float, default=12.)
    parser.add_argument("--observation-index", type=int, help="Zero-based policy inference index; overrides video-time")
    parser.add_argument("--side", choices=("old", "new", "both"), default="both")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.video_time < 0 or (args.observation_index is not None and args.observation_index < 0):
        parser.error("Time and observation index must be nonnegative")
    args.case_dir = args.case_dir.resolve()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    args.output = (args.output or args.case_dir / f"policy_inputs_{stamp}").resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = load_json(args.case_dir / "scene.json")
    reports, sheets = {}, []
    for side in (("old", "new") if args.side == "both" else (args.side,)):
        reports[side], sheet = export_side(args, side, scene)
        sheets.append(sheet)
        print(json.dumps(reports[side], indent=2), flush=True)
    combined = Image.new("RGB", (sum(sheet.width for sheet in sheets), max(sheet.height for sheet in sheets)), "#202020")
    offset = 0
    for sheet in sheets:
        combined.paste(sheet, (offset, 0))
        offset += sheet.width
    combined.save(args.output / "policy_inputs.png")
    (args.output / "verification.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"Verified lossless input images: {args.output / 'policy_inputs.png'}", flush=True)


if __name__ == "__main__":
    main()
