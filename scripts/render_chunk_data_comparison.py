"""Export old/new images, states and actions for paired loss-example chunks.

Recorded images use the production training decoder and clean preprocessing.
Export the first conditioning image and all 30 frames of saved state/action.
"""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from abc_minimal.config import DiTConfig
from abc_minimal.dataloader import decode_frame, read_state_action_rows
from abc_minimal.preprocess import NORM_PRESETS, augment_and_normalize, normalize, preset_for_backbone
import torch
from abc_minimal.preprocess import parse_norm_stats


DEFAULT_CHUNKS = (
    "episode_298becaa-b397-4766-9c80-6fb74e182f4b:1336",
    "episode_5d3515a9-3f0d-4275-a5bd-1daa9db9c78a:1076",
)
CELLS = ("old_on_old", "new_on_old", "old_on_new", "new_on_new")
JOINT_NAMES = [f"{side}_{'gripper' if j == 6 else 'j'+str(j+1)}"
               for side in ("left", "right") for j in range(7)]


def pixels(tensor):
    return (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def panel_grid(panels, columns):
    """Labelled panels; preserve each panel's original pixels and resolution."""
    width = max(image.shape[1] for _, image in panels)
    height = max(image.shape[0] for _, image in panels)
    canvas = Image.new("RGB", (columns * width, ((len(panels)+columns-1)//columns)*(height+28)), "#202020")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x, y = (index % columns)*width, (index // columns)*(height+28)
        draw.text((x+6, y+6), label, fill="white")
        canvas.paste(Image.fromarray(image), (x, y+28))
    return np.asarray(canvas)


def decode_side(row, side, metadata, frame):
    folder = Path(row[f"{side}_episode_path"])
    model = DiTConfig(**metadata["train"]["model"])
    saved = json.loads((folder / "episode_metadata.json").read_text())
    length = (folder / "states_actions.bin").stat().st_size // (28 * 8)
    local = row[f"{side}_frame"] + frame
    raw = decode_frame(folder, local, length, saved.get("cameras") or model.camera_keys, model.camera_keys)
    preset = preset_for_backbone(model.vision_backbone)
    network = augment_and_normalize(raw, False, norm_preset=preset)
    mean, std = NORM_PRESETS[preset]
    previews = {cam: pixels(image * (std+1e-6) + mean) for cam, image in network.items()}
    return raw, network, previews


def plot_joints(folder, frames, arrays):
    for stream in ("state", "action"):
        figure, axes = plt.subplots(7, 2, figsize=(14, 17), sharex=True, constrained_layout=True)
        for arm, side in enumerate(("left", "right")):
            for joint in range(7):
                axis = axes[joint, arm]
                column = arm*7+joint
                for version, color in (("old", "tab:blue"), ("new", "tab:orange")):
                    axis.plot(frames, arrays[f"{version}_{stream}"][:, column],
                              label=version, color=color, linestyle="-" if version == "old" else "--")
                axis.set_ylabel("opening fraction" if joint == 6 else "rad")
                axis.set_title(JOINT_NAMES[column])
                axis.grid(alpha=.25)
                axis.legend()
        for axis in axes[-1]:
            axis.set_xlabel("Original source frame")
        figure.suptitle(f"{stream}: saved old/new data; policy state condition uses the first frame")
        figure.savefig(folder / f"{stream}_joints.png", dpi=150)
        plt.close(figure)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checkpoint_metadata(path):
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return dict(train=saved["train_config"], stats=parse_norm_stats(saved["norm_stats"]))


def export_chunk(row, source_index, metadata, source, output):
    first, count = row["source_frame"], row["chunk_length"]
    folder = output / f"{row['source_episode']}_frames_{first}_{first+count-1}"
    folder.mkdir()
    arrays, information = {}, {}
    cameras = tuple(DiTConfig(**metadata["old"]["train"]["model"]).camera_keys)
    for side in ("old", "new"):
        destination = folder / side
        destination.mkdir()
        episode = Path(row[f"{side}_episode_path"])
        local = row[f"{side}_frame"]
        values = read_state_action_rows(episode, local, local+count,
                                       DiTConfig(**metadata[side]["train"]["model"]))
        if values.shape != (count, 28) or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved state/action data: {episode}")
        for stream, columns in (("state", slice(0, 14)), ("action", slice(14, 28))):
            arrays[f"{side}_{stream}"] = values[:, columns].copy()
            arrays[f"{side}_{stream}_normalized"] = normalize(values[:, columns], metadata[side]["stats"]["state" if stream == "state" else "actions"]).astype(np.float32)
        saved = json.loads((episode / "episode_metadata.json").read_text())
        alias = "top"
        source_cameras = saved.get("cameras") or cameras
        if "top" not in source_cameras and {"top_left", "top_right"} <= set(source_cameras):
            alias = "top_left" if hashlib.sha1(episode.name.encode()).digest()[0] % 2 == 0 else "top_right"
        information[side] = dict(episode_path=str(episode), local_start=local, selected_top_camera=alias,
                                 source_cameras=list(source_cameras))
    frames = np.arange(first, first+count)
    keys = ("old_state", "new_state", "old_action", "new_action")
    with (folder / "state_action.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_frame"] + [f"{key}_{joint}" for key in keys for joint in JOINT_NAMES])
        writer.writerows([int(frame), *np.concatenate([arrays[key][i] for key in keys])]
                         for i, frame in enumerate(frames))
    plot_joints(folder, frames, arrays)
    image_differences = {}
    panels, native = [], {}
    for side in ("old", "new"):
        raw, network, previews = decode_side(row, side, metadata[side], 0)
        native[side] = network
        for cam in cameras:
            panels.append((f"{side} / {cam} / source frame {first}", previews[cam]))
            Image.fromarray(pixels(raw[cam])).save(folder / side / f"{cam}_raw.png")
            Image.fromarray(previews[cam]).save(folder / side / f"{cam}_policy_input.png")
            arrays[f"{side}_image_{cam}_normalized"] = network[cam].numpy()
    Image.fromarray(panel_grid(panels, len(cameras))).save(folder / "policy_images.png")
    for cam in cameras:
        a, b = native["old"][cam].numpy(), native["new"][cam].numpy()
        image_differences[cam] = dict(exact_equal=bool(np.array_equal(a, b)),
                                     mean_abs_normalized_difference=float(np.abs(a-b).mean()))
    np.savez(folder / "data.npz", **arrays)
    with np.load(source / "screen_draws.npz", allow_pickle=False) as archive:
        losses = {cell: float(archive[cell][source_index, 0].mean()) for cell in CELLS}
    differences = {}
    for stream in ("state", "action"):
        delta = arrays[f"new_{stream}"] - arrays[f"old_{stream}"]
        differences[stream] = dict(max_joint_difference_rad=float(np.abs(delta[:, [*range(6), *range(7,13)]]).max()),
                                   max_gripper_difference=float(np.abs(delta[:, [6,13]]).max()))
    report = dict(chunk=row, data_sources=information, single_draw_losses=losses,
                  gap=losses["new_on_new"]-losses["old_on_old"],
                  image_differences=image_differences, data_differences=differences,
                  notes=["Images use the native training decoder and clean preprocessing; PNG encoding is lossless.",
                         "data.npz preserves exact normalized image tensors, original joints and normalized joints.",
                         "Policy conditions use only the first image/state; full 30-frame data is provided for context.",
                         "No joint clipping, IK recomputation, policy generation or physics stepping."])
    save_json(folder / "report.json", report)
    print(json.dumps(dict(output=str(folder), losses=losses, image_differences=image_differences,
                          data_differences=differences), indent=2), flush=True)
    return str(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-run", type=Path, required=True)
    parser.add_argument("--chunks", nargs="+", default=list(DEFAULT_CHUNKS), help="Original episode:start_frame")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        parser.error("Use a new output directory")
    source = json.loads((args.loss_run / "input.json").read_text())
    locations = json.loads((args.loss_run / "sampled_chunks.json").read_text())
    selections = []
    for request in args.chunks:
        episode, frame = request.rsplit(":", 1)
        matches = [(i, row) for i, row in enumerate(locations)
                   if row["source_episode"] == episode and row["source_frame"] == int(frame)]
        if len(matches) != 1:
            raise ValueError(f"Expected one saved chunk for {request}, found {len(matches)}")
        selections.append(matches[0])
    metadata = {side: checkpoint_metadata(Path(source[f"{side}_checkpoint"])) for side in ("old", "new")}
    args.output.mkdir(parents=True, exist_ok=True)
    print("Exporting saved images and state/action curves...", flush=True)
    folders = [export_chunk(row, index, metadata, args.loss_run, args.output)
               for index, row in selections]
    save_json(args.output / "manifest.json", dict(status="complete", loss_run=str(args.loss_run), chunks=folders))


if __name__ == "__main__":
    main()
