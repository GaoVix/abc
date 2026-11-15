"""Reevaluate saved paired chunks at fixed flow times, keeping draw-0 noise."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.compare_chunk_val_loss import (
    CELLS, SERVER, CLIPTextEmbedder, ClipConfig, DiTConfig, DiTPolicy,
    EpisodeDataset, PromptConfig, batch_to_device, checkpoint_metadata, collate,
    configure_determinism, load_operator_label_maps, load_pretrained,
    noise_and_times, save_csv, save_json,
)

DEFAULT_CHUNKS = [
    "episode_298becaa-b397-4766-9c80-6fb74e182f4b:1336",
    "episode_5d3515a9-3f0d-4275-a5bd-1daa9db9c78a:1076",
]


@torch.inference_mode()
def sweep_losses(models, batches, identifiers, seed, levels):
    shape = batches["old"]["actions"].shape[1:]
    if batches["new"]["actions"].shape != batches["old"]["actions"].shape:
        raise ValueError("Old/new action shapes differ")
    noise, original_times = noise_and_times(identifiers, shape, 1, seed, 0,
                                             batches["old"]["actions"].device)
    noise = noise[0]
    vision = {f"{side}_on_{data}": model.build_vision_tokens(batch["images"])
              for side, model in models.items() for data, batch in batches.items()}
    results = []
    for level in [None, *levels]:
        t = original_times[0] if level is None else torch.full_like(original_times[0], level)
        losses = {}
        for data, batch in batches.items():
            actions = batch["actions"]
            x_t, target = (1-t)*actions + t*noise, noise-actions
            for side, model in models.items():
                condition = model.compute_cond(batch["state"], batch["task_vec_clip"], t[:, 0, 0])
                prediction = model.predict_velocity(x_t, condition, vision[f"{side}_on_{data}"])
                values = (prediction-target).square().mean(dim=(1, 2))
                if level is None:
                    native = model(batch, noise=noise, t=t, max_action_prefix=0,
                                   prefix_conditioning_prob=0.)
                    torch.testing.assert_close(values.mean(), native, rtol=2e-5, atol=1e-7)
                if not torch.isfinite(values).all():
                    raise ValueError("Non-finite loss")
                losses[f"{side}_on_{data}"] = values.cpu().numpy()
        results.append(("original" if level is None else "fixed", t[:, 0, 0].cpu().numpy(), losses))
    return results, noise.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-run", type=Path,
                        default=SERVER / "output/compare_chunk_loss_20261105_125057")
    parser.add_argument("--chunks", nargs="+", default=DEFAULT_CHUNKS)
    parser.add_argument("--times", nargs="+", type=float, default=[.1, .3, .5, .7, .9])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not np.isfinite(t) or not 0 <= t <= 1 for t in args.times):
        parser.error("Each t must be finite and in [0, 1]")
    if any((args.output / name).exists() for name in ("summary.json", "loss_by_t.csv", "noise.npz")):
        parser.error("Use a new output directory")
    if not 0 <= args.gpu_id < torch.cuda.device_count():
        parser.error(f"GPU {args.gpu_id} unavailable")
    started = time.perf_counter()
    source = json.loads((args.loss_run / "input.json").read_text())
    saved = json.loads((args.loss_run / "sampled_chunks.json").read_text())
    locations, indices = [], []
    for request in args.chunks:
        episode, frame = request.rsplit(":", 1)
        matches = [(i, row) for i, row in enumerate(saved)
                   if row["source_episode"] == episode and row["source_frame"] == int(frame)]
        if len(matches) != 1:
            raise ValueError(f"Expected one saved chunk: {request}")
        index, row = matches[0]
        locations.append(row)
        indices.append(index)
    configure_determinism(source["seed"])
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"Checking {len(locations)} chunks: original t and {args.times}; fixed original draw-0 noise", flush=True)
    metadata = {side: checkpoint_metadata(Path(source[f"{side}_checkpoint"])) for side in ("old", "new")}
    configs = {side: DiTConfig(**meta["train"]["model"]) for side, meta in metadata.items()}
    if metadata["old"]["step"] != metadata["new"]["step"] or asdict(configs["old"]) != asdict(configs["new"]):
        raise ValueError("Checkpoint steps or architectures differ")
    for stream in ("state", "actions"):
        for key in ("mean", "std"):
            np.testing.assert_array_equal(metadata["old"]["stats"][stream][key], metadata["new"]["stats"][stream][key])
    text_configs = [ClipConfig(**meta["train"].get("clip", {})) for meta in metadata.values()]
    if asdict(text_configs[0]) != asdict(text_configs[1]):
        raise ValueError("Text encoder configurations differ")
    cameras = tuple(configs["old"].camera_keys)
    models, samples = {}, {}
    for side in ("old", "new"):
        print(f"Loading {side} model and saved chunks...", flush=True)
        meta = metadata[side]
        prompt = PromptConfig(**meta["train"].get("prompt", {}))
        labels = load_operator_label_maps(prompt.operator_label_map_path) if prompt.use_operator_id_as_prompt else {}
        datasets = {}
        for split in dict.fromkeys(row["split"] for row in locations):
            task = ("sim_" if split.endswith("sim") else "") + "load_the_plates_into_the_dish_rack"
            datasets[split] = EpisodeDataset(Path(source["cache_roots"][side]) / split,
                meta["stats"], train=False, mask_state_ratio=0., model_config=configs[side],
                default_task_name=task, prompt_config=prompt, operator_label_maps=labels)
        selected = []
        for row in locations:
            dataset = datasets[row["split"]]
            index = row[f"{side}_dataset_index"]
            ep = int(np.searchsorted(dataset.cum, index, side="right"))
            local = int(index - (dataset.cum[ep-1] if ep else 0))
            if (ep >= len(dataset.episodes) or dataset.episodes[ep][0].resolve() != Path(row[f"{side}_episode_path"]).resolve()
                    or local != row[f"{side}_frame"]):
                raise ValueError("Saved chunk mapping changed")
            selected.append(dataset[index])
        samples[side] = collate(selected, cameras)
        model = DiTPolicy(configs[side])
        load_pretrained(model, Path(source[f"{side}_checkpoint"]))
        model.img_backbone.set_bfloat16(meta["train"].get("dino_bf16", True))
        models[side] = model.to(device).eval()
    print("Loading text encoder and evaluating noise levels...", flush=True)
    embedder = CLIPTextEmbedder(text_configs[0], device=device)
    batches = {side: batch_to_device(batch, device, embedder, cameras) for side, batch in samples.items()}
    results, noise = sweep_losses(models, batches, [row["sample_id"] for row in locations], source["seed"], args.times)
    rows = []
    with np.load(args.loss_run / "screen_draws.npz", allow_pickle=False) as original:
        for kind, times, losses in results:
            for i, location in enumerate(locations):
                row = dict(source_episode=location["source_episode"], source_frame=location["source_frame"],
                           source_frame_last=location["source_frame"]+location["chunk_length"]-1,
                           sample_id=location["sample_id"], mode=kind, t=float(times[i]))
                row.update({f"{cell}_loss": float(losses[cell][i]) for cell in CELLS})
                row.update({f"gap_{name}": row[f"{new}_loss"]-row[f"{old}_loss"]
                            for name, old, new in (("own_data", "old_on_old", "new_on_new"),
                                                   ("on_old_data", "old_on_old", "new_on_old"),
                                                   ("on_new_data", "old_on_new", "new_on_new"))})
                row["original_replay_max_abs_difference"] = (max(abs(row[f"{cell}_loss"]-float(original[cell][indices[i], 0].mean()))
                                                               for cell in CELLS) if kind == "original" else None)
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(args.output / "loss_by_t.csv", rows)
    np.savez(args.output / "noise.npz", noise=noise, sample_ids=[row["sample_id"] for row in locations])
    save_json(args.output / "summary.json", dict(status="complete", loss_run=str(args.loss_run),
        seed=source["seed"], draw=0, fixed_times=args.times,
        noise="Original phase-0 draw-0 epsilon held fixed across all times and four model/data cells",
        old_checkpoint=source["old_checkpoint"], new_checkpoint=source["new_checkpoint"],
        results=rows, seconds=time.perf_counter()-started))
    print(f"Wrote {args.output / 'loss_by_t.csv'}", flush=True)


if __name__ == "__main__":
    main()
