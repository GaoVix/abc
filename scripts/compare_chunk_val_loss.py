"""Compare old/new policies on both processed datasets, paired by original frames.

Uses the official clean, unprefixed validation objective. Screening and paired
confirmation use independent noise/time draws. No training updates or rollouts.
Primary selection compares each model on its own data; cross-evaluations are diagnostic.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import torch
from torch.utils.data import DataLoader, Dataset

from abc_minimal.config import ClipConfig, DiTConfig, PromptConfig
from abc_minimal.dataloader import EpisodeDataset, collate
from abc_minimal.determinism import configure_determinism
from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained
from abc_minimal.operator import load_operator_label_maps
from abc_minimal.preprocess import parse_norm_stats
from abc_minimal.train_loop import batch_to_device


SERVER = Path("/mnt/afs/L202500703/abc_storage")
CELLS = ("old_on_old", "new_on_old", "old_on_new", "new_on_new")
COMPARISONS = {
    "on_old_data": ("old_on_old", "new_on_old"),
    "on_new_data": ("old_on_new", "new_on_new"),
    "own_data": ("old_on_old", "new_on_new"),
}


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_metadata(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "train_config" not in checkpoint or "norm_stats" not in checkpoint:
        raise ValueError(f"Missing training config or normalization metadata: {path}")
    return dict(step=int(checkpoint["global_step"]), train=checkpoint["train_config"],
                stats=parse_norm_stats(checkpoint["norm_stats"]))


class ChunkSamples(Dataset):
    def __init__(self, datasets, locations):
        self.datasets = datasets
        self.locations = locations

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        location = self.locations[index]
        samples = {side: datasets[location["split"]][location[f"{side}_dataset_index"]]
                   for side, datasets in self.datasets.items()}
        return samples, location["sample_id"]


def collate_samples(samples, cameras):
    rows, identifiers = zip(*samples)
    return {side: collate([row[side] for row in rows], cameras) for side in ("old", "new")}, list(identifiers)


def source_spans(dataset):
    """Valid chunk-start ranges in original frames, without expanding every frame."""
    sources = {}
    for number, (folder, length, usable, *_) in enumerate(dataset.episodes):
        metadata = json.loads((folder / "episode_metadata.json").read_text())
        origin = metadata.get("sequence_ik", {}).get("source_episode", folder.name)
        origin = Path(origin).name
        begin, end = metadata.get("source_frame_range", [0, length])
        if begin < 0 or end - begin != length:
            raise ValueError(f"Cannot map contiguous original frames for {folder}")
        sources.setdefault(origin, []).append(dict(
            begin=int(begin), end=int(begin + usable), episode=folder.name,
            episode_path=str(folder), dataset_start=int(dataset.cum[number - 1]) if number else 0,
            metadata=metadata))
    for origin, spans in sources.items():
        spans.sort(key=lambda span: span["begin"])
        if any(a["end"] > b["begin"] for a, b in zip(spans, spans[1:])):
            raise ValueError(f"Ambiguous overlapping source chunks: {origin}")
    return sources


def sample_locations(datasets, per_split, seed):
    locations, coverage = [], {}
    for split_number, split in enumerate(datasets["new"]):
        old, new = (datasets[side][split] for side in ("old", "new"))
        sources = {side: source_spans(dataset) for side, dataset in (("old", old), ("new", new))}
        intervals = []
        for origin in sorted(sources["old"].keys() & sources["new"].keys()):
            old_spans, new_spans = sources["old"][origin], sources["new"][origin]
            i = j = 0
            while i < len(old_spans) and j < len(new_spans):
                a, b = old_spans[i], new_spans[j]
                begin, end = max(a["begin"], b["begin"]), min(a["end"], b["end"])
                if begin < end:
                    intervals.append((origin, begin, end, a, b))
                if a["end"] <= b["end"]:
                    i += 1
                else:
                    j += 1
        cumulative = np.cumsum([end - begin for _, begin, end, _, _ in intervals])
        total = int(cumulative[-1]) if len(cumulative) else 0
        if not total:
            raise ValueError(f"No common original-frame chunks in {split}")
        generator = np.random.default_rng([seed, split_number])
        indices = np.sort(generator.choice(total, min(per_split, total), replace=False))
        for index in indices:
            interval = int(np.searchsorted(cumulative, index, side="right"))
            origin, begin, _, a, b = intervals[interval]
            frame = int(begin + index - (cumulative[interval - 1] if interval else 0))
            row = dict(sample_id=len(locations), split=split, source_episode=origin,
                       source_frame=frame, source_frame_last=frame + new.model_config.chunk_length - 1,
                       chunk_length=new.model_config.chunk_length)
            timestamps, clocks = [], []
            for side, span in (("old", a), ("new", b)):
                local = frame - span["begin"]
                row.update({f"{side}_episode": span["episode"], f"{side}_frame": local,
                            f"{side}_episode_path": span["episode_path"],
                            f"{side}_dataset_index": span["dataset_start"] + local})
                ticks = span["metadata"].get("top_camera_timestamps_ns")
                timestamps.append(ticks[local:local + row["chunk_length"]] if ticks is not None else None)
                meta = span["metadata"]
                clocks.append((meta["t0_ns"] + local * meta["tick_ns"], meta["tick_ns"])
                              if "t0_ns" in meta and "tick_ns" in meta else None)
            if all(ticks is not None for ticks in timestamps) and timestamps[0] != timestamps[1]:
                raise ValueError(f"Original camera timestamps differ: {origin}, frame {frame}")
            if all(clock is not None for clock in clocks) and clocks[0] != clocks[1]:
                raise ValueError(f"Original frame clocks differ: {origin}, frame {frame}")
            locations.append(row)
        coverage[split] = dict(old_chunks=len(old), new_chunks=len(new),
                               common_chunks=total, sampled_chunks=len(indices))
    return locations, coverage


def noise_and_times(identifiers, shape, draws, seed, phase, device):
    noises = np.empty((draws, len(identifiers), *shape), np.float32)
    times = np.empty((draws, len(identifiers), 1, 1), np.float32)
    for column, identifier in enumerate(identifiers):
        for draw in range(draws):
            rng = np.random.default_rng([seed, phase, identifier, draw])
            noises[draw, column] = rng.standard_normal(shape, dtype=np.float32)
            times[draw, column, 0, 0] = rng.random()
    return torch.from_numpy(noises).to(device), torch.from_numpy(times).to(device)


@torch.inference_mode()
def paired_losses(models, batches, identifiers, draws, seed, phase, verify=False):
    actions = batches["old"]["actions"]
    if actions.shape != batches["new"]["actions"].shape:
        raise ValueError("Paired action chunk shapes differ")
    noise, times = noise_and_times(identifiers, actions.shape[1:], draws, seed, phase, actions.device)
    # The images do not depend on the flow-matching noise/time: encode once.
    vision = {f"{side}_on_{data}": model.build_vision_tokens(batch["images"])
              for side, model in models.items() for data, batch in batches.items()}
    losses = {cell: [] for cell in CELLS}
    for draw in range(draws):
        t = times[draw]
        for data, batch in batches.items():
            actions = batch["actions"]
            x_t = (1 - t) * actions + t * noise[draw]
            target = noise[draw] - actions
            for side, model in models.items():
                cell = f"{side}_on_{data}"
                condition = model.compute_cond(batch["state"], batch["task_vec_clip"], t[:, 0, 0])
                prediction = model.predict_velocity(x_t, condition, vision[cell])
                per_dimension = (target - prediction).square().mean(dim=1)
                if verify and draw == 0:
                    reference = model(batch, noise=noise[draw], t=t,
                                      max_action_prefix=0, prefix_conditioning_prob=0.)
                    torch.testing.assert_close(per_dimension.mean(), reference, rtol=2e-5, atol=1e-7)
                losses[cell].append(per_dimension.cpu().numpy())
    result = {cell: np.stack(values, axis=1) for cell, values in losses.items()}
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError(f"Non-finite losses in paired samples {identifiers}")
    return result


def evaluate(models, embedder, datasets, locations, args, phase, draws):
    samples = ChunkSamples(datasets, locations)
    cameras = tuple(next(iter(datasets["new"].values())).model_config.camera_keys)
    loader = DataLoader(samples, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=partial(collate_samples, cameras=cameras), pin_memory=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        multiprocessing_context="spawn" if args.num_workers else None)
    collected = {cell: [] for cell in CELLS}
    started = time.perf_counter()
    count = 0
    for batch_index, (batches, identifiers) in enumerate(loader):
        with torch.inference_mode():
            batches = {side: batch_to_device(batch, args.device, embedder, cameras)
                       for side, batch in batches.items()}
            values = paired_losses(models, batches, identifiers, draws, args.seed, phase,
                                   verify=batch_index == 0)
        for cell in CELLS:
            collected[cell].append(values[cell])
        count += len(identifiers)
        elapsed = time.perf_counter() - started
        remaining = elapsed / count * (len(samples) - count)
        print(f"{'screen' if phase == 0 else 'confirm'}: {count}/{len(samples)} chunks; "
              f"elapsed={elapsed:.1f}s; estimated_remaining={remaining:.1f}s", flush=True)
    return {cell: np.concatenate(values) for cell, values in collected.items()}


def mean_rows(locations, losses):
    means = {cell: values.mean(axis=(1, 2)) for cell, values in losses.items()}
    rows = []
    for index, location in enumerate(locations):
        row = dict(location, **{f"{cell}_loss": float(values[index]) for cell, values in means.items()})
        for comparison, (old, new) in COMPARISONS.items():
            a, b = float(means[old][index]), float(means[new][index])
            row[f"gap_{comparison}"] = b - a
            row[f"ratio_{comparison}"] = b / max(a, 1e-12)
        rows.append(row)
    return rows


def pick_candidates(rows, count):
    selected, origins = [], set()
    for row in sorted(rows, key=lambda row: -row["gap_own_data"]):
        key = (row["split"], row["source_episode"])
        if row["gap_own_data"] <= 0 or key in origins:
            continue
        selected.append(row)
        origins.add(key)
        if len(selected) == count:
            break
    return selected


def confirm_rows(locations, losses, min_delta, min_ratio, top_k=5):
    rows = mean_rows(locations, losses)
    for index, row in enumerate(rows):
        for comparison, (old, new) in COMPARISONS.items():
            difference = (losses[new][index] - losses[old][index]).mean(axis=-1)
            standard_error = float(difference.std(ddof=1) / math.sqrt(len(difference)))
            # Independent confirmation; control candidate tests within each comparison.
            critical = float(student_t.ppf(1 - .05 / len(rows), len(difference) - 1))
            lower = float(difference.mean()) - critical * standard_error
            row[f"{comparison}_gap_standard_error"] = standard_error
            row[f"{comparison}_gap_lower_bound_fwer95"] = lower
            row[f"{comparison}_new_worse_draw_fraction"] = float((difference > 0).mean())
            row[f"{comparison}_significantly_worse"] = bool(
                lower > 0 and row[f"gap_{comparison}"] >= min_delta
                and row[f"ratio_{comparison}"] >= min_ratio)
        row["noise_time_draws"] = len(difference)
        row["significantly_worse"] = row["own_data_significantly_worse"]
        for cell in CELLS:
            dimensions = losses[cell][index].mean(axis=0)
            row[f"{cell}_left_joints_loss"] = float(dimensions[:6].mean())
            row[f"{cell}_right_joints_loss"] = float(dimensions[7:13].mean())
            row[f"{cell}_grippers_loss"] = float(dimensions[[6, 13]].mean())
    rows.sort(key=lambda row: -row["gap_own_data"])
    qualified = 0
    for rank, row in enumerate(rows, 1):
        row["primary_rank"] = rank
        qualified += int(row["significantly_worse"])
        row["is_top_chunk"] = bool(row["significantly_worse"] and qualified <= top_k)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-checkpoint", type=Path,
                        default=SERVER / "output/dit_joint_retargeted_official_timing_20261022_125331/100000.pt")
    parser.add_argument("--new-checkpoint", type=Path,
                        default=SERVER / "output/dit_xml_default_real_ik_raw_sim_20261103_112255/100000.pt")
    parser.add_argument("--old-cache-root", type=Path, help="Defaults to the old checkpoint's training cache_root")
    parser.add_argument("--new-cache-root", type=Path, help="Defaults to the new checkpoint's training cache_root")
    parser.add_argument("--splits", nargs="+", choices=("train_real", "train_sim", "val_real", "val_sim"),
                        default=["train_real", "train_sim"])
    parser.add_argument("--chunks-per-split", type=int, default=1024)
    parser.add_argument("--screen-draws", type=int, default=4)
    parser.add_argument("--confirm-draws", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=.005)
    parser.add_argument("--min-ratio", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.chunks_per_split, args.screen_draws, args.candidates, args.top_k, args.batch_size) < 1:
        parser.error("chunk/draw/candidate/batch counts must be positive")
    if args.confirm_draws < 2 or args.num_workers < 0 or args.seed < 0:
        parser.error("confirmation requires >=2 draws; workers and seed must be nonnegative")
    if args.min_delta < 0 or args.min_ratio < 1:
        parser.error("min-delta must be nonnegative and min-ratio must be >=1")
    if (args.output / "input.json").exists():
        parser.error("Use a new output directory")
    if not 0 <= args.gpu_id < torch.cuda.device_count():
        parser.error(f"GPU {args.gpu_id} unavailable; visible GPUs={torch.cuda.device_count()}")
    configure_determinism(args.seed)
    torch.cuda.set_device(args.gpu_id)
    args.device = torch.device(f"cuda:{args.gpu_id}")
    metadata = {side: checkpoint_metadata(getattr(args, f"{side}_checkpoint")) for side in ("old", "new")}
    if metadata["old"]["step"] != metadata["new"]["step"]:
        raise ValueError("Compare checkpoints at the same training step")
    architectures = {side: DiTConfig(**row["train"]["model"]) for side, row in metadata.items()}
    if asdict(architectures["old"]) != asdict(architectures["new"]):
        raise ValueError("Different architectures/camera keys: compare matching model configurations")
    for stream in ("state", "actions"):
        for key in ("mean", "std"):
            a, b = (metadata[side]["stats"][stream][key] for side in ("old", "new"))
            if not np.array_equal(a, b):
                raise ValueError(f"Normalization differs for {stream}.{key}; native losses are not comparable")
    saved = metadata["new"]["train"]
    architecture = architectures["new"]
    datasets, roots = {}, {}
    for side in ("old", "new"):
        train = metadata[side]["train"]
        root = getattr(args, f"{side}_cache_root") or Path(train["cache_root"])
        roots[side] = root.expanduser().resolve()
        if not roots[side].is_dir():
            raise ValueError(f"Dataset missing: {roots[side]}; set --{side}-cache-root to its current location")
        prompt = PromptConfig(**train.get("prompt", {}))
        labels = load_operator_label_maps(prompt.operator_label_map_path) if prompt.use_operator_id_as_prompt else {}
        datasets[side] = {}
        for split in dict.fromkeys(args.splits):
            task = ("sim_" if split.endswith("sim") else "") + "load_the_plates_into_the_dish_rack"
            datasets[side][split] = EpisodeDataset(
                roots[side] / split, metadata[side]["stats"], train=False,
                default_task_name=task, mask_state_ratio=0., model_config=architecture,
                prompt_config=prompt, operator_label_maps=labels)
    if ClipConfig(**metadata["old"]["train"].get("clip", {})).model_name != ClipConfig(**saved.get("clip", {})).model_name:
        raise ValueError("Different CLIP text models: cannot share the text embedder")
    locations, coverage = sample_locations(datasets, args.chunks_per_split, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "input.json", dict(
        old_checkpoint=str(args.old_checkpoint), new_checkpoint=str(args.new_checkpoint),
        step=metadata["old"]["step"], cache_roots={side: str(root) for side, root in roots.items()},
        pairing="Same source episode and original frame range; each model uses its own processed data",
        primary_comparison="new_on_new loss minus old_on_old loss; cross-evaluations are diagnostic only",
        coverage=coverage,
        splits=args.splits, seed=args.seed, normalization_verified_equal=True,
        screen_draws=args.screen_draws, confirm_draws=args.confirm_draws,
        min_delta=args.min_delta, min_ratio=args.min_ratio,
        objective="Official validation flow-matching MSE; clean images/state; no prefix",
        old_dino_bf16=metadata["old"]["train"].get("dino_bf16", True),
        new_dino_bf16=metadata["new"]["train"].get("dino_bf16", True)))
    save_json(args.output / "sampled_chunks.json", locations)
    print(json.dumps(dict(paired_chunks=len(locations), coverage=coverage,
                          cache_roots={side: str(root) for side, root in roots.items()}), indent=2), flush=True)
    started = time.perf_counter()
    models = {}
    for side in ("old", "new"):
        model = DiTPolicy(architectures[side])
        load_pretrained(model, getattr(args, f"{side}_checkpoint"))
        model.img_backbone.set_bfloat16(metadata[side]["train"].get("dino_bf16", True))
        models[side] = model.to(args.device).eval()
        gc.collect()
    embedder = CLIPTextEmbedder(ClipConfig(**saved.get("clip", {})), device=args.device)
    screened = evaluate(models, embedder, datasets, locations, args, phase=0, draws=args.screen_draws)
    rows = mean_rows(locations, screened)
    save_csv(args.output / "screen_chunks.csv", rows)
    np.savez_compressed(args.output / "screen_draws.npz", **screened)
    candidates = pick_candidates(rows, args.candidates)
    selected_locations = [locations[row["sample_id"]] for row in candidates]
    confirmed_rows = []
    if selected_locations:
        confirmed = evaluate(models, embedder, datasets, selected_locations, args,
                             phase=1, draws=args.confirm_draws)
        confirmed_rows = confirm_rows(selected_locations, confirmed, args.min_delta, args.min_ratio, args.top_k)
        save_csv(args.output / "confirmed_chunks.csv", confirmed_rows)
        np.savez_compressed(args.output / "confirmation_draws.npz", **confirmed,
                            sample_ids=[row["sample_id"] for row in selected_locations])
    worse = [row for row in confirmed_rows if row["significantly_worse"]]
    summary = dict(status="found" if worse else "no_confirmed_large_gap", sampled_chunks=len(locations),
                   primary_comparison="own_data",
                   confirmed_candidates=len(confirmed_rows), significantly_worse_chunks=len(worse),
                   top_chunks=worse[:args.top_k], largest_confirmed_gaps=confirmed_rows[:args.top_k],
                   comparisons={comparison: dict(
                       significantly_worse_chunks=sum(row[f"{comparison}_significantly_worse"] for row in confirmed_rows),
                       top_chunks=sorted(
                           [row for row in confirmed_rows if row[f"{comparison}_significantly_worse"]],
                           key=lambda row: -row[f"gap_{comparison}"])[:args.top_k])
                       for comparison in COMPARISONS},
                   screen_mean_losses={split: {
                       cell: float(screened[cell][[i for i, row in enumerate(locations) if row["split"] == split]].mean())
                       for cell in CELLS} for split in datasets["new"]},
                   seconds=time.perf_counter() - started,
                   coverage=coverage,
                   notes=["2x2 comparison: each model is evaluated on both processed datasets, paired by original frames.",
                          "All four cells use matching noise and flow times; normalization statistics are identical.",
                          "Screening, ranking and top_chunks use only new_on_new minus old_on_old; cross-evaluations do not select cases.",
                          "on_old_data/on_new_data compare weights on fixed inputs and labels; own_data also changes preprocessing.",
                          "Training splits use clean validation transforms; this is not held-out validation.",
                          "Each candidate comes from a different source episode.",
                          "Confirmation uses new draws; paired-t bounds have Bonferroni correction across candidates within each comparison.",
                          "Significance concerns noise/time draws for each fixed chunk, not dataset-level generalization."])
    save_json(args.output / "summary.json", summary)
    print(json.dumps(dict(
        status=summary["status"], sampled_chunks=len(locations),
        confirmed_candidates=len(confirmed_rows), seconds=summary["seconds"],
        screen_mean_losses=summary["screen_mean_losses"],
        primary_comparison="new_on_new vs old_on_old",
        top_chunks=[{key: row[key] for key in (
            "split", "source_episode", "source_frame", "source_frame_last",
            "old_on_old_loss", "new_on_new_loss", "gap_own_data", "ratio_own_data",
            "own_data_gap_lower_bound_fwer95", "own_data_new_worse_draw_fraction",
            "old_on_new_loss", "new_on_old_loss", "gap_on_old_data", "gap_on_new_data")}
            for row in worse[:args.top_k]],
        summary=str(args.output / "summary.json")), indent=2), flush=True)


if __name__ == "__main__":
    main()
