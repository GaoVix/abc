"""Export complete final three-stage XML IK, without the pose-quality filter.

Reuse the frozen exporter and full trajectories in the existing upload bundle.
This does not solve IK or launch training. Original sim splits are linked as-is.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from functools import lru_cache, partial
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time


@lru_cache(maxsize=1)
def exporter(bundle):
    code = bundle / "code"
    path = code / "scripts/export_sequence_ik.py"
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location("_unfiltered_frozen_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def export_one(job, args, settings):
    # Infinity is used only in memory: finite pose errors cannot fail this
    # comparison. Metadata stores null tolerances and pose_filter=False.
    # The frozen exporter still excludes missing/NaN values and index gaps.
    args.position_tolerance_mm = float("inf")
    args.rotation_tolerance_deg = float("inf")
    return exporter(args.bundle).export_episode(job, args, settings)


def original_sim(source):
    rows = []
    for split in ("train_sim", "val_sim"):
        for episode in sorted((source / split).glob("episode_*")):
            binary = episode / "states_actions.bin"
            size = binary.stat().st_size
            if size % (28 * 8):
                raise ValueError(f"Invalid original sim binary: {binary}")
            frames = size // (28 * 8)
            meta = json.loads((episode / "episode_metadata.json").read_text())
            if (frames < 30 or meta["num_steps"] != frames or "sequence_ik" in meta
                    or "joint_conversion" in meta
                    or not (episode / "combined_camera-images-rgb.mp4").is_file()):
                raise ValueError(f"Expected complete, original sim episode: {episode}")
            rows.append((episode.name, frames))
    if len(rows) != len({name for name, _ in rows}) or len(rows) != 3253:
        raise ValueError("Expected 3253 unique original sim sequences")
    frames = sum(n for _, n in rows)
    chunks = sum(n - 29 for _, n in rows)
    if (frames, chunks) != (3952729, 3858392):
        raise ValueError(f"Original sim inventory changed: {frames} frames / {chunks} chunks")
    return dict(frames=frames, retained_frames=frames, segments=len(rows),
                original_chunks=chunks, usable_chunks=chunks, usable_chunk_percent=100.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Existing abc_xml_default_real_ik_raw_sim_20261103 bundle")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Original official dataset, before any IK conversion/filtering")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-jump-filter", action="store_true",
                        help="Still split at raw joint changes >30 degrees; default disables this")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episodes", nargs="+", help="Pilot episode IDs/prefixes; never a full training set")
    args = parser.parse_args()
    for name in ("bundle", "source_root", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.workers < 1:
        parser.error("--workers must be positive")
    for source in (args.bundle, args.source_root):
        if args.output.is_relative_to(source) or source.is_relative_to(args.output):
            raise ValueError("Use a separate new output directory, outside the source and bundle")
    args.base_mode = "xml"
    args.chunk_length = 30
    args.jump_threshold_deg = 30. if args.keep_jump_filter else 0.
    mode = "no_pose_filter_keep_jump" if args.keep_jump_filter else "finite_only"
    expected = (dict(retained_frames=1751696, segments=1415, usable_chunks=1710661)
                if args.keep_jump_filter else
                dict(retained_frames=1753897, segments=1262, usable_chunks=1717299))
    frozen = args.bundle / "code/scripts/export_sequence_ik.py"
    settings = dict(
        format_version=1, bundle=str(args.bundle), source_root=str(args.source_root),
        base_mode="xml", sim_handling="original", filter_mode=mode,
        filters=dict(pose_filter=False, position_tolerance_mm=None, rotation_tolerance_deg=None,
                     jump_threshold_deg=args.jump_threshold_deg, chunk_length=30),
        frozen_exporter_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        episodes=args.episodes,
    )
    action = {p.parent.name: p for p in (args.bundle / "action_xml/real").glob("*/trajectory.npz")}
    state = {p.parent.name: p for p in (args.bundle / "state_xml/real").glob("*/trajectory.npz")}
    if action.keys() != state.keys() or (not args.episodes and len(action) != 1261):
        raise ValueError("Expected matching complete final state/action results for 1261 real sequences")
    sources = {}
    for split in ("train_real", "val_real"):
        for episode in (args.source_root / split).glob("episode_*"):
            if (episode / "states_actions.bin").is_file():
                key = "real_" + episode.name
                if key in sources:
                    raise ValueError(f"Duplicate source sequence across train/val: {key}")
                sources[key] = (split, episode)
    keys = sorted(action)
    if args.episodes:
        keys = [key for key in keys if any(part in key for part in args.episodes)]
    if not keys or any(key not in sources for key in keys):
        raise ValueError("Missing original real sequences or empty pilot selection")
    sim = None if args.episodes else original_sim(args.source_root)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "export.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = args.output / "config.json"
        if config.exists():
            if json.loads(config.read_text()) != settings:
                raise ValueError("Existing output uses different settings; choose a new directory")
        elif any(p.name != "export.lock" for p in args.output.iterdir()):
            raise ValueError("Output is not empty; choose a new directory")
        write_json(config, settings)
        jobs = [(key, *sources[key], action[key], state[key]) for key in keys]
        reports, started = [], time.monotonic()
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for row in pool.map(partial(export_one, args=args, settings=settings), jobs, chunksize=1):
                reports.append(row)
                print(json.dumps(dict(completed=len(reports), selected=len(jobs), **row)), flush=True)
        failed = sum(r["status"] != "complete" for r in reports)
        totals = {k: sum(r.get(k, 0) for r in reports) for k in
                  ("frames", "retained_frames", "segments", "original_chunks", "usable_chunks")}
        totals["usable_chunk_percent"] = 100 * totals["usable_chunks"] / max(1, totals["original_chunks"])
        real_summary = dict(status="failed" if failed else "complete", selected=len(keys), failed=failed,
                            scope="pilot" if args.episodes else "full", dry_run=args.dry_run,
                            filter_mode=mode, totals=totals, episodes=reports,
                            seconds=time.monotonic() - started)
        write_json(args.output / ("dry_run_summary.json" if args.dry_run else "real_export_summary.json"), real_summary)
        if failed:
            raise SystemExit("Export failed; inspect per-episode errors before training")
        if args.episodes:
            print("Pilot only; no complete training dataset was created", flush=True)
            return
        if totals["frames"] != 1753906 or totals["original_chunks"] != 1717337:
            raise ValueError("Unexpected real sequence/frame inventory")
        if any(totals[k] != v for k, v in expected.items()):
            raise ValueError(f"Export differs from verified full-data counts: {totals}")
        if args.dry_run:
            print(json.dumps({k: v for k, v in real_summary.items() if k != "episodes"}, indent=2))
            return
        for split in ("train_sim", "val_sim"):
            destination = args.output / split
            if destination.is_symlink():
                if destination.resolve() != (args.source_root / split).resolve():
                    raise ValueError(f"Wrong existing sim link: {destination}")
            elif destination.exists():
                raise ValueError(f"Refusing to replace an existing sim directory: {destination}")
            else:
                destination.symlink_to(args.source_root / split, target_is_directory=True)
        stats = args.source_root / "norm_stats.json"
        if stats.is_file():
            target = args.output / "norm_stats.json"
            if target.exists() and target.read_bytes() != stats.read_bytes():
                raise ValueError("Existing norm_stats differs from the original source")
            if not target.exists():
                shutil.copy2(stats, target)
        combined = {k: totals[k] + sim[k] for k in
                    ("frames", "retained_frames", "segments", "original_chunks", "usable_chunks")}
        combined["usable_chunk_percent"] = 100 * combined["usable_chunks"] / combined["original_chunks"]
        summary = dict(status="complete", selected=4514, failed=0, dry_run=False,
                       base_mode="xml", sim_handling="original", filter_mode=mode,
                       source_root=str(args.source_root), real_export_summary="real_export_summary.json",
                       totals=combined, totals_by_kind=dict(real=totals, sim=sim))
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
