"""Add original raw joints on exactly the saved IK-overlap frame pairs/triples.

Reuses the Mink/three-stage histograms, computes only raw derivatives, and saves
all raw signed float64 samples. No IK, clipping, interpolation, or angle unwrap.
"""

import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np

import plot_ik_smoothness as plots


JOINT_COLUMNS = np.array([offset + column for offset in (0, 14)
                          for columns in (range(6), range(7, 13)) for column in columns])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_arrays(path, arrays):
    temporary = path.with_suffix(".npz.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
    temporary.replace(path)


def extend_range(distributions, order, stream, arm, maximum):
    """Append shared bins, leaving every existing bin/count unchanged."""
    reference = distributions[order, stream, arm, "raw"]
    previous = reference.absolute_edges
    if maximum < previous[-1]:
        return
    steps = int(np.ceil(np.log(maximum / previous[-1]) / np.log(1.2))) + 1
    extra = np.geomspace(previous[-1], np.nextafter(maximum, np.inf), steps + 2)[1:]
    positive = np.r_[previous, extra]
    signed = np.r_[-positive[:0:-1], positive]
    for version in plots.versions_in(distributions):
        distribution = distributions[order, stream, arm, version]
        np.testing.assert_array_equal(distribution.absolute_edges, previous)
        distribution.absolute_counts = np.pad(distribution.absolute_counts, ((0, 0), (0, len(extra))))
        distribution.signed_counts = np.pad(distribution.signed_counts, ((0, 0), (len(extra), len(extra))))
        distribution.absolute_edges = positive
        distribution.signed_edges = signed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True, help="Completed comparison containing summary.json")
    parser.add_argument("--raw-root", type=Path, required=True, help="Original official 30 Hz exported dataset")
    parser.add_argument("--histogram-dir", type=Path, help="Default: <analysis-dir>/plots_overlap")
    parser.add_argument("--output-dir", type=Path, required=True, help="New empty directory for raw samples and combined histograms")
    parser.add_argument("--expected-sequences", type=int, default=1261)
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    cache = (args.histogram_dir or analysis_dir / "plots_overlap").expanduser().resolve()
    if output.is_relative_to(raw_root) or output == analysis_dir or output == cache:
        raise ValueError("Use a new analysis directory outside the source datasets")
    summary = plots.read_json(analysis_dir / "summary.json")
    if summary.get("status") != "complete" or summary["sequences"] != args.expected_sequences:
        raise ValueError("Original IK comparison is incomplete")
    derivative_dir = plots.resolve_derivatives(analysis_dir, summary)
    manifest = plots.read_json(derivative_dir / "manifest.json")
    files = manifest.get("files", [])
    if (manifest.get("status") != "complete" or len(files) != summary["sequences"]
            or len({row["file"] for row in files}) != len(files)):
        raise ValueError("Original derivative manifest is incomplete or duplicated")
    old_report, distributions = plots.read_histograms(cache)
    if (old_report["sequences"] != summary["sequences"]
            or Path(old_report["input_summary"]).resolve() != (analysis_dir / "summary.json")
            or "raw" in plots.versions_in(distributions)):
        raise ValueError("Histogram cache does not match the original two-version comparison")
    for order in plots.ORDERS:
        for stream in plots.STREAMS:
            for arm in plots.ARMS:
                a = distributions[order, stream, arm, "mink"]
                b = distributions[order, stream, arm, "three_stage"]
                np.testing.assert_array_equal(a.signed_edges, b.signed_edges)
                np.testing.assert_array_equal(a.absolute_edges, b.absolute_edges)
                distributions[order, stream, arm, "raw"] = plots.Distribution(a.signed_edges, a.absolute_edges)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; previous results are preserved")
    raw_dir = output / "raw.derivatives"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(output / "distribution_summary.json", {"status": "running", "view": plots.VIEW})
    started = time.monotonic()
    raw_files = []
    prefix = plots.VIEW + "__mink__"
    raw_prefix = plots.VIEW + "__raw__"
    for number, row in enumerate(files, 1):
        episode = row.get("episode", Path(row["file"]).stem)
        if Path(episode).name != episode:
            raise ValueError("Invalid episode name")
        split = row["split"]
        split = split + "_real" if split in ("train", "val") else split
        if split not in ("train_real", "val_real"):
            raise ValueError("This comparison covers real sequences only")
        folder = raw_root / split / episode
        metadata = plots.read_json(folder / "episode_metadata.json")
        if "joint_conversion" in metadata or "sequence_ik" in metadata:
            raise ValueError(f"Raw source has already been converted: {folder}")
        binary = folder / "states_actions.bin"
        frames = metadata["num_steps"]
        if binary.stat().st_size != frames * 28 * 8 or frames != row["source_frames"]:
            raise ValueError(f"Raw source frame count differs from the comparison: {folder}")
        source = np.memmap(binary, dtype=np.float64, mode="r", shape=(frames, 28))
        archive_path = (derivative_dir / row["file"]).resolve()
        if not archive_path.is_relative_to(derivative_dir):
            raise ValueError("Derivative archive leaves its directory")
        arrays = {"stream_labels": np.array(plots.STREAMS), "arm_labels": np.array(plots.ARMS),
                  "joint_numbers": np.arange(1, 7), "source_frame_count": np.int64(frames)}
        with np.load(archive_path, allow_pickle=False) as saved:
            fps = float(saved["fps"]) if "fps" in saved else float(summary["fps"])
            if not np.isclose(fps, 30) or not np.isclose(metadata["tick_ns"] / 1e9, 1 / fps, rtol=1e-6, atol=0):
                raise ValueError(f"Raw source sampling is not the same official 30 Hz clock: {folder}")
            if "metadata_t0_ns" in saved and "tick_ns" in saved:
                t0, tick = int(saved["metadata_t0_ns"]), int(saved["tick_ns"])
            else:
                old_folder = Path(summary["inputs"]["mink_root"]) / split / episode
                old_metadata = plots.read_json(old_folder / "episode_metadata.json")
                t0, tick = old_metadata["t0_ns"], old_metadata["tick_ns"]
            if metadata["t0_ns"] != t0 or metadata["tick_ns"] != tick:
                raise ValueError(f"Raw source timestamps differ from the comparison: {folder}")
            arrays.update(fps=np.float64(fps), metadata_t0_ns=np.int64(t0), tick_ns=np.int64(tick))
            used = []
            for order, (data_key, index_key, _, _, _) in plots.ORDERS.items():
                indices = saved[prefix + index_key]
                np.testing.assert_array_equal(indices, saved[plots.VIEW + "__three_stage__" + index_key])
                width = 2 if order == "velocity" else 3
                if (indices.ndim != 2 or indices.shape[1] != width
                        or not np.issubdtype(indices.dtype, np.integer)
                        or np.any(indices < 0) or np.any(indices >= frames)
                        or not np.all(np.diff(indices, axis=1) == 1)):
                    raise ValueError("Invalid saved overlap frame indices")
                used.append(indices.ravel())
                first = source[indices[:, 0]][:, JOINT_COLUMNS].reshape(-1, 2, 2, 6)
                middle = source[indices[:, 1]][:, JOINT_COLUMNS].reshape(-1, 2, 2, 6)
                if width == 2:
                    difference = middle - first
                else:
                    last = source[indices[:, 2]][:, JOINT_COLUMNS].reshape(-1, 2, 2, 6)
                    difference = (last - middle) - (middle - first)
                values = np.rad2deg(difference) * fps ** (width - 1)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite raw derivative on a matched frame: {folder}")
                arrays[raw_prefix + data_key] = values
                arrays[raw_prefix + index_key] = indices
                for suffix in ("intervals", "velocity_segment_ids", "acceleration_segment_ids"):
                    if prefix + suffix in saved:
                        arrays[raw_prefix + suffix] = saved[prefix + suffix]
                for si, stream in enumerate(plots.STREAMS):
                    for ai, arm in enumerate(plots.ARMS):
                        data = values[:, si, ai]
                        extend_range(distributions, order, stream, arm, float(np.abs(data).max(initial=0)))
                        distributions[order, stream, arm, "raw"].add(data)
            frame_indices = np.unique(np.concatenate(used))
            arrays["source_frame_indices"] = frame_indices
            arrays["raw__state_action_rows"] = np.asarray(source[frame_indices])
        arrays[raw_prefix + "velocity_time_s"] = (arrays[raw_prefix + "velocity_frame_indices"][:, 0] + 0.5) / fps
        arrays[raw_prefix + "acceleration_time_s"] = arrays[raw_prefix + "acceleration_frame_indices"][:, 1] / fps
        path = raw_dir / (episode + ".npz")
        save_arrays(path, arrays)
        raw_files.append({"episode": episode, "split": split, "file": path.name,
                          "source_frames": frames, "bytes": path.stat().st_size})
        if number % 100 == 0 or number == len(files):
            print(f"Added raw overlap: {number}/{len(files)} sequences", flush=True)
    histograms, reports = {}, {}
    for (order, stream, arm, version), distribution in distributions.items():
        key = f"{order}__{stream}__{arm}__{version}"
        for name in ("signed_edges", "signed_counts", "absolute_edges", "absolute_counts"):
            histograms[key + "__" + name] = getattr(distribution, name)
        reports[key] = distribution.report()
        if version == "raw":
            for other in plots.VERSIONS:
                if distribution.report()["samples_per_joint"] != distributions[order, stream, arm, other].report()["samples_per_joint"]:
                    raise ValueError("Raw derivative count differs from the matched IK frame pairs/triples")
    np.savez_compressed(output / "histograms.npz", **histograms)
    write_json(raw_dir / "manifest.json", {"status": "complete", "sequences": len(raw_files),
        "format_version": 2, "files": raw_files, "fps": 30, "precision": "float64", "signed": True,
        "sampling": False, "clip": False, "angle_unwrap": False,
        "frame_indices": "Exactly the original matched_training_intervals Mink/three-stage pairs and triples",
        "joint_rows": "raw__state_action_rows indexed by source_frame_indices, original radians/grippers"})
    report = {**old_report, "status": "complete", "versions": ["raw", *plots.VERSIONS],
        "distributions": reports, "figures": [], "seconds": time.monotonic() - started,
        "raw_data": {"root": str(raw_root), "clip": False, "interpolation": False, "angle_unwrap": False,
                     "selection": "Exact previously saved overlap frame pairs/triples; no additional filtering"},
        "raw_derivatives": {"directory": str(raw_dir), "manifest": str(raw_dir / "manifest.json")},
        "histograms_source": str(cache), "raw_derivatives_computed": True,
        "derivatives_recomputed": False,
        "summary_checks": "Original statistics unchanged; raw sample counts match both IK versions; no histogram samples lost"}
    write_json(output / "distribution_summary.json", report)
    print(f"\nHISTOGRAMS={output}\nRAW_DERIVATIVES={raw_dir}", flush=True)


if __name__ == "__main__":
    main()
