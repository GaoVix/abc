"""Compare official-timing Mink and three-stage real-data joint trajectories.

Reads only the two exported training datasets. No IK, video decoding, angle
unwrapping, or data modifications occur.
Only NumPy and the Python standard library are required.
"""

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np


ARMS = {"left": np.arange(6), "right": np.arange(7, 13)}
STREAMS = {"state": 0, "action": 14}
VIEWS = ("as_trained", "matched_training_intervals")
VERSIONS = ("mink", "three_stage")


@dataclass
class Moments:
    count: int = 0
    absolute: np.ndarray = field(default_factory=lambda: np.zeros(6))
    square: np.ndarray = field(default_factory=lambda: np.zeros(6))
    maximum: np.ndarray = field(default_factory=lambda: np.zeros(6))
    maximum_l2: float = 0.0

    def add(self, difference):
        if not len(difference):
            return
        degrees = np.rad2deg(difference)
        if not np.isfinite(degrees).all():
            raise ValueError("Non-finite difference on an included frame")
        self.count += len(degrees)
        self.absolute += np.abs(degrees).sum(axis=0)
        squares = np.square(degrees)
        self.square += squares.sum(axis=0)
        self.maximum = np.maximum(self.maximum, np.abs(degrees).max(axis=0))
        self.maximum_l2 = max(self.maximum_l2, float(np.sqrt(squares.sum(axis=1).max())))

    def report(self, order, fps):
        if not self.count:
            return {"samples_per_joint": 0}
        scale = fps ** order
        return {
            "samples_per_joint": self.count,
            "units": "deg/s" if order == 1 else "deg/s^2",
            "mean_absolute": float(self.absolute.mean() / self.count * scale),
            "rms": float(np.sqrt(self.square.mean() / self.count) * scale),
            "maximum_absolute": float(self.maximum.max() * scale),
            "rms_joint_difference_deg": float(np.sqrt(self.square.mean() / self.count)),
            "mean_squared_l2_joint_difference_deg2": float(self.square.sum() / self.count),
            "rms_l2": float(np.sqrt(self.square.sum() / self.count) * scale),
            "maximum_l2": self.maximum_l2 * scale,
            "per_joint_mean_absolute": (self.absolute / self.count * scale).tolist(),
            "per_joint_rms": (np.sqrt(self.square / self.count) * scale).tolist(),
            "per_joint_maximum_absolute": (self.maximum * scale).tolist(),
        }


@dataclass
class Motion:
    first: Moments = field(default_factory=Moments)
    second: Moments = field(default_factory=Moments)
    jumps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))

    def add(self, q, valid):
        edges = valid[:-1] & valid[1:]
        triples = valid[:-2] & valid[1:-1] & valid[2:]
        first = np.diff(q, axis=0)[edges]
        second = np.diff(q, n=2, axis=0)[triples]
        self.first.add(first)
        self.second.add(second)
        if len(first):
            largest = np.rad2deg(np.abs(first).max(axis=1))
            self.jumps += (largest[:, None] > np.array([30.0, 60.0, 90.0])).sum(axis=0)

    def report(self, fps):
        return {
            "first_derivative": self.first.report(1, fps),
            "second_derivative": self.second.report(2, fps),
            "jump_edges": {f"gt_{threshold}_deg": int(count)
                           for threshold, count in zip((30, 60, 90), self.jumps)},
        }


def read_json(path):
    return json.loads(path.read_text())


def read_rows(folder, metadata):
    binary = folder / "states_actions.bin"
    size = binary.stat().st_size
    if not size or size % (28 * 8):
        raise ValueError(f"Invalid float64 (N,28) file: {binary}")
    rows = np.memmap(binary, dtype=np.float64, mode="r", shape=(size // (28 * 8), 28))
    if len(rows) != metadata["num_steps"]:
        raise ValueError(f"Metadata frame count mismatch: {folder}")
    return rows


def discover_mink(root):
    episodes = {}
    for split in ("train_real", "val_real"):
        for binary in sorted((root / split).glob("episode_*/states_actions.bin")):
            name = binary.parent.name
            if name in episodes:
                raise ValueError(f"Duplicate Mink episode: {name}")
            meta = read_json(binary.parent / "episode_metadata.json")
            conversion = meta.get("joint_conversion", {})
            if (conversion.get("solver") != "MinkArmController"
                    or conversion.get("interpolation") is not False
                    or conversion.get("time_offset_s") != 0
                    or conversion.get("sampling") != "unchanged official export"):
                raise ValueError(f"Not the official-timing Mink export: {binary.parent}")
            episodes[name] = (split, binary.parent, meta)
    return episodes


def discover_three_stage(root):
    episodes = {}
    for split in ("train_real", "val_real"):
        for binary in sorted((root / split).glob("episode_*/states_actions.bin")):
            meta = read_json(binary.parent / "episode_metadata.json")
            conversion = meta.get("sequence_ik", {})
            if (conversion.get("base_mode") != "xml"
                    or conversion.get("action_solver") != "trust-constr"
                    or conversion.get("state_solver") != "trust-constr"
                    or conversion.get("normalized") is not False):
                raise ValueError(f"Not final XML-base state/action IK: {binary.parent}")
            name = conversion["source_episode"]
            begin, end = meta["source_frame_range"]
            if not (isinstance(begin, int) and isinstance(end, int) and 0 <= begin < end):
                raise ValueError(f"Invalid source frame range: {binary.parent}")
            episodes.setdefault(name, []).append((begin, end, split, binary.parent, meta))
    for segments in episodes.values():
        segments.sort(key=lambda segment: segment[:2])
        if any(a[1] > b[0] for a, b in zip(segments, segments[1:])):
            raise ValueError("Overlapping exported intervals")
    return episodes


def add_intervals(global_stats, view, version, rows, valid, intervals, fps, writer, episode, split):
    local = {(stream, arm): Motion() for stream in STREAMS for arm in ARMS}
    for begin, end in intervals:
        for stream, offset in STREAMS.items():
            for arm, columns in ARMS.items():
                q = rows[begin:end, offset + columns]
                mask = valid[begin:end]
                global_stats[view, version, stream, arm].add(q, mask)
                local[stream, arm].add(q, mask)
    for (stream, arm), motion in local.items():
        report = motion.report(fps)
        first, second = report["first_derivative"], report["second_derivative"]
        writer.writerow({
            "view": view, "episode": episode, "split": split, "version": version,
            "stream": stream, "arm": arm,
            "velocity_samples": first["samples_per_joint"],
            "velocity_rms_deg_s": first.get("rms"),
            "velocity_max_deg_s": first.get("maximum_absolute"),
            "acceleration_samples": second["samples_per_joint"],
            "acceleration_rms_deg_s2": second.get("rms"),
            "acceleration_max_deg_s2": second.get("maximum_absolute"),
            **report["jump_edges"],
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mink-root", type=Path, required=True)
    parser.add_argument("--three-stage-root", type=Path, required=True,
                        help="New exported training data, with source_frame_range metadata")
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON; also writes a per-episode CSV")
    parser.add_argument("--expected-sequences", type=int, default=1261)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.fps <= 0 or not np.isfinite(args.fps):
        raise ValueError("FPS must be finite and positive")
    for name in ("mink_root", "three_stage_root", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for root in (args.mink_root, args.three_stage_root):
        if args.output.is_relative_to(root):
            raise ValueError("Write analysis reports outside the input datasets")
    mink, new = discover_mink(args.mink_root), discover_three_stage(args.three_stage_root)
    if set(mink) != set(new) or len(mink) != args.expected_sequences:
        raise ValueError(f"Episode sets differ or incomplete: Mink={len(mink)}, three-stage={len(new)}, "
                         f"only Mink={sorted(set(mink)-set(new))[:3]}, only new={sorted(set(new)-set(mink))[:3]}")
    stats = {(view, version, stream, arm): Motion()
             for view in VIEWS for version in VERSIONS for stream in STREAMS for arm in ARMS}
    fields = ("view", "episode", "split", "version", "stream", "arm", "velocity_samples",
              "velocity_rms_deg_s", "velocity_max_deg_s", "acceleration_samples",
              "acceleration_rms_deg_s2", "acceleration_max_deg_s2",
              "gt_30_deg", "gt_60_deg", "gt_90_deg")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".episodes.csv")
    csv_temp = csv_path.with_suffix(".csv.tmp")
    started, frames, segments = time.monotonic(), 0, 0
    retained, old_nonfinite, excluded_paired = 0, 0, 0
    with csv_temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for number, episode in enumerate(sorted(mink), 1):
            split, folder, old_meta = mink[episode]
            old_rows = read_rows(folder, old_meta)
            new_rows = np.full(old_rows.shape, np.nan)
            old_valid = np.isfinite(old_rows).all(axis=1)
            frames += len(old_rows)
            old_nonfinite += int((~old_valid).sum())
            intervals = []
            for begin, end, new_split, new_folder, meta in new[episode]:
                if new_split != split or end > len(old_rows):
                    raise ValueError(f"Split or frame range mismatch: {new_folder}")
                if meta.get("tick_ns") != old_meta.get("tick_ns"):
                    raise ValueError(f"Different sampling clocks: {new_folder}")
                if "t0_ns" not in old_meta or meta.get("t0_ns") != old_meta["t0_ns"] + begin * old_meta["tick_ns"]:
                    raise ValueError(f"Timestamp/frame offset mismatch: {new_folder}")
                if not np.isclose(old_meta["tick_ns"] / 1e9, 1 / args.fps, rtol=1e-6, atol=0):
                    raise ValueError(f"FPS does not match episode timing: {folder}")
                saved = read_rows(new_folder, meta)
                if len(saved) != end - begin or not np.isfinite(saved).all() or end - begin < 30:
                    raise ValueError(f"Invalid frame in final training interval: {new_folder}")
                np.testing.assert_array_equal(saved[:, [6, 13, 20, 27]], old_rows[begin:end, [6, 13, 20, 27]])
                new_rows[begin:end] = saved
                retained += end - begin
                excluded_paired += int((~old_valid[begin:end]).sum())
                intervals.append((begin, end))
            new_valid = np.isfinite(new_rows).all(axis=1)
            common = old_valid & new_valid
            whole = [(0, len(old_rows))]
            for version, rows in (("mink", old_rows), ("three_stage", new_rows)):
                add_intervals(stats, "as_trained", version, rows,
                              old_valid if version == "mink" else new_valid,
                              whole if version == "mink" else intervals,
                              args.fps, writer, episode, split)
                add_intervals(stats, "matched_training_intervals", version, rows, common, intervals,
                              args.fps, writer, episode, split)
            segments += len(intervals)
            if number % 100 == 0 or number == len(mink):
                print(f"Compared {number}/{len(mink)} real sequences", flush=True)
    csv_temp.replace(csv_path)
    views = {view: {version: {stream: {arm: stats[view, version, stream, arm].report(args.fps)
                                     for arm in ARMS} for stream in STREAMS} for version in VERSIONS}
             for view in VIEWS}
    summary = {
        "status": "complete", "sequences": len(mink), "source_frames": frames,
        "new_training_segments": segments, "new_retained_frames": retained,
        "old_nonfinite_frames": old_nonfinite, "excluded_paired_training_frames": excluded_paired,
        "fps": args.fps, "grippers_included": False, "angle_unwrap": False,
        "first_derivative": "(q[t+1]-q[t])*fps",
        "second_derivative": "(q[t+2]-2*q[t+1]+q[t])*fps**2",
        "rms_definition": "sqrt(mean across included times and six joint components of derivative**2)",
        "view_definitions": {
            "matched_training_intervals": "Both versions on the exact intervals retained in the new training data; no differences across interval boundaries",
            "as_trained": "Each version's own exported real training sequences; retained frames/intervals can differ",
        },
        "inputs": {name: str(getattr(args, name)) for name in
                   ("mink_root", "three_stage_root")},
        "views": views, "per_episode_csv": str(csv_path), "seconds": time.monotonic() - started,
    }
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    for view in VIEWS:
        print(f"\n{view}: component RMS and maximum; acceleration measures changes in joint velocity")
        print("stream arm        vel_RMS(deg/s) old -> new     acc_RMS(deg/s^2) old -> new     acc_MAX old -> new")
        for name in STREAMS:
            for arm in ARMS:
                a, b = (views[view][version][name][arm] for version in VERSIONS)
                def fmt(row, order, metric):
                    value = row[order].get(metric)
                    return f"{value:.4f}" if value is not None else "n/a"
                print(f"{name:6} {arm:5}  "
                      f"{fmt(a, 'first_derivative', 'rms'):>12} -> {fmt(b, 'first_derivative', 'rms'):<12} "
                      f"{fmt(a, 'second_derivative', 'rms'):>12} -> {fmt(b, 'second_derivative', 'rms'):<12} "
                      f"{fmt(a, 'second_derivative', 'maximum_absolute')} -> {fmt(b, 'second_derivative', 'maximum_absolute')}")
    print(f"\nSUMMARY={args.output}\nEPISODES={csv_path}", flush=True)


if __name__ == "__main__":
    main()
