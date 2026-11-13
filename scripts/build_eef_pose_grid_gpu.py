"""Build a lossless GPU FK pose table and 6D pose-to-joint index."""

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from abc_minimal.eef_pose_grid import write_cell_index
from build_eef_pose_grid_fine import GpuFK
from build_eef_workspace import REPO, arm_model, extract_poses


DEFAULT_STEPS = (0.05, 0.06, 0.08, 0.1, 0.13, 0.18)
BUCKETS = 512
BUCKET_SHIFT = 39


@wp.kernel
def fill_sequential(start: wp.int64, count: int,
                    lower: wp.array(dtype=wp.float32),
                    upper: wp.array(dtype=wp.float32),
                    counts: wp.array(dtype=wp.int32),
                    steps: wp.array(dtype=wp.float32),
                    qpos: wp.array2d(dtype=wp.float32)):
    world = wp.tid()
    if world < count:
        row = start + wp.int64(world)
        for joint in range(6):
            digit = row % wp.int64(counts[joint])
            row = row // wp.int64(counts[joint])
            qpos[world, joint] = wp.min(lower[joint] + wp.float32(digit) * steps[joint],
                                        upper[joint])


@wp.kernel
def fill_indexed(start: int, count: int, rows: wp.array(dtype=wp.int64),
                 lower: wp.array(dtype=wp.float32),
                 upper: wp.array(dtype=wp.float32),
                 counts: wp.array(dtype=wp.int32),
                 steps: wp.array(dtype=wp.float32),
                 qpos: wp.array2d(dtype=wp.float32)):
    world = wp.tid()
    if world < count:
        row = rows[start + world]
        for joint in range(6):
            digit = row % wp.int64(counts[joint])
            row = row // wp.int64(counts[joint])
            qpos[world, joint] = wp.min(lower[joint] + wp.float32(digit) * steps[joint],
                                        upper[joint])


@wp.kernel
def encode_poses(start: int, count: int, poses: wp.array2d(dtype=wp.float32),
                 keys: wp.array(dtype=wp.int64), outside: wp.array(dtype=wp.int32)):
    world = wp.tid()
    if world < count:
        key = wp.int64(0)
        for axis in range(6):
            if axis < 3:
                digit = int(wp.floor(poses[world, axis] / wp.float32(0.01))) + 256
                if digit < 0 or digit >= 512:
                    wp.atomic_add(outside, 0, 1)
                key = (key << wp.int64(9)) | wp.int64(digit)
            else:
                digit = int(wp.floor(poses[world, axis] / wp.float32(0.1))) + 64
                if digit < 0 or digit >= 128:
                    wp.atomic_add(outside, 0, 1)
                key = (key << wp.int64(7)) | wp.int64(digit)
        keys[start + world] = key


def save_metadata(path, metadata):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def stage(args, gpu, metadata):
    pose_path = args.output / "poses.bin"
    spool_dir = args.scratch / "spool"
    if not args.poses_only:
        spool_dir.mkdir(parents=True, exist_ok=True)
    committed = metadata["staged_rows"]
    stream_context = nullcontext(None) if args.index_only else pose_path.open("r+b" if pose_path.exists() else "w+b")
    with stream_context as pose_stream:
        if pose_stream is not None:
            pose_stream.truncate(committed * 24)
            pose_stream.seek(committed * 24)
        for bucket, count in enumerate(metadata["bucket_counts"]):
            path = spool_dir / f"{bucket:03d}.rows"
            if path.exists():
                with path.open("r+b") as stream:
                    stream.truncate(count * 8)
            elif count:
                raise FileNotFoundError(path)
        files = {}
        keys = outside = None
        if not args.poses_only:
            keys = wp.empty(args.batch_size, dtype=wp.int64, device=args.device)
            outside = wp.zeros(1, dtype=wp.int32, device=args.device)
        started = last_checkpoint = time.monotonic()
        for start in range(committed, metadata["total_rows"], args.batch_size):
            count = min(args.batch_size, metadata["total_rows"] - start)
            wp.launch(fill_sequential, dim=args.batch_size, inputs=[
                wp.int64(start), count, gpu.lower, gpu.upper,
                gpu.counts, gpu.steps, gpu.data.qpos,
            ])
            mjw.kinematics(gpu.model, gpu.data)
            wp.launch(extract_poses, dim=args.batch_size, inputs=[
                gpu.site, gpu.data.site_xpos, gpu.data.site_xmat, gpu.poses_device,
            ])
            if pose_stream is not None:
                pose_stream.write(memoryview(gpu.poses_device.numpy()[:count]))
            if not args.poses_only:
                wp.launch(encode_poses, dim=args.batch_size,
                          inputs=[0, count, gpu.poses_device, keys, outside])
                bucket_ids = wp.to_torch(keys)[:count] >> BUCKET_SHIFT
                sorted_buckets, order = torch.sort(bucket_ids)
                sizes = torch.bincount(bucket_ids, minlength=BUCKETS).cpu().numpy()
                rows = (order + start).cpu().numpy()
                offset = 0
                for bucket in np.flatnonzero(sizes):
                    number = int(bucket)
                    size = int(sizes[bucket])
                    if number not in files:
                        files[number] = (spool_dir / f"{number:03d}.rows").open("ab")
                    files[number].write(memoryview(rows[offset:offset + size]))
                    metadata["bucket_counts"][number] += size
                    offset += size
            metadata["staged_rows"] = start + count
            if time.monotonic() - last_checkpoint >= args.log_every or start + count == metadata["total_rows"]:
                if outside is not None and outside.numpy()[0]:
                    raise ValueError("FK pose exceeds the 6D key range")
                if pose_stream is not None:
                    pose_stream.flush()
                    os.fsync(pose_stream.fileno())
                for stream in files.values():
                    stream.flush()
                    os.fsync(stream.fileno())
                save_metadata(args.output / "metadata.json", metadata)
                now = time.monotonic()
                rate = (metadata["staged_rows"] - committed) / (now - started)
                print(f"stage {metadata['staged_rows']:,}/{metadata['total_rows']:,}, "
                      f"{rate / 1e6:.1f} M rows/s", flush=True)
                last_checkpoint = now
        for stream in files.values():
            stream.close()
    metadata["phase"] = "complete" if args.poses_only else "index"
    save_metadata(args.output / "metadata.json", metadata)


def index(args, gpu, metadata):
    cells = args.output / "cells"
    cells.mkdir(exist_ok=True)
    for bucket, count in enumerate(metadata["bucket_counts"]):
        source = args.scratch / "spool" / f"{bucket:03d}.rows"
        if bucket in metadata["finished_buckets"]:
            source.unlink(missing_ok=True)
            continue
        if not count:
            metadata["finished_buckets"].append(bucket)
            save_metadata(args.output / "metadata.json", metadata)
            continue
        if count >= 2**32:
            raise ValueError("A bucket exceeds the 32-bit directory offset")
        rows_cpu = np.memmap(source, mode="r", dtype="<i8", shape=(count,))
        rows_gpu = wp.array(rows_cpu, dtype=wp.int64, device=args.device)
        keys = wp.empty(count, dtype=wp.int64, device=args.device)
        outside = wp.zeros(1, dtype=wp.int32, device=args.device)
        for start in range(0, count, args.batch_size):
            size = min(args.batch_size, count - start)
            wp.launch(fill_indexed, dim=args.batch_size, inputs=[
                start, size, rows_gpu, gpu.lower, gpu.upper,
                gpu.counts, gpu.steps, gpu.data.qpos,
            ])
            mjw.kinematics(gpu.model, gpu.data)
            wp.launch(extract_poses, dim=args.batch_size, inputs=[
                gpu.site, gpu.data.site_xpos, gpu.data.site_xmat, gpu.poses_device,
            ])
            wp.launch(encode_poses, dim=args.batch_size,
                      inputs=[start, size, gpu.poses_device, keys, outside])
        wp.synchronize_device()
        if outside.numpy()[0]:
            raise ValueError("FK pose exceeds the 6D key range")
        sorted_keys, order = torch.sort(wp.to_torch(keys))
        unique, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        if bool(torch.any((sorted_keys >> BUCKET_SHIFT) != bucket)):
            raise ValueError(f"Wrong pose bucket {bucket}")
        local_keys = (unique & ((1 << BUCKET_SHIFT) - 1)).cpu().numpy().astype(np.uint64)
        cell_counts = counts.cpu().numpy().astype(np.uint32)
        base = cells / f"{bucket:03d}"
        target = base.with_suffix(".rows5")
        temporary = base.with_suffix(".rows5.tmp")
        source_rows = wp.to_torch(rows_gpu)
        with temporary.open("wb") as stream:
            for start in range(0, count, 4_000_000):
                selected = source_rows[order[start:start + 4_000_000]].cpu().numpy()
                packed = np.ascontiguousarray(selected.view(np.uint8).reshape(-1, 8)[:, :5])
                stream.write(memoryview(packed))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        write_cell_index(base, local_keys, cell_counts)
        metadata["indexed_rows"] += count
        metadata["finished_buckets"].append(bucket)
        save_metadata(args.output / "metadata.json", metadata)
        del rows_cpu, rows_gpu, keys, sorted_keys, order, unique, counts, source_rows
        torch.cuda.empty_cache()
        source.unlink()
        print(f"index {metadata['indexed_rows']:,}/{metadata['total_rows']:,} rows", flush=True)
    if metadata["indexed_rows"] != metadata["total_rows"]:
        raise ValueError("Indexed row count differs from FK row count")
    metadata["phase"] = "complete"
    save_metadata(args.output / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=REPO / "abc_sim/models/yam_dishrack_base.xml")
    parser.add_argument("--joint-steps", type=float, nargs=6, default=DEFAULT_STEPS)
    parser.add_argument("--limit-margin-deg", type=float, default=10.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--poses-only", action="store_true", help="Save the FK table without a pose-to-joint lookup index")
    mode.add_argument("--index-only", action="store_true", help="Build the pose-to-joint index without writing another FK table")
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--log-every", type=float, default=20)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    steps = np.asarray(args.joint_steps, dtype=np.float32)
    model = arm_model(args.model, "left")
    model.jnt_range[:] += np.deg2rad(args.limit_margin_deg) * np.array([-1, 1])
    joint_counts = np.array([len(np.arange(lo, hi, step)) + 1
                             for (lo, hi), step in zip(model.jnt_range, steps)], dtype=np.int64)
    full_rows = math.prod(joint_counts.tolist())
    total_rows = min(full_rows, args.limit_rows) if args.limit_rows else full_rows
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    config = {"model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "limit_margin_deg": args.limit_margin_deg,
              "joint_counts": joint_counts.tolist(), "joint_steps_rad": steps.tolist(),
              "joint_ranges_rad": model.jnt_range.tolist(), "total_rows": total_rows,
              "pose_index": not args.poses_only,
              "site": "left_tcp_site", "frame": "left_arm base",
              "pose_columns": ["x", "y", "z", "rotvec_x", "rotvec_y", "rotvec_z"],
              "position_step_m": 0.01, "rotation_step_rad": 0.1}
    if args.index_only:
        config["pose_table"] = False
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["config"] != config or metadata["scratch"] != str(args.scratch.resolve()):
            raise ValueError("Existing build has different settings or scratch path")
    else:
        bytes_per_row = 5 if args.index_only else 24 if args.poses_only else 29
        if shutil.disk_usage(args.output).free < total_rows * bytes_per_row + 5 * 2**30:
            raise OSError("Insufficient output space for all poses and joint row IDs")
        if not args.poses_only and shutil.disk_usage(args.scratch).free < total_rows * 8 + 5 * 2**30:
            raise OSError("Insufficient scratch space for partitioned joint row IDs")
        metadata = {"config": config, "scratch": str(args.scratch.resolve()),
                    "phase": "stage", "total_rows": total_rows, "staged_rows": 0,
                    "bucket_counts": [0] * BUCKETS, "indexed_rows": 0,
                    "finished_buckets": []}
        save_metadata(metadata_path, metadata)
    if metadata["phase"] == "complete":
        print(f"Already complete: {args.output}", flush=True)
        return
    gpu = GpuFK(model, model.site("left_tcp_site").id,
                joint_counts, args.batch_size, args.device, steps)
    print(f"{total_rows:,} rows, counts={joint_counts.tolist()}, phase={metadata['phase']}", flush=True)
    if metadata["phase"] == "stage":
        stage(args, gpu, metadata)
    if metadata["phase"] == "index":
        index(args, gpu, metadata)


if __name__ == "__main__":
    main()
