"""Build a lossless, searchable EEF pose table for a six-joint lattice."""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import shutil
import struct
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
import zstandard as zstd
from scipy.spatial.transform import Rotation

from build_eef_workspace import REPO, arm_model, extract_poses


STEPS = np.array([0.05, 0.06, 0.08, 0.1, 0.1, 0.1], dtype=np.float32)
RECORD = np.dtype([("row", "u1", 5), ("pose", "<f4", 6)])
RUN_HEADER = struct.Struct("<IQ")
BLOCK_ROWS = 16384
BUCKETS = 512
BUCKET_SHIFT = 39


@wp.kernel
def fill_gpu_joints(count: int, rows: wp.array(dtype=wp.int64),
                    lower: wp.array(dtype=wp.float32),
                    upper: wp.array(dtype=wp.float32),
                    counts: wp.array(dtype=wp.int32),
                    steps: wp.array(dtype=wp.float32),
                    qpos: wp.array2d(dtype=wp.float32)):
    world = wp.tid()
    if world < count:
        index = rows[world]
        for joint in range(6):
            digit = index % wp.int64(counts[joint])
            index = index // wp.int64(counts[joint])
            qpos[world, joint] = wp.min(lower[joint] + wp.float32(digit) * steps[joint],
                                        upper[joint])


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def joint_counts(model):
    return np.array([len(np.arange(lo, hi, step)) + 1
                     for (lo, hi), step in zip(model.jnt_range, STEPS)], dtype=np.int64)


def cube_ranges(counts, width):
    axes = [range(0, int(count), width) for count in counts]
    for begins in itertools.product(*axes):
        yield tuple((begin, min(begin + width, int(count)))
                    for begin, count in zip(begins, counts))


def cube_rows(ranges, counts):
    sizes = [end - begin for begin, end in ranges]
    local = np.arange(math.prod(sizes), dtype=np.int64)
    strides = np.r_[1, np.cumprod(counts[:-1], dtype=np.int64)]
    rows = np.zeros(len(local), dtype=np.int64)
    for (begin, _), size, stride in zip(ranges, sizes, strides):
        rows += (local % size + begin) * stride
        local //= size
    return rows


def fk_poses_cpu(model, site, rows, counts):
    data = mujoco.MjData(model)
    strides = np.r_[1, np.cumprod(counts[:-1], dtype=np.int64)]
    poses = np.empty((len(rows), 6), dtype=np.float32)
    matrices = np.empty((65536, 3, 3), dtype=np.float64)
    for start in range(0, len(rows), len(matrices)):
        end = min(start + len(matrices), len(rows))
        digits = rows[start:end, None] // strides % counts
        joints = np.minimum(model.jnt_range[:, 0] + digits * STEPS,
                            model.jnt_range[:, 1])
        for offset, joints_row in enumerate(joints):
            data.qpos[:6] = joints_row
            mujoco.mj_kinematics(model, data)
            poses[start + offset, :3] = data.site_xpos[site]
            matrices[offset] = data.site_xmat[site].reshape(3, 3)
        poses[start:end, 3:] = Rotation.from_matrix(matrices[:end - start]).as_rotvec()
    return poses


class GpuFK:
    def __init__(self, model, site, counts, batch_size, device, steps=None):
        wp.init()
        wp.set_device(device)
        cpu_data = mujoco.MjData(model)
        self.model = mjw.put_model(model)
        self.data = mjw.put_data(model, cpu_data, nworld=batch_size, nconmax=0, njmax=0)
        self.site = site
        self.batch_size = batch_size
        self.lower = wp.array(model.jnt_range[:, 0].astype(np.float32), device=device)
        self.upper = wp.array(model.jnt_range[:, 1].astype(np.float32), device=device)
        self.counts = wp.array(counts.astype(np.int32), device=device)
        self.steps = wp.array(np.asarray(STEPS if steps is None else steps, dtype=np.float32), device=device)
        self.poses_device = wp.empty((batch_size, 6), dtype=wp.float32, device=device)

    def poses(self, rows):
        output = np.empty((len(rows), 6), dtype=np.float32)
        for start in range(0, len(rows), self.batch_size):
            end = min(start + self.batch_size, len(rows))
            count = end - start
            device_rows = wp.array(rows[start:end], dtype=wp.int64, device=wp.get_device())
            wp.launch(fill_gpu_joints, dim=self.batch_size, inputs=[
                count, device_rows, self.lower, self.upper,
                self.counts, self.steps, self.data.qpos,
            ])
            mjw.kinematics(self.model, self.data)
            wp.launch(extract_poses, dim=self.batch_size, inputs=[
                self.site, self.data.site_xpos, self.data.site_xmat, self.poses_device,
            ])
            output[start:end] = self.poses_device.numpy()[:count]
        return output


def pose_keys(poses):
    positions = np.floor(poses[:, :3] / np.float32(0.01)).astype(np.int64) + 256
    rotations = np.floor(poses[:, 3:] / np.float32(0.1)).astype(np.int64) + 64
    digits = np.column_stack((positions, rotations))
    if np.any(digits[:, :3] < 0) or np.any(digits[:, :3] >= 512) or np.any(digits[:, 3:] < 0) or np.any(digits[:, 3:] >= 128):
        raise ValueError("FK pose falls outside the 6D key range")
    keys = np.zeros(len(poses), dtype=np.uint64)
    for column, bits in zip(digits.T, (9, 9, 9, 7, 7, 7)):
        keys = (keys << bits) | column.astype(np.uint64)
    return keys


def pack_records(rows, poses):
    records = np.empty(len(rows), dtype=RECORD)
    bytes_view = rows.astype("<u8", copy=False).view(np.uint8).reshape(-1, 8)
    records["row"] = bytes_view[:, :5]
    records["pose"] = poses
    return records


def spool_path(output, scratch, bucket):
    root = scratch if bucket % 3 == 0 else output
    return root / "spool" / f"{bucket:03d}.run"


def cube_payload(model, site, ranges, counts, compressor, gpu=None):
    rows = cube_rows(ranges, counts)
    poses = gpu.poses(rows) if gpu is not None else fk_poses_cpu(model, site, rows, counts)
    keys = pose_keys(poses)
    buckets = (keys >> np.uint64(BUCKET_SHIFT)).astype(np.int32)
    pieces = []
    for bucket in np.unique(buckets):
        selection = np.flatnonzero(buckets == bucket)
        order = np.argsort(keys[selection], kind="stable")
        records = pack_records(rows[selection[order]], poses[selection[order]])
        pieces.append((int(bucket), len(records), compressor.compress(records.tobytes())))
    return len(rows), pieces


def init_worker(model_path):
    global WORKER_MODEL, WORKER_SITE, WORKER_COUNTS, WORKER_COMPRESSOR
    WORKER_MODEL = arm_model(model_path, "left")
    WORKER_SITE = WORKER_MODEL.site("left_tcp_site").id
    WORKER_COUNTS = joint_counts(WORKER_MODEL)
    WORKER_COMPRESSOR = zstd.ZstdCompressor(level=3)


def worker_payload(ranges):
    return cube_payload(WORKER_MODEL, WORKER_SITE, ranges, WORKER_COUNTS,
                        WORKER_COMPRESSOR)


def stage(args, model, counts, metadata):
    site = model.site("left_tcp_site").id
    compressor = zstd.ZstdCompressor(level=3)
    ranges = list(cube_ranges(counts, args.cube_width))
    sizes = metadata["spool_sizes"]
    for bucket, size in enumerate(sizes):
        path = spool_path(args.output, args.scratch, bucket)
        if path.exists():
            if path.stat().st_size < size:
                raise OSError(f"Incomplete committed spool: {path}")
            with path.open("r+b") as stream:
                stream.truncate(size)
        elif size:
            raise FileNotFoundError(path)
    started = time.monotonic()
    begin = metadata["completed_cubes"]
    stop = min(len(ranges), args.max_cubes) if args.max_cubes else len(ranges)
    if args.workers == 1:
        gpu = GpuFK(model, site, counts, args.batch_size, args.device) if args.backend == "mjwarp" else None
        results = (cube_payload(model, site, ranges[number], counts, compressor, gpu)
                   for number in range(begin, stop))
    else:
        if args.backend == "mjwarp":
            raise ValueError("MJWarp uses one process to share the GPU context")
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker, initargs=(args.model,))
        pending = {number: pool.submit(worker_payload, ranges[number])
                   for number in range(begin, min(stop, begin + 2 * args.workers))}

        def completed():
            for number in range(begin, stop):
                result = pending.pop(number).result()
                next_number = number + 2 * args.workers
                if next_number < stop:
                    pending[next_number] = pool.submit(worker_payload, ranges[next_number])
                yield result

        results = completed()
    try:
        for cube_number, (row_count, pieces) in enumerate(results, start=begin):
            for bucket, count, payload in pieces:
                path = spool_path(args.output, args.scratch, bucket)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab") as stream:
                    stream.write(RUN_HEADER.pack(count, len(payload)))
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                    sizes[bucket] = stream.tell()
            metadata["completed_cubes"] = cube_number + 1
            metadata["completed_rows"] += row_count
            save_json(args.output / "metadata.json", metadata)
            if (cube_number + 1) % args.log_every == 0 or cube_number + 1 == stop:
                elapsed = time.monotonic() - started
                print(f"FK {cube_number + 1:,}/{len(ranges):,} cubes, "
                      f"{metadata['completed_rows']:,}/{metadata['total_rows']:,} rows "
                      f"({metadata['completed_rows'] / metadata['total_rows']:.2%}), "
                      f"{len(pieces)} buckets, {elapsed:.0f}s", flush=True)
    finally:
        if args.workers != 1:
            pool.shutdown(cancel_futures=True)
    if metadata["completed_cubes"] == len(ranges):
        metadata["phase"] = "finalize"
        save_json(args.output / "metadata.json", metadata)


def load_bucket(path):
    decoder = zstd.ZstdDecompressor()
    runs = []
    with path.open("rb") as stream:
        while header := stream.read(RUN_HEADER.size):
            count, size = RUN_HEADER.unpack(header)
            payload = stream.read(size)
            if len(payload) != size:
                raise OSError(f"Truncated spool run: {path}")
            records = np.frombuffer(decoder.decompress(payload), dtype=RECORD)
            if len(records) != count:
                raise OSError(f"Incorrect spool count: {path}")
            runs.append(records)
    return np.concatenate(runs) if runs else np.empty(0, dtype=RECORD)


def finalize(args, metadata):
    compressor = zstd.ZstdCompressor(level=3)
    output_dir = args.output / "buckets"
    output_dir.mkdir(exist_ok=True)
    for bucket in range(BUCKETS):
        split_dir = args.output / "split" / f"{bucket:03d}"
        if bucket in metadata["finished_buckets"]:
            spool_path(args.output, args.scratch, bucket).unlink(missing_ok=True)
            shutil.rmtree(split_dir, ignore_errors=True)
            continue
        source = spool_path(args.output, args.scratch, bucket)
        if not source.exists():
            metadata["finished_buckets"].append(bucket)
            save_json(args.output / "metadata.json", metadata)
            continue
        shutil.rmtree(split_dir, ignore_errors=True)
        split_dir.mkdir(parents=True)
        decoder = zstd.ZstdDecompressor()
        source_count = 0
        with source.open("rb") as stream:
            while header := stream.read(RUN_HEADER.size):
                count, size = RUN_HEADER.unpack(header)
                records = np.frombuffer(decoder.decompress(stream.read(size)), dtype=RECORD)
                if len(records) != count:
                    raise OSError(f"Incorrect spool run: {source}")
                keys = pose_keys(records["pose"])
                if np.any((keys >> np.uint64(BUCKET_SHIFT)) != bucket):
                    raise ValueError(f"Bucket {bucket} has wrong keys")
                ys, starts, lengths = np.unique((keys >> np.uint64(30)) & 511,
                                                return_index=True, return_counts=True)
                for y, start, length in zip(ys, starts, lengths):
                    payload = compressor.compress(records[start:start + length].tobytes())
                    with (split_dir / f"{int(y):03d}.run").open("ab") as target:
                        target.write(RUN_HEADER.pack(int(length), len(payload)))
                        target.write(payload)
                source_count += count
        indexed_count = 0
        for split_path in sorted(split_dir.glob("*.run")):
            y = int(split_path.stem)
            records = load_bucket(split_path)
            keys = pose_keys(records["pose"])
            if np.any((keys >> np.uint64(30)) != (bucket << 9) + y):
                raise ValueError(f"Bucket {bucket}, {y} has wrong keys")
            order = np.argsort(keys, kind="stable")
            records = records[order]
            keys = keys[order]
            name = f"{bucket:03d}_{y:03d}"
            path = output_dir / f"{name}.zst"
            temporary = path.with_suffix(".zst.tmp")
            first, last, offsets = [], [], [0]
            with temporary.open("wb") as stream:
                for begin in range(0, len(records), BLOCK_ROWS):
                    end = min(begin + BLOCK_ROWS, len(records))
                    stream.write(compressor.compress(records[begin:end].tobytes()))
                    first.append(keys[begin])
                    last.append(keys[end - 1])
                    offsets.append(stream.tell())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            index = output_dir / f"{name}.npz"
            temporary_index = index.with_suffix(".npz.tmp")
            with temporary_index.open("wb") as stream:
                np.savez(stream, first=np.asarray(first, dtype=np.uint64),
                         last=np.asarray(last, dtype=np.uint64),
                         offsets=np.asarray(offsets, dtype=np.uint64),
                         count=np.uint64(len(records)))
                stream.flush()
                os.fsync(stream.fileno())
            temporary_index.replace(index)
            indexed_count += len(records)
            del records, keys, order
            split_path.unlink()
        if indexed_count != source_count:
            raise ValueError(f"Bucket {bucket} row count differs after splitting")
        metadata["indexed_rows"] += indexed_count
        metadata["finished_buckets"].append(bucket)
        save_json(args.output / "metadata.json", metadata)
        source.unlink()
        split_dir.rmdir()
        print(f"index bucket {bucket + 1}/{BUCKETS}, "
              f"{metadata['indexed_rows']:,}/{metadata['total_rows']:,} rows", flush=True)
    if metadata["indexed_rows"] != metadata["total_rows"]:
        raise ValueError("Final row count differs from joint lattice")
    metadata["phase"] = "complete"
    save_json(args.output / "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=REPO / "abc_sim/models/yam_dishrack_base.xml")
    parser.add_argument("--cube-width", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-cubes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("cpu", "mjwarp"), default="mjwarp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    model = arm_model(args.model, "left")
    counts = joint_counts(model)
    total = math.prod(counts.tolist())
    config = {"model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "joint_steps_rad": STEPS.tolist(), "joint_counts": counts.tolist(),
              "joint_ranges_rad": model.jnt_range.tolist(), "cube_width": args.cube_width,
              "position_step_m": 0.01, "rotation_step_rad": 0.1,
              "pose_columns": ["x", "y", "z", "rotvec_x", "rotvec_y", "rotvec_z"],
              "pose_dtype": "float32", "row_dtype": "uint40_le",
              "frame": "left_arm base", "site": "left_tcp_site",
              "backend": args.backend}
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["config"] != config or metadata["scratch"] != str(args.scratch.resolve()):
            raise ValueError("Existing table has different configuration or scratch path")
    else:
        metadata = {"config": config, "scratch": str(args.scratch.resolve()),
                    "phase": "stage", "total_rows": total, "completed_cubes": 0,
                    "completed_rows": 0, "spool_sizes": [0] * BUCKETS,
                    "finished_buckets": [], "indexed_rows": 0}
        save_json(metadata_path, metadata)
    print(f"{total:,} rows, counts={counts.tolist()}, phase={metadata['phase']}", flush=True)
    if metadata["phase"] == "stage":
        stage(args, model, counts, metadata)
    if metadata["phase"] == "finalize":
        finalize(args, metadata)


if __name__ == "__main__":
    main()
