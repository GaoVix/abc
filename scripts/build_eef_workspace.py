"""Build recorded EEF-pose intervals from a complete joint grid.

Poses use the selected arm's base frame and TCP site. Queries accept component
values between recorded minima and maxima in the same position cell, without
padding. This is an approximation, not an exact reachability test.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp


REPO = Path(__file__).resolve().parents[1]


def arm_model(path, side):
    spec = mujoco.MjSpec.from_file(str(path))
    for field in ("sensors", "equalities", "actuators", "tendons", "keys"):
        for element in list(getattr(spec, field)):
            spec.delete(element)
    for body in list(spec.worldbody.bodies):
        if body.name != f"{side}_arm":
            spec.delete(body)
    for body in list(spec.body(f"{side}_link_6").bodies):
        spec.delete(body)
    for geom in list(spec.geoms):
        spec.delete(geom)
    base = spec.body(f"{side}_arm")
    base.pos[:] = 0
    base.quat[:] = [1, 0, 0, 0]
    return spec.compile()


@wp.kernel
def fill_joints(
    start: wp.int64,
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    step: wp.float32,
    qpos: wp.array2d(dtype=wp.float32),
):
    world = wp.tid()
    index = start + wp.int64(world)
    for joint in range(6):
        digit = index % wp.int64(counts[joint])
        index = index // wp.int64(counts[joint])
        qpos[world, joint] = wp.min(lower[joint] + wp.float32(digit) * step, upper[joint])


@wp.kernel
def extract_poses(
    site: int,
    positions: wp.array2d(dtype=wp.vec3),
    rotations: wp.array2d(dtype=wp.mat33),
    poses: wp.array2d(dtype=wp.float32),
):
    world = wp.tid()
    axis, angle = wp.quat_to_axis_angle(wp.quat_from_matrix(rotations[world, site]))
    for component in range(3):
        poses[world, component] = positions[world, site][component]
        poses[world, component + 3] = axis[component] * angle


def save_all(args):
    started = time.monotonic()
    model = arm_model(args.model, args.side)
    counts_np = np.array([len(np.arange(lo, hi, args.joint_step)) + 1
                          for lo, hi in model.jnt_range], dtype=np.int32)
    total = math.prod(int(count) for count in counts_np)
    config = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "frame": f"{args.side}_arm",
        "site": f"{args.side}_tcp_site",
        "joint_ranges_rad": model.jnt_range.tolist(),
        "joint_step_rad": args.joint_step,
        "joint_grid_counts": counts_np.tolist(),
        "joint_index_order": "joint1 fastest, joint6 slowest; mixed-radix row index",
        "joint_value_rule": "min(lower + digit * joint_step, upper), float32",
        "pose_columns": ["x_m", "y_m", "z_m", "rotvec_x_rad", "rotvec_y_rad", "rotvec_z_rad"],
        "rotation_representation": "axis multiplied by angle, principal angle in [0, pi]",
        "dtype": "float32",
        "shape": [total, 6],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.json"
    partial_path = args.output / "poses.partial.npy"
    final_path = args.output / "poses.npy"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["config"] != config:
            raise ValueError("Saved sample configuration differs; choose another output directory")
        if metadata["status"] == "complete":
            saved = np.load(final_path, mmap_mode="r", allow_pickle=False)
            assert saved.shape == (total, 6)
            print(f"Already complete: {final_path}, {total:,} poses", flush=True)
            return
        completed = metadata["completed"]
        saved = np.load(partial_path, mmap_mode="r+", allow_pickle=False)
    else:
        required = total * 6 * np.dtype(np.float32).itemsize + 4096
        if shutil.disk_usage(args.output).free < required + 5 * 2**30:
            raise OSError(f"Need {required / 1e9:.2f} GB plus 5 GiB free space")
        saved = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=(total, 6))
        with partial_path.open("r+b") as stream:
            os.posix_fallocate(stream.fileno(), 0, partial_path.stat().st_size)
        completed = 0
        metadata = {"config": config, "completed": 0, "status": "running", "file": partial_path.name}

    def checkpoint():
        saved.flush()
        with partial_path.open("rb") as stream:
            os.fsync(stream.fileno())
        metadata["completed"] = completed
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)

    checkpoint()
    wp.init()
    wp.set_device(args.device)
    cpu_data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, cpu_data)
    warp_model = mjw.put_model(model)
    data = mjw.put_data(model, cpu_data, nworld=args.batch_size, nconmax=0, njmax=0)
    joint_min = wp.array(model.jnt_range[:, 0].astype(np.float32))
    joint_max = wp.array(model.jnt_range[:, 1].astype(np.float32))
    counts = wp.array(counts_np)
    poses = wp.empty((args.batch_size, 6), dtype=wp.float32)
    site = model.site(config["site"]).id
    print(f"Saving {total:,} full poses ({total * 24 / 1e9:.2f} GB) to {args.output}", flush=True)
    print(f"joint grid={counts_np.tolist()}, resuming at {completed:,}", flush=True)
    initial_completed = completed
    last_log = sampled_at = time.monotonic()
    try:
        for start in range(completed, total, args.batch_size):
            count = min(args.batch_size, total - start)
            wp.launch(fill_joints, dim=args.batch_size, inputs=[
                wp.int64(start), joint_min, joint_max, counts, args.joint_step, data.qpos,
            ])
            mjw.kinematics(warp_model, data)
            wp.launch(extract_poses, dim=args.batch_size,
                      inputs=[site, data.site_xpos, data.site_xmat, poses])
            saved[start:start + count] = poses.numpy()[:count]
            completed = start + count
            if time.monotonic() - last_log >= 10 or completed == total:
                checkpoint()
                last_log = time.monotonic()
                rate = (completed - initial_completed) / (last_log - sampled_at)
                print(f"{completed:,}/{total:,} ({100 * completed / total:.1f}%), "
                      f"{rate / 1e6:.2f} M poses/s saved", flush=True)
    except KeyboardInterrupt:
        checkpoint()
        raise
    partial_path.replace(final_path)
    metadata.update(status="complete", file=final_path.name,
                    last_run_seconds=time.monotonic() - started)
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
    print(f"Saved all {total:,} poses: {final_path}", flush=True)


@wp.kernel
def accumulate_poses(
    valid_count: int,
    site: int,
    origin: wp.vec3,
    voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
    positions: wp.array2d(dtype=wp.vec3),
    rotations: wp.array2d(dtype=wp.mat33),
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
    outside: wp.array(dtype=wp.int32),
):
    world = wp.tid()
    if world < valid_count:
        position = (positions[world, site] - origin) / voxel_size
        x = int(wp.floor(position[0]))
        y = int(wp.floor(position[1]))
        z = int(wp.floor(position[2]))
        if x >= 0 and x < nx and y >= 0 and y < ny and z >= 0 and z < nz:
            cell = (x * ny + y) * nz + z
            for component in range(3):
                value = positions[world, site][component]
                if value < lower[component, cell]:
                    wp.atomic_min(lower, component, cell, value)
                if value > upper[component, cell]:
                    wp.atomic_max(upper, component, cell, value)
            rotation = rotations[world, site]
            for row in range(3):
                for column in range(3):
                    component = 3 + row * 3 + column
                    value = rotation[row, column]
                    if value < lower[component, cell]:
                        wp.atomic_min(lower, component, cell, value)
                    if value > upper[component, cell]:
                        wp.atomic_max(upper, component, cell, value)
        else:
            wp.atomic_add(outside, 0, 1)


class EEFWorkspace:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as saved:
            self.metadata = json.loads(str(saved["metadata"]))
            self.origin = saved["origin"].astype(np.float32)
            self.voxel_size = float(saved["voxel_size"])
            self.position_min = saved["position_min"]
            self.position_max = saved["position_max"]
            self.rotation_min = saved["rotation_min"]
            self.rotation_max = saved["rotation_max"]

    def is_reachable(self, poses, base_from_frame=None):
        """Accept (..., 4, 4) poses inside recorded intervals, approximately."""
        poses = np.asarray(poses, dtype=np.float64)
        if base_from_frame is not None:
            poses = np.asarray(base_from_frame) @ poses
        shape = poses.shape[:-2]
        poses = poses.reshape(-1, 4, 4)
        positions = poses[:, :3, 3].astype(np.float32)
        cells = np.floor((positions - self.origin) / np.float32(self.voxel_size)).astype(np.int64)
        inside = ((cells >= 0) & (cells < self.rotation_min.shape[1:])).all(axis=1)
        accepted = np.zeros(len(poses), dtype=bool)
        indices = np.flatnonzero(inside)
        x, y, z = cells[indices].T
        positions = positions[indices]
        rotations = poses[indices, :3, :3].reshape(-1, 9).astype(np.float32)
        accepted[indices] = (
            ((positions >= self.position_min[:, x, y, z].T)
             & (positions <= self.position_max[:, x, y, z].T)).all(axis=1)
            & ((rotations >= self.rotation_min[:, x, y, z].T)
               & (rotations <= self.rotation_max[:, x, y, z].T)).all(axis=1)
        )
        return accepted.reshape(shape)

    def is_unreachable(self, poses, base_from_frame=None):
        """Return uncovered poses; this does not prove physical unreachability."""
        return ~self.is_reachable(poses, base_from_frame)


def build(args):
    started = time.monotonic()
    model = arm_model(args.model, args.side)
    cpu_data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, cpu_data)
    site = model.site(f"{args.side}_tcp_site").id
    axes = [np.r_[np.arange(lo, hi, args.joint_step), hi] for lo, hi in model.jnt_range]
    counts_np = np.array([len(axis) for axis in axes], dtype=np.int32)
    total = math.prod(int(count) for count in counts_np)
    anchors = np.vstack((cpu_data.xanchor, cpu_data.site_xpos[site]))
    lengths = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
    levers = np.cumsum(lengths[::-1])[::-1]
    extent = levers[0] + args.voxel_size
    origin = (np.floor((anchors[0] - extent) / args.voxel_size) * args.voxel_size).astype(np.float32)
    grid_shape = np.ceil((anchors[0] + extent - origin) / args.voxel_size).astype(int)
    num_cells = math.prod(int(n) for n in grid_shape)

    wp.init()
    wp.set_device(args.device)
    warp_model = mjw.put_model(model)
    data = mjw.put_data(model, cpu_data, nworld=args.batch_size, nconmax=0, njmax=0)
    joint_min = wp.array(model.jnt_range[:, 0].astype(np.float32))
    joint_max = wp.array(model.jnt_range[:, 1].astype(np.float32))
    counts = wp.array(counts_np)
    lower = wp.full((12, num_cells), float("inf"), dtype=wp.float32)
    upper = wp.full((12, num_cells), -float("inf"), dtype=wp.float32)
    outside = wp.zeros(1, dtype=wp.int32)
    print(f"{args.side} arm, base frame, TCP={model.site(site).name}", flush=True)
    print(f"joint grid={counts_np.tolist()}, combinations={total:,}, voxel grid={grid_shape.tolist()}", flush=True)
    print("Recorded position/rotation component intervals only; no padding", flush=True)

    sampled_at = last_log = time.monotonic()
    for start in range(0, total, args.batch_size):
        count = min(args.batch_size, total - start)
        wp.launch(fill_joints, dim=args.batch_size, inputs=[
            wp.int64(start), joint_min, joint_max, counts, args.joint_step, data.qpos,
        ])
        mjw.kinematics(warp_model, data)
        wp.launch(accumulate_poses, dim=args.batch_size, inputs=[
            count, site, wp.vec3(*origin), args.voxel_size, *grid_shape,
            data.site_xpos, data.site_xmat, lower, upper,
            outside,
        ])
        if time.monotonic() - last_log >= 5 or start + count == total:
            wp.synchronize_device()
            last_log = time.monotonic()
            completed = start + count
            rate = completed / (last_log - sampled_at)
            print(f"{completed:,}/{total:,} ({100 * completed / total:.1f}%), {rate / 1e6:.1f} M combinations/s", flush=True)
    if outside.numpy()[0]:
        raise RuntimeError("FK samples fell outside the allocated voxel grid")
    sample_seconds = time.monotonic() - sampled_at

    lower = lower.numpy().reshape(12, *grid_shape)
    upper = upper.numpy().reshape(12, *grid_shape)
    metadata = {
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "side": args.side,
        "frame": f"{args.side}_arm",
        "site": f"{args.side}_tcp_site",
        "joint_ranges_rad": model.jnt_range.tolist(),
        "joint_step_rad": args.joint_step,
        "joint_grid_counts": counts_np.tolist(),
        "combinations": total,
        "voxel_size_m": args.voxel_size,
        "position_coverage_margin_m": 0,
        "rotation_coverage_margin_rad": 0,
        "rotation_matrix_entry_margin": 0,
        "position_padding_cells": 0,
        "rotation_representation": "row-major 3x3 matrix component bounds",
        "query_rule": "componentwise intervals within the same recorded position cell; inclusive endpoints; no padding",
        "sample_seconds": sample_seconds,
        "meaning": "inside recorded intervals is treated as reachable approximately; outside is uncovered, not proven unreachable",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), origin=origin,
                            voxel_size=args.voxel_size,
                            position_min=lower[:3], position_max=upper[:3],
                            rotation_min=lower[3:], rotation_max=upper[3:])
    temporary.replace(args.output)
    print(f"Saved {args.output}: {args.output.stat().st_size / 1e6:.2f} MB, {time.monotonic() - started:.1f}s total", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO / "abc_sim/models/yam_dishrack_base.xml")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--joint-step", type=float, default=0.1, help="radians")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="metres")
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-all", action="store_true", help="save every FK pose without voxel aggregation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.joint_step, args.voxel_size, args.batch_size) <= 0:
        parser.error("joint-step, voxel-size and batch-size must be positive")
    if args.save_all:
        args.output = args.output or REPO / "outputs/eef_fk_all_0p1rad"
        save_all(args)
    else:
        args.output = args.output or REPO / "outputs/eef_workspace_0p1rad_1cm_no_padding.npz"
        build(args)


if __name__ == "__main__":
    main()
