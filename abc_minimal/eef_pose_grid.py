"""Sparse six-coordinate EEF cells containing every matching joint-grid row."""

import json
import os
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def write_cell_index(base, keys, counts, block_size=4096):
    byte_offsets, row_offsets = [0], [0]
    dtype = np.dtype([("delta", "<u8"), ("count", "<u4")])
    path = base.with_suffix(".blocks")
    temporary = path.with_suffix(".blocks.tmp")
    with temporary.open("wb") as stream:
        for start in range(0, len(keys), block_size):
            end = min(start + block_size, len(keys))
            block = np.empty(end - start, dtype=dtype)
            block["delta"] = np.diff(keys[start:end], prepend=np.uint64(0))
            block["count"] = counts[start:end]
            stream.write(zlib.compress(block.tobytes(), level=1))
            byte_offsets.append(stream.tell())
            row_offsets.append(row_offsets[-1] + int(counts[start:end].sum()))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    path = base.with_suffix(".directory.npz")
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, first_keys=keys[::block_size],
                            byte_offsets=np.asarray(byte_offsets, dtype=np.uint64),
                            row_offsets=np.asarray(row_offsets, dtype=np.uint32))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class EEFPoseGrid:
    def __init__(self, path, source=None):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        if self.metadata["status"] != "complete":
            raise ValueError("Pose grid is not complete")
        self.config = self.metadata["config"]
        self.grid = self.config["grid"]
        self.joint_grid = self.config["joint_grid"]
        self.origin = np.asarray(self.grid["position_origin"], dtype=np.float32)
        self.bits = [self.grid["position_bits"]] * 3 + [self.grid["rotation_bits"]] * 3
        self.bucket_shift = sum(self.bits[1:])
        self.local_mask = np.uint64((1 << self.bucket_shift) - 1)
        self.steps = np.array([self.grid["position_step_m"]] * 3
                              + [self.grid["rotation_step_rad"]] * 3, dtype=np.float32)
        self.offset = np.r_[self.origin, np.zeros(3, dtype=np.float32)]
        self.bias = np.array([0, 0, 0] + [self.grid["rotation_bias"]] * 3, dtype=np.int64)
        directory = Path(source or self.config["source"])
        self.poses = np.load(directory / "poses.npy", mmap_mode="r", allow_pickle=False)

    def encode(self, pose_vectors):
        values = np.asarray(pose_vectors, dtype=np.float32)
        cells = np.floor((values - self.offset) / self.steps).astype(np.int64) + self.bias
        return self.encode_cells(cells)

    def encode_cells(self, cells):
        cells = np.asarray(cells, dtype=np.int64)
        valid = ((cells >= 0) & (cells < (1 << np.array(self.bits)))).all(axis=-1)
        keys = np.zeros(cells.shape[:-1], dtype=np.uint64)
        for axis, bits in enumerate(self.bits):
            keys = keys * np.uint64(1 << bits) + np.maximum(cells[..., axis], 0).astype(np.uint64)
        return keys, valid

    def joints(self, rows):
        counts = np.asarray(self.joint_grid["joint_grid_counts"], dtype=np.int64)
        strides = np.r_[1, np.cumprod(counts[:-1])]
        digits = np.asarray(rows, dtype=np.int64)[..., None] // strides % counts
        limits = np.asarray(self.joint_grid["joint_ranges_rad"], dtype=np.float32)
        return np.minimum(limits[:, 0] + digits.astype(np.float32)
                          * np.float32(self.joint_grid["joint_step_rad"]), limits[:, 1])

    @lru_cache(maxsize=8)
    def _bucket(self, bucket):
        base = self.path / "cells" / f"{bucket:04d}"
        directory = base.with_suffix(".directory.npz")
        if not directory.exists():
            return {"first_keys": np.empty(0, dtype=np.uint64)}, np.empty(0, dtype=np.uint32)
        with np.load(directory, allow_pickle=False) as saved:
            index = {name: saved[name] for name in saved.files}
        return index, np.memmap(base.with_suffix(".rows"), mode="r", dtype=np.uint32)

    @lru_cache(maxsize=1024)
    def _block(self, bucket, block):
        index, _ = self._bucket(bucket)
        begin, end = index["byte_offsets"][block:block + 2]
        with (self.path / "cells" / f"{bucket:04d}.blocks").open("rb") as stream:
            stream.seek(int(begin))
            data = zlib.decompress(stream.read(int(end - begin)))
        values = np.frombuffer(data, dtype=[("delta", "<u8"), ("count", "<u4")])
        keys = np.cumsum(values["delta"], dtype=np.uint64)
        start = index["row_offsets"][block]
        offsets = np.r_[start, start + np.cumsum(values["count"], dtype=np.uint32)]
        return keys, offsets

    def rows_in_cells(self, keys):
        keys = np.unique(np.asarray(keys, dtype=np.uint64).reshape(-1))
        buckets = keys >> np.uint64(self.bucket_shift)
        found = []
        for bucket in np.unique(buckets):
            index, rows = self._bucket(int(bucket))
            queries = keys[buckets == bucket] & self.local_mask
            blocks = np.searchsorted(index["first_keys"], queries, side="right") - 1
            for block in np.unique(blocks[blocks >= 0]):
                stored, offsets = self._block(int(bucket), int(block))
                query = queries[blocks == block]
                indices = np.searchsorted(stored, query)
                inside = indices < len(stored)
                indices, query = indices[inside], query[inside]
                for cell in indices[stored[indices] == query]:
                    found.append(rows[offsets[cell]:offsets[cell + 1]])
        return np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.uint32)

    def cell_rows(self, target):
        """All row IDs in the exact cell of a base-frame (4, 4) TCP pose."""
        target = np.asarray(target, dtype=np.float64)
        values = np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_rotvec()]
        keys, valid = self.encode(values[None])
        return self.rows_in_cells(keys[valid])

    def query(self, target, reference=None, k=8, neighbors=1):
        """Rank IK seeds; reference is a physical six-joint vector in radians."""
        target = np.asarray(target, dtype=np.float64)
        rotation = Rotation.from_matrix(target[:3, :3])
        rotvec = rotation.as_rotvec()
        representations = [rotvec]
        angle = np.linalg.norm(rotvec)
        if angle > 1e-12:
            representations.extend((rotvec * (1 + 2 * np.pi / angle), rotvec * (1 - 2 * np.pi / angle)))
        offsets = np.array(np.meshgrid(*([np.arange(-neighbors, neighbors + 1)] * 6),
                                       indexing="ij")).reshape(6, -1).T
        cells = []
        for representation in representations:
            values = np.r_[target[:3, 3], representation].astype(np.float32)
            center = np.floor((values - self.offset) / self.steps).astype(np.int64) + self.bias
            cells.append(center + offsets)
        keys, valid = self.encode_cells(np.concatenate(cells))
        rows = self.rows_in_cells(keys[valid])
        poses = self.poses[rows]
        joints = self.joints(rows)
        position_error = np.linalg.norm(poses[:, :3] - target[:3, 3], axis=1)
        rotation_error = (rotation.inv() * Rotation.from_rotvec(poses[:, 3:])).magnitude() if len(rows) else np.empty(0)
        pose_score = (position_error / self.steps[0]) ** 2 + (rotation_error / self.steps[3]) ** 2
        joint_distance = np.linalg.norm(joints - reference, axis=1) if reference is not None else pose_score
        order = np.lexsort((pose_score, joint_distance))[:k]
        return {"rows": rows[order], "joints": joints[order], "poses": poses[order],
                "position_error_m": position_error[order], "rotation_error_rad": rotation_error[order]}
