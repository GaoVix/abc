"""Query a GPU-built lossless EEF pose-to-joint grid."""

import json
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class GPUEEFPoseGrid:
    def __init__(self, path):
        self.path = Path(path)
        metadata = json.loads((self.path / "metadata.json").read_text())
        if metadata["phase"] != "complete":
            raise ValueError("EEF pose grid is not complete")
        config = metadata["config"]
        self.total_rows = metadata["total_rows"]
        self.poses = np.memmap(self.path / "poses.bin", mode="r", dtype="<f4",
                               shape=(self.total_rows, 6))
        self.counts = np.asarray(config["joint_counts"], dtype=np.int64)
        self.strides = np.r_[1, np.cumprod(self.counts[:-1], dtype=np.int64)]
        self.steps = np.asarray(config["joint_steps_rad"], dtype=np.float32)
        self.limits = np.asarray(config["joint_ranges_rad"], dtype=np.float32)

    @lru_cache(maxsize=8)
    def _bucket(self, bucket):
        base = self.path / "cells" / f"{bucket:03d}"
        directory = base.with_suffix(".directory.npz")
        if not directory.exists():
            return None
        with np.load(directory, allow_pickle=False) as saved:
            index = {name: saved[name] for name in saved.files}
        count = int(index["row_offsets"][-1])
        rows = np.memmap(base.with_suffix(".rows5"), mode="r", dtype=np.uint8,
                         shape=(count, 5))
        return index, rows

    @lru_cache(maxsize=1024)
    def _block(self, bucket, number):
        index, _ = self._bucket(bucket)
        begin, end = index["byte_offsets"][number:number + 2]
        with (self.path / "cells" / f"{bucket:03d}.blocks").open("rb") as stream:
            stream.seek(int(begin))
            values = np.frombuffer(zlib.decompress(stream.read(int(end - begin))),
                                   dtype=[("delta", "<u8"), ("count", "<u4")])
        keys = np.cumsum(values["delta"], dtype=np.uint64)
        start = index["row_offsets"][number]
        offsets = np.r_[start, start + np.cumsum(values["count"], dtype=np.uint32)]
        return keys, offsets

    @staticmethod
    def encode_cells(cells):
        bits = (9, 9, 9, 7, 7, 7)
        cells = np.asarray(cells, dtype=np.int64)
        valid = ((cells >= 0) & (cells < (1 << np.asarray(bits)))).all(axis=-1)
        keys = np.zeros(cells.shape[:-1], dtype=np.uint64)
        for axis, width in enumerate(bits):
            keys = (keys << width) | np.maximum(cells[..., axis], 0).astype(np.uint64)
        return keys, valid

    def rows_in_cells(self, keys):
        keys = np.unique(np.asarray(keys, dtype=np.uint64).reshape(-1))
        buckets = keys >> np.uint64(39)
        found = []
        for bucket in np.unique(buckets):
            stored = self._bucket(int(bucket))
            if stored is None:
                continue
            index, rows = stored
            queries = keys[buckets == bucket] & np.uint64((1 << 39) - 1)
            blocks = np.searchsorted(index["first_keys"], queries, side="right") - 1
            for block in np.unique(blocks[blocks >= 0]):
                cell_keys, offsets = self._block(int(bucket), int(block))
                selected = queries[blocks == block]
                positions = np.searchsorted(cell_keys, selected)
                valid = positions < len(cell_keys)
                for position in positions[valid & (cell_keys[np.minimum(positions, len(cell_keys) - 1)] == selected)]:
                    packed = rows[offsets[position]:offsets[position + 1]]
                    unpacked = np.zeros((len(packed), 8), dtype=np.uint8)
                    unpacked[:, :5] = packed
                    found.append(unpacked.view("<u8").reshape(-1))
        return np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.uint64)

    def joints(self, rows):
        digits = np.asarray(rows, dtype=np.uint64)[..., None] // self.strides % self.counts
        return np.minimum(self.limits[:, 0] + digits.astype(np.float32) * self.steps,
                          self.limits[:, 1])

    def query(self, target, reference=None, k=8, neighbors=1):
        target = np.asarray(target, dtype=np.float64)
        rotation = Rotation.from_matrix(target[:3, :3])
        rotvec = rotation.as_rotvec()
        representations = [rotvec]
        angle = np.linalg.norm(rotvec)
        if angle > 1e-12:
            representations.extend((rotvec * (1 + 2 * np.pi / angle),
                                    rotvec * (1 - 2 * np.pi / angle)))
        offsets = np.stack(np.meshgrid(*([np.arange(-neighbors, neighbors + 1)] * 6),
                                       indexing="ij"), axis=-1).reshape(-1, 6)
        cells = [np.floor(np.r_[target[:3, 3], vector].astype(np.float32)
                          / np.float32([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])).astype(np.int64)
                 + np.array([256, 256, 256, 64, 64, 64]) + offsets
                 for vector in representations]
        keys, valid = self.encode_cells(np.concatenate(cells))
        rows = self.rows_in_cells(keys[valid])
        poses = self.poses[rows]
        joints = self.joints(rows)
        position_error = np.linalg.norm(poses[:, :3] - target[:3, 3], axis=1)
        rotation_error = ((rotation.inv() * Rotation.from_rotvec(poses[:, 3:])).magnitude()
                          if len(rows) else np.empty(0))
        pose_score = (position_error / 0.01) ** 2 + (rotation_error / 0.1) ** 2
        joint_score = (np.linalg.norm(joints - reference, axis=1)
                       if reference is not None else pose_score)
        order = np.lexsort((pose_score, joint_score))[:k]
        return {"rows": rows[order], "joints": joints[order], "poses": poses[order],
                "position_error_m": position_error[order],
                "rotation_error_rad": rotation_error[order]}
