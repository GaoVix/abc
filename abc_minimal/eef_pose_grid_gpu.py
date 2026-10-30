import json
import mmap
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
        self.has_pose_index = config.get("pose_index", True)
        self.cells_path = Path(metadata.get("index_source", self.path)) / "cells"
        self.total_rows = metadata["total_rows"]
        self.poses = np.memmap(self.path / "poses.bin", mode="r", dtype="<f4", shape=(self.total_rows, 6))
        self.poses._mmap.madvise(mmap.MADV_RANDOM)
        self.counts = np.asarray(config["joint_counts"], dtype=np.int64)
        self.strides = np.r_[1, np.cumprod(self.counts[:-1], dtype=np.int64)]
        self.steps = np.asarray(config["joint_steps_rad"], dtype=np.float32)
        self.limits = np.asarray(config["joint_ranges_rad"], dtype=np.float32)

    def query(self, target, reference=None, k=8, neighbors=1, side=None, distance_limit=0.5):
        if not self.has_pose_index:
            raise ValueError("The FK table's EEF-to-joint index is not built yet")
        target = np.asarray(target, dtype=np.float64)
        rotation = Rotation.from_matrix(target[:3, :3])
        if side is None:
            rows = self._nearby_rows(target[:3, 3], rotation.as_rotvec(), neighbors)
            poses = self.poses[rows]
        else:
            centers = self._cell_centers(target[:3, 3], rotation.as_rotvec(), neighbors)
            rows, poses, scores = self._filtered_neighborhood(centers, neighbors, side, distance_limit)
        position_error = np.linalg.norm(poses[:, :3] - target[:3, 3], axis=1)
        rotation_error = (rotation.inv() * Rotation.from_rotvec(poses[:, 3:])).magnitude() if len(rows) else np.empty(0)
        pose_score = (position_error / 0.01) ** 2 + (rotation_error / 0.1) ** 2
        if reference is None:
            eligible = (np.flatnonzero(pose_score <= np.partition(pose_score, k - 1)[k - 1])
                        if 0 < k < len(rows) else np.arange(len(rows)))
            order = eligible[np.argsort(pose_score[eligible], kind="stable")[:k]]
        else:
            joint_distance = np.linalg.norm(self.joints(rows) - reference, axis=1)
            order = np.lexsort((pose_score, joint_distance))[:k]
        result = {"rows": rows[order], "joints": self.joints(rows[order]), "poses": poses[order],
                  "position_error_m": position_error[order], "rotation_error_rad": rotation_error[order]}
        if side is not None:
            result["teleop_distance"] = scores[order]
        return result

    def joints(self, rows):
        digits = np.asarray(rows, dtype=np.uint64)[..., None] // self.strides % self.counts
        return np.minimum(self.limits[:, 0] + digits.astype(np.float32) * self.steps, self.limits[:, 1])

    def teleop_distance(self, rows, side):
        rows = np.asarray(rows, dtype=np.int64)
        if np.any((rows < 0) | (rows >= self.total_rows)):
            raise IndexError("Grid row is outside the score table")
        result = np.full(rows.shape, np.nan, dtype=np.float32)
        for start, end, values in self._teleop_score_shards(side):
            selected = (rows >= start) & (rows < end)
            result[selected] = values[rows[selected] - start]
        return result

    def _nearby_rows(self, position, rotvec, neighbors):
        return self.rows_in_cells(self._nearby_keys(position, rotvec, neighbors))

    def _nearby_keys(self, position, rotvec, neighbors):
        return self._keys_near_centers(self._cell_centers(position, rotvec, neighbors), neighbors)

    @staticmethod
    def _cell_centers(position, rotvec, neighbors=1):
        representations = [rotvec]
        angle = np.linalg.norm(rotvec)
        if angle > 1e-12:
            representations.extend((rotvec * (1 + 2 * np.pi / angle), rotvec * (1 - 2 * np.pi / angle)))
        cell_size = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1], dtype=np.float32)
        poses = np.array([np.r_[position, vector] for vector in representations], dtype=np.float32)
        centers = np.floor(poses / cell_size).astype(np.int64) + [256, 256, 256, 64, 64, 64]
        # Stored rotvecs have norm <= pi; keep neighborhoods crossing that boundary.
        low = (centers[:, 3:] - 64 - neighbors) * float(cell_size[3])
        high = (centers[:, 3:] - 64 + neighbors + 1) * float(cell_size[3])
        closest = np.maximum(np.maximum(low, -high), 0.)
        centers = centers[np.linalg.norm(closest, axis=1) <= np.pi + 1e-5]
        return tuple(centers.ravel().tolist())

    def _keys_near_centers(self, centers, neighbors):
        centers = np.asarray(centers).reshape(-1, 6)
        axis_offsets = np.arange(-neighbors, neighbors + 1)
        offsets = np.stack(np.meshgrid(*([axis_offsets] * 6), indexing="ij"), axis=-1).reshape(-1, 6)
        cells = (centers[:, None, :] + offsets).reshape(-1, 6)
        keys, valid = self.encode_cells(cells)
        return keys[valid]

    @lru_cache(maxsize=2)
    def _teleop_score_shards(self, side):
        metadata = json.loads((self.path / "metadata.json").read_text())
        by_side = metadata.get("similarity_by_side") or {}
        if "similarity" in metadata:
            previous = metadata["similarity"]
            by_side.setdefault(previous.get("reference_side", "left"), previous)
        score = by_side[side]
        shards = score.get("shards") or [dict(start_row=0, end_row=self.total_rows, path=score["path"])]
        tables = []
        for part in shards:
            start, end = part["start_row"], part["end_row"]
            values = np.memmap(part["path"], mode="r", dtype="<f4", shape=(end - start,))
            values._mmap.madvise(mmap.MADV_RANDOM)
            tables.append((start, end, values))
        return tables

    @lru_cache(maxsize=256)
    def _bucket(self, bucket):
        base = self.cells_path / f"{bucket:03d}"
        directory = base.with_suffix(".directory.npz")
        if not directory.exists():
            return None
        with np.load(directory, allow_pickle=False) as saved:
            index = {name: saved[name] for name in saved.files}
        index["payload"] = np.memmap(base.with_suffix(".blocks"), mode="r", dtype=np.uint8)
        count = int(index["row_offsets"][-1])
        rows = np.memmap(base.with_suffix(".rows5"), mode="r", dtype=np.uint8, shape=(count, 5))
        rows._mmap.madvise(mmap.MADV_RANDOM)
        return index, rows

    @lru_cache(maxsize=16384)
    def _block(self, bucket, block):
        index, _ = self._bucket(bucket)
        begin, end = index["byte_offsets"][block:block + 2]
        payload = zlib.decompress(index["payload"][begin:end], bufsize=48 * 1024)
        entries = np.frombuffer(payload, dtype=[("delta", "<u8"), ("count", "<u4")])
        keys = np.cumsum(entries["delta"], dtype=np.uint64)
        start = index["row_offsets"][block]
        offsets = np.r_[start, start + np.cumsum(entries["count"], dtype=np.uint32)]
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

    def _matching_cells(self, keys):
        bucket_shift = 39
        keys = np.unique(np.asarray(keys, dtype=np.uint64).reshape(-1))
        buckets = keys >> np.uint64(bucket_shift)
        for bucket in np.unique(buckets):
            stored = self._bucket(int(bucket))
            if stored is None:
                continue
            index, _ = stored
            queries = keys[buckets == bucket] & np.uint64((1 << bucket_shift) - 1)
            blocks = np.searchsorted(index["first_keys"], queries, side="right") - 1
            for block in np.unique(blocks[blocks >= 0]):
                cell_keys, _ = self._block(int(bucket), int(block))
                selected = queries[blocks == block]
                positions = np.searchsorted(cell_keys, selected)
                valid = positions < len(cell_keys)
                positions, selected = positions[valid], selected[valid]
                matches = positions[cell_keys[positions] == selected]
                for position in matches:
                    yield int(bucket), int(block), int(position)

    def _cell_rows(self, bucket, block, position):
        _, offsets = self._block(bucket, block)
        _, rows = self._bucket(bucket)
        packed_rows = rows[offsets[position]:offsets[position + 1]]
        row_bytes = np.zeros((len(packed_rows), 8), dtype=np.uint8)
        row_bytes[:, :5] = packed_rows
        return row_bytes.view("<u8").reshape(-1)

    @lru_cache(maxsize=65536)
    def _filtered_cell(self, bucket, block, position, side, distance_limit):
        rows = self._cell_rows(bucket, block, position)
        scores = self.teleop_distance(rows, side)
        keep = scores < distance_limit
        rows, scores = rows[keep], scores[keep]
        return rows, self.poses[rows], scores

    @lru_cache(maxsize=128)
    def _filtered_neighborhood(self, centers, neighbors, side, distance_limit):
        keys = self._keys_near_centers(centers, neighbors)
        cells = [self._filtered_cell(bucket, block, position, side, distance_limit)
                 for bucket, block, position in self._matching_cells(keys)]
        cells = [cell for cell in cells if len(cell[0])]
        if not cells:
            return np.empty(0, dtype=np.uint64), np.empty((0, 6), dtype=np.float32), np.empty(0, dtype=np.float32)
        rows, indices = np.unique(np.concatenate([cell[0] for cell in cells]), return_index=True)
        poses = np.concatenate([cell[1] for cell in cells])[indices]
        scores = np.concatenate([cell[2] for cell in cells])[indices]
        return rows, poses, scores

    def rows_in_cells(self, keys):
        found = [self._cell_rows(bucket, block, position)
                 for bucket, block, position in self._matching_cells(keys)]
        return np.unique(np.concatenate(found)) if found else np.empty(0, dtype=np.uint64)
