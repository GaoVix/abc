"""Streaming statistics shared by data export and normalization tools."""

import numpy as np


class RunningStats:
    """Streaming mean/std/min/max over (N, dim) batches."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None
        self.min = None
        self.max = None

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if self.mean is None:
            dim = batch.shape[1]
            self.mean = np.zeros(dim)
            self.m2 = np.zeros(dim)
            self.min = np.full(dim, np.inf)
            self.max = np.full(dim, -np.inf)
        for row in batch:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)
        self.min = np.minimum(self.min, batch.min(axis=0))
        self.max = np.maximum(self.max, batch.max(axis=0))

    def statistics(self) -> dict:
        std = np.sqrt(self.m2 / max(self.count - 1, 1))
        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }
