"""Simple local ring buffer for sub-second EEW-style ingestion."""

from __future__ import annotations

from collections import deque

import numpy as np


class WaveformRingBuffer:
    """
    Fixed-capacity multi-channel ring buffer.

    Designed for local/replay ingestion so inference does not wait on HTTP.
    """

    def __init__(self, n_channels: int = 3, capacity_samples: int = 1000) -> None:
        if n_channels < 1 or capacity_samples < 1:
            raise ValueError("n_channels and capacity_samples must be >= 1")
        self.n_channels = n_channels
        self.capacity = capacity_samples
        self._buf = deque(maxlen=capacity_samples)
        self.total_pushed = 0

    def push(self, samples: np.ndarray) -> None:
        """
        Append samples.

        Accepts shape ``(channels,)`` for one sample or ``(channels, n)`` / ``(n, channels)``.
        """
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != self.n_channels:
                raise ValueError(f"Expected {self.n_channels} channels, got {arr.shape}")
            self._buf.append(arr.copy())
            self.total_pushed += 1
            return
        if arr.ndim != 2:
            raise ValueError(f"Expected 1-D or 2-D array, got shape {arr.shape}")
        if arr.shape[0] == self.n_channels:
            cols = arr
        elif arr.shape[1] == self.n_channels:
            cols = arr.T
        else:
            raise ValueError(f"Cannot interpret channels from shape {arr.shape}")
        for i in range(cols.shape[1]):
            self._buf.append(cols[:, i].copy())
        self.total_pushed += cols.shape[1]

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.capacity

    def get_window(self) -> np.ndarray:
        """Return ``(channels, capacity)`` oldest→newest, or raise if not full."""
        if not self.ready:
            raise RuntimeError(
                f"Buffer not full: {len(self._buf)}/{self.capacity} samples"
            )
        return np.stack(list(self._buf), axis=1).astype(np.float32)
