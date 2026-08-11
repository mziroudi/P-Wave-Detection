"""PyTorch Dataset wrappers for windowed STEAD arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        assert len(x) == len(y)
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def save_window_cache(
    out_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", x)
    np.save(out_dir / "y.npy", y)
    # Lightweight JSON-serializable meta via npz of object arrays is awkward;
    # store as CSV-friendly npy of structured fields.
    names = np.array([m["trace_name"] for m in meta], dtype=object)
    starts = np.array([m["start_sample"] for m in meta], dtype=np.int32)
    labels = np.array([m["label"] for m in meta], dtype=np.int64)
    np.savez(out_dir / "meta.npz", trace_name=names, start_sample=starts, label=labels)


def load_window_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(cache_dir / "X.npy")
    y = np.load(cache_dir / "y.npy")
    return x, y
