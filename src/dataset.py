"""PyTorch Dataset wrappers for windowed STEAD arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Classification dataset (integer labels)."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        assert len(x) == len(y)
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class RegressionWindowDataset(Dataset):
    """P-arrival regression dataset (float sample-index targets)."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        assert len(x) == len(y)
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

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
    names = np.array([m["trace_name"] for m in meta], dtype=object)
    starts = np.array([m["start_sample"] for m in meta], dtype=np.int32)
    payload = {"trace_name": names, "start_sample": starts}
    if meta and "label" in meta[0]:
        payload["label"] = np.array([m["label"] for m in meta], dtype=np.int64)
    if meta and "p_offset" in meta[0]:
        payload["p_offset"] = np.array(
            [(-1.0 if m["p_offset"] is None else m["p_offset"]) for m in meta],
            dtype=np.float32,
        )
    np.savez(out_dir / "meta.npz", **payload)


def load_window_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(cache_dir / "X.npy")
    y = np.load(cache_dir / "y.npy")
    return x, y
