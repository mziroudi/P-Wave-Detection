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
    if meta and "station_id" in meta[0]:
        payload["station_id"] = np.array([m["station_id"] for m in meta], dtype=object)
    if meta and "event_id" in meta[0]:
        payload["event_id"] = np.array([m["event_id"] for m in meta], dtype=object)
    np.savez(out_dir / "meta.npz", **payload)


def load_window_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(cache_dir / "X.npy")
    y = np.load(cache_dir / "y.npy")
    return x, y


def load_window_meta(cache_dir: Path) -> list[dict]:
    """Load per-window metadata saved by ``save_window_cache``."""
    path = cache_dir / "meta.npz"
    if not path.exists():
        return []
    raw = np.load(path, allow_pickle=True)
    n = len(raw["trace_name"])
    meta: list[dict] = []
    for i in range(n):
        row = {
            "trace_name": str(raw["trace_name"][i]),
            "start_sample": int(raw["start_sample"][i]),
        }
        if "label" in raw.files:
            row["label"] = int(raw["label"][i])
        if "p_offset" in raw.files:
            val = float(raw["p_offset"][i])
            row["p_offset"] = None if val < 0 else val
        if "station_id" in raw.files:
            row["station_id"] = str(raw["station_id"][i])
        if "event_id" in raw.files:
            row["event_id"] = str(raw["event_id"][i])
        meta.append(row)
    return meta
