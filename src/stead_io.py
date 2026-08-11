"""Load STEAD waveforms from official HDF5+CSV or the Zenodo subsample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import h5py
import numpy as np
import pandas as pd

from src.utils import decode_name

SplitName = Literal["train", "test", "all"]


@dataclass
class TraceRecord:
    name: str
    waveform: np.ndarray  # (samples, channels) float32
    category: str  # "earthquake_local" | "noise"
    p_arrival: int | None = None
    s_arrival: int | None = None


def _iter_subsample_file(
    path: Path,
    category: str,
    limit: int | None = None,
) -> Iterator[TraceRecord]:
    with h5py.File(path, "r") as f:
        traces = f["traces"]
        names = f["metadata"]
        n = traces.shape[0]
        if limit is not None:
            n = min(n, limit)
        has_p = "p_arrival" in f
        has_s = "s_arrival" in f
        p_arr = f["p_arrival"] if has_p else None
        s_arr = f["s_arrival"] if has_s else None
        for i in range(n):
            yield TraceRecord(
                name=decode_name(names[i]),
                waveform=np.asarray(traces[i], dtype=np.float32),
                category=category,
                p_arrival=int(p_arr[i]) if p_arr is not None else None,
                s_arrival=int(s_arr[i]) if s_arr is not None else None,
            )


def iter_subsample_traces(
    data_dir: Path,
    split: SplitName = "all",
    max_earthquake: int | None = None,
    max_noise: int | None = None,
) -> Iterator[TraceRecord]:
    """Yield traces from Zenodo STEAD subsample HDF5 files."""
    mapping: list[tuple[str, str, int | None]] = []
    if split in ("train", "all"):
        mapping.append(("train.hdf5", "earthquake_local", max_earthquake))
        mapping.append(("train_noise.hdf5", "noise", max_noise))
    if split in ("test", "all"):
        mapping.append(("test.hdf5", "earthquake_local", max_earthquake))
        mapping.append(("test_noise.hdf5", "noise", max_noise))

    # Skip missing or partially-downloaded HDF5 files.
    existing = [
        (fn, cat, lim)
        for fn, cat, lim in mapping
        if (data_dir / fn).exists() and _is_readable_hdf5(data_dir / fn)
    ]
    if not existing and split == "train":
        existing = [
            ("test.hdf5", "earthquake_local", max_earthquake),
            ("test_noise.hdf5", "noise", max_noise),
        ]
        existing = [
            (fn, cat, lim)
            for fn, cat, lim in existing
            if (data_dir / fn).exists() and _is_readable_hdf5(data_dir / fn)
        ]

    yielded_names: set[str] = set()
    for filename, category, limit in existing:
        for rec in _iter_subsample_file(data_dir / filename, category, limit=limit):
            if rec.name in yielded_names:
                continue
            yielded_names.add(rec.name)
            yield rec


def _is_readable_hdf5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            _ = list(f.keys())
        return True
    except OSError:
        return False


def export_subsample_csv(data_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Build a STEAD-like CSV index from subsample HDF5 metadata."""
    rows: list[dict] = []
    for split_prefix, category in (
        ("train", "earthquake_local"),
        ("train_noise", "noise"),
        ("test", "earthquake_local"),
        ("test_noise", "noise"),
    ):
        path = data_dir / f"{split_prefix}.hdf5"
        if not path.exists():
            continue
        if not _is_readable_hdf5(path):
            print(f"[warn] skipping incomplete/unreadable HDF5: {path.name}")
            continue
        with h5py.File(path, "r") as f:
            n = f["traces"].shape[0]
            names = f["metadata"]
            p_arr = f["p_arrival"] if "p_arrival" in f else None
            s_arr = f["s_arrival"] if "s_arrival" in f else None
            for i in range(n):
                row = {
                    "trace_name": decode_name(names[i]),
                    "trace_category": category,
                    "split": "train" if split_prefix.startswith("train") else "test",
                    "source_file": path.name,
                    "source_index": i,
                    "p_arrival_sample": int(p_arr[i]) if p_arr is not None else np.nan,
                    "s_arrival_sample": int(s_arr[i]) if s_arr is not None else np.nan,
                }
                rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_official_trace(hdf5_path: Path, trace_name: str) -> TraceRecord:
    """Load one waveform from official STEAD merged/chunk HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        dataset = f.get(f"data/{trace_name}")
        if dataset is None:
            raise KeyError(f"Trace not found: {trace_name}")
        data = np.asarray(dataset, dtype=np.float32)
        category = decode_name(dataset.attrs.get("trace_category", "unknown"))
        p = dataset.attrs.get("p_arrival_sample", None)
        s = dataset.attrs.get("s_arrival_sample", None)
        return TraceRecord(
            name=trace_name,
            waveform=data,
            category=category,
            p_arrival=int(p) if p is not None and not np.isnan(p) else None,
            s_arrival=int(s) if s is not None and not np.isnan(s) else None,
        )


def iter_official_traces(
    csv_path: Path,
    hdf5_path: Path,
    categories: tuple[str, ...] = ("earthquake_local", "noise"),
    limit_per_category: int | None = None,
) -> Iterator[TraceRecord]:
    """Stream traces from official STEAD CSV + HDF5 pair."""
    df = pd.read_csv(csv_path)
    df = df[df["trace_category"].isin(categories)]
    counts = {c: 0 for c in categories}
    with h5py.File(hdf5_path, "r") as f:
        for _, row in df.iterrows():
            cat = row["trace_category"]
            if limit_per_category is not None and counts[cat] >= limit_per_category:
                if all(
                    limit_per_category is None or counts[c] >= limit_per_category
                    for c in categories
                ):
                    break
                continue
            name = row["trace_name"]
            dataset = f.get(f"data/{name}")
            if dataset is None:
                continue
            data = np.asarray(dataset, dtype=np.float32)
            p = row.get("p_arrival_sample", np.nan)
            s = row.get("s_arrival_sample", np.nan)
            yield TraceRecord(
                name=name,
                waveform=data,
                category=cat,
                p_arrival=int(p) if pd.notna(p) else None,
                s_arrival=int(s) if pd.notna(s) else None,
            )
            counts[cat] += 1
