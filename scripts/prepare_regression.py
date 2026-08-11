"""Prepare cached windows for P-arrival sample-index regression."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import save_window_cache
from src.stead_io import iter_subsample_traces
from src.utils import DATA_DIR, STEAD_SUBSAMPLE_DIR, ensure_dirs
from src.windows import build_regression_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare P-arrival regression windows")
    parser.add_argument("--data-dir", type=Path, default=STEAD_SUBSAMPLE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "windows_regression")
    parser.add_argument("--max-earthquake", type=int, default=4000)
    parser.add_argument("--max-windows", type=int, default=8000)
    parser.add_argument("--n-jitters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer-split", choices=["train", "test", "all"], default="test")
    args = parser.parse_args()
    ensure_dirs()

    records = list(
        iter_subsample_traces(
            args.data_dir,
            split=args.prefer_split,
            max_earthquake=args.max_earthquake,
            max_noise=0,
        )
    )
    records = [r for r in records if r.category != "noise"]
    print(f"[info] loaded {len(records)} earthquake traces")

    x, y, meta = build_regression_arrays(
        records,
        seed=args.seed,
        max_windows=args.max_windows,
        n_jitters=args.n_jitters,
    )
    print(
        f"[info] windows={len(y)}  p_offset mean={y.mean():.1f} "
        f"std={y.std():.1f} range=[{y.min():.1f}, {y.max():.1f}]"
    )

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=args.seed)

    for name, subset in ("train", train_idx), ("val", val_idx), ("test", test_idx):
        save_window_cache(args.out_dir / name, x[subset], y[subset], [meta[i] for i in subset])
        print(f"[ok] {name}: {len(subset)} → {args.out_dir / name}")

    summary = {
        "task": "p_arrival_regression",
        "n_traces": len(records),
        "n_windows": int(len(y)),
        "target": "p_offset_samples_within_10s_window",
        "splits": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
