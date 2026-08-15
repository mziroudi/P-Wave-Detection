"""Prepare cached 10-second Noise vs Earthquake windows from STEAD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import save_window_cache
from src.stead_io import iter_official_traces, iter_subsample_traces
from src.utils import WINDOWS_DIR, ensure_dirs
from src.windows import build_window_arrays


def _parse_group(trace_name: str, group_by: str) -> str:
    """event = timestamp token (STATION.NET_YYYYMMDDhhmmss_EV); station = leading token."""
    if group_by == "station":
        return trace_name.split(".")[0]
    parts = trace_name.split("_")
    return parts[1] if len(parts) > 1 else trace_name


def _groups_from_meta(meta: list[dict], group_by: str) -> np.ndarray:
    return np.array([_parse_group(m["trace_name"], group_by) for m in meta])


def _report_group_leakage(meta, train_idx, test_idx, group_by: str) -> None:
    ev = lambda i: _parse_group(meta[i]["trace_name"], "event")
    stn = lambda i: _parse_group(meta[i]["trace_name"], "station")
    tr_ev = {ev(i) for i in train_idx}
    tr_stn = {stn(i) for i in train_idx}
    same_ev = np.mean([ev(i) in tr_ev for i in test_idx])
    same_stn = np.mean([stn(i) in tr_stn for i in test_idx])
    print(
        f"[leakage] grouped by {group_by}: test windows sharing a TRAIN "
        f"event={same_ev:.1%}  station={same_stn:.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare STEAD classification windows")
    parser.add_argument("--source", choices=["subsample", "official"], default="subsample")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None, help="Official STEAD CSV")
    parser.add_argument("--hdf5", type=Path, default=None, help="Official STEAD HDF5")
    parser.add_argument("--max-earthquake", type=int, default=4000)
    parser.add_argument("--max-noise", type=int, default=4000)
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=WINDOWS_DIR)
    parser.add_argument("--prefer-split", choices=["train", "test", "all"], default="all")
    parser.add_argument(
        "--norm",
        choices=["zscore", "agc"],
        default="zscore",
        help="Per-window normalization: zscore (original) or agc (breaks the amplitude shortcut)",
    )
    parser.add_argument(
        "--group-by",
        choices=["none", "event", "station"],
        default="none",
        help="Leakage-free split: keep all windows of the same event/station in one split",
    )
    args = parser.parse_args()
    ensure_dirs()

    if args.source == "subsample":
        data_dir = args.data_dir or (ROOT / "data" / "stead_subsample")
        # Prefer train files when present; fall back to test.
        train_eq = data_dir / "train.hdf5"
        split = args.prefer_split
        if split == "all" and not train_eq.exists():
            split = "test"
            print("[info] train.hdf5 not found — using test split")
        records = list(
            iter_subsample_traces(
                data_dir,
                split=split if split != "all" else "all",
                max_earthquake=args.max_earthquake,
                max_noise=args.max_noise,
            )
        )
        # Deduplicate by name if all pulled both train+test with overlapping limits logic
        uniq = {}
        for r in records:
            uniq[r.name] = r
        records = list(uniq.values())
    else:
        if not args.csv or not args.hdf5:
            raise SystemExit("--csv and --hdf5 are required for --source official")
        records = list(
            iter_official_traces(
                args.csv,
                args.hdf5,
                limit_per_category=max(args.max_earthquake, args.max_noise),
            )
        )

    print(f"[info] loaded {len(records)} traces")
    x, y, meta = build_window_arrays(
        records,
        seed=args.seed,
        include_pre_p_noise=True,
        max_per_class=args.max_per_class,
        norm=args.norm,
    )
    print(
        f"[info] windows: {len(y)}  class counts: noise={(y==0).sum()} "
        f"earthquake={(y==1).sum()}  norm={args.norm}  group_by={args.group_by}"
    )

    idx = np.arange(len(y))
    if args.group_by == "none":
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=args.seed, stratify=y
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, random_state=args.seed, stratify=y[train_idx]
        )
    else:
        groups = _groups_from_meta(meta, args.group_by)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx, test_idx = next(gss1.split(idx, y, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
        rel_train, rel_val = next(
            gss2.split(train_idx, y[train_idx], groups[train_idx])
        )
        val_idx = train_idx[rel_val]
        train_idx = train_idx[rel_train]
        _report_group_leakage(meta, train_idx, test_idx, args.group_by)

    out = args.out_dir
    for name, subset in ("train", train_idx), ("val", val_idx), ("test", test_idx):
        subset_meta = [meta[i] for i in subset]
        save_window_cache(out / name, x[subset], y[subset], subset_meta)
        print(f"[ok] {name}: {len(subset)} → {out / name}")

    summary = {
        "n_traces": len(records),
        "n_windows": int(len(y)),
        "window_shape": list(x.shape[1:]),
        "class_counts": {"noise": int((y == 0).sum()), "earthquake": int((y == 1).sum())},
        "splits": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "source": args.source,
        "norm": args.norm,
        "group_by": args.group_by,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
