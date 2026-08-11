"""Prepare cached 10-second Noise vs Earthquake windows from STEAD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import save_window_cache
from src.seed import set_global_seed
from src.splits import assert_no_group_leakage, group_train_val_test_indices
from src.stead_io import iter_official_traces, iter_subsample_traces
from src.utils import WINDOWS_DIR, ensure_dirs
from src.windows import build_window_arrays


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
        "--group-by",
        choices=["event", "station", "window"],
        default="event",
        help="Leakage-safe split key (default: event). Use 'window' for legacy random split.",
    )
    args = parser.parse_args()
    ensure_dirs()
    set_global_seed(args.seed)

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
    )
    print(f"[info] windows: {len(y)}  class counts: noise={(y==0).sum()} earthquake={(y==1).sum()}")

    train_idx, val_idx, test_idx = group_train_val_test_indices(
        meta,
        y,
        group_by=args.group_by,
        seed=args.seed,
    )
    leakage = assert_no_group_leakage(meta, train_idx, val_idx, test_idx, group_by=args.group_by)
    print(f"[info] split group_by={args.group_by} leakage_check={leakage}")

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
        "group_by": args.group_by,
        "leakage_check": leakage,
        "source": args.source,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
