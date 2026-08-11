"""5-fold group cross-validation for SeismicCNN1D with mean±std metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WindowDataset, load_window_cache, load_window_meta
from src.model import SeismicCNN1D, count_parameters
from src.seed import set_global_seed
from src.splits import kfold_group_indices
from src.trace_ids import enrich_meta_ids, parse_event_id, parse_station_id
from src.utils import ARTIFACTS_DIR, WINDOWS_DIR, ensure_dirs


def _load_all_splits(windows_dir: Path):
    xs, ys, metas = [], [], []
    for split in ("train", "val", "test"):
        d = windows_dir / split
        if not (d / "X.npy").exists():
            continue
        x, y = load_window_cache(d)
        meta = load_window_meta(d)
        if not meta:
            meta = [
                {
                    "trace_name": f"unknown_{i}",
                    "start_sample": 0,
                    "label": int(y[i]),
                    "station_id": f"unknown_{i}",
                    "event_id": f"unknown_{i}",
                }
                for i in range(len(y))
            ]
        else:
            # Ensure IDs exist even for older caches
            for i, m in enumerate(meta):
                name = str(m.get("trace_name", f"unknown_{i}"))
                m.setdefault("station_id", parse_station_id(name))
                m.setdefault(
                    "event_id",
                    parse_event_id(name, "noise" if int(y[i]) == 0 else "earthquake_local"),
                )
            meta = enrich_meta_ids(meta)
        xs.append(x)
        ys.append(y)
        metas.extend(meta)
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return x, y, metas


def _train_one(
    x_tr,
    y_tr,
    x_te,
    y_te,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    model = SeismicCNN1D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(WindowDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(WindowDataset(x_te, y_te), batch_size=batch_size, shuffle=False)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold group CV")
    parser.add_argument("--windows-dir", type=Path, default=WINDOWS_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--group-by", choices=["event", "station"], default="event")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "cv")
    args = parser.parse_args()
    ensure_dirs()
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y, meta = _load_all_splits(args.windows_dir)
    folds = kfold_group_indices(meta, group_by=args.group_by, n_splits=args.n_splits, seed=args.seed)
    print(f"[info] windows={len(y)} folds={len(folds)} group_by={args.group_by} device={device}")

    fold_metrics = []
    for i, (tr, te) in enumerate(folds, start=1):
        set_global_seed(args.seed + i)
        m = _train_one(
            x[tr],
            y[tr],
            x[te],
            y[te],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        m["fold"] = i
        fold_metrics.append(m)
        print(
            f"[fold {i}] acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
            f"auc={m['roc_auc']:.4f} n_test={m['n_test']}"
        )

    def _agg(key: str) -> dict:
        vals = np.array([m[key] for m in fold_metrics], dtype=np.float64)
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)}

    summary = {
        "n_splits": len(folds),
        "group_by": args.group_by,
        "epochs": args.epochs,
        "parameters": count_parameters(SeismicCNN1D()),
        "folds": fold_metrics,
        "accuracy": _agg("accuracy"),
        "f1_macro": _agg("f1_macro"),
        "roc_auc": _agg("roc_auc"),
    }
    out = args.out_dir / "cv_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(
        f"[summary] acc={summary['accuracy']['mean']:.4f}±{summary['accuracy']['std']:.4f} "
        f"f1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f} "
        f"auc={summary['roc_auc']['mean']:.4f}±{summary['roc_auc']['std']:.4f}"
    )
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
