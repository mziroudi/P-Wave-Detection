"""Train P-arrival sample-index regressor."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import RegressionWindowDataset, load_window_cache
from src.model import SeismicCNN1DRegressor, count_parameters
from src.utils import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, SAMPLE_RATE_HZ, ensure_dirs


def eval_regressor(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    errs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += float(loss.item()) * len(yb)
            errs.append((pred - yb).detach().cpu().numpy())
            n += len(yb)
    err = np.concatenate(errs)
    mae_samples = float(np.mean(np.abs(err)))
    mae_ms = mae_samples * (1000.0 / SAMPLE_RATE_HZ)
    return {
        "loss": total_loss / max(n, 1),
        "mae_samples": mae_samples,
        "mae_ms": mae_ms,
        "rmse_samples": float(np.sqrt(np.mean(err**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P-arrival regressor")
    parser.add_argument("--windows-dir", type=Path, default=DATA_DIR / "windows_regression")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained-classifier", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_dirs()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = load_window_cache(args.windows_dir / "train")
    x_val, y_val = load_window_cache(args.windows_dir / "val")
    x_test, y_test = load_window_cache(args.windows_dir / "test")

    train_loader = DataLoader(RegressionWindowDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RegressionWindowDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(RegressionWindowDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = SeismicCNN1DRegressor().to(device)
    if args.pretrained_classifier.exists():
        ckpt = torch.load(args.pretrained_classifier, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_pretrained_features(state)
        print(f"[info] loaded pretrained features from {args.pretrained_classifier}")
    print(f"[info] parameters={count_parameters(model):,} device={device}")

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_path = MODELS_DIR / "seismic_cnn1d_regressor_best.pt"
    history = []
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(yb)
            seen += len(yb)
            pbar.set_postfix(loss=running / max(seen, 1))

        train_loss = running / max(seen, 1)
        val_metrics = eval_regressor(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_mae={val_metrics['mae_samples']:.2f} samples "
            f"({val_metrics['mae_ms']:.1f} ms)"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, best_path)
            print(f"[ok] saved best → {best_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = eval_regressor(model, test_loader, device, criterion)
    print(
        f"[test] MAE={test_metrics['mae_samples']:.2f} samples "
        f"({test_metrics['mae_ms']:.1f} ms)  RMSE={test_metrics['rmse_samples']:.2f} samples"
    )

    results = {
        "task": "p_arrival_regression",
        "device": str(device),
        "seconds": round(time.time() - t0, 2),
        "history": history,
        "test": test_metrics,
        "best_checkpoint": str(best_path),
    }
    out = ARTIFACTS_DIR / "regression_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
