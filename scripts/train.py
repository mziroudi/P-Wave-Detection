"""Train the 1D CNN Noise vs Earthquake classifier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WindowDataset, load_window_cache
from src.model import SeismicCNN1D, count_parameters
from src.utils import ARTIFACTS_DIR, CLASS_NAMES, MODELS_DIR, WINDOWS_DIR, ensure_dirs


def evaluate_loader(model, loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    n = 0
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            if criterion is not None:
                loss = criterion(logits, yb)
                total_loss += float(loss.item()) * len(yb)
            pred = logits.argmax(dim=1)
            ys.append(yb.cpu().numpy())
            preds.append(pred.cpu().numpy())
            n += len(yb)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    metrics = {
        "loss": total_loss / max(n, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    return metrics, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SeismicCNN1D")
    parser.add_argument("--windows-dir", type=Path, default=WINDOWS_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_dirs()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    x_train, y_train = load_window_cache(args.windows_dir / "train")
    x_val, y_val = load_window_cache(args.windows_dir / "val")
    x_test, y_test = load_window_cache(args.windows_dir / "test")

    train_loader = DataLoader(
        WindowDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        WindowDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        WindowDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SeismicCNN1D().to(device)
    print(f"[info] parameters={count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = []
    best_val = float("inf")
    best_path = MODELS_DIR / "seismic_cnn1d_best.pt"
    last_path = MODELS_DIR / "seismic_cnn1d_last.pt"

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
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(yb)
            seen += len(yb)
            pbar.set_postfix(loss=running / max(seen, 1))

        train_loss = running / max(seen, 1)
        val_metrics, _, _ = evaluate_loader(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, last_path)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": row,
                    "class_names": CLASS_NAMES,
                },
                best_path,
            )
            print(f"[ok] saved best → {best_path}")

    # Final test with best checkpoint
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics, y_true, y_pred = evaluate_loader(model, test_loader, device, criterion)
    report = classification_report(y_true, y_pred, target_names=list(CLASS_NAMES), digits=4)
    print("\n=== Test set ===")
    print(report)

    results = {
        "device": str(device),
        "epochs": args.epochs,
        "seconds": round(time.time() - t0, 2),
        "parameters": count_parameters(model),
        "history": history,
        "test": test_metrics,
        "classification_report": report,
        "best_checkpoint": str(best_path),
    }
    out_json = ARTIFACTS_DIR / "train_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    (ARTIFACTS_DIR / "classification_report.txt").write_text(report)
    print(f"[ok] wrote {out_json}")


if __name__ == "__main__":
    main()
