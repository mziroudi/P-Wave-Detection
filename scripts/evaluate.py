"""Evaluate a trained checkpoint and write confusion-matrix / prediction plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WindowDataset, load_window_cache
from src.model import SeismicCNN1D
from src.utils import ARTIFACTS_DIR, CLASS_NAMES, MODELS_DIR, WINDOWS_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SeismicCNN1D")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--windows-dir", type=Path, default=WINDOWS_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_window_cache(args.windows_dir / args.split)
    loader = DataLoader(WindowDataset(x, y), batch_size=args.batch_size, shuffle=False)

    model = SeismicCNN1D().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(yb.numpy())
            preds.append(pred)
            probs.append(prob)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)

    report = classification_report(y_true, y_pred, target_names=list(CLASS_NAMES), digits=4)
    cm = confusion_matrix(y_true, y_pred)
    auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    print(report)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Noise vs Earthquake — Confusion Matrix")
    fig.tight_layout()
    cm_path = args.out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#1d4ed8", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9ca3af", ls="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Earthquake class")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    roc_path = args.out_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Precision-Recall (preferred under class imbalance)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#0f766e", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall — Earthquake class")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pr_path = args.out_dir / "pr_curve.png"
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Example windows with predictions
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharey=False)
    rng = np.random.default_rng(0)
    for row, label in enumerate([0, 1]):
        idxs = np.where(y_true == label)[0]
        pick = rng.choice(idxs, size=min(3, len(idxs)), replace=False)
        for col, idx in enumerate(pick):
            ax = axes[row, col]
            ax.plot(x[idx, 2], color="#111827", lw=0.7)
            correct = "OK" if y_pred[idx] == y_true[idx] else "MISS"
            ax.set_title(
                f"true={CLASS_NAMES[label]}  pred={CLASS_NAMES[y_pred[idx]]} ({correct})\n"
                f"p(eq)={y_prob[idx]:.2f}",
                fontsize=9,
            )
            ax.set_xticks([])
    axes[0, 0].set_ylabel("Noise examples\nZ channel")
    axes[1, 0].set_ylabel("Earthquake examples\nZ channel")
    fig.suptitle("Sample 10 s windows and model predictions", fontsize=12, fontweight="bold")
    fig.tight_layout()
    samples_path = args.out_dir / "prediction_samples.png"
    fig.savefig(samples_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    (args.out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[ok] {cm_path}")
    print(f"[ok] {roc_path}")
    print(f"[ok] {pr_path}")
    print(f"[ok] {samples_path}")


if __name__ == "__main__":
    main()
