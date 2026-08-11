"""Evaluate classifier under realistic class imbalance + PR-AUC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WindowDataset, load_window_cache
from src.model import SeismicCNN1D
from src.seed import set_global_seed
from src.utils import ARTIFACTS_DIR, CLASS_NAMES, MODELS_DIR, WINDOWS_DIR, ensure_dirs


def _subsample_imbalanced(
    x: np.ndarray,
    y: np.ndarray,
    noise_to_eq: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    eq_idx = np.where(y == 1)[0]
    noise_idx = np.where(y == 0)[0]
    if len(eq_idx) == 0 or len(noise_idx) == 0:
        raise SystemExit("Need both noise and earthquake samples for imbalanced eval")
    n_eq = len(eq_idx)
    n_noise_target = int(min(len(noise_idx), max(1, round(n_eq * noise_to_eq))))
    # If we don't have enough noise in the split, keep all noise and note the ratio.
    pick_noise = rng.choice(noise_idx, size=n_noise_target, replace=False)
    pick_eq = eq_idx
    idx = np.concatenate([pick_noise, pick_eq])
    rng.shuffle(idx)
    return x[idx], y[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Imbalanced / PR-AUC evaluation")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--windows-dir", type=Path, default=WINDOWS_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--noise-to-eq",
        type=float,
        default=100.0,
        help="Target noise:earthquake ratio (capped by available noise windows).",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "imbalanced")
    args = parser.parse_args()
    ensure_dirs()
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_window_cache(args.windows_dir / args.split)
    x_eval, y_eval = _subsample_imbalanced(x, y, args.noise_to_eq, args.seed)
    actual_ratio = float((y_eval == 0).sum() / max((y_eval == 1).sum(), 1))
    print(
        f"[info] eval windows={len(y_eval)} "
        f"noise={(y_eval==0).sum()} eq={(y_eval==1).sum()} ratio={actual_ratio:.1f}:1"
    )

    loader = DataLoader(WindowDataset(x_eval, y_eval), batch_size=args.batch_size, shuffle=False)
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

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / max(tn + fp, 1))
    report = classification_report(y_true, y_pred, target_names=list(CLASS_NAMES), digits=4)
    print(report)
    print(f"PR-AUC: {pr_auc:.4f}  ROC-AUC: {roc_auc:.4f}  FPR: {fpr:.6f}")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#0f766e", lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall (noise:eq ≈ {actual_ratio:.0f}:1)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pr_path = args.out_dir / "pr_curve.png"
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "split": args.split,
        "requested_noise_to_eq": args.noise_to_eq,
        "actual_noise_to_eq": actual_ratio,
        "n_noise": int((y_true == 0).sum()),
        "n_earthquake": int((y_true == 1).sum()),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "false_positive_rate": fpr,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "checkpoint": str(args.checkpoint),
    }
    (args.out_dir / "imbalanced_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[ok] {pr_path}")
    print(f"[ok] {args.out_dir / 'imbalanced_metrics.json'}")


if __name__ == "__main__":
    main()
