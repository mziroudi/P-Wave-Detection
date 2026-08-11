"""Evaluate P-arrival regressor and plot prediction scatter / error hist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import RegressionWindowDataset, load_window_cache
from src.model import SeismicCNN1DRegressor
from src.utils import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, SAMPLE_RATE_HZ, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate P-arrival regressor")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_regressor_best.pt")
    parser.add_argument("--windows-dir", type=Path, default=DATA_DIR / "windows_regression")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "regression")
    args = parser.parse_args()
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_window_cache(args.windows_dir / args.split)
    loader = DataLoader(RegressionWindowDataset(x, y), batch_size=128, shuffle=False)

    model = SeismicCNN1DRegressor().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds)
    err = y_pred - y
    mae_s = float(np.mean(np.abs(err)))
    mae_ms = mae_s * (1000.0 / SAMPLE_RATE_HZ)
    print(f"MAE: {mae_s:.2f} samples ({mae_ms:.1f} ms)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y, y_pred, s=8, alpha=0.35, color="#1d4ed8")
    lims = [0, max(y.max(), y_pred.max())]
    axes[0].plot(lims, lims, color="#9ca3af", ls="--")
    axes[0].set_xlabel("True P offset (samples)")
    axes[0].set_ylabel("Predicted P offset (samples)")
    axes[0].set_title("P-arrival regression")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(err * (1000.0 / SAMPLE_RATE_HZ), bins=40, color="#0f766e", alpha=0.85)
    axes[1].set_xlabel("Error (ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Error histogram (MAE={mae_ms:.1f} ms)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = args.out_dir / "p_arrival_regression.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "split": args.split,
        "mae_samples": mae_s,
        "mae_ms": mae_ms,
        "rmse_samples": float(np.sqrt(np.mean(err**2))),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[ok] {plot_path}")


if __name__ == "__main__":
    main()
