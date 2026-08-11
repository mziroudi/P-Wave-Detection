"""Benchmark STA/LTA P-picker vs CNN regressor on the same test windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from src.seed import set_global_seed
from src.sta_lta import evaluate_sta_lta_mae
from src.utils import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, SAMPLE_RATE_HZ, ensure_dirs


def _cnn_preds(model, loader, device) -> tuple[np.ndarray, float]:
    preds = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    elapsed = time.perf_counter() - t0
    y_pred = np.concatenate(preds)
    return y_pred, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="STA/LTA vs CNN regression benchmark")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_regressor_best.pt")
    parser.add_argument("--windows-dir", type=Path, default=DATA_DIR / "windows_regression")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sta-samples", type=int, default=25)
    parser.add_argument("--lta-samples", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "baselines")
    args = parser.parse_args()
    ensure_dirs()
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_window_cache(args.windows_dir / args.split)
    t0 = time.perf_counter()
    sta = evaluate_sta_lta_mae(
        x,
        y,
        sta_samples=args.sta_samples,
        lta_samples=args.lta_samples,
        threshold=args.threshold,
    )
    sta_seconds = time.perf_counter() - t0
    sta_per_window_ms = (sta_seconds / max(len(y), 1)) * 1000.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicCNN1DRegressor().to(device)
    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        loader = DataLoader(RegressionWindowDataset(x, y), batch_size=128, shuffle=False)
        y_pred, cnn_elapsed = _cnn_preds(model, loader, device)
        cnn_err = y_pred - y
        cnn_mae_s = float(np.mean(np.abs(cnn_err)))
        cnn_mae_ms = cnn_mae_s * (1000.0 / SAMPLE_RATE_HZ)
        cnn_per_window_ms = (cnn_elapsed / max(len(y), 1)) * 1000.0
        cnn_metrics = {
            "mae_samples": cnn_mae_s,
            "mae_ms": cnn_mae_ms,
            "inference_seconds_total": cnn_elapsed,
            "inference_ms_per_window": cnn_per_window_ms,
            "checkpoint": str(args.checkpoint),
        }
    else:
        print(f"[warn] missing regressor checkpoint: {args.checkpoint}")
        cnn_metrics = None
        cnn_err = None

    comparison = {
        "split": args.split,
        "n": int(len(y)),
        "sta_lta": {
            "mae_samples": sta["mae_samples"],
            "mae_ms": sta["mae_ms"],
            "rmse_samples": sta["rmse_samples"],
            "seconds_total": sta_seconds,
            "ms_per_window": sta_per_window_ms,
            "sta_samples": args.sta_samples,
            "lta_samples": args.lta_samples,
            "threshold": args.threshold,
        },
        "cnn_regressor": cnn_metrics,
    }
    if cnn_metrics is not None:
        comparison["delta_mae_ms_cnn_minus_sta"] = cnn_metrics["mae_ms"] - sta["mae_ms"]
        comparison["cnn_better_mae"] = bool(cnn_metrics["mae_ms"] < sta["mae_ms"])

    out_json = args.out_dir / "sta_lta_vs_cnn.json"
    out_json.write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))

    # Plot error histograms
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        sta["errors"] * (1000.0 / SAMPLE_RATE_HZ),
        bins=40,
        alpha=0.7,
        label=f"STA/LTA MAE={sta['mae_ms']:.1f} ms",
        color="#b45309",
    )
    if cnn_err is not None:
        ax.hist(
            cnn_err * (1000.0 / SAMPLE_RATE_HZ),
            bins=40,
            alpha=0.7,
            label=f"CNN MAE={cnn_metrics['mae_ms']:.1f} ms",
            color="#1d4ed8",
        )
    ax.set_xlabel("Error (ms)")
    ax.set_ylabel("Count")
    ax.set_title("P-arrival pick error: STA/LTA vs CNN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = args.out_dir / "sta_lta_vs_cnn.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_json}")
    print(f"[ok] {plot_path}")


if __name__ == "__main__":
    main()
