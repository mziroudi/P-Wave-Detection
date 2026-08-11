"""Run sliding-window inference on a continuous MiniSEED hour and plot results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from obspy import UTCDateTime, read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import load_classifier_checkpoint
from src.sliding_window import (
    preprocess_stream,
    resample_to_hz,
    sliding_window_predict,
    stream_to_array,
)
from src.utils import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, SAMPLE_RATE_HZ, ensure_dirs


def plot_continuous_results(
    wave: np.ndarray,
    sample_rate: float,
    result,
    out_path: Path,
    origin_offset_s: float | None = None,
    title: str = "Continuous sliding-window inference",
    threshold: float = 0.85,
) -> Path:
    t_wave = np.arange(wave.shape[1]) / sample_rate
    z = wave[2]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(t_wave, z, color="#111827", lw=0.4)
    axes[0].set_ylabel("Z counts")
    axes[0].grid(True, alpha=0.25)
    if origin_offset_s is not None:
        axes[0].axvline(origin_offset_s, color="#dc2626", ls="--", lw=1.5, label="Catalog origin")
        axes[0].legend(loc="upper right")

    axes[1].plot(result.times_s, result.probs, color="#1d4ed8", lw=1.2, label="P(earthquake)")
    axes[1].axhline(threshold, color="#9ca3af", ls="--", lw=1, label=f"threshold={threshold}")
    if result.alerts.any():
        axes[1].fill_between(
            result.times_s,
            0,
            1,
            where=result.alerts,
            color="#f59e0b",
            alpha=0.25,
            label="alert (consec. rule)",
        )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xlabel("Time from stream start (s)")
    axes[1].set_ylabel("Earthquake probability")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window continuous inference")
    parser.add_argument(
        "--mseed",
        type=Path,
        default=DATA_DIR / "continuous" / "ridgecrest_m71_2019.mseed",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=DATA_DIR / "continuous" / "ridgecrest_m71_2019.json",
    )
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--hop-s", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-filter", action="store_true", help="Skip bandpass preprocessing")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "continuous")
    args = parser.parse_args()
    ensure_dirs()

    if not args.mseed.exists():
        raise SystemExit(
            f"Missing {args.mseed}. Run: python scripts/download_continuous.py"
        )
    if not args.checkpoint.exists():
        raise SystemExit(
            f"Missing {args.checkpoint}. Run: python scripts/train.py"
        )

    st = read(str(args.mseed))
    if not args.no_filter:
        st = preprocess_stream(st)
    wave = stream_to_array(st)
    sr = float(st[0].stats.sampling_rate)
    if abs(sr - SAMPLE_RATE_HZ) > 1e-3:
        print(f"[info] resampling {sr} Hz → {SAMPLE_RATE_HZ} Hz")
        wave = resample_to_hz(wave, sr, SAMPLE_RATE_HZ)
        sr = float(SAMPLE_RATE_HZ)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier_checkpoint(args.checkpoint, device)

    hop = int(round(args.hop_s * sr))
    result = sliding_window_predict(
        wave,
        model,
        device,
        hop_samples=hop,
        batch_size=args.batch_size,
        sample_rate=sr,
        alert_threshold=args.threshold,
        alert_consecutive=args.consecutive,
    )

    origin_offset = None
    title = f"Sliding window — {args.mseed.name}"
    if args.meta.exists():
        meta = json.loads(args.meta.read_text())
        start = UTCDateTime(meta["starttime"])
        origin = UTCDateTime(meta["origin_time"])
        origin_offset = float(origin - start)
        title = (
            f"{meta.get('name', 'event')} @ {meta['network']}.{meta['station']} "
            f"(M{meta.get('magnitude', '?')})"
        )

    n_alerts = int(result.alerts.sum())
    n_raw = int((result.probs >= args.threshold).sum())
    print(f"[info] windows={len(result.probs)}")
    print(f"[info] raw threshold crossings (p>={args.threshold}): {n_raw}")
    print(f"[info] alerts after {args.consecutive} consecutive rule: {n_alerts}")
    if result.alerts.any():
        first = float(result.times_s[np.argmax(result.alerts)])
        print(f"[info] first alert at t={first:.1f}s from stream start")
        if origin_offset is not None:
            print(f"[info] catalog origin at t={origin_offset:.1f}s")

    out_png = plot_continuous_results(
        wave,
        sr,
        result,
        args.out_dir / "sliding_window_probs.png",
        origin_offset_s=origin_offset,
        title=title,
        threshold=args.threshold,
    )
    summary = {
        "mseed": str(args.mseed),
        "checkpoint": str(args.checkpoint),
        "n_windows": int(len(result.probs)),
        "threshold": args.threshold,
        "consecutive": args.consecutive,
        "raw_threshold_crossings": n_raw,
        "n_alerts": n_alerts,
        "max_prob": float(result.probs.max()),
        "mean_prob": float(result.probs.mean()),
        "origin_offset_s": origin_offset,
    }
    summary_path = args.out_dir / "sliding_window_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[ok] {out_png}")
    print(f"[ok] {summary_path}")


if __name__ == "__main__":
    main()
