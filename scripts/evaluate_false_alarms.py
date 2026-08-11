"""Estimate false alarms per 24 h on continuous (mostly noise) waveforms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from obspy import read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import load_classifier_checkpoint
from src.seed import set_global_seed
from src.sliding_window import (
    preprocess_stream,
    resample_to_hz,
    sliding_window_predict,
    stream_to_array,
)
from src.utils import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, SAMPLE_RATE_HZ, ensure_dirs


def count_alert_events(alerts: np.ndarray) -> int:
    """Count rising-edge alert onsets (contiguous True runs)."""
    if len(alerts) == 0:
        return 0
    prev = False
    n = 0
    for flag in alerts:
        if flag and not prev:
            n += 1
        prev = bool(flag)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="False-alarm rate on continuous data")
    parser.add_argument(
        "--mseed",
        type=Path,
        default=DATA_DIR / "continuous" / "ridgecrest_m71_2019.mseed",
    )
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--hop-samples", type=int, default=SAMPLE_RATE_HZ)
    parser.add_argument(
        "--exclude-event-start",
        type=float,
        default=None,
        help="If set, only use samples before this time (s) as noise-only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "false_alarms")
    args = parser.parse_args()
    ensure_dirs()
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.mseed.exists():
        raise SystemExit(f"Missing continuous file: {args.mseed}")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier_checkpoint(args.checkpoint, device)

    st = read(str(args.mseed))
    st = preprocess_stream(st)
    wave = stream_to_array(st)
    orig_hz = float(st[0].stats.sampling_rate)
    wave = resample_to_hz(wave, orig_hz, SAMPLE_RATE_HZ)

    if args.exclude_event_start is not None:
        cut = int(args.exclude_event_start * SAMPLE_RATE_HZ)
        wave = wave[:, :cut]
        print(f"[info] truncated to first {args.exclude_event_start:.0f}s as noise proxy")

    duration_s = wave.shape[1] / float(SAMPLE_RATE_HZ)
    result = sliding_window_predict(
        wave,
        model,
        device,
        hop_samples=args.hop_samples,
        alert_threshold=args.threshold,
        alert_consecutive=args.consecutive,
    )
    n_alerts = count_alert_events(result.alerts)
    # Extrapolate to 24 h
    hours = duration_s / 3600.0
    far_per_hour = n_alerts / max(hours, 1e-9)
    far_per_24h = far_per_hour * 24.0
    # Also report raw FPR at window level (above threshold before consecutive rule)
    window_fpr = float(np.mean(result.probs >= args.threshold))

    metrics = {
        "mseed": str(args.mseed),
        "duration_seconds": duration_s,
        "duration_hours": hours,
        "n_windows": int(len(result.probs)),
        "n_alert_events": n_alerts,
        "false_alarms_per_hour": far_per_hour,
        "false_alarms_per_24h": far_per_24h,
        "window_positive_rate": window_fpr,
        "threshold": args.threshold,
        "consecutive": args.consecutive,
        "note": (
            "Ridgecrest hour includes a real M7.1; treat FAR as an upper bound unless "
            "--exclude-event-start is used to crop to a noise-only segment."
        ),
    }
    out = args.out_dir / "false_alarm_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
