"""Side-by-side continuous-data comparison: CNN vs classical STA/LTA.

This is the regression test for the whole project: every time the model changes,
re-run this to see false-alarms/hour on quiet noise and whether the mainshock is
detected — for the CNN and for the STA/LTA baseline on the *same* recording.

    python scripts/evaluate_baselines.py \
        --checkpoint models/seismic_cnn1d_best.pt --norm zscore

Compare two models by running it twice with different --checkpoint/--norm and
different --tag values; results are written to artifacts/baselines/<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from obspy import read  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import detected_within, estimate_p_onset, sta_lta_onsets  # noqa: E402
from src.model import load_classifier_checkpoint  # noqa: E402
from src.sliding_window import (  # noqa: E402
    preprocess_stream,
    resample_to_hz,
    sliding_window_predict,
    stream_to_array,
)
from src.utils import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR, SAMPLE_RATE_HZ, ensure_dirs


def main() -> None:
    p = argparse.ArgumentParser(description="CNN vs STA/LTA on continuous data")
    p.add_argument("--mseed", type=Path, default=DATA_DIR / "continuous" / "ridgecrest_m71_2019.mseed")
    p.add_argument("--meta", type=Path, default=DATA_DIR / "continuous" / "ridgecrest_m71_2019.json")
    p.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    p.add_argument("--norm", choices=["zscore", "agc"], default="zscore")
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--sta-s", type=float, default=1.0)
    p.add_argument("--lta-s", type=float, default=20.0)
    p.add_argument("--thr-on", type=float, default=4.0)
    p.add_argument("--thr-off", type=float, default=1.5)
    p.add_argument("--tag", default="cnn")
    p.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "baselines")
    args = p.parse_args()
    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.mseed.exists():
        raise SystemExit(f"Missing {args.mseed}. Run scripts/download_continuous.py")

    meta = json.loads(args.meta.read_text()) if args.meta.exists() else {}
    origin = float(meta.get("pre_event_s", 600.0))

    st = preprocess_stream(read(str(args.mseed)))
    wave = stream_to_array(st)
    sr = float(st[0].stats.sampling_rate)
    if abs(sr - SAMPLE_RATE_HZ) > 1e-3:
        wave = resample_to_hz(wave, sr, SAMPLE_RATE_HZ)
        sr = float(SAMPLE_RATE_HZ)

    p_onset = estimate_p_onset(wave[2], sr, origin)
    quiet_hours = (origin) / 3600.0

    # ---- CNN ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier_checkpoint(args.checkpoint, device)
    res = sliding_window_predict(
        wave, model, device, hop_samples=int(sr), sample_rate=sr,
        alert_threshold=args.threshold, alert_consecutive=3, norm=args.norm,
    )
    t, prob = res.times_s, res.probs
    quiet = t < origin
    cnn = {
        "false_alarms_per_hour": float((prob[quiet] >= args.threshold).sum() / quiet_hours),
        "mean_prob_quiet": float(prob[quiet].mean()),
        "detects_mainshock": bool(np.any((t >= p_onset - 2) & (t <= p_onset + 10) & (prob >= args.threshold))),
    }

    # ---- STA/LTA ----
    onsets, cft = sta_lta_onsets(wave[2], sr, args.sta_s, args.lta_s, args.thr_on, args.thr_off)
    i0, i1 = int((p_onset - 2) * sr), int((p_onset + 10) * sr)
    stalta_detect = bool(cft[i0:i1].max() >= args.thr_on) if i1 <= len(cft) else detected_within(onsets, p_onset)
    stalta = {
        "false_alarms_per_hour": float((onsets < origin).sum() / quiet_hours),
        "detects_mainshock": stalta_detect,
        "params": {"sta_s": args.sta_s, "lta_s": args.lta_s, "thr_on": args.thr_on, "thr_off": args.thr_off},
    }

    summary = {
        "tag": args.tag, "checkpoint": str(args.checkpoint), "norm": args.norm,
        "p_onset_s": round(p_onset, 2), "quiet_minutes": round(origin / 60, 1),
        "cnn": cnn, "sta_lta": stalta,
    }
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- figure ----
    fig, ax = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    tw = np.arange(wave.shape[1]) / sr / 60.0
    ax[0].plot(tw, wave[2], color="#111827", lw=0.3)
    ax[0].axvspan(0, origin / 60, color="#93c5fd", alpha=.3, label="quiet pre-event")
    ax[0].axvline(p_onset / 60, color="#dc2626", ls="--", lw=1, label="P onset")
    ax[0].set_ylabel("Z"); ax[0].legend(loc="upper right")
    ax[0].set_title(f"CNN ({args.tag}, norm={args.norm}) vs STA/LTA — Ridgecrest hour")
    ax[1].plot(t / 60, prob, color="#1d4ed8", lw=.7); ax[1].axhline(args.threshold, color="#9ca3af", ls="--", lw=1)
    ax[1].axvspan(0, origin / 60, color="#93c5fd", alpha=.3)
    ax[1].set_ylabel("CNN P(eq)"); ax[1].set_ylim(-.02, 1.02)
    ax[1].set_title(f"CNN: {cnn['false_alarms_per_hour']:.0f} false alarms/hour on quiet noise")
    tc = np.arange(len(cft)) / sr / 60.0
    ax[2].plot(tc, cft, color="#0f766e", lw=.5); ax[2].axhline(args.thr_on, color="#9ca3af", ls="--", lw=1)
    ax[2].axvspan(0, origin / 60, color="#93c5fd", alpha=.3)
    ax[2].set_ylabel("STA/LTA"); ax[2].set_xlabel("time from stream start (min)")
    ax[2].set_title(f"STA/LTA: {stalta['false_alarms_per_hour']:.0f} false alarms/hour on quiet noise")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"{args.tag}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {args.out_dir / (args.tag + '.json')} and .png")


if __name__ == "__main__":
    main()
