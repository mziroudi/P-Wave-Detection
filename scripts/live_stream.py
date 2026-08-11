"""
Live (near-real-time) FDSN polling loop for P-wave detection alerts.

Polls the most recent ~10 s of data from a USGS/EarthScope station every second,
runs the classifier, and prints an alert when the consecutive-threshold rule fires.

Note: FDSN archive feeds typically lag wall-clock by tens of seconds; this is a
practical engineering demo of continuous inference, not a SeedLink ultra-low-latency feed.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import load_classifier_checkpoint
from src.sliding_window import apply_alert_rule, resample_to_hz, stream_to_array
from src.utils import MODELS_DIR, SAMPLE_RATE_HZ, WINDOW_SAMPLES
from src.windows import _normalize


@torch.inference_mode()
def predict_prob(model, wave_window: np.ndarray, device: torch.device) -> float:
    x = _normalize(wave_window.astype(np.float32))
    tensor = torch.from_numpy(x[None, ...]).to(device)
    logits = model(tensor)
    return float(torch.softmax(logits, dim=1)[0, 1].item())


def fetch_recent_window(
    client: Client,
    network: str,
    station: str,
    location: str,
    channel: str,
    window_s: float,
    latency_s: float,
) -> tuple[np.ndarray, float, UTCDateTime]:
    end = UTCDateTime() - latency_s
    start = end - window_s
    st = client.get_waveforms(network, station, location, channel, start, end)
    st.merge(method=1, fill_value="interpolate")
    wave = stream_to_array(st)
    sr = float(st[0].stats.sampling_rate)
    if abs(sr - SAMPLE_RATE_HZ) > 1e-3:
        wave = resample_to_hz(wave, sr, SAMPLE_RATE_HZ)
        sr = float(SAMPLE_RATE_HZ)
    # Trim / pad to exact window length
    need = int(round(window_s * sr))
    if wave.shape[1] > need:
        wave = wave[:, -need:]
    elif wave.shape[1] < need:
        pad = need - wave.shape[1]
        wave = np.pad(wave, ((0, 0), (pad, 0)), mode="edge")
    return wave, sr, end


def main() -> None:
    parser = argparse.ArgumentParser(description="Live FDSN sliding-window alert loop")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--client", default="EARTHSCOPE")
    parser.add_argument("--network", default="IU")
    parser.add_argument("--station", default="ANMO")
    parser.add_argument("--location", default="00")
    parser.add_argument("--channel", default="BH?")
    parser.add_argument("--latency-s", type=float, default=30.0, help="Archive lag behind wall clock")
    parser.add_argument("--poll-s", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, default=30, help="Stop after N polls (0 = forever)")
    parser.add_argument("--demo-replay", type=Path, default=None, help="Optional MiniSEED to replay instead of live FDSN")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier_checkpoint(args.checkpoint, device)
    recent_probs: deque[float] = deque(maxlen=max(args.consecutive, 8))

    print(
        f"[live] client={args.client} {args.network}.{args.station} "
        f"threshold={args.threshold} consecutive={args.consecutive}"
    )

    # Optional offline replay for CI / demos without depending on live FDSN
    if args.demo_replay is not None:
        from obspy import read

        from src.ringbuffer import WaveformRingBuffer

        st = read(str(args.demo_replay))
        wave = stream_to_array(st)
        sr = float(st[0].stats.sampling_rate)
        if abs(sr - SAMPLE_RATE_HZ) > 1e-3:
            wave = resample_to_hz(wave, sr, SAMPLE_RATE_HZ)
        # Local ringbuffer path: push 1 s chunks, infer when full — no HTTP wait.
        ring = WaveformRingBuffer(n_channels=3, capacity_samples=WINDOW_SAMPLES)
        hop = SAMPLE_RATE_HZ
        n_alerts = 0
        n_windows = 0
        recent_probs.clear()
        for start in range(0, wave.shape[1], hop):
            if args.max_iterations and n_windows >= args.max_iterations:
                break
            chunk = wave[:, start : start + hop]
            if chunk.shape[1] == 0:
                break
            ring.push(chunk)
            if not ring.ready:
                continue
            window = ring.get_window()
            prob = predict_prob(model, window, device)
            recent_probs.append(prob)
            alerts = apply_alert_rule(
                np.asarray(recent_probs, dtype=np.float64),
                threshold=args.threshold,
                consecutive=args.consecutive,
            )
            t = (start + hop) / float(SAMPLE_RATE_HZ)
            n_windows += 1
            print(f"[ringbuffer t={t:7.1f}s] p(eq)={prob:.3f}")
            if alerts[-1]:
                n_alerts += 1
                print("ALERT: P-Wave Detected!")
        print(f"[done] ringbuffer windows={n_windows} alerts={n_alerts}")
        return

    client = Client(args.client)
    iteration = 0
    try:
        while True:
            iteration += 1
            try:
                wave, sr, end = fetch_recent_window(
                    client,
                    args.network,
                    args.station,
                    args.location,
                    args.channel,
                    window_s=WINDOW_SAMPLES / SAMPLE_RATE_HZ,
                    latency_s=args.latency_s,
                )
                prob = predict_prob(model, wave, device)
                recent_probs.append(prob)
                alerts = apply_alert_rule(
                    np.asarray(recent_probs, dtype=np.float64),
                    threshold=args.threshold,
                    consecutive=args.consecutive,
                )
                stamp = str(end)
                print(f"[{stamp}] p(eq)={prob:.3f}")
                if alerts[-1]:
                    print("ALERT: P-Wave Detected!")
            except Exception as exc:  # keep loop alive on transient FDSN gaps
                print(f"[warn] fetch/predict failed: {type(exc).__name__}: {exc}")

            if args.max_iterations and iteration >= args.max_iterations:
                print(f"[done] reached max_iterations={args.max_iterations}")
                break
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("\n[stop] interrupted by user")


if __name__ == "__main__":
    main()
