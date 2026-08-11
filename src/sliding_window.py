"""Sliding-window inference and alert thresholding for continuous waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.utils import SAMPLE_RATE_HZ, WINDOW_SAMPLES
from src.windows import _normalize


@dataclass
class SlidingWindowResult:
    times_s: np.ndarray  # window end time in seconds from stream start
    probs: np.ndarray  # earthquake probability per window
    alerts: np.ndarray  # bool, after consecutive-threshold rule


def preprocess_stream(stream, freqmin: float = 1.0, freqmax: float = 20.0):
    """Detrend, taper, and bandpass — closer to STEAD-style ML inputs."""
    st = stream.copy()
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="hann")
    st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    return st


def stream_to_array(stream, prefer_channels: tuple[str, ...] = ("E", "N", "Z")) -> np.ndarray:
    """
    Convert an ObsPy Stream to (channels, samples) float32 at a common length.

    Prefers E/N/Z ordering when channel codes end with those letters.
    """
    if len(stream) == 0:
        raise ValueError("Empty stream")

    # Merge / sort for continuous traces
    st = stream.copy()
    st.merge(method=1, fill_value="interpolate")
    st.sort(keys=["channel"])

    by_comp: dict[str, np.ndarray] = {}
    for tr in st:
        comp = tr.stats.channel[-1].upper()
        by_comp[comp] = np.asarray(tr.data, dtype=np.float32)

    channels = []
    for letter in prefer_channels:
        # HHZ -> Z, BHE -> E, BH1/BH2 fallback handled below
        if letter in by_comp:
            channels.append(by_comp[letter])
        elif letter == "E" and "1" in by_comp:
            channels.append(by_comp["1"])
        elif letter == "N" and "2" in by_comp:
            channels.append(by_comp["2"])

    if len(channels) == 1:
        # Duplicate vertical to 3 channels if only Z available
        channels = [channels[0], channels[0], channels[0]]
    if len(channels) != 3:
        # Fall back to first three traces in stream order
        data = [np.asarray(tr.data, dtype=np.float32) for tr in st[:3]]
        while len(data) < 3:
            data.append(data[-1])
        channels = data[:3]

    n = min(len(c) for c in channels)
    arr = np.stack([c[:n] for c in channels], axis=0)
    return arr


def resample_to_hz(wave: np.ndarray, orig_hz: float, target_hz: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """Linear-resample (channels, samples) to target_hz."""
    if abs(orig_hz - target_hz) < 1e-6:
        return wave.astype(np.float32)
    n_old = wave.shape[-1]
    duration = n_old / orig_hz
    n_new = int(round(duration * target_hz))
    x_old = np.linspace(0.0, 1.0, n_old, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
    out = np.stack([np.interp(x_new, x_old, ch) for ch in wave], axis=0)
    return out.astype(np.float32)


def apply_alert_rule(
    probs: np.ndarray,
    threshold: float = 0.85,
    consecutive: int = 3,
) -> np.ndarray:
    """Alert only if probability stays above threshold for N consecutive windows."""
    above = probs >= threshold
    alerts = np.zeros_like(above, dtype=bool)
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= consecutive:
            alerts[i] = True
    return alerts


@torch.inference_mode()
def sliding_window_predict(
    wave: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = SAMPLE_RATE_HZ,
    batch_size: int = 64,
    sample_rate: float = SAMPLE_RATE_HZ,
    alert_threshold: float = 0.85,
    alert_consecutive: int = 3,
) -> SlidingWindowResult:
    """
    Slide a fixed window across a continuous (channels, samples) array.

    Returns probabilities aligned to the *end* time of each window.
    """
    if wave.ndim != 2:
        raise ValueError(f"Expected (C, T) array, got {wave.shape}")
    if wave.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {wave.shape[0]}")
    n = wave.shape[1]
    if n < window_samples:
        raise ValueError(f"Waveform shorter than window ({n} < {window_samples})")

    starts = list(range(0, n - window_samples + 1, hop_samples))
    windows = np.stack(
        [_normalize(wave[:, s : s + window_samples]) for s in starts],
        axis=0,
    ).astype(np.float32)

    model.eval()
    probs = []
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
        logits = model(batch)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(p)
    probs_arr = np.concatenate(probs, axis=0)
    times_s = (np.asarray(starts) + window_samples) / float(sample_rate)
    alerts = apply_alert_rule(probs_arr, threshold=alert_threshold, consecutive=alert_consecutive)
    return SlidingWindowResult(times_s=times_s, probs=probs_arr, alerts=alerts)
