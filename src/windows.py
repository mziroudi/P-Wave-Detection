"""Extract labeled 10-second windows for Noise vs Earthquake classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.stead_io import TraceRecord
from src.utils import (
    LABEL_EARTHQUAKE,
    LABEL_NOISE,
    WINDOW_SAMPLES,
)


@dataclass
class WindowExample:
    waveform: np.ndarray  # (channels, samples)
    label: int
    trace_name: str
    start_sample: int
    p_offset: float | None = None  # P sample index within the window (regression target)


def _normalize(window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score normalization."""
    mean = window.mean(axis=-1, keepdims=True)
    std = window.std(axis=-1, keepdims=True)
    return (window - mean) / (std + eps)


def _movmean(a: np.ndarray, win: int) -> np.ndarray:
    """O(T) moving average with edge padding, same length as input."""
    if win <= 1:
        return a
    c = np.cumsum(np.insert(a, 0, 0.0))
    m = (c[win:] - c[:-win]) / win
    pad_l = win // 2
    pad_r = a.shape[0] - m.shape[0] - pad_l
    return np.pad(m, (pad_l, max(pad_r, 0)), mode="edge")[: a.shape[0]]


def agc_normalize(window: np.ndarray, win: int = 100, eps: float = 1e-6) -> np.ndarray:
    """
    Automatic Gain Control: divide each sample by the local RMS over a sliding
    window (~1 s at 100 Hz), then z-score per channel.

    This flattens the amplitude *envelope* — the quiet-lead-in-then-loud-coda
    contrast that the classifier was exploiting as a shortcut — forcing the model
    to rely on waveform *shape* (frequency content of the P onset) instead of gross
    energy. Applied identically at train time and during continuous inference.
    """
    out = np.empty_like(window, dtype=np.float32)
    for c in range(window.shape[0]):
        local_rms = np.sqrt(_movmean(window[c] ** 2, win) + eps)
        out[c] = window[c] / (local_rms + eps)
    mean = out.mean(axis=-1, keepdims=True)
    std = out.std(axis=-1, keepdims=True)
    return ((out - mean) / (std + eps)).astype(np.float32)


def normalize(window: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """Dispatch normalization by name: 'zscore' (default) or 'agc'."""
    if mode == "agc":
        return agc_normalize(window)
    return _normalize(window)


def _to_channels_first(wave: np.ndarray) -> np.ndarray:
    """STEAD stores (samples, channels); model expects (channels, samples)."""
    if wave.ndim != 2:
        raise ValueError(f"Expected 2D waveform, got shape {wave.shape}")
    if wave.shape[0] < wave.shape[1]:
        # already channels-first-ish; still transpose if channels last
        pass
    if wave.shape[1] in (1, 3) and wave.shape[0] > wave.shape[1]:
        return wave.T.astype(np.float32)
    return wave.astype(np.float32)


def extract_earthquake_window(
    record: TraceRecord,
    pre_p_samples: int = 200,
    rng: np.random.Generator | None = None,
    norm: str = "zscore",
) -> WindowExample | None:
    """
    Build a 10 s window that includes the P-wave onset.

    Default: start ~2 s before P so the window captures the first tremor
    (P) while ideally remaining before the destructive S-wave when possible.
    """
    if record.p_arrival is None:
        return None
    wave = record.waveform
    n = wave.shape[0]
    if n < WINDOW_SAMPLES:
        return None

    p = int(record.p_arrival)
    # Jitter start a little so the model does not memorize a fixed P index.
    jitter = 0
    if rng is not None:
        jitter = int(rng.integers(-50, 51))
    start = p - pre_p_samples + jitter
    start = max(0, min(start, n - WINDOW_SAMPLES))
    end = start + WINDOW_SAMPLES
    window = _to_channels_first(wave[start:end])
    if window.shape[-1] != WINDOW_SAMPLES:
        return None
    return WindowExample(
        waveform=normalize(window, norm),
        label=LABEL_EARTHQUAKE,
        trace_name=record.name,
        start_sample=start,
        p_offset=float(p - start),
    )


def extract_noise_window(
    record: TraceRecord,
    rng: np.random.Generator,
    norm: str = "zscore",
) -> WindowExample | None:
    wave = record.waveform
    n = wave.shape[0]
    if n < WINDOW_SAMPLES:
        return None
    start = int(rng.integers(0, n - WINDOW_SAMPLES + 1))
    window = _to_channels_first(wave[start : start + WINDOW_SAMPLES])
    return WindowExample(
        waveform=normalize(window, norm),
        label=LABEL_NOISE,
        trace_name=record.name,
        start_sample=start,
    )


def extract_pre_p_noise_window(
    record: TraceRecord,
    rng: np.random.Generator,
    margin: int = 50,
    norm: str = "zscore",
) -> WindowExample | None:
    """Optional hard negative: noise taken from before the P arrival on EQ traces."""
    if record.p_arrival is None:
        return None
    wave = record.waveform
    max_end = int(record.p_arrival) - margin
    if max_end < WINDOW_SAMPLES:
        return None
    start = int(rng.integers(0, max_end - WINDOW_SAMPLES + 1))
    window = _to_channels_first(wave[start : start + WINDOW_SAMPLES])
    return WindowExample(
        waveform=normalize(window, norm),
        label=LABEL_NOISE,
        trace_name=record.name,
        start_sample=start,
    )


def build_window_arrays(
    records: list[TraceRecord],
    seed: int = 42,
    include_pre_p_noise: bool = True,
    max_per_class: int | None = None,
    norm: str = "zscore",
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Convert TraceRecords into (X, y, meta).

    X shape: (N, 3, WINDOW_SAMPLES)
    y shape: (N,)

    `norm` selects the per-window normalization: "zscore" (original) or "agc"
    (Automatic Gain Control, which removes the amplitude-envelope shortcut).
    """
    rng = np.random.default_rng(seed)
    examples: list[WindowExample] = []

    eq_records = [r for r in records if r.category != "noise"]
    noise_records = [r for r in records if r.category == "noise"]

    for rec in eq_records:
        ex = extract_earthquake_window(rec, rng=rng, norm=norm)
        if ex is not None:
            examples.append(ex)
        if include_pre_p_noise:
            neg = extract_pre_p_noise_window(rec, rng=rng, norm=norm)
            if neg is not None:
                examples.append(neg)

    for rec in noise_records:
        ex = extract_noise_window(rec, rng=rng, norm=norm)
        if ex is not None:
            examples.append(ex)

    if max_per_class is not None:
        by_label: dict[int, list[WindowExample]] = {LABEL_NOISE: [], LABEL_EARTHQUAKE: []}
        for ex in examples:
            by_label[ex.label].append(ex)
        trimmed: list[WindowExample] = []
        for label, items in by_label.items():
            rng.shuffle(items)
            trimmed.extend(items[:max_per_class])
        examples = trimmed

    rng.shuffle(examples)
    if not examples:
        raise RuntimeError("No windows extracted — check that STEAD files are present.")

    x = np.stack([ex.waveform for ex in examples], axis=0).astype(np.float32)
    y = np.array([ex.label for ex in examples], dtype=np.int64)
    meta = [
        {
            "trace_name": ex.trace_name,
            "start_sample": ex.start_sample,
            "label": ex.label,
            "p_offset": ex.p_offset,
        }
        for ex in examples
    ]
    return x, y, meta


def build_regression_arrays(
    records: list[TraceRecord],
    seed: int = 42,
    max_windows: int | None = None,
    pre_p_samples: int = 200,
    n_jitters: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build earthquake-only windows for P-arrival regression.

    Target `y` is the P-wave sample index within each 10 s window
    (convertible to ms via y * 1000 / SAMPLE_RATE_HZ).
    """
    rng = np.random.default_rng(seed)
    examples: list[WindowExample] = []
    eq_records = [r for r in records if r.category != "noise" and r.p_arrival is not None]

    for rec in eq_records:
        for _ in range(n_jitters):
            ex = extract_earthquake_window(rec, pre_p_samples=pre_p_samples, rng=rng)
            if ex is None or ex.p_offset is None:
                continue
            if not (0 <= ex.p_offset < WINDOW_SAMPLES):
                continue
            examples.append(ex)

    if max_windows is not None and len(examples) > max_windows:
        rng.shuffle(examples)
        examples = examples[:max_windows]

    rng.shuffle(examples)
    if not examples:
        raise RuntimeError("No regression windows extracted — need earthquake traces with P picks.")

    x = np.stack([ex.waveform for ex in examples], axis=0).astype(np.float32)
    y = np.array([ex.p_offset for ex in examples], dtype=np.float32)
    meta = [
        {
            "trace_name": ex.trace_name,
            "start_sample": ex.start_sample,
            "p_offset": ex.p_offset,
        }
        for ex in examples
    ]
    return x, y, meta
