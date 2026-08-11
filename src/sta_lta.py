"""Classical STA/LTA P-arrival picker baseline."""

from __future__ import annotations

import numpy as np

from src.utils import SAMPLE_RATE_HZ, WINDOW_SAMPLES


def classic_sta_lta(
    trace: np.ndarray,
    sta_samples: int = 25,
    lta_samples: int = 200,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Cumulative-sum STA/LTA ratio on a 1-D trace.

    Defaults are tuned for **10 s @ 100 Hz** windows (STA≈0.25 s, LTA≈2.0 s).
    Classic ObsPy longer LTA (5 s) is a poor fit inside a 10 s analysis window.
    """
    x = np.asarray(trace, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D trace, got shape {x.shape}")
    sq = x * x
    # Cumulative sum for O(n) window means
    c = np.concatenate([[0.0], np.cumsum(sq)])
    n = len(x)
    sta = np.zeros(n, dtype=np.float64)
    lta = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sta_start = max(0, i - sta_samples + 1)
        lta_start = max(0, i - lta_samples + 1)
        sta[i] = (c[i + 1] - c[sta_start]) / max(i - sta_start + 1, 1)
        lta[i] = (c[i + 1] - c[lta_start]) / max(i - lta_start + 1, 1)
    return sta / (lta + eps)


def pick_p_sta_lta(
    window: np.ndarray,
    sta_samples: int = 25,
    lta_samples: int = 200,
    threshold: float = 2.0,
    channel: int = 2,
) -> float:
    """
    Pick P-arrival sample index inside a (channels, samples) window.

    Uses the Z channel by default. Returns the first index where STA/LTA
    exceeds ``threshold`` after the LTA warm-up *and* is locally rising;
    falls back to the global post-warmup argmax.
    """
    wave = np.asarray(window, dtype=np.float64)
    if wave.ndim == 2:
        trace = wave[channel]
    else:
        trace = wave
    ratio = classic_sta_lta(trace, sta_samples=sta_samples, lta_samples=lta_samples)
    start = min(lta_samples, max(len(ratio) - 1, 0))
    seg = ratio[start:]
    if len(seg) == 0:
        return 0.0
    # Prefer first rising threshold crossing (classic trigger behavior).
    above = seg >= threshold
    rising = np.empty_like(above)
    rising[0] = above[0]
    rising[1:] = above[1:] & (~above[:-1])
    crossings = np.where(rising)[0]
    if len(crossings):
        return float(start + int(crossings[0]))
    return float(start + int(np.argmax(seg)))


def evaluate_sta_lta_mae(
    x: np.ndarray,
    y_true: np.ndarray,
    sta_samples: int = 25,
    lta_samples: int = 200,
    threshold: float = 2.0,
    sample_rate: float = SAMPLE_RATE_HZ,
) -> dict:
    """Compare STA/LTA picks to labeled P offsets; return MAE in samples and ms."""
    preds = np.array(
        [
            pick_p_sta_lta(
                x[i],
                sta_samples=sta_samples,
                lta_samples=lta_samples,
                threshold=threshold,
            )
            for i in range(len(x))
        ],
        dtype=np.float64,
    )
    err = preds - np.asarray(y_true, dtype=np.float64)
    mae_samples = float(np.mean(np.abs(err)))
    return {
        "n": int(len(y_true)),
        "mae_samples": mae_samples,
        "mae_ms": mae_samples * (1000.0 / sample_rate),
        "rmse_samples": float(np.sqrt(np.mean(err**2))),
        "sta_samples": sta_samples,
        "lta_samples": lta_samples,
        "threshold": threshold,
        "window_samples": int(WINDOW_SAMPLES),
        "predictions": preds,
        "errors": err,
    }
