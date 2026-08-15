"""Classical detector baselines for fair comparison against the CNN.

STA/LTA (Short-Term Average / Long-Term Average) is the workhorse trigger used in
operational seismology for decades. Any learned detector must be shown to beat it
on the same continuous data — otherwise the added complexity is unjustified.
"""

from __future__ import annotations

import numpy as np
from obspy.signal.trigger import classic_sta_lta, trigger_onset


def sta_lta_onsets(
    z: np.ndarray,
    sample_rate: float,
    sta_s: float = 1.0,
    lta_s: float = 20.0,
    thr_on: float = 4.0,
    thr_off: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run classic STA/LTA on a single channel and return (onset_times_s, cft).

    onset_times_s are the start times (seconds from stream start) of each trigger.
    """
    cft = classic_sta_lta(z.astype(np.float64), int(sta_s * sample_rate), int(lta_s * sample_rate))
    trigs = trigger_onset(cft, thr_on, thr_off)
    onsets = np.array([t[0] / sample_rate for t in trigs], dtype=np.float64) if len(trigs) else np.array([])
    return onsets, cft


def detected_within(onsets: np.ndarray, target_s: float, lo: float = 2.0, hi: float = 10.0) -> bool:
    """True if any onset falls in [target-lo, target+hi]."""
    if onsets.size == 0:
        return False
    return bool(np.any((onsets >= target_s - lo) & (onsets <= target_s + hi)))


def estimate_p_onset(z: np.ndarray, sample_rate: float, origin_s: float, k: float = 8.0) -> float:
    """
    Robust P-onset estimate: first sample after (origin-5 s) whose |amplitude|
    exceeds k × the pre-event standard deviation.
    """
    pre_end = max(int((origin_s - 10.0) * sample_rate), int(0.5 * len(z)))
    pre_std = z[:pre_end].std() if pre_end > 0 else z.std()
    search_from = max(int((origin_s - 5.0) * sample_rate), 0)
    idx = np.where(np.abs(z[search_from:]) > k * pre_std)[0]
    return (search_from + idx[0]) / sample_rate if idx.size else origin_s
