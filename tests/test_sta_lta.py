"""Tests for classical STA/LTA picker."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sta_lta import classic_sta_lta, pick_p_sta_lta


def test_sta_lta_spikes_after_onset():
    n = 1000
    x = np.zeros(n, dtype=np.float64)
    x[400:] = 5.0
    x += 0.01 * np.random.default_rng(0).normal(size=n)
    ratio = classic_sta_lta(x, sta_samples=20, lta_samples=200)
    assert ratio[450] > ratio[200]


def test_pick_near_synthetic_onset():
    n = 1000
    wave = np.zeros((3, n), dtype=np.float64)
    onset = 350
    # Deterministic step onset (no pre-event noise) — STA/LTA should lock on.
    wave[:, onset:] = 5.0
    pick = pick_p_sta_lta(wave, sta_samples=20, lta_samples=200, threshold=2.0)
    assert abs(pick - onset) < 5
