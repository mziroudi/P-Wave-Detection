"""Tests for local waveform ring buffer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ringbuffer import WaveformRingBuffer


def test_ringbuffer_fills_and_windows():
    rb = WaveformRingBuffer(n_channels=3, capacity_samples=10)
    assert not rb.ready
    for i in range(10):
        rb.push(np.array([i, i, i], dtype=np.float32))
    assert rb.ready
    w = rb.get_window()
    assert w.shape == (3, 10)
    assert w[0, 0] == 0
    assert w[0, -1] == 9
    # Overwrite oldest
    rb.push(np.array([99, 99, 99], dtype=np.float32))
    w2 = rb.get_window()
    assert w2[0, 0] == 1
    assert w2[0, -1] == 99


def test_ringbuffer_rejects_bad_shape():
    rb = WaveformRingBuffer(n_channels=3, capacity_samples=4)
    with pytest.raises(ValueError):
        rb.push(np.zeros(2, dtype=np.float32))
