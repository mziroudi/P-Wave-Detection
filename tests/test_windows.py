"""Unit tests for window extraction / P-pick alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.splits import assert_no_group_leakage, group_train_val_test_indices
from src.stead_io import TraceRecord
from src.trace_ids import parse_event_id, parse_station_id
from src.utils import LABEL_EARTHQUAKE, LABEL_NOISE, WINDOW_SAMPLES
from src.windows import (
    extract_earthquake_window,
    extract_noise_window,
    extract_pre_p_noise_window,
)


def _synth_eq(name: str, p: int = 900, n: int = 6000) -> TraceRecord:
    rng = np.random.default_rng(0)
    wave = rng.normal(0, 1, size=(n, 3)).astype(np.float32)
    # spike at P so alignment is observable
    wave[p : p + 5] += 20.0
    return TraceRecord(name=name, waveform=wave, category="earthquake_local", p_arrival=p, s_arrival=p + 800)


def test_parse_trace_ids():
    assert parse_station_id("AMT.HP_20120416124803_EV") == "AMT.HP"
    assert parse_event_id("AMT.HP_20120416124803_EV", "earthquake_local") == "20120416124803"
    assert parse_event_id("AKRB.AV_20180116185030_NO", "noise") == "AKRB.AV_20180116185030_NO"


def test_earthquake_window_includes_p_and_reports_offset():
    rec = _synth_eq("STA.NET_20100101000000_EV", p=900)
    ex = extract_earthquake_window(rec, pre_p_samples=200, rng=None)
    assert ex is not None
    assert ex.label == LABEL_EARTHQUAKE
    assert ex.waveform.shape == (3, WINDOW_SAMPLES)
    assert ex.p_offset == pytest.approx(200.0)
    # P sample inside window equals original absolute P
    assert int(ex.start_sample + ex.p_offset) == 900


def test_earthquake_window_clamps_near_trace_start():
    rec = _synth_eq("STA.NET_20100101000000_EV", p=50)
    ex = extract_earthquake_window(rec, pre_p_samples=200, rng=None)
    assert ex is not None
    assert ex.start_sample == 0
    assert ex.p_offset == pytest.approx(50.0)
    assert 0 <= ex.p_offset < WINDOW_SAMPLES


def test_earthquake_window_clamps_near_trace_end():
    n = 6000
    p = n - 10
    rec = _synth_eq("STA.NET_20100101000000_EV", p=p, n=n)
    ex = extract_earthquake_window(rec, pre_p_samples=200, rng=None)
    assert ex is not None
    assert ex.start_sample == n - WINDOW_SAMPLES
    assert int(ex.start_sample + ex.p_offset) == p


def test_noise_window_shape_and_label():
    rng = np.random.default_rng(1)
    wave = rng.normal(0, 1, size=(6000, 3)).astype(np.float32)
    rec = TraceRecord(name="STA.NET_20180101000000_NO", waveform=wave, category="noise")
    ex = extract_noise_window(rec, rng=rng)
    assert ex is not None
    assert ex.label == LABEL_NOISE
    assert ex.waveform.shape == (3, WINDOW_SAMPLES)
    assert ex.p_offset is None


def test_pre_p_noise_never_crosses_p():
    rng = np.random.default_rng(2)
    rec = _synth_eq("STA.NET_20100101000000_EV", p=1500)
    ex = extract_pre_p_noise_window(rec, rng=rng, margin=50)
    assert ex is not None
    assert ex.label == LABEL_NOISE
    assert ex.start_sample + WINDOW_SAMPLES <= 1500 - 50


def test_event_level_split_has_no_leakage():
    meta = []
    y = []
    # 20 events × 2 stations, plus unique noise
    for e in range(20):
        ts = f"20100101{e:06d}"
        for sta in ("A.AA", "B.BB"):
            meta.append(
                {
                    "trace_name": f"{sta}_{ts}_EV",
                    "event_id": ts,
                    "station_id": sta,
                    "label": 1,
                }
            )
            y.append(1)
    for i in range(20):
        name = f"N.NN_20180202{i:06d}_NO"
        meta.append(
            {
                "trace_name": name,
                "event_id": name,
                "station_id": "N.NN",
                "label": 0,
            }
        )
        y.append(0)
    y = np.asarray(y)
    tr, va, te = group_train_val_test_indices(meta, y, group_by="event", seed=0)
    stats = assert_no_group_leakage(meta, tr, va, te, group_by="event")
    assert stats["train_test_overlap"] == 0
    # Same event must not appear in both train and test
    train_events = {meta[i]["event_id"] for i in tr}
    test_events = {meta[i]["event_id"] for i in te}
    assert train_events.isdisjoint(test_events)
