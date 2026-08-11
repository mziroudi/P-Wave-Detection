"""Visualize STEAD waveforms with ObsPy + Matplotlib."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace, UTCDateTime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stead_io import TraceRecord, iter_subsample_traces
from src.utils import ARTIFACTS_DIR, SAMPLE_RATE_HZ, STEAD_SUBSAMPLE_DIR, ensure_dirs


CHANNEL_NAMES = ("E", "N", "Z")


def to_obspy_stream(record: TraceRecord, starttime: str | None = None) -> Stream:
    wave = record.waveform
    if wave.shape[1] != 3 and wave.shape[0] == 3:
        wave = wave.T
    t0 = UTCDateTime(starttime or "2015-01-01T00:00:00")
    traces = []
    for i, ch in enumerate(CHANNEL_NAMES):
        tr = Trace(data=np.asarray(wave[:, i], dtype=np.float64))
        tr.stats.sampling_rate = SAMPLE_RATE_HZ
        tr.stats.starttime = t0
        tr.stats.channel = ch
        tr.stats.station = record.name.split(".")[0][:5]
        tr.stats.network = "XX"
        traces.append(tr)
    return Stream(traces)


def plot_record(
    record: TraceRecord,
    out_path: Path,
    title: str | None = None,
) -> Path:
    st = to_obspy_stream(record)
    n = record.waveform.shape[0]
    times = np.arange(n) / SAMPLE_RATE_HZ

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(title or f"{record.name} ({record.category})", fontsize=13, fontweight="bold")

    for i, (ax, ch) in enumerate(zip(axes, CHANNEL_NAMES)):
        ax.plot(times, st[i].data, color="#1a1a1a", lw=0.6)
        ax.set_ylabel(f"{ch}\ncounts")
        ax.grid(True, alpha=0.25)
        ymin, ymax = ax.get_ylim()
        if record.p_arrival is not None:
            ax.axvline(record.p_arrival / SAMPLE_RATE_HZ, color="#2563eb", lw=1.8, label="P-wave")
        if record.s_arrival is not None:
            ax.axvline(record.s_arrival / SAMPLE_RATE_HZ, color="#dc2626", lw=1.8, label="S-wave")
        if i == 0 and record.p_arrival is not None:
            ax.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_window_comparison(
    eq: TraceRecord,
    noise: TraceRecord,
    out_path: Path,
    window_samples: int = 1000,
) -> Path:
    """Side-by-side 10 s windows: P-onset vs noise."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    fig.suptitle("10-second windows: Earthquake (P-onset) vs Noise", fontsize=13, fontweight="bold")

    # EQ window around P
    p = eq.p_arrival or 900
    start = max(0, p - 200)
    end = start + window_samples
    eq_w = eq.waveform[start:end, 2]  # Z
    t = np.arange(len(eq_w)) / SAMPLE_RATE_HZ
    axes[0].plot(t, eq_w, color="#111827", lw=0.8)
    axes[0].axvline((p - start) / SAMPLE_RATE_HZ, color="#2563eb", lw=1.8, label="P-wave")
    if eq.s_arrival is not None and start <= eq.s_arrival < end:
        axes[0].axvline((eq.s_arrival - start) / SAMPLE_RATE_HZ, color="#dc2626", lw=1.8, label="S-wave")
    axes[0].set_title(f"Earthquake  |  {eq.name}")
    axes[0].set_ylabel("Z counts")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    noise_w = noise.waveform[:window_samples, 2]
    axes[1].plot(np.arange(len(noise_w)) / SAMPLE_RATE_HZ, noise_w, color="#111827", lw=0.8)
    axes[1].set_title(f"Noise  |  {noise.name}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Z counts")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize STEAD waveforms")
    parser.add_argument("--data-dir", type=Path, default=STEAD_SUBSAMPLE_DIR)
    parser.add_argument("--n-earthquake", type=int, default=3)
    parser.add_argument("--n-noise", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "waveforms")
    args = parser.parse_args()
    ensure_dirs()

    eq_recs = list(
        iter_subsample_traces(args.data_dir, split="test", max_earthquake=args.n_earthquake, max_noise=0)
    )
    # Filter only earthquake from iterator quirk — iter yields both file types
    eq_recs = [r for r in eq_recs if r.category != "noise"][: args.n_earthquake]
    noise_recs = list(
        iter_subsample_traces(args.data_dir, split="test", max_earthquake=0, max_noise=args.n_noise)
    )
    noise_recs = [r for r in noise_recs if r.category == "noise"][: args.n_noise]

    if not eq_recs or not noise_recs:
        raise SystemExit(
            f"No traces found under {args.data_dir}. Run: python scripts/download_stead.py --test-only"
        )

    for i, rec in enumerate(eq_recs):
        path = plot_record(rec, args.out_dir / f"earthquake_{i:02d}.png")
        print(f"[ok] {path}")
        # Also save ObsPy quick plot
        st = to_obspy_stream(rec)
        obspy_path = args.out_dir / f"earthquake_{i:02d}_obspy.png"
        fig = plt.figure(figsize=(12, 6))
        st.plot(fig=fig, equal_scale=False, show=False)
        fig.savefig(obspy_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] {obspy_path}")

    for i, rec in enumerate(noise_recs):
        path = plot_record(rec, args.out_dir / f"noise_{i:02d}.png")
        print(f"[ok] {path}")

    cmp_path = plot_window_comparison(eq_recs[0], noise_recs[0], args.out_dir / "window_comparison.png")
    print(f"[ok] {cmp_path}")


if __name__ == "__main__":
    main()
