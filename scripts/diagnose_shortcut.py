"""Diagnostics for the "flat-then-spike" concern and balanced-eval optimism.

Reproduces the evidence discussed in docs/shortcut_and_leakage_analysis.md:

  1. Is the pre-P segment zero-padded or genuine sensor noise? (raw HDF5 stats)
  2. Does a trivial amplitude-envelope feature match the CNN on the balanced
     test set? (degenerate-shortcut baseline)
  3. Event/station/trace leakage between the train and test window splits.
  4. Real continuous data: false-alarm rate on the quiet pre-event minutes of
     the Ridgecrest hour (requires scripts/download_continuous.py first).

Figures are written to artifacts/analysis/ and are NOT tracked in git.

Run:
    python scripts/diagnose_shortcut.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve  # noqa: E402

from src.model import SeismicCNN1D  # noqa: E402
from src.utils import MODELS_DIR, WINDOWS_DIR  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def _load_windows(split: str):
    x = np.load(WINDOWS_DIR / split / "X.npy")
    y = np.load(WINDOWS_DIR / split / "y.npy")
    return x, y


def _envelope_feats(x: np.ndarray) -> np.ndarray:
    e = np.sqrt((x**2).mean(axis=1))  # (N, T) RMS-over-channels envelope
    first = e[:, :150].mean(axis=1)
    last = e[:, 350:].mean(axis=1)
    ratio = last / (first + 1e-9)
    early_std = x[:, :, :150].std(axis=(1, 2))
    return np.column_stack([first, last, ratio, early_std, np.log(ratio + 1e-9)])


def check_zero_padding(findings: dict) -> None:
    import h5py

    eq = DATA / "stead_subsample" / "test.hdf5"
    if not eq.exists():
        print("[skip] zero-padding check — download_stead.py --test-only not run")
        return
    with h5py.File(eq, "r") as f:
        traces, p_arr = f["traces"], np.asarray(f["p_arrival"])
        n = min(800, traces.shape[0])
        pre_std, post_std, zero_frac, pre_max = [], [], [], []
        for i in range(n):
            w = np.asarray(traces[i])
            p = int(p_arr[i])
            pre = w[: max(p - 50, 1)]
            post = w[p + 50 : p + 850]
            pre_std.append(pre.std())
            post_std.append(post.std())
            pre_max.append(np.abs(pre).max())
            zero_frac.append(np.mean(pre == 0.0))
    pre_std, post_std = np.array(pre_std), np.array(post_std)
    findings["zero_padding"] = {
        "pre_p_std_median": float(np.median(pre_std)),
        "post_p_std_median": float(np.median(post_std)),
        "post_over_pre_ratio_median": float(np.median(post_std / (pre_std + 1e-12))),
        "frac_traces_all_zero_pre_p": float(np.mean(np.array(zero_frac) == 1.0)),
        "mean_exact_zero_fraction_pre_p": float(np.mean(zero_frac)),
        "pre_p_absmax_median": float(np.median(pre_max)),
    }
    print("[1] zero-padding check:", json.dumps(findings["zero_padding"], indent=2))


def shortcut_baseline(findings: dict) -> None:
    xtr, ytr = _load_windows("train")
    xte, yte = _load_windows("test")
    ftr, fte = _envelope_feats(xtr), _envelope_feats(xte)

    early_auc = roc_auc_score(yte, -fte[:, 3])
    ratio_auc = roc_auc_score(yte, fte[:, 2])

    est = ftr[:, 3]
    best_acc, best_thr = 0.0, 0.0
    for t in np.unique(est)[:: max(1, len(np.unique(est)) // 300)]:
        acc = accuracy_score(ytr, (est < t).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, float(t)
    rule_acc = accuracy_score(yte, (fte[:, 3] < best_thr).astype(int))

    clf = LogisticRegression(max_iter=2000).fit(ftr, ytr)
    p_lr = clf.predict_proba(fte)[:, 1]
    lr_acc = accuracy_score(yte, (p_lr > 0.5).astype(int))
    lr_auc = roc_auc_score(yte, p_lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicCNN1D().to(device)
    ckpt = torch.load(MODELS_DIR / "seismic_cnn1d_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        p_cnn = (
            torch.softmax(model(torch.from_numpy(xte).to(device)), dim=1)[:, 1]
            .cpu()
            .numpy()
        )
    cnn_acc = accuracy_score(yte, (p_cnn > 0.5).astype(int))
    cnn_auc = roc_auc_score(yte, p_cnn)

    findings["shortcut"] = {
        "early_std_only_AUC": float(early_auc),
        "late_over_early_ratio_AUC": float(ratio_auc),
        "threshold_rule_acc": float(rule_acc),
        "threshold_rule": f"early_std < {best_thr:.4f} => earthquake",
        "logreg_envelope_acc": float(lr_acc),
        "logreg_envelope_AUC": float(lr_auc),
        "cnn_acc": float(cnn_acc),
        "cnn_AUC": float(cnn_auc),
        "normalized_env_std": {
            "noise_first150": float(np.sqrt((xte[yte == 0, :, :150] ** 2).mean())),
            "noise_last650": float(np.sqrt((xte[yte == 0, :, 350:] ** 2).mean())),
            "eq_first150": float(np.sqrt((xte[yte == 1, :, :150] ** 2).mean())),
            "eq_last650": float(np.sqrt((xte[yte == 1, :, 350:] ** 2).mean())),
        },
    }
    print("[2] shortcut baseline:", json.dumps(findings["shortcut"], indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].hist(fte[yte == 0, 3], bins=50, alpha=0.6, label="noise", color="#2563eb")
    ax[0].hist(fte[yte == 1, 3], bins=50, alpha=0.6, label="earthquake", color="#dc2626")
    ax[0].axvline(best_thr, color="k", ls="--", lw=1, label="trivial threshold")
    ax[0].set_xlabel("amplitude std of first 1.5 s of the (normalized) window")
    ax[0].set_ylabel("count")
    ax[0].set_title("A single trivial feature already separates the classes")
    ax[0].legend()

    for label, prob, color in [("CNN", p_cnn, "#111827"), ("envelope logreg", p_lr, "#f59e0b")]:
        fpr, tpr, _ = roc_curve(yte, prob)
        ax[1].plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc_score(yte, prob):.3f})")
    fpr, tpr, _ = roc_curve(yte, -fte[:, 3])
    ax[1].plot(fpr, tpr, color="#2563eb", lw=1.5, ls=":", label=f"early_std only (AUC={early_auc:.3f})")
    ax[1].plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--")
    ax[1].set_xlabel("false positive rate")
    ax[1].set_ylabel("true positive rate")
    ax[1].set_title("Trivial amplitude baseline ≈ CNN on the balanced test set")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "shortcut_baseline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def windowing_figure() -> None:
    import h5py

    eq = DATA / "stead_subsample" / "test.hdf5"
    if not eq.exists():
        return
    with h5py.File(eq, "r") as f:
        w = np.asarray(f["traces"][0])  # (6000, 3)
        p = int(f["p_arrival"][0])
    from src.windows import _normalize

    xte, yte = _load_windows("test")
    eq_win = xte[np.argmax(yte == 1)]
    noise_win = xte[np.argmax(yte == 0)]

    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    t = np.arange(w.shape[0]) / 100.0
    ax[0, 0].plot(t, w[:, 2], color="#111827", lw=0.4)
    ax[0, 0].axvline(p / 100.0, color="#dc2626", ls="--", lw=1.2, label="P pick")
    ax[0, 0].set_title("Raw earthquake trace (Z) — pre-P is genuine ambient noise, NOT zeros")
    ax[0, 0].set_xlabel("time (s)")
    ax[0, 0].legend(loc="upper right")

    pre = w[max(p - 250, 0) : p, 2]
    ax[0, 1].plot(np.arange(len(pre)) / 100.0, pre, color="#2563eb", lw=0.7)
    ax[0, 1].set_title(f"Zoom: 2.5 s before P (std={pre.std():.1f} counts) — real wiggles")
    ax[0, 1].set_xlabel("time before P (s)")

    tw = np.arange(eq_win.shape[1]) / 100.0
    ax[1, 0].plot(tw, eq_win[2], color="#dc2626", lw=0.6)
    ax[1, 0].axvspan(0, 1.5, color="#fca5a5", alpha=0.3, label="first 1.5 s (quiet after z-score)")
    ax[1, 0].set_title("Normalized EARTHQUAKE window — 'flat-then-spike' is a z-score artifact")
    ax[1, 0].set_xlabel("time (s)")
    ax[1, 0].legend(loc="upper right")

    ax[1, 1].plot(tw, noise_win[2], color="#2563eb", lw=0.6)
    ax[1, 1].set_title("Normalized NOISE window — uniform energy throughout")
    ax[1, 1].set_xlabel("time (s)")
    ax[1, 1].set_ylim(ax[1, 0].get_ylim())
    fig.tight_layout()
    fig.savefig(OUT / "windowing_diagnostic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def leakage(findings: dict) -> None:
    def load_names(split):
        m = np.load(WINDOWS_DIR / split / "meta.npz", allow_pickle=True)
        return m["trace_name"].astype(str), m["label"]

    trn, _ = load_names("train")
    tst, yte = load_names("test")
    ev = lambda n: n.split("_")[1] if len(n.split("_")) > 1 else n
    stn = lambda n: n.split(".")[0]
    tr_events, tr_stations, tr_traces = {ev(x) for x in trn}, {stn(x) for x in trn}, set(trn)
    findings["leakage"] = {
        "test_windows": int(len(tst)),
        "same_event_as_train": float(np.mean([ev(t) in tr_events for t in tst])),
        "same_station_as_train": float(np.mean([stn(t) in tr_stations for t in tst])),
        "same_trace_as_train": float(np.mean([t in tr_traces for t in tst])),
    }
    print("[3] leakage:", json.dumps(findings["leakage"], indent=2))


def continuous_false_alarms(findings: dict) -> None:
    mseed = DATA / "continuous" / "ridgecrest_m71_2019.mseed"
    if not mseed.exists():
        print("[skip] continuous check — run scripts/download_continuous.py first")
        return
    from obspy import read

    from src.model import load_classifier_checkpoint
    from src.sliding_window import (
        preprocess_stream,
        resample_to_hz,
        sliding_window_predict,
        stream_to_array,
    )

    meta = json.loads((DATA / "continuous" / "ridgecrest_m71_2019.json").read_text())
    origin_offset = float(meta.get("pre_event_s", 600))
    st = preprocess_stream(read(str(mseed)))
    wave = stream_to_array(st)
    sr = float(st[0].stats.sampling_rate)
    if abs(sr - 100.0) > 1e-3:
        wave = resample_to_hz(wave, sr, 100.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_classifier_checkpoint(MODELS_DIR / "seismic_cnn1d_best.pt", device)
    res = sliding_window_predict(wave, model, device, hop_samples=100, sample_rate=100.0)
    t, p = res.times_s, res.probs
    pre = t < origin_offset
    minutes = pre.sum() / 60.0
    findings["continuous"] = {
        "quiet_minutes": float(minutes),
        "mean_prob_on_quiet_noise": float(p[pre].mean()),
        "median_prob_on_quiet_noise": float(np.median(p[pre])),
        "frac_quiet_windows_prob_gt_0.5": float((p[pre] > 0.5).mean()),
        "false_alarms_per_hour_at_0.85": float((p[pre] >= 0.85).sum() / (minutes / 60.0)),
        "false_alarms_per_hour_at_0.95": float((p[pre] >= 0.95).sum() / (minutes / 60.0)),
    }
    print("[4] continuous false alarms:", json.dumps(findings["continuous"], indent=2))

    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    ax[0].plot(t / 60.0, wave[2][:: int(len(wave[2]) / len(t))][: len(t)], color="#111827", lw=0.3)
    ax[0].axvspan(0, origin_offset / 60.0, color="#93c5fd", alpha=0.3, label="quiet pre-event (no earthquake)")
    ax[0].axvline(origin_offset / 60.0, color="#dc2626", ls="--", lw=1.2, label="M7.1 origin")
    ax[0].set_ylabel("Z (bandpassed)")
    ax[0].set_title("Ridgecrest hour @ CI.CLC — model fires all over the quiet pre-event window")
    ax[0].legend(loc="upper right")
    ax[1].plot(t / 60.0, p, color="#1d4ed8", lw=0.8)
    ax[1].axhline(0.85, color="#9ca3af", ls="--", lw=1, label="alert threshold 0.85")
    ax[1].axvspan(0, origin_offset / 60.0, color="#93c5fd", alpha=0.3)
    ax[1].axvline(origin_offset / 60.0, color="#dc2626", ls="--", lw=1.2)
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_ylabel("P(earthquake)")
    ax[1].set_xlabel("time from stream start (min)")
    ax[1].set_title(
        f"~{findings['continuous']['false_alarms_per_hour_at_0.85']:.0f} false alarms/hour on quiet noise "
        f"(mean P={findings['continuous']['mean_prob_on_quiet_noise']:.2f})"
    )
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "continuous_false_alarms.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def failure_cases() -> None:
    xte, yte = _load_windows("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeismicCNN1D().to(device)
    ckpt = torch.load(MODELS_DIR / "seismic_cnn1d_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(torch.from_numpy(xte).to(device)), dim=1)[:, 1].cpu().numpy()
    pred = (prob > 0.5).astype(int)
    fp = np.where((pred == 1) & (yte == 0))[0]
    fn = np.where((pred == 0) & (yte == 1))[0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    fig.suptitle("Real FAILURE cases (not cherry-picked successes)", fontweight="bold")
    tw = np.arange(xte.shape[2]) / 100.0
    for j in range(3):
        if j < len(fp):
            i = fp[j]
            axes[0, j].plot(tw, xte[i, 2], color="#2563eb", lw=0.6)
            axes[0, j].set_title(f"FALSE ALARM: true=noise, p(eq)={prob[i]:.2f}")
        if j < len(fn):
            i = fn[j]
            axes[1, j].plot(tw, xte[i, 2], color="#dc2626", lw=0.6)
            axes[1, j].set_title(f"MISS: true=earthquake, p(eq)={prob[i]:.2f}")
    for a in axes.ravel():
        a.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "failure_cases.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[5] failure cases: {len(fp)} false alarms, {len(fn)} misses on balanced test set")


def main() -> None:
    findings: dict = {}
    check_zero_padding(findings)
    shortcut_baseline(findings)
    windowing_figure()
    leakage(findings)
    continuous_false_alarms(findings)
    failure_cases()
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2))
    print(f"\n[ok] wrote figures + findings to {OUT}")


if __name__ == "__main__":
    main()
