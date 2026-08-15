"""Export real chart data for the interactive study guide (docs/index.html).

Writes docs/assets/data/charts.js as `window.PWDATA = {...}` so the page can render
interactive Plotly charts client-side. Numbers come straight from the trained models
and the audit, so the interactive charts show the same truth as the static figures.

    python scripts/export_chart_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import SeismicCNN1D, SeismicCNN1DRegressor, load_classifier_checkpoint
from src.sliding_window import preprocess_stream, resample_to_hz, sliding_window_predict, stream_to_array
from src.utils import DATA_DIR, MODELS_DIR, SAMPLE_RATE_HZ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(split_dir: Path):
    return np.load(split_dir / "X.npy"), np.load(split_dir / "y.npy")


def _cnn_probs(ckpt: Path, X: np.ndarray) -> np.ndarray:
    model = SeismicCNN1D().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i : i + 256]).to(DEVICE)
            out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def _early_std(X: np.ndarray) -> np.ndarray:
    return X[:, :, :150].std(axis=(1, 2))


def _downsample_roc(y, s, n=140):
    fpr, tpr, _ = roc_curve(y, s)
    if len(fpr) > n:
        idx = np.linspace(0, len(fpr) - 1, n).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return fpr.round(4).tolist(), tpr.round(4).tolist()


def confusion_and_roc(data: dict) -> None:
    X, y = _load(DATA_DIR / "windows" / "test")
    p_cnn = _cnn_probs(MODELS_DIR / "seismic_cnn1d_best.pt", X)
    pred = (p_cnn > 0.5).astype(int)
    cm = confusion_matrix(y, pred)

    feats = np.column_stack([
        np.sqrt((X ** 2).mean(1))[:, :150].mean(1),
        np.sqrt((X ** 2).mean(1))[:, 350:].mean(1),
        _early_std(X),
    ])
    lr = LogisticRegression(max_iter=1000).fit(feats, y)
    p_lr = lr.predict_proba(feats)[:, 1]
    es = _early_std(X)

    fpr_c, tpr_c = _downsample_roc(y, p_cnn)
    fpr_l, tpr_l = _downsample_roc(y, p_lr)
    fpr_e, tpr_e = _downsample_roc(y, -es)
    data["confusion"] = {"z": cm.tolist(), "labels": ["noise", "earthquake"]}
    data["roc"] = {
        "cnn": {"fpr": fpr_c, "tpr": tpr_c, "auc": round(float(roc_auc_score(y, p_cnn)), 3)},
        "envelope_lr": {"fpr": fpr_l, "tpr": tpr_l, "auc": round(float(roc_auc_score(y, p_lr)), 3)},
        "early_std": {"fpr": fpr_e, "tpr": tpr_e, "auc": round(float(max(roc_auc_score(y, es), roc_auc_score(y, -es))), 3)},
    }


def shortcut_histograms(data: dict) -> None:
    bins = np.linspace(0, 2.4, 41)
    centers = ((bins[:-1] + bins[1:]) / 2).round(3).tolist()
    out = {"bin_centers": centers, "modes": {}}
    for mode, d in (("zscore", "windows_gz"), ("agc", "windows_ga")):
        X, y = _load(DATA_DIR / d / "test")
        es = _early_std(X)
        hn, _ = np.histogram(es[y == 0], bins=bins)
        he, _ = np.histogram(es[y == 1], bins=bins)
        auc = float(max(roc_auc_score(y, es), roc_auc_score(y, -es)))
        out["modes"][mode] = {"noise": hn.tolist(), "earthquake": he.tolist(), "auc": round(auc, 3)}
    data["shortcut_hist"] = out


def before_after(data: dict) -> None:
    data["before_after"] = {
        "labels": ["Original\n(leaky)", "Leak-free\n(grouped)", "Shortcut-fixed\n(AGC)", "STA/LTA"],
        "false_alarms": [1416, 1248, 1500, 126],
        "accuracy_labels": ["Original", "Leak-free", "Shortcut-fixed"],
        "accuracy": [97.4, 96.9, 95.0],
        "shortcut_auc": [0.986, 0.986, 0.613],
    }


def continuous(data: dict) -> None:
    from obspy import read

    meta = json.loads((DATA_DIR / "continuous" / "ridgecrest_m71_2019.json").read_text())
    origin_min = float(meta.get("pre_event_s", 600)) / 60.0
    st = preprocess_stream(read(str(DATA_DIR / "continuous" / "ridgecrest_m71_2019.mseed")))
    wave = stream_to_array(st)
    sr = float(st[0].stats.sampling_rate)
    if abs(sr - SAMPLE_RATE_HZ) > 1e-3:
        wave = resample_to_hz(wave, sr, SAMPLE_RATE_HZ)
        sr = float(SAMPLE_RATE_HZ)
    series = {}
    for tag, ckpt, norm in (("before", "seismic_cnn1d_best.pt", "zscore"), ("agc", "cnn_grouped_agc_best.pt", "agc")):
        model = load_classifier_checkpoint(MODELS_DIR / ckpt, DEVICE)
        res = sliding_window_predict(wave, model, DEVICE, hop_samples=int(sr), sample_rate=sr, norm=norm)
        t_min = (res.times_s / 60.0)
        step = max(1, len(t_min) // 1000)
        series[tag] = {"t": t_min[::step].round(3).tolist(), "p": res.probs[::step].round(3).tolist()}
    data["continuous"] = {"origin_min": round(origin_min, 2), "series": series}


def regression(data: dict) -> None:
    X, y = _load(DATA_DIR / "windows_regression" / "test")
    model = SeismicCNN1DRegressor().to(DEVICE)
    state = torch.load(MODELS_DIR / "seismic_cnn1d_regressor_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), 256):
            preds.append(model(torch.from_numpy(X[i : i + 256]).to(DEVICE)).cpu().numpy())
    pred = np.concatenate(preds)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(500, len(y)), replace=False)
    mae_ms = float(np.mean(np.abs(pred - y)) * (1000.0 / SAMPLE_RATE_HZ))
    data["regression"] = {
        "true": y[idx].round(1).tolist(),
        "pred": pred[idx].round(1).tolist(),
        "mae_ms": round(mae_ms, 1),
    }


def main() -> None:
    data: dict = {}
    confusion_and_roc(data)
    shortcut_histograms(data)
    before_after(data)
    continuous(data)
    regression(data)
    out = ROOT / "docs" / "assets" / "data" / "charts.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.PWDATA = " + json.dumps(data) + ";\n")
    print(f"[ok] wrote {out} ({out.stat().st_size/1024:.1f} KB)")
    for k in data:
        print("  -", k)


if __name__ == "__main__":
    main()
