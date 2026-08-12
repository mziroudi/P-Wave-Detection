<div align="center">

# P-Wave Detection

**Real-time earthquake detection from raw seismograms with a compact 1D CNN.**

Classify a 10-second, 3-channel seismic window as **Noise** or **Earthquake**, and
regress the **P-wave arrival time** inside that window — trained on waveforms from
[STEAD](https://github.com/smousavi05/STEAD), the Stanford Earthquake Dataset.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

```text
Sensor stream ──► 10 s moving window ──► 1D CNN ──► Noise | Earthquake ──► alert rule ──► 🚨
```

---

## Table of contents

- [What this is](#what-this-is)
- [Why it matters (the physics)](#why-it-matters-the-physics)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Quickstart](#quickstart-5-minutes-cpu)
- [The full pipeline](#the-full-pipeline)
  - [1. Classification](#1-classification-noise-vs-earthquake)
  - [2. Continuous sliding-window inference](#2-continuous-sliding-window-inference)
  - [3. P-wave arrival regression](#3-p-wave-arrival-regression)
  - [4. Live FDSN streaming alerts](#4-live-fdsn-streaming-alerts)
- [How it works under the hood](#how-it-works-under-the-hood)
- [Model architecture](#model-architecture)
- [Results (demo run)](#results-demo-run)
- [Honest evaluation: what the model actually learned](#honest-evaluation-what-the-model-actually-learned)
- [CLI reference](#cli-reference)
- [Configuration reference](#configuration-reference)
- [Artifacts reference](#artifacts-reference)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Production roadmap](#production-roadmap)
- [Citation](#citation)
- [License & acknowledgments](#license--acknowledgments)

---

## What this is

An end-to-end, reproducible research/portfolio project that treats a three-component
seismogram like a short multi-channel audio clip and applies a small 1D convolutional
network to it. It ships two learning tasks and three inference modes:

| Task | Model | Input | Output |
|------|-------|-------|--------|
| **Classification** | `SeismicCNN1D` | `(3, 1000)` E/N/Z @ 100 Hz | logits for `[noise, earthquake]` |
| **P-arrival regression** | `SeismicCNN1DRegressor` | `(3, 1000)` | P-wave sample index within the window |

Inference modes: single-window prediction, **continuous** sliding-window inference over
an hour of real data, and a **live** near-real-time FDSN polling loop with an alert rule.

Everything is a command-line script — there are no servers, databases, or web UI. Two
pre-trained demo checkpoints are committed to `models/`, so you can evaluate and run
inference **without retraining**.

> **Read this too:** [`docs/shortcut_and_leakage_analysis.md`](docs/shortcut_and_leakage_analysis.md)
> is a candid audit of what the demo model really keys on. The
> [Honest evaluation](#honest-evaluation-what-the-model-actually-learned) section below
> summarizes it. This project is as much about *how to critique a seismic ML model* as
> it is about building one.

## Why it matters (the physics)

An earthquake radiates two main body waves:

- The **P-wave** (primary) is faster (~6 km/s) but weaker — it arrives first.
- The **S-wave** (secondary) is slower (~3.5 km/s) but carries most of the destructive
  shaking — it arrives later.

That speed gap is the entire basis of **Earthquake Early Warning (EEW)**: if you can
recognize the P-wave the instant it reaches a sensor, you may get seconds to tens of
seconds of warning before the S-wave hits. This is a *real-time signal-classification*
problem (a smoke detector), **not** forecasting a future event.

Earthquake windows here are anchored ~2 s **before** the catalog P pick, so the network
sees the pre-event context and the first P tremor rather than waiting for the full S-wave
coda.

## Repository layout

```text
P-Wave-Detection/
├── src/                        # Library code (importable as `src`)
│   ├── utils.py                #   shared constants + paths (sample rate, window size, dirs)
│   ├── stead_io.py             #   STEAD loaders (Zenodo subsample HDF5 + official HDF5/CSV)
│   ├── windows.py              #   10 s window extraction + per-window normalization
│   ├── dataset.py              #   PyTorch Datasets + .npy/.npz window cache I/O
│   ├── model.py                #   SeismicCNN1D classifier + SeismicCNN1DRegressor
│   └── sliding_window.py       #   continuous inference, preprocessing, alert rule
├── scripts/                    # Command-line entry points (run from repo root)
│   ├── download_stead.py       #   fetch the STEAD subsample from Zenodo
│   ├── download_continuous.py  #   fetch 1 h of real continuous data (SCEDC)
│   ├── visualize_waveforms.py  #   ObsPy/Matplotlib waveform + window plots
│   ├── prepare_windows.py      #   build cached classification windows + splits
│   ├── prepare_regression.py   #   build cached regression windows + splits
│   ├── train.py                #   train the classifier
│   ├── train_regression.py     #   train the P-arrival regressor
│   ├── evaluate.py             #   classifier metrics + confusion matrix / ROC plots
│   ├── evaluate_regression.py  #   regressor MAE + scatter / error-histogram plots
│   ├── predict.py              #   single-window inference demo
│   ├── continuous_inference.py #   sliding-window inference over a MiniSEED hour
│   ├── live_stream.py          #   live FDSN polling loop (or offline --demo-replay)
│   └── diagnose_shortcut.py    #   reproducible audit of the demo model (see analysis doc)
├── models/                     # Committed demo checkpoints (*.pt)
├── artifacts/                  # Generated plots, metrics, and reports
├── notebooks/                  # Exploration + a self-contained Colab notebook
├── docs/                       # shortcut_and_leakage_analysis.md (evaluation audit)
├── requirements.txt
└── AGENTS.md                   # environment/setup notes for automated agents
```

## Installation

Python **3.10+** is recommended (developed and tested on 3.12; CPU is fully supported —
CUDA is auto-detected when present).

```bash
git clone https://github.com/mziroudi/P-Wave-Detection.git
cd P-Wave-Detection

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

<details>
<summary><b>Dependencies</b></summary>

| Package | Role |
|---------|------|
| `torch` | 1D CNN model + training loop |
| `numpy`, `pandas` | array + tabular handling |
| `h5py` | read STEAD HDF5 waveform files |
| `obspy` | seismology I/O, FDSN clients, filtering, MiniSEED |
| `scikit-learn` | metrics + stratified train/val/test splits |
| `matplotlib`, `seaborn` | plots and figures |
| `tqdm`, `requests` | progress bars + Zenodo downloads |

</details>

### Google Colab

1. Open [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** and
   upload `notebooks/PWave_Detection_Colab.ipynb` (optionally **Runtime → Change runtime
   type → GPU**), then **Run all**.
2. Or paste this into a fresh cell:

   ```python
   !git clone https://github.com/mziroudi/P-Wave-Detection.git
   %cd P-Wave-Detection
   !pip install -q h5py matplotlib seaborn obspy scikit-learn tqdm requests
   !python scripts/download_stead.py --test-only
   !python scripts/prepare_windows.py --prefer-split test
   !python scripts/train.py --epochs 12
   !python scripts/evaluate.py
   ```

## Data

This project uses **STEAD** (Mousavi et al., 2019) — ~1.2M labeled three-component
seismograms, each 60 s @ 100 Hz (6000 samples) with P/S phase picks.

By default it pulls a public **STEAD subsample** hosted on Zenodo
([record 11094536](https://zenodo.org/records/11094536)), which keeps STEAD-style
waveforms plus P/S sample indices and is practical for a single-machine run:

```bash
# ~800 MB test split only — enough to train a demo model end to end
python scripts/download_stead.py --test-only

# Full subsample (train + test, ~5.8 GB)
python scripts/download_stead.py

# Just print official STEAD download links / mirrors (no download)
python scripts/download_stead.py --source official-info
```

After downloading you get:

```text
data/stead_subsample/
├── test.hdf5              # earthquake waveforms  (keys: traces, metadata, p_arrival, s_arrival)
├── test_noise.hdf5       # noise waveforms
├── train.hdf5            # (only with the full download)
├── train_noise.hdf5      # (only with the full download)
└── stead_subsample_index.csv   # STEAD-like metadata index, auto-generated
```

Each HDF5 holds `traces` of shape `(N, 6000, 3)` (samples × E/N/Z), a `metadata` name
array (e.g. `AMT.HP_20120416124803_EV` = `STATION.NETWORK_EVENTTIME_EV`), and integer
`p_arrival` / `s_arrival` sample indices for earthquake files.

> `data/` is git-ignored — regenerate it any time with the download script.

<details>
<summary><b>Using the official STEAD chunks instead</b></summary>

The official chunks (~14 GB each) are HDF5 + CSV pairs. Point the pipeline at them with:

```bash
python scripts/prepare_windows.py --source official \
  --csv  data/stead_official/chunk2.csv \
  --hdf5 data/stead_official/chunk2.hdf5
```

</details>

## Quickstart (5 minutes, CPU)

Evaluate the **committed** demo model — no training required:

```bash
python scripts/download_stead.py --test-only        # get the data (~800 MB)
python scripts/prepare_windows.py --prefer-split test   # build cached windows
python scripts/evaluate.py                           # metrics + plots in artifacts/
python scripts/predict.py --index 0                  # classify one window
```

Expected `predict.py` output:

```text
index=0
true_label=earthquake
pred_label=earthquake
p(noise)=0.0009  p(earthquake)=0.9991
```

## The full pipeline

Run all commands from the repository root.

### 1. Classification (Noise vs Earthquake)

```bash
# a) Visualize raw waveforms + a P-onset-vs-noise window comparison
python scripts/visualize_waveforms.py

# b) Build labeled 10 s windows and stratified train/val/test splits
python scripts/prepare_windows.py --prefer-split test --max-earthquake 3000 --max-noise 3000

# c) Train the 1D CNN (writes best/last checkpoints to models/)
python scripts/train.py --epochs 12 --batch-size 64

# d) Evaluate + write confusion matrix, ROC, and sample-prediction plots
python scripts/evaluate.py

# e) Classify a single cached window
python scripts/predict.py --index 0
```

### 2. Continuous sliding-window inference

Score every hop across a real one-hour recording and apply the false-alarm-control alert
rule (probability must stay above the threshold for N consecutive hops).

```bash
# 1 h of the 2019 Ridgecrest M7.1 sequence at CI.CLC (100 Hz) from SCEDC
python scripts/download_continuous.py
python scripts/continuous_inference.py --threshold 0.85 --consecutive 3
```

Produces `artifacts/continuous/sliding_window_probs.png` (raw Z waveform + P(earthquake)
over time, with the catalog origin marked) and a JSON summary.

### 3. P-wave arrival regression

Predict *where inside the 10 s window* the P-wave arrives (sample index → milliseconds via
`sample × 10` at 100 Hz).

```bash
python scripts/prepare_regression.py --prefer-split test
python scripts/train_regression.py --epochs 15
python scripts/evaluate_regression.py
```

The regressor reuses the classifier's convolutional backbone (`load_pretrained_features`)
and adds a sigmoid head that outputs a sample index in `[0, 999]`.

### 4. Live FDSN streaming alerts

```bash
# Near-real-time: poll the last 10 s from a live FDSN station every second
python scripts/live_stream.py --max-iterations 30

# Offline replay of the downloaded continuous hour (no live network needed)
python scripts/live_stream.py --demo-replay data/continuous/ridgecrest_m71_2019.mseed
```

> FDSN archive feeds typically lag wall-clock by tens of seconds (`--latency-s`), so this
> is a practical continuous-inference demo, not an ultra-low-latency SeedLink feed.

## How it works under the hood

**Window extraction** (`src/windows.py`):

- **Earthquake window:** start at `p_arrival − 200 samples` (≈2 s of pre-P context), with a
  small random jitter of ±50 samples so the model can't memorize a fixed P index. The start
  is clamped to a valid range and a fixed 1000-sample (10 s) window is cut. The regression
  target `p_offset = p_arrival − start` is recorded.
- **Noise window:** a random 1000-sample window from a dedicated noise trace.
- **Pre-P hard negative (optional, on by default):** a noise-labeled window taken from
  *before* the P arrival on an earthquake trace, to discourage the model from firing on the
  quiet lead-in.
- **Normalization:** each window is **per-channel z-scored** (mean/std over time).

**Splitting** (`scripts/prepare_windows.py`): windows are cached as `.npy`/`.npz`, then split
80/20 into train/test and the train set split 85/15 into train/val (classification splits are
stratified by label). See the [Honest evaluation](#honest-evaluation-what-the-model-actually-learned)
section for an important caveat about this split.

**Continuous / live preprocessing** (`src/sliding_window.py`): detrend (demean + linear),
5% Hann taper, zero-phase 1–20 Hz bandpass, map channels to E/N/Z, resample to 100 Hz, then
z-score each window before scoring.

## Model architecture

`SeismicCNN1D` — four Conv1d blocks + global average pooling + an MLP head
(**110,466 parameters**):

```text
Input  (B, 3, 1000)
  ├─ Conv1d(3→32,  k=7) ─ BatchNorm ─ ReLU ─ MaxPool(2)   →  (B, 32, 500)
  ├─ Conv1d(32→64, k=5) ─ BatchNorm ─ ReLU ─ MaxPool(2)   →  (B, 64, 250)
  ├─ Conv1d(64→128,k=5) ─ BatchNorm ─ ReLU ─ MaxPool(2)   →  (B,128, 125)
  ├─ Conv1d(128→128,k=3)─ BatchNorm ─ ReLU ─ AdaptiveAvgPool1d(1) → (B,128,1)
  └─ Flatten ─ Dropout(0.3) ─ Linear(128→64) ─ ReLU ─ Dropout(0.3) ─ Linear(64→2)
Output (B, 2)   # logits: [noise, earthquake]
```

`SeismicCNN1DRegressor` shares the identical backbone and replaces the head with a single
sigmoid unit scaled to `[0, window_samples−1]`, so it emits a P-arrival **sample index**.

**Training details:** AdamW (`lr=1e-3`, `weight_decay=1e-4`), `ReduceLROnPlateau`
(factor 0.5, patience 2), CrossEntropy for classification / SmoothL1 for regression,
best checkpoint selected by validation loss, seed 42.

## Results (demo run)

Balanced STEAD-subsample windows (10 s × 3 channels), 12 epochs, CPU:

| Metric | Classification (test, 1,211 windows) |
|--------|--------------------------------------|
| Accuracy | **97.4 %** |
| Macro F1 | **0.974** |
| ROC-AUC (earthquake) | **0.994** |

```text
              precision    recall  f1-score   support
       noise     0.9769    0.9705    0.9737       611
  earthquake     0.9702    0.9767    0.9734       600
    accuracy                         0.9736      1211
```

Confusion matrix: `[[593, 18], [14, 586]]` — see `artifacts/confusion_matrix.png`,
`artifacts/roc_curve.png`, and `artifacts/prediction_samples.png`.

**P-arrival regression (demo):** test **MAE ≈ 23 samples (~233 ms)** at 100 Hz — see
`artifacts/regression/p_arrival_regression.png`.

## Honest evaluation: what the model actually learned

> The headline metrics above are **optimistic**, and a discerning reviewer should not trust
> them at face value. Below is the summary of a reproducible audit
> (`python scripts/diagnose_shortcut.py`); the full write-up with figures is in
> [`docs/shortcut_and_leakage_analysis.md`](docs/shortcut_and_leakage_analysis.md).

**1. The classifier learned an amplitude-envelope shortcut, not P-wave morphology.**
Because P is pre-aligned near a fixed index and each window is per-window z-scored, the
genuine (but ~27× smaller) pre-P noise is scaled down to look flat, while noise windows have
uniform energy throughout. A **single trivial feature** — the amplitude std of the first
1.5 s — already separates the classes:

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| `early_std` threshold rule | 95.4 % | 0.979 |
| Logistic regression on 5 envelope features | 95.3 % | 0.991 |
| **SeismicCNN1D** | 97.4 % | 0.994 |

The CNN buys only ~2 points over a hand-coded amplitude detector. (Note: the pre-P segment
is *genuine ambient noise*, **not** zero-padding — that specific hypothesis was tested and
rejected; the flatness is a normalization artifact.)

**2. It does not transfer to real continuous data.** Sliding-window inference over the
~10 minutes of quiet noise *before* the Ridgecrest M7.1 origin yields mean `P(eq) = 0.54`
and **~1,440 false alarms/hour** at threshold 0.85. On a balanced holdout a 3 % false-positive
rate looks great; on a stream that is 99 %+ noise it is operationally useless.

**3. There is train/test leakage.** The split is at the *window* level with no grouping:
**24.9 %** of test windows share the same seismic event as a training window and **94.1 %**
share the same station, inflating the reported metrics.

**Takeaway:** report **false alarms per hour of quiet data**, **PR-AUC**, and **detection
latency** on continuous recordings — not just balanced-set accuracy — and split by event and
station. See the [Production roadmap](#production-roadmap).

## CLI reference

Every script accepts `-h/--help`. Key flags and defaults:

| Script | Important flags (defaults) | Main outputs |
|--------|----------------------------|--------------|
| `download_stead.py` | `--source {subsample,official-info,both}` (subsample), `--test-only`, `--force` | `data/stead_subsample/*.hdf5` + index CSV |
| `download_continuous.py` | `--event ridgecrest`, `--client/--network/--station`, `--pre-event-s 600`, `--duration-s 3600` | `data/continuous/ridgecrest_m71_2019.{mseed,json}` |
| `visualize_waveforms.py` | `--n-earthquake 3`, `--n-noise 2`, `--out-dir artifacts/waveforms` | per-trace + comparison PNGs |
| `prepare_windows.py` | `--source {subsample,official}`, `--prefer-split {train,test,all}` (all), `--max-earthquake 4000`, `--max-noise 4000`, `--max-per-class 5000`, `--seed 42` | `data/windows/{train,val,test}` + `summary.json` |
| `train.py` | `--epochs 12`, `--batch-size 64`, `--lr 1e-3`, `--weight-decay 1e-4`, `--seed 42` | `models/seismic_cnn1d_{best,last}.pt`, `artifacts/train_results.json` |
| `evaluate.py` | `--checkpoint models/seismic_cnn1d_best.pt`, `--split {val,test}` (test), `--batch-size 128` | `artifacts/{confusion_matrix,roc_curve,prediction_samples}.png`, `eval_metrics.json` |
| `predict.py` | `--checkpoint …best.pt`, `--index 0` | printed prediction |
| `prepare_regression.py` | `--prefer-split {train,test,all}` (test), `--max-earthquake 4000`, `--max-windows 8000`, `--n-jitters 3` | `data/windows_regression/{train,val,test}` |
| `train_regression.py` | `--epochs 15`, `--batch-size 64`, `--lr 1e-3`, `--pretrained-classifier …best.pt` | `models/seismic_cnn1d_regressor_best.pt`, `artifacts/regression_results.json` |
| `evaluate_regression.py` | `--checkpoint …regressor_best.pt`, `--split test` | `artifacts/regression/p_arrival_regression.png`, `metrics.json` |
| `continuous_inference.py` | `--threshold 0.85`, `--consecutive 3`, `--hop-s 1.0`, `--no-filter` | `artifacts/continuous/sliding_window_probs.png`, summary JSON |
| `live_stream.py` | `--client EARTHSCOPE`, `--network IU`, `--station ANMO`, `--channel BH?`, `--latency-s 30`, `--max-iterations 30`, `--demo-replay <mseed>` | streamed probabilities + alerts |
| `diagnose_shortcut.py` | *(no args)* | `artifacts/analysis/*.png` + `findings.json` |

## Configuration reference

Global constants live in `src/utils.py` (no environment variables are used):

| Constant | Value | Meaning |
|----------|-------|---------|
| `SAMPLE_RATE_HZ` | 100 | sampling rate of all waveforms |
| `TRACE_LENGTH` | 6000 | samples per STEAD trace (60 s) |
| `WINDOW_SECONDS` | 10 | window length in seconds |
| `WINDOW_SAMPLES` | 1000 | window length in samples |
| `LABEL_NOISE` / `LABEL_EARTHQUAKE` | 0 / 1 | class labels |
| `CLASS_NAMES` | `("noise", "earthquake")` | display names |

Directory layout (all under the repo root, auto-created by `ensure_dirs()`): `data/`,
`data/stead_subsample/`, `data/stead_official/`, `data/windows/`, `models/`, `artifacts/`.

## Artifacts reference

| Path | Produced by | Contents |
|------|-------------|----------|
| `artifacts/waveforms/` | `visualize_waveforms.py` | raw traces + P/S picks, window comparison |
| `artifacts/confusion_matrix.png`, `roc_curve.png`, `prediction_samples.png` | `evaluate.py` | classifier evaluation plots |
| `artifacts/eval_metrics.json`, `classification_report.txt`, `train_results.json` | `evaluate.py` / `train.py` | metrics + training history |
| `artifacts/continuous/` | `continuous_inference.py` | sliding-window probability plot + summary |
| `artifacts/regression/` | `evaluate_regression.py` | scatter + error histogram + metrics |
| `artifacts/analysis/` | `diagnose_shortcut.py` | audit figures + `findings.json` (git-ignored) |
| `models/seismic_cnn1d_best.pt`, `seismic_cnn1d_regressor_best.pt` | `train*.py` | committed demo checkpoints |

## Reproducibility

- Seeds are set (`torch.manual_seed`, `np.random.seed`, `default_rng(42)`) for windowing,
  splitting, and training, so runs are close to deterministic on CPU.
- The two committed checkpoints let you reproduce every evaluation number without training.
- The audit is fully scripted: `python scripts/diagnose_shortcut.py` regenerates every figure
  and number in [`docs/shortcut_and_leakage_analysis.md`](docs/shortcut_and_leakage_analysis.md).

> Heads-up: `train.py` and `train_regression.py` **overwrite** the committed checkpoints in
> `models/`. To keep the demo weights after experimenting, restore them with
> `git checkout -- models/`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No windows extracted — check that STEAD files are present` | Run `python scripts/download_stead.py --test-only` first |
| `Missing data/continuous/…mseed` | Run `python scripts/download_continuous.py` (needs SCEDC network access) |
| `python -m venv` fails on Debian/Ubuntu | `sudo apt install python3-venv` (or your `python3.X-venv`) |
| Live stream prints `fetch/predict failed` | Transient FDSN gaps are tolerated; the loop keeps going. Try `--demo-replay` offline |
| Very slow training | CPU is expected; reduce `--epochs`, `--max-earthquake/--max-noise`, or use a GPU |

## Production roadmap

Moving from this prototype to a credible EEW system requires closing the gaps surfaced by the
audit, in priority order:

1. **Break the envelope shortcut** — consistent/global normalization instead of per-window
   z-score, amplitude/gain augmentation, and envelope-matched hard negatives so absolute and
   relative energy stop being informative.
2. **Evaluate like production** — report false alarms per 24 h of quiet noise, switch the
   primary metric to **PR-AUC**, and measure detection latency on continuous recordings.
3. **Show failure cases** — publish representative false alarms and misses, not only successes.
4. **Grouped, leakage-free splits** — split by event *and* station.
5. **Classical baseline** — implement **STA/LTA** on the same test set and beat it on both MAE
   and latency before claiming the CNN earns its complexity.
6. **Latency & edge deployment** — benchmark per-window inference on commodity CPUs, export to
   ONNX/TensorRT, and replace the HTTP poller with a true streaming feed (SeedLink/WebSocket).
7. **Engineering rigor** — pin dependencies, add k-fold cross-validation with variance
   reporting, unit-test the windowing/indexing logic, and wire up CI.

## Citation

```bibtex
@article{mousavi2019stanford,
  title={STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI},
  author={Mousavi, S Mostafa and Sheng, Yixiao and Zhu, Weiqiang and Beroza, Gregory C},
  journal={IEEE Access},
  year={2019},
  publisher={IEEE}
}
```

## License & acknowledgments

Code in this repository is released under the **MIT License** (see [`LICENSE`](LICENSE)).
STEAD data follows its own license — see the
[STEAD repository](https://github.com/smousavi05/STEAD) and paper.

Built with [PyTorch](https://pytorch.org/) and [ObsPy](https://docs.obspy.org/); continuous
data courtesy of the [Southern California Earthquake Data Center (SCEDC)](https://scedc.caltech.edu/)
and [EarthScope/IRIS](https://www.earthscope.org/) FDSN services.
