# P-Wave Detection

Train a **1D CNN** to classify a **10-second** seismic window as **Noise** vs **Earthquake**, using waveforms from the [STEAD](https://github.com/smousavi05/STEAD) (Stanford Earthquake Dataset).

The goal is early-warning style detection: recognize the first, weaker **P-wave** as soon as it hits a sensor — before the destructive **S-wave** arrives. This is signal classification in real time (smoke detector), not forecasting the future.

```text
Sensor stream ──► 10 s moving window ──► 1D CNN ──► Noise | Earthquake
```

## Why this works

| Idea | Detail |
|------|--------|
| Treat seismograms like audio | 3-channel (E/N/Z) time series @ 100 Hz |
| Fixed window | 10 s → 1000 samples |
| Labels from STEAD | Noise traces vs earthquake traces with P/S picks |
| Model | Compact PyTorch 1D CNN |

Earthquake windows are anchored near the **P-arrival** (≈2 s of pre-event context + P onset), so the network learns the first tremor rather than waiting for the full S-wave coda.

## Dataset

**STEAD** (Mousavi et al., 2019) — ~1.2M labeled three-component seismograms.

Official chunks (~14 GB each): [smousavi05/STEAD](https://github.com/smousavi05/STEAD)

This repo defaults to a public **STEAD subsample** on Zenodo ([record 11094536](https://zenodo.org/records/11094536)) that keeps STEAD-style waveforms + P/S sample indices and is practical for a solo portfolio run.

```bash
# ~800 MB test split (enough to train a demo model)
python scripts/download_stead.py --test-only

# Or full subsample train+test (~5.8 GB)
python scripts/download_stead.py

# Print official STEAD download links / MEGA mirrors
python scripts/download_stead.py --source official-info
```

After download you get:

- `data/stead_subsample/*.hdf5` — waveforms
- `data/stead_subsample/stead_subsample_index.csv` — STEAD-like metadata index

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

1. Open [Google Colab](https://colab.research.google.com/) → **File → Upload notebook**
2. Upload `notebooks/PWave_Detection_Colab.ipynb` (or open it from the repo after cloning in a blank notebook)
3. Optional: **Runtime → Change runtime type → GPU**
4. Run all cells (clone → install → download → visualize → prepare → train → evaluate)

Or paste this in a fresh Colab cell first:

```python
!git clone https://github.com/mziroudi/P-Wave-Detection.git
%cd P-Wave-Detection
!pip install -q h5py matplotlib seaborn obspy scikit-learn tqdm requests
!python scripts/download_stead.py --test-only
!python scripts/prepare_windows.py --prefer-split test
!python scripts/train.py --epochs 12
!python scripts/evaluate.py
```

## Pipeline

```bash
# 1) Download STEAD subsample
python scripts/download_stead.py --test-only

# 2) Visualize with ObsPy / Matplotlib
python scripts/visualize_waveforms.py

# 3) Build labeled 10 s windows (Noise vs Earthquake; event-level split by default)
python scripts/prepare_windows.py --max-earthquake 3000 --max-noise 3000 --group-by event

# 4) Train classifier 1D CNN
python scripts/train.py --epochs 12 --batch-size 64

# 5) Evaluate + plots
python scripts/evaluate.py

# 6) Single-window demo
python scripts/predict.py --index 0
```

### Continuous sliding-window inference (Step 1)

```bash
# 1-hour Ridgecrest M7.1 recording from SCEDC (CI.CLC @ 100 Hz)
python scripts/download_continuous.py
python scripts/continuous_inference.py --threshold 0.85 --consecutive 3
```

Plots `artifacts/continuous/sliding_window_probs.png` (raw waveform + P(earthquake) over time). Alerts only fire if probability stays above the threshold for N consecutive hops — the same false-alarm control EEW engineers use.

### P-wave arrival regression (Step 2)

```bash
python scripts/prepare_regression.py --prefer-split test
python scripts/train_regression.py --epochs 15
python scripts/evaluate_regression.py
```

The regression head predicts the **sample index of the P-wave inside the 10 s window** (→ milliseconds via `pred * 10`).

### Live FDSN streaming alerts (Step 3)

```bash
# Near-real-time poll of the last 10 s from a live FDSN station
python scripts/live_stream.py --max-iterations 30

# Offline replay of the downloaded continuous hour (no live network needed)
python scripts/live_stream.py --demo-replay data/continuous/ridgecrest_m71_2019.mseed
```

Artifacts land in `artifacts/` (waveforms, confusion matrix, ROC, continuous/regression plots). Checkpoints land in `models/`.

### Official STEAD HDF5 + CSV

```bash
python scripts/prepare_windows.py --source official \
  --csv data/stead_official/chunk2.csv \
  --hdf5 data/stead_official/chunk2.hdf5
```

## Model

`SeismicCNN1D` — four Conv1d blocks + global average pool + MLP head.

- **Classifier input/output:** `(batch, 3, 1000)` → logits for `[noise, earthquake]`
- **Regressor:** `SeismicCNN1DRegressor` — same backbone, sigmoid head → P-arrival sample index in the window

## Project layout

```text
src/
  stead_io.py         # STEAD subsample + official loaders
  windows.py          # 10 s window extraction (classify + regress)
  splits.py           # event/station leakage-safe splits
  trace_ids.py        # station/event IDs from STEAD names
  dataset.py          # PyTorch Dataset + cache I/O
  model.py            # 1D CNN classifier + P-arrival regressor
  sliding_window.py   # continuous inference + alert rule
  sta_lta.py          # classical STA/LTA P-picker baseline
  ringbuffer.py       # local ringbuffer for low-latency replay
  seed.py             # global RNG seeding
scripts/
  download_stead.py
  download_continuous.py
  visualize_waveforms.py
  prepare_windows.py          # --group-by event|station|window
  prepare_regression.py
  train.py
  train_regression.py
  evaluate.py                 # ROC-AUC + PR-AUC
  evaluate_imbalanced.py      # high noise:eq ratio + PR-AUC
  evaluate_false_alarms.py    # FAR / 24 h on continuous data
  evaluate_regression.py
  benchmark_sta_lta.py        # STA/LTA vs CNN MAE + latency
  export_onnx.py
  benchmark_latency.py
  cross_validate.py           # 5-fold group CV
  continuous_inference.py
  live_stream.py              # FDSN poll or --demo-replay ringbuffer
  predict.py
tests/
  test_windows.py
  test_sta_lta.py
  test_ringbuffer.py
.github/workflows/ci.yml
notebooks/
  explore_stead.ipynb
  PWave_Detection_Colab.ipynb
```

## Results (demo run)

Trained on ~7k STEAD-subsample windows (10 s × 3 channels), 12 epochs, CPU:

| Metric | Value |
|--------|-------|
| Test accuracy | **97.4%** |
| Macro F1 | **0.974** |
| ROC-AUC (earthquake) | **0.996** |

```text
              precision    recall  f1-score   support
       noise     0.9694    0.9789    0.9741       712
  earthquake     0.9784    0.9686    0.9734       700
    accuracy                         0.9738      1412
```

See `artifacts/confusion_matrix.png` and `artifacts/roc_curve.png`.

**P-arrival regression (demo):** test MAE ≈ **23 samples (~230 ms)** at 100 Hz — see `artifacts/regression/p_arrival_regression.png`.

**Continuous sliding window:** Ridgecrest M7.1 hour at CI.CLC — see `artifacts/continuous/sliding_window_probs.png`. Expect false alarms on raw continuous data; tune `--threshold` / `--consecutive`.

## Production evaluation (roadmap)

These commands implement the production gaps below. Prefer **event-level** splits for new training runs:

```bash
# Leakage-safe windows (default --group-by event)
python scripts/prepare_windows.py --prefer-split test --group-by event
python scripts/prepare_regression.py --prefer-split test --group-by event

# Imbalanced eval + PR-AUC (ratio capped by available noise windows)
python scripts/evaluate_imbalanced.py --noise-to-eq 100

# False alarms / 24 h on continuous waveform
python scripts/evaluate_false_alarms.py --threshold 0.85 --consecutive 3

# STA/LTA baseline vs CNN regressor
python scripts/benchmark_sta_lta.py

# Latency + ONNX
python scripts/export_onnx.py
python scripts/benchmark_latency.py

# 5-fold event-grouped CV (mean ± std)
python scripts/cross_validate.py --n-splits 5 --epochs 5

# Unit tests / CI locally
pytest -q
```

## Limitations & Production Roadmap

This project demonstrates an end-to-end ML pipeline for seismic detection. The items below were the remaining gaps for production-minded EEW work — **status reflects what this repo now implements**.

### 1. Data Provenance & Leakage Control (Critical) — addressed in-repo

**Was:** Random window-level train/test split on a Zenodo STEAD subsample.

**Now:**
- `prepare_windows.py` / `prepare_regression.py` default to **`--group-by event`** (also supports `station` or legacy `window`).
- Splits assert **zero event/station overlap** across train/val/test (`src/splits.py`).
- Official STEAD chunk path remains available via `--source official --csv … --hdf5 …` when you have the ~14 GB files locally (download still blocked in many environments).

### 2. Realistic Class Imbalance & False Alarms (Critical) — addressed in-repo

**Was:** Balanced 50/50 accuracy / ROC-AUC only.

**Now:**
- `evaluate.py` reports **PR-AUC** alongside ROC-AUC.
- `evaluate_imbalanced.py` re-evaluates at a high noise:earthquake ratio and writes `artifacts/imbalanced/`.
- `evaluate_false_alarms.py` reports **false alarms per 24 h** on continuous MiniSEED (`artifacts/false_alarms/`).

### 3. Benchmarking Against Classical Baselines — addressed in-repo

**Was:** CNN regression MAE (~230 ms) with no classical comparison.

**Now:** `src/sta_lta.py` + `scripts/benchmark_sta_lta.py` compare STA/LTA vs CNN MAE and per-window latency on the same test set (`artifacts/baselines/`).

### 4. Inference Latency & Edge Deployment — addressed in-repo

**Was:** Unmeasured inference latency; HTTP FDSN polling only.

**Now:**
- `export_onnx.py` exports `models/seismic_cnn1d.onnx`.
- `benchmark_latency.py` reports PyTorch CPU and ONNX Runtime ms/window (`artifacts/latency/`).
- `live_stream.py --demo-replay` feeds a **local ringbuffer** (`src/ringbuffer.py`) for sub-second offline inference without HTTP. Live FDSN remains a lagged archive poll (not SeedLink).

### 5. Software Engineering Rigor — addressed in-repo

**Was:** Single run, unpinned deps, no windowing tests.

**Now:**
- Global seeds via `src/seed.py` (used by train / prepare / eval scripts).
- `cross_validate.py` — **5-fold group CV** with mean ± std.
- Pinned `requirements.txt`.
- Unit tests for window P-alignment, STA/LTA, and ringbuffer under `tests/`.
- GitHub Actions CI (`.github/workflows/ci.yml`) runs `pytest` on every push/PR.

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

## License

Code in this repository is MIT. STEAD data follows its own license (see the STEAD repository / paper).
