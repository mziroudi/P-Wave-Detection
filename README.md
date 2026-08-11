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

# 3) Build labeled 10 s windows (Noise vs Earthquake)
python scripts/prepare_windows.py --max-earthquake 3000 --max-noise 3000

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
  dataset.py          # PyTorch Dataset + cache I/O
  model.py            # 1D CNN classifier + P-arrival regressor
  sliding_window.py   # continuous inference + alert rule
scripts/
  download_stead.py
  download_continuous.py
  visualize_waveforms.py
  prepare_windows.py
  prepare_regression.py
  train.py
  train_regression.py
  evaluate.py
  evaluate_regression.py
  continuous_inference.py
  live_stream.py
  predict.py
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

## Limitations & Production Roadmap

This project successfully demonstrates an end-to-end ML pipeline for seismic detection. However, transitioning this from a functional prototype to a production-grade Earthquake Early Warning (EEW) system requires addressing several critical gaps in data integrity, realistic evaluation, and engineering rigor.

### 1. Data Provenance & Leakage Control (Critical)

**Current State:** The model is trained on a 3rd-party Zenodo subsample of STEAD, and the train/test split is performed randomly at the window level.

**The Problem:** STEAD contains multiple traces from the same earthquake recorded at different stations. Randomly splitting windows risks data leakage, where the model sees data from the same seismic event in both training and test sets, artificially inflating the 97.4% accuracy. Furthermore, the Zenodo subsample's curation criteria are unverified.

**Production Fix:** Migrate to the official 14GB STEAD chunks. Implement event-level and station-level train/test splitting to guarantee zero overlap of seismic events across splits.

### 2. Realistic Class Imbalance & False Alarms (Critical)

**Current State:** Evaluation uses an artificially balanced 50/50 dataset (3,000 noise vs. 3,000 earthquakes).

**The Problem:** Real seismic streams are 99%+ noise. A 97.4% accuracy on a balanced dataset tells us nothing about the real-world False Positive Rate (FPR), which is the primary bottleneck in EEW systems. A model that cries wolf is operationally useless.

**Production Fix:** Re-evaluate on highly imbalanced datasets (e.g., 1000:1 noise-to-event ratio). Shift the primary evaluation metric from ROC-AUC to Precision-Recall AUC (PR-AUC). Report metrics as False Alarms per 24 hours of continuous noise data.

### 3. Benchmarking Against Classical Baselines

**Current State:** The P-wave arrival regression achieves an MAE of ~230ms.

**The Problem:** This metric exists in a vacuum. In seismology, deep learning models must be benchmarked against classical algorithms—specifically STA/LTA (Short-Term Average / Long-Term Average)—which are computationally cheap and widely deployed. State-of-the-art pickers (e.g., EQTransformer, PhaseNet) operate at <50ms MAE.

**Production Fix:** Implement a standard STA/LTA algorithm on the same test set and compare MAE and latency against the 1D CNN. The deep learning model must justify its complexity by outperforming STA/LTA.

### 4. Inference Latency & Edge Deployment

**Current State:** The core value proposition of EEW is "beating the S-wave," but model inference latency is unmeasured. The live stream script polls via requests, introducing network overhead.

**Production Fix:** Benchmark inference time per 10s window on standard CPUs and edge devices (e.g., Raspberry Pi). Optimize via ONNX export or TensorRT. Replace the HTTP poller with a true streaming protocol (WebSocket/UDP) or a local ringbuffer to guarantee sub-second latency.

### 5. Software Engineering Rigor

**Current State:** Single training run, no variance reporting, unpinned dependencies, and no unit tests for the windowing logic.

**Production Fix:**

- Set global random seeds (PyTorch, NumPy, Python `random`).
- Perform 5-fold cross-validation and report standard deviation.
- Pin all dependencies in `requirements.txt` (e.g., `torch==2.0.1`).
- Add unit tests specifically for `windows.py` to prevent off-by-one indexing errors in P-pick alignment.
- Implement GitHub Actions CI to ensure the pipeline runs from scratch on a clean environment.

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
