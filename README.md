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
!git checkout cursor/p-wave-detection-5a2b
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

## Limitations & Next Steps

The classifier on pre-cut STEAD windows is the baseline. This repo now also implements the senior-level follow-ons; remaining polish is noted below.

1. **Continuous Data Testing** — **implemented** in `scripts/continuous_inference.py`  
   Sliding 10 s windows over a 1-hour USGS/SCEDC recording (Ridgecrest M7.1 at CI.CLC), with probability traces and a consecutive-threshold alert rule to control false positives. Next: scale to a full 24-hour quiet stretch for a proper false-alarm rate study.

2. **P-Wave Arrival Time Prediction** — **implemented** in `scripts/train_regression.py`  
   Regression head predicts the P-wave sample index inside each 10 s window (millisecond-convertible). Next: joint multi-task training (detect + pick) and comparison against catalog picks on continuous data.

3. **Inference Speed**  
   An EEW system requires sub-second latency. Future work will benchmark the model's inference time on edge devices (like a Raspberry Pi) to ensure alerts can be sent faster than the S-wave travels. The live FDSN poller (`scripts/live_stream.py`) is the software-side streaming half of that goal.

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
