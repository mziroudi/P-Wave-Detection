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

## Pipeline

```bash
# 1) Download data
python scripts/download_stead.py --test-only

# 2) Visualize with ObsPy / Matplotlib
python scripts/visualize_waveforms.py

# 3) Build labeled 10 s windows (Noise vs Earthquake)
python scripts/prepare_windows.py --max-earthquake 3000 --max-noise 3000

# 4) Train 1D CNN
python scripts/train.py --epochs 12 --batch-size 64

# 5) Evaluate + plots
python scripts/evaluate.py

# 6) Single-window demo
python scripts/predict.py --index 0
```

Artifacts land in `artifacts/` (waveforms, confusion matrix, ROC). Checkpoints land in `models/`.

### Official STEAD HDF5 + CSV

```bash
python scripts/prepare_windows.py --source official \
  --csv data/stead_official/chunk2.csv \
  --hdf5 data/stead_official/chunk2.hdf5
```

## Model

`SeismicCNN1D` — four Conv1d blocks + global average pool + MLP head.

- **Input:** `(batch, 3, 1000)`
- **Output:** logits for `[noise, earthquake]`

## Project layout

```text
src/
  stead_io.py      # STEAD subsample + official loaders
  windows.py       # 10 s window extraction / normalization
  dataset.py       # PyTorch Dataset + cache I/O
  model.py         # 1D CNN
scripts/
  download_stead.py
  visualize_waveforms.py
  prepare_windows.py
  train.py
  evaluate.py
  predict.py
notebooks/
  explore_stead.ipynb
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
