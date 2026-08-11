"""
Quick inference demo: classify a single cached window or a random test example.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import load_window_cache
from src.model import SeismicCNN1D
from src.utils import CLASS_NAMES, MODELS_DIR, WINDOWS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one window")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--windows-dir", type=Path, default=WINDOWS_DIR)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_window_cache(args.windows_dir / "test")
    idx = args.index % len(y)

    model = SeismicCNN1D().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x[idx : idx + 1]).to(device))
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(prob.argmax())
    print(f"index={idx}")
    print(f"true_label={CLASS_NAMES[int(y[idx])]}")
    print(f"pred_label={CLASS_NAMES[pred]}")
    print(f"p(noise)={prob[0]:.4f}  p(earthquake)={prob[1]:.4f}")


if __name__ == "__main__":
    main()
