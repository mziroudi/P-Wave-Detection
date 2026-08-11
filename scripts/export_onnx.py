"""Export SeismicCNN1D to ONNX and optionally verify with onnxruntime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import SeismicCNN1D
from src.utils import MODELS_DIR, WINDOW_SAMPLES, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export classifier to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "seismic_cnn1d.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    ensure_dirs()

    device = torch.device("cpu")
    model = SeismicCNN1D().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, WINDOW_SAMPLES, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"[ok] wrote {args.out}")

    # Verify numerical agreement if onnxruntime is installed
    report = {"onnx_path": str(args.out), "opset": args.opset}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        x = dummy.numpy()
        ort_out = sess.run(None, {"waveform": x})[0]
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        max_abs = float(np.max(np.abs(ort_out - pt_out)))
        report["max_abs_diff_vs_pytorch"] = max_abs
        print(f"[ok] onnxruntime max|diff|={max_abs:.6e}")
    except ImportError:
        report["onnxruntime"] = "not installed — skip numeric check"
        print("[info] onnxruntime not installed; skipped verification")

    meta_path = args.out.with_suffix(".onnx.json")
    meta_path.write_text(json.dumps(report, indent=2))
    print(f"[ok] {meta_path}")


if __name__ == "__main__":
    main()
