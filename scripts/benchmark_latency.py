"""Benchmark per-window inference latency (PyTorch CPU and optional ONNX)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import SeismicCNN1D
from src.seed import set_global_seed
from src.utils import ARTIFACTS_DIR, MODELS_DIR, WINDOW_SAMPLES, ensure_dirs


def _bench_callable(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "runs": runs,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference latency benchmark")
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "seismic_cnn1d_best.pt")
    parser.add_argument("--onnx", type=Path, default=MODELS_DIR / "seismic_cnn1d.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "latency")
    args = parser.parse_args()
    ensure_dirs()
    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = SeismicCNN1D().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = torch.randn(args.batch_size, 3, WINDOW_SAMPLES, dtype=torch.float32)

    def pt_fn():
        with torch.no_grad():
            _ = model(x)

    results = {
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "window_samples": WINDOW_SAMPLES,
        "pytorch_cpu": _bench_callable(pt_fn, args.warmup, args.runs),
    }

    if args.onnx.exists():
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
            xin = x.numpy()

            def ort_fn():
                _ = sess.run(None, {"waveform": xin})

            results["onnxruntime_cpu"] = _bench_callable(ort_fn, args.warmup, args.runs)
            results["onnx_path"] = str(args.onnx)
        except ImportError:
            results["onnxruntime_cpu"] = "onnxruntime not installed"
    else:
        results["onnxruntime_cpu"] = f"missing ONNX file: {args.onnx}"

    out = args.out_dir / "latency_benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
