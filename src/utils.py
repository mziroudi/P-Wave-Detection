"""Shared paths and helpers."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
STEAD_SUBSAMPLE_DIR = DATA_DIR / "stead_subsample"
STEAD_OFFICIAL_DIR = DATA_DIR / "stead_official"
WINDOWS_DIR = DATA_DIR / "windows"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

SAMPLE_RATE_HZ = 100
TRACE_LENGTH = 6000  # 60 s @ 100 Hz
WINDOW_SECONDS = 10
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE_HZ  # 1000

LABEL_NOISE = 0
LABEL_EARTHQUAKE = 1
CLASS_NAMES = ("noise", "earthquake")


def ensure_dirs() -> None:
    for path in (
        DATA_DIR,
        STEAD_SUBSAMPLE_DIR,
        STEAD_OFFICIAL_DIR,
        WINDOWS_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
