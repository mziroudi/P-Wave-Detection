"""Download STEAD data for this project.

Preferred path (works in restricted environments):
  Zenodo STEAD subsample (HDF5 with waveforms + P/S arrivals)

Official path (large; often behind Cloudflare/Google Drive quotas):
  https://github.com/smousavi05/STEAD  — chunk HDF5 + CSV pairs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Allow `python scripts/...` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stead_io import export_subsample_csv
from src.utils import STEAD_OFFICIAL_DIR, STEAD_SUBSAMPLE_DIR, ensure_dirs

ZENODO_RECORD = "11094536"
ZENODO_FILES = {
    "test.hdf5": f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/test.hdf5/content",
    "test_noise.hdf5": f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/test_noise.hdf5/content",
    "train.hdf5": f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/train.hdf5/content",
    "train_noise.hdf5": f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/train_noise.hdf5/content",
}

OFFICIAL_LINKS = {
    "chunk1_noise": "https://rebrand.ly/chunk1",
    "chunk2_earthquakes": "https://rebrand.ly/chunk2",
    "chunk3_earthquakes": "https://rebrand.ly/chunk3",
    "whole_merged": "https://rebrand.ly/whole",
}


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    mode = "wb"
    existing = 0
    if tmp.exists():
        existing = tmp.stat().st_size
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total_i = int(total) + existing if total else None
        with open(tmp, mode) as f, tqdm(
            total=total_i,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(dest)


def download_subsample(include_train: bool = True, force: bool = False) -> None:
    ensure_dirs()
    files = dict(ZENODO_FILES)
    if not include_train:
        files = {k: v for k, v in files.items() if k.startswith("test")}

    for name, url in files.items():
        dest = STEAD_SUBSAMPLE_DIR / name
        if dest.exists() and not force:
            print(f"[skip] {dest} already exists")
            continue
        print(f"[download] {name}")
        download_file(url, dest)

    csv_path = STEAD_SUBSAMPLE_DIR / "stead_subsample_index.csv"
    df = export_subsample_csv(STEAD_SUBSAMPLE_DIR, csv_path)
    print(f"[ok] wrote CSV index with {len(df)} traces → {csv_path}")


def print_official_instructions() -> None:
    print(
        """
Official STEAD download (Stanford Earthquake Dataset)
=====================================================
Repo: https://github.com/smousavi05/STEAD

Each chunk is ~14 GB and contains one HDF5 + one CSV (~200k 3C waveforms).

  chunk1 (noise):              https://rebrand.ly/chunk1
  chunk2–6 (local earthquakes): https://rebrand.ly/chunk2 ... chunk6
  whole merged (~85 GB):       https://rebrand.ly/whole

MEGA mirrors (if rebrand.ly / Drive quotas fail) are listed in community forks
such as https://github.com/Amandah21/STEAD

After download, place files here:
  {official}/

Then point training at them with:
  python scripts/prepare_windows.py --source official \\
      --csv data/stead_official/chunk2.csv \\
      --hdf5 data/stead_official/chunk2.hdf5
""".format(official=STEAD_OFFICIAL_DIR)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download STEAD data for P-wave detection")
    parser.add_argument(
        "--source",
        choices=["subsample", "official-info", "both"],
        default="subsample",
        help="subsample=Zenodo STEAD subsample (recommended); official-info=print links",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only download test_*.hdf5 from Zenodo (~800 MB total)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.source in ("subsample", "both"):
        download_subsample(include_train=not args.test_only, force=args.force)
    if args.source in ("official-info", "both"):
        print_official_instructions()


if __name__ == "__main__":
    main()
