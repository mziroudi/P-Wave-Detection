"""Download continuous USGS/SCEDC waveforms that contain a known earthquake."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import DATA_DIR, ensure_dirs

# 2019 Ridgecrest M7.1 — clear local recording at CI.CLC (100 Hz)
DEFAULT_EVENT = {
    "name": "ridgecrest_m71_2019",
    "origin_time": "2019-07-06T03:19:53",
    "magnitude": 7.1,
    "network": "CI",
    "station": "CLC",
    "location": "*",
    "channel": "HH?",
    "client": "SCEDC",
    "pre_event_s": 600,
    "duration_s": 3600,
}


def download_event_hour(cfg: dict, out_dir: Path) -> Path:
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    origin = UTCDateTime(cfg["origin_time"])
    start = origin - float(cfg["pre_event_s"])
    end = start + float(cfg["duration_s"])

    client = Client(cfg["client"])
    print(
        f"[download] {cfg['client']} {cfg['network']}.{cfg['station']} "
        f"{cfg['channel']}  {start} → {end}"
    )
    st = client.get_waveforms(
        cfg["network"],
        cfg["station"],
        cfg["location"],
        cfg["channel"],
        start,
        end,
    )
    st.merge(method=1, fill_value="interpolate")
    st.sort()

    mseed_path = out_dir / f"{cfg['name']}.mseed"
    meta_path = out_dir / f"{cfg['name']}.json"
    st.write(str(mseed_path), format="MSEED")
    meta = {
        **cfg,
        "starttime": str(start),
        "endtime": str(end),
        "traces": [
            {
                "id": tr.id,
                "sampling_rate": float(tr.stats.sampling_rate),
                "npts": int(tr.stats.npts),
                "starttime": str(tr.stats.starttime),
            }
            for tr in st
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[ok] wrote {mseed_path} ({mseed_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[ok] wrote {meta_path}")
    return mseed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 1-hour continuous waveform with an earthquake")
    parser.add_argument("--event", default="ridgecrest", choices=["ridgecrest"])
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "continuous")
    parser.add_argument("--client", default=None)
    parser.add_argument("--network", default=None)
    parser.add_argument("--station", default=None)
    parser.add_argument("--pre-event-s", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULT_EVENT)
    if args.client:
        cfg["client"] = args.client
    if args.network:
        cfg["network"] = args.network
    if args.station:
        cfg["station"] = args.station
    if args.pre_event_s is not None:
        cfg["pre_event_s"] = args.pre_event_s
    if args.duration_s is not None:
        cfg["duration_s"] = args.duration_s

    path = download_event_hour(cfg, args.out_dir)
    # Quick sanity read
    st = read(str(path))
    print(st)


if __name__ == "__main__":
    main()
