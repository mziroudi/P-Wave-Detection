"""Parse station / event identifiers from STEAD-style trace names."""

from __future__ import annotations

import re

# STA.NET_YYYYMMDDHHMMSS_EV  or  STA.NET_YYYYMMDDHHMM_NO (noise sometimes 12 digits)
_TRACE_RE = re.compile(
    r"^(?P<station>[^_]+)_(?P<timestamp>\d{10,14})(?:_(?P<suffix>.+))?$"
)


def parse_station_id(trace_name: str) -> str:
    """Return station key (e.g. ``AMT.HP``) from a STEAD-like trace name."""
    m = _TRACE_RE.match(str(trace_name))
    if m:
        return m.group("station")
    return str(trace_name).split("_", 1)[0]


def parse_event_id(trace_name: str, category: str | None = None) -> str:
    """
    Return a grouping key used for leakage-safe splits.

    Earthquakes: origin timestamp (shared across stations for the same event when
    STEAD naming aligns). Noise: full trace name so noise segments stay unique.
    """
    name = str(trace_name)
    cat = (category or "").lower()
    m = _TRACE_RE.match(name)
    if m is None:
        return name
    suffix = (m.group("suffix") or "").upper()
    is_noise = cat == "noise" or suffix.startswith("NO")
    if is_noise:
        return name
    return m.group("timestamp")


def enrich_meta_ids(meta: list[dict], category_key: str = "label") -> list[dict]:
    """Add ``station_id`` / ``event_id`` fields onto window metadata dicts."""
    out: list[dict] = []
    for m in meta:
        row = dict(m)
        name = str(row.get("trace_name", ""))
        # label 0 == noise in this project
        cat = None
        if "category" in row:
            cat = str(row["category"])
        elif category_key in row and row[category_key] == 0:
            cat = "noise"
        elif name.upper().endswith("_NO") or "_NO" in name.upper():
            cat = "noise"
        row.setdefault("station_id", parse_station_id(name))
        row.setdefault("event_id", parse_event_id(name, cat))
        out.append(row)
    return out
