"""Leakage-safe train/val/test splits by event or station groups."""

from __future__ import annotations

from typing import Literal

import numpy as np

GroupBy = Literal["event", "station", "window"]


def _group_keys(meta: list[dict], group_by: GroupBy) -> np.ndarray:
    if group_by == "window":
        return np.arange(len(meta))
    key_name = "event_id" if group_by == "event" else "station_id"
    keys = []
    for i, m in enumerate(meta):
        if key_name in m and m[key_name] is not None:
            keys.append(str(m[key_name]))
        else:
            keys.append(str(m.get("trace_name", i)))
    return np.asarray(keys, dtype=object)


def group_train_val_test_indices(
    meta: list[dict],
    y: np.ndarray | None = None,
    *,
    group_by: GroupBy = "event",
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split window indices so that groups (events or stations) never cross splits.

    ``val_size`` is the fraction of the *train+val* pool used for validation
    (same convention as nested ``train_test_split``).
    """
    n = len(meta)
    if n == 0:
        raise ValueError("Empty meta for split")
    groups = _group_keys(meta, group_by)
    rng = np.random.default_rng(seed)

    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    # Stratify groups by majority label when labels are available.
    if y is not None and len(y) == n and group_by != "window":
        group_label: dict[str, int] = {}
        for g, label in zip(groups, y):
            g = str(g)
            # prefer earthquake presence in the group
            prev = group_label.get(g, 0)
            group_label[g] = max(prev, int(label))
        g0 = [g for g in unique_groups if group_label.get(str(g), 0) == 0]
        g1 = [g for g in unique_groups if group_label.get(str(g), 0) == 1]
        rng.shuffle(g0)
        rng.shuffle(g1)

        def _split_list(items: list, test_frac: float, val_frac: float):
            n_items = len(items)
            n_test = max(1, int(round(n_items * test_frac))) if n_items >= 3 else max(0, n_items // 5)
            n_test = min(n_test, max(0, n_items - 2)) if n_items >= 3 else n_test
            rest = items[n_test:]
            n_val = max(1, int(round(len(rest) * val_frac))) if len(rest) >= 2 else 0
            n_val = min(n_val, max(0, len(rest) - 1)) if len(rest) >= 2 else n_val
            return items[n_test + n_val :], items[n_test : n_test + n_val], items[:n_test]

        tr0, va0, te0 = _split_list(g0, test_size, val_size)
        tr1, va1, te1 = _split_list(g1, test_size, val_size)
        train_g = set(tr0 + tr1)
        val_g = set(va0 + va1)
        test_g = set(te0 + te1)
    else:
        n_g = len(unique_groups)
        n_test = max(1, int(round(n_g * test_size))) if n_g >= 3 else max(0, n_g // 5)
        n_test = min(n_test, max(0, n_g - 2)) if n_g >= 3 else n_test
        test_g = set(unique_groups[:n_test].tolist())
        rest = unique_groups[n_test:]
        n_val = max(1, int(round(len(rest) * val_size))) if len(rest) >= 2 else 0
        n_val = min(n_val, max(0, len(rest) - 1)) if len(rest) >= 2 else n_val
        val_g = set(rest[:n_val].tolist())
        train_g = set(rest[n_val:].tolist())

    train_idx = np.array([i for i, g in enumerate(groups) if str(g) in train_g], dtype=np.int64)
    val_idx = np.array([i for i, g in enumerate(groups) if str(g) in val_g], dtype=np.int64)
    test_idx = np.array([i for i, g in enumerate(groups) if str(g) in test_g], dtype=np.int64)

    # Fall back if a split emptied (tiny datasets)
    if len(train_idx) == 0 or len(test_idx) == 0:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_size)))
        n_val = max(1, int(round((n - n_test) * val_size)))
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]

    return train_idx, val_idx, test_idx


def assert_no_group_leakage(
    meta: list[dict],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    group_by: GroupBy = "event",
) -> dict:
    """Return overlap stats; raise if train/test groups intersect when group_by != window."""
    if group_by == "window":
        return {"group_by": group_by, "train_test_overlap": 0}

    key = "event_id" if group_by == "event" else "station_id"
    def keys(idxs):
        return {str(meta[i].get(key, meta[i].get("trace_name"))) for i in idxs}

    tr, va, te = keys(train_idx), keys(val_idx), keys(test_idx)
    overlap_tt = tr & te
    overlap_tv = tr & va
    overlap_vt = va & te
    stats = {
        "group_by": group_by,
        "n_train_groups": len(tr),
        "n_val_groups": len(va),
        "n_test_groups": len(te),
        "train_test_overlap": len(overlap_tt),
        "train_val_overlap": len(overlap_tv),
        "val_test_overlap": len(overlap_vt),
    }
    if overlap_tt or overlap_tv or overlap_vt:
        raise RuntimeError(f"Group leakage detected: {stats}")
    return stats


def kfold_group_indices(
    meta: list[dict],
    *,
    group_by: GroupBy = "event",
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) folds with disjoint groups."""
    groups = _group_keys(meta, group_by)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    n_splits = max(2, min(n_splits, len(unique)))
    fold_groups = np.array_split(unique, n_splits)
    for i in range(n_splits):
        test_g = set(fold_groups[i].tolist())
        train_g = set(unique.tolist()) - test_g
        train_idx = np.array([j for j, g in enumerate(groups) if str(g) in train_g], dtype=np.int64)
        test_idx = np.array([j for j, g in enumerate(groups) if str(g) in test_g], dtype=np.int64)
        folds.append((train_idx, test_idx))
    return folds
