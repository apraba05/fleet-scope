"""Rarity and priority scoring of mined scenarios.

A scenario is valuable for training/validation when it is *severe* and *rare*. Rarity
combines two signals:

* **type surprisal**  ``-log p(type)`` with add-one smoothing, normalised by its maximum
  (the rarest type scores 1.0). A swerve in a fleet of hard brakes is interesting.
* **within-type novelty**  distance to the k-th nearest neighbour of the same type in
  robust-standardised feature space, as a percentile rank (1.0 = most isolated). The
  1000th ordinary hard brake at 25 m/s is not; one at 3 m/s with a 9 m/s^3 jerk is.

    rarity   = 0.5 * type_surprisal + 0.5 * novelty
    priority = 0.5 * severity       + 0.5 * rarity        (all in [0, 1])
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from fleetscope.scenarios.detectors import FEATURES

KNN_K = 5


def robust_standardize(x: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Column-wise (x - median) / IQR, clipped. Robust to the heavy tails of event features."""
    med = np.median(x, axis=0)
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
    std = x.std(axis=0)
    scale = np.where(iqr > 1e-9, iqr, np.where(std > 1e-9, std, 1.0))
    return np.clip((x - med) / scale, -clip, clip)


def feature_matrix(table: pa.Table) -> np.ndarray:
    if table.num_rows == 0:
        return np.empty((0, len(FEATURES)))
    x = np.column_stack([table.column(f).to_numpy(zero_copy_only=False) for f in FEATURES])
    return robust_standardize(np.nan_to_num(x.astype(np.float64)))


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    if len(x) == 1:
        return np.ones(1)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return order / (len(x) - 1)


def knn_novelty(x: np.ndarray, groups: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """k-NN distance within each group, as percentile rank within that group."""
    out = np.ones(len(x))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) <= k:
            continue  # too few members to call any of them ordinary
        xs = x[idx]
        # blockwise to keep memory O(block * n)
        kth = np.empty(len(idx))
        for s in range(0, len(idx), 2048):
            d = np.sqrt(((xs[s : s + 2048, None, :] - xs[None, :, :]) ** 2).sum(-1))
            kth[s : s + 2048] = np.partition(d, k, axis=1)[:, k]  # index 0 is self
        out[idx] = _percentile_rank(kth)
    return out


def score_scenarios(table: pa.Table) -> pa.Table:
    n = table.num_rows
    if n == 0:
        empty = pa.array([], pa.float64())
        return (
            table.append_column("type_rarity", empty)
            .append_column("novelty", empty)
            .append_column("rarity", empty)
            .append_column("priority", empty)
            .append_column("rank", pa.array([], pa.int64()))
        )
    types = table.column("scenario_type").to_numpy(zero_copy_only=False)
    uniq, inv, counts = np.unique(types, return_inverse=True, return_counts=True)
    p = (counts + 1) / (n + len(uniq))
    surprisal = -np.log(p)
    type_rarity = (surprisal / surprisal.max())[inv] if surprisal.max() > 0 else np.ones(n)
    novelty = knn_novelty(feature_matrix(table), types)
    rarity = 0.5 * type_rarity + 0.5 * novelty
    sev = table.column("severity").to_numpy()
    priority = 0.5 * sev + 0.5 * rarity
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(-priority, kind="stable")] = np.arange(1, n + 1)
    out = (
        table.append_column("type_rarity", pa.array(type_rarity))
        .append_column("novelty", pa.array(novelty))
        .append_column("rarity", pa.array(rarity))
        .append_column("priority", pa.array(priority))
        .append_column("rank", pa.array(rank))
    )
    return out.take(pc.sort_indices(out, [("rank", "ascending")]))
