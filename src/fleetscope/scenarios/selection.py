"""Choose which scenario clips to upload under a fixed storage budget.

Every mined scenario is a candidate clip (event +- padding of multi-camera video and
logs). Uploading all of them is not possible, so we pick a subset ``S`` with total size
<= budget that is *valuable* (high priority) and *diverse* (does not spend the budget on
hundreds of near-identical hard brakes).

Objective: type-balanced weighted facility location

    f(S) = sum_i  w_i * max_{j in S} sim(i, j)

over all candidates ``i``, with ``sim(i, j) = exp(-||x_i - x_j||^2 / (2 sigma^2))`` in
robust-standardised feature space when ``i`` and ``j`` are of the same type, 0 otherwise.
``f`` rewards a clip that *represents* many valuable, similar scenarios; once one is
selected, its neighbours add little.

The weights split each type's value ``v_t`` (its mean priority) over its members in
proportion to priority: ``w_i = v_t * p_i / sum_{j in t} p_j``. Without this, a type's
value grows with its count, and 1000 ordinary hard brakes would outweigh the one swerve
the fleet saw; with it, ``f(S) / f(all)`` is the priority-weighted average over types of
the fraction of each type's variety that ``S`` captures.

``f`` is monotone submodular, so under a knapsack constraint the better of

* cost-benefit lazy greedy (pick max marginal gain / bytes), and
* the best single clip that fits

is a (1 - 1/e) / 2 approximation of the optimum (Khuller, Moss & Naor 1999; Leskovec et
al. 2007). Lazy evaluation (CELF) is exact because marginal gains only shrink.

Baselines for comparison: ``random``, ``top_priority``, ``top_severity`` and
``stratified`` (round-robin across types in priority order), each filling the budget
greedily in its own order.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from fleetscope.scenarios.rarity import feature_matrix

CLIP_BYTES_PER_S = 6_000_000  # 8 cameras at ~6 Mbit/s, plus CAN / IMU logs
CLIP_PADDING_S = 5.0
METHODS = ("diverse", "top_priority", "top_severity", "stratified", "random")


@dataclass(frozen=True)
class SelectionConfig:
    budget_bytes: int
    bytes_per_s: float = CLIP_BYTES_PER_S
    padding_s: float = CLIP_PADDING_S
    sigma: float = 1.0  # similarity length scale, in IQR units of each feature
    seed: int = 0  # for the random baseline


@dataclass
class Selection:
    method: str
    indices: list[int]  # row indices into the candidate table, in pick order
    cost_bytes: int
    objective: float  # f(S)
    max_objective: float  # f(all candidates): the value of uploading everything

    @property
    def coverage(self) -> float:
        return self.objective / self.max_objective if self.max_objective > 0 else 0.0


def type_balanced_weights(types: np.ndarray, priority: np.ndarray) -> np.ndarray:
    w = np.zeros(len(types))
    for t in np.unique(types):
        idx = np.flatnonzero(types == t)
        total = priority[idx].sum()
        if total > 0:
            w[idx] = priority[idx].mean() * priority[idx] / total
    return w


def clip_costs(table: pa.Table, cfg: SelectionConfig) -> np.ndarray:
    dur = table.column("duration_s").to_numpy(zero_copy_only=False)
    return np.ceil((dur + 2 * cfg.padding_s) * cfg.bytes_per_s).astype(np.int64)


class FacilityLocation:
    """f(S) = sum_i w_i max_{j in S} sim(i, j), with sim block-diagonal by type."""

    def __init__(self, x: np.ndarray, groups: np.ndarray, weights: np.ndarray, sigma: float):
        self.n = len(x)
        self.w = weights.astype(np.float64)
        self.group_of = np.empty(self.n, dtype=np.int64)
        self.members: list[np.ndarray] = []
        self.sims: list[np.ndarray] = []  # per group: (m, m) float32
        for gi, g in enumerate(np.unique(groups)):
            idx = np.flatnonzero(groups == g)
            xs = x[idx].astype(np.float32)
            d2 = ((xs[:, None, :] - xs[None, :, :]) ** 2).sum(-1)
            self.members.append(idx)
            self.sims.append(np.exp(-d2 / np.float32(2 * sigma**2)))
            self.group_of[idx] = gi
        self.local = np.empty(self.n, dtype=np.int64)  # position inside its group
        for idx in self.members:
            self.local[idx] = np.arange(len(idx))
        # current best similarity of each candidate to the selected set
        self.best = [np.zeros(len(idx), dtype=np.float32) for idx in self.members]

    def reset(self) -> None:
        for b in self.best:
            b[:] = 0

    def gain(self, j: int) -> float:
        g, k = self.group_of[j], self.local[j]
        lift = np.maximum(self.sims[g][:, k] - self.best[g], 0)
        return float(self.w[self.members[g]] @ lift)

    def add(self, j: int) -> None:
        g, k = self.group_of[j], self.local[j]
        np.maximum(self.best[g], self.sims[g][:, k], out=self.best[g])

    def value(self, selected: list[int]) -> float:
        self.reset()
        for j in selected:
            self.add(j)
        total = sum(float(self.w[idx] @ b) for idx, b in zip(self.members, self.best, strict=True))
        self.reset()
        return total

    def value_all(self) -> float:
        return float(self.w.sum())  # every i is its own best match: sim(i, i) = 1


def lazy_greedy(fl: FacilityLocation, costs: np.ndarray, budget: int) -> list[int]:
    """Cost-benefit CELF: repeatedly add the max gain-per-byte candidate that fits.

    Heap entries carry the size of S their ratio was computed against. By submodularity a
    stale ratio is an upper bound, so a fresh ratio at the top of the heap is the true max.
    """
    fl.reset()
    heap = [(-fl.gain(j) / costs[j], j, 0) for j in range(fl.n) if costs[j] <= budget]
    heapq.heapify(heap)
    chosen: list[int] = []
    spent = 0
    while heap:
        neg_ratio, j, at = heapq.heappop(heap)
        if spent + costs[j] > budget:
            continue  # never fits again: the remaining budget only shrinks
        if at != len(chosen):
            heapq.heappush(heap, (-fl.gain(j) / costs[j], j, len(chosen)))
            continue
        if neg_ratio >= 0:
            break  # nothing left adds value
        chosen.append(j)
        spent += int(costs[j])
        fl.add(j)
    fl.reset()
    return chosen


def _fill(order: np.ndarray, costs: np.ndarray, budget: int) -> list[int]:
    chosen, spent = [], 0
    for j in order.tolist():
        if spent + costs[j] <= budget:
            chosen.append(j)
            spent += int(costs[j])
    return chosen


def _stratified_order(types: np.ndarray, priority: np.ndarray) -> np.ndarray:
    queues = []
    for t in np.unique(types):
        idx = np.flatnonzero(types == t)
        queues.append(idx[np.argsort(-priority[idx], kind="stable")].tolist())
    order: list[int] = []
    depth = 0
    while len(order) < len(types):
        order.extend(q[depth] for q in queues if depth < len(q))
        depth += 1
    return np.array(order, dtype=np.int64)


def select(table: pa.Table, cfg: SelectionConfig, method: str = "diverse") -> Selection:
    """Select clips from a scored scenarios table (output of ``score_scenarios``)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    n = table.num_rows
    if n == 0:
        return Selection(method, [], 0, 0.0, 0.0)
    costs = clip_costs(table, cfg)
    types = table.column("scenario_type").to_numpy(zero_copy_only=False)
    priority = table.column("priority").to_numpy()
    weights = type_balanced_weights(types, priority)
    fl = FacilityLocation(feature_matrix(table), types, weights, cfg.sigma)

    if method == "diverse":
        greedy = lazy_greedy(fl, costs, cfg.budget_bytes)
        fits = np.flatnonzero(costs <= cfg.budget_bytes)
        chosen = greedy
        if len(fits):
            fl.reset()
            single = int(fits[np.argmax([fl.gain(int(j)) for j in fits])])
            if fl.value([single]) > fl.value(greedy):
                chosen = [single]
    elif method == "top_priority":
        chosen = _fill(np.argsort(-priority, kind="stable"), costs, cfg.budget_bytes)
    elif method == "top_severity":
        sev = table.column("severity").to_numpy()
        chosen = _fill(np.argsort(-sev, kind="stable"), costs, cfg.budget_bytes)
    elif method == "stratified":
        chosen = _fill(_stratified_order(types, priority), costs, cfg.budget_bytes)
    else:
        rng = np.random.default_rng(cfg.seed)
        chosen = _fill(rng.permutation(n), costs, cfg.budget_bytes)

    return Selection(
        method=method,
        indices=chosen,
        cost_bytes=int(costs[chosen].sum()) if chosen else 0,
        objective=fl.value(chosen),
        max_objective=fl.value_all(),
    )


def compare(table: pa.Table, cfg: SelectionConfig) -> pa.Table:
    """Run every method on the same candidates and budget; one summary row per method."""
    types = table.column("scenario_type").to_numpy(zero_copy_only=False)
    priority = table.column("priority").to_numpy() if table.num_rows else np.empty(0)
    n_types = len(np.unique(types))
    rows = []
    for m in METHODS:
        s = select(table, cfg, m)
        rows.append(
            {
                "method": m,
                "clips": len(s.indices),
                "cost_bytes": s.cost_bytes,
                "budget_used": s.cost_bytes / cfg.budget_bytes,
                "objective": s.objective,
                "coverage": s.coverage,
                "types_covered": len(set(types[s.indices].tolist())),
                "types_total": n_types,
                "mean_priority": float(priority[s.indices].mean()) if s.indices else 0.0,
            }
        )
    return pa.Table.from_pylist(rows)


def selection_table(table: pa.Table, s: Selection, cfg: SelectionConfig) -> pa.Table:
    """The selected rows, in pick order, with their clip cost and pick order."""
    costs = clip_costs(table, cfg)
    out = table.take(pa.array(s.indices, pa.int64()))
    return (
        out.select(["scenario_id", "vehicle_id", "trip_id", "scenario_type", "start_ts", "end_ts"])
        .append_column("severity", out.column("severity"))
        .append_column("priority", out.column("priority"))
        .append_column("pick_order", pa.array(np.arange(1, len(s.indices) + 1), pa.int64()))
        .append_column("clip_bytes", pa.array(costs[s.indices] if s.indices else [], pa.int64()))
        .append_column("method", pa.array([s.method] * len(s.indices), pa.string()))
    )
