import itertools
import math

import numpy as np
import pyarrow as pa
import pytest

from fleetscope.scenarios.detectors import FEATURES
from fleetscope.scenarios.rarity import feature_matrix
from fleetscope.scenarios.selection import (
    METHODS,
    FacilityLocation,
    SelectionConfig,
    clip_costs,
    compare,
    lazy_greedy,
    select,
    selection_table,
    type_balanced_weights,
)

MB = 1_000_000
BOUND = (1 - 1 / math.e) / 2


def candidates(rng, n: int, types: list[str], dup_type: str | None = None) -> pa.Table:
    """Scored candidates; ``dup_type`` rows are near-duplicates with high priority."""
    feats = rng.normal(0, 1, (n, len(FEATURES)))
    kinds = rng.choice(types, n)
    prio = rng.uniform(0.2, 0.9, n)
    if dup_type is not None:
        dup = kinds == dup_type
        feats[dup] = rng.normal(0, 0.01, (dup.sum(), len(FEATURES)))
        prio[dup] = rng.uniform(0.9, 1.0, dup.sum())
    cols = {
        "scenario_id": [f"s{i}" for i in range(n)],
        "vehicle_id": ["v"] * n,
        "trip_id": ["t"] * n,
        "scenario_type": kinds,
        "start_ts": pa.array(np.arange(n) * 10**9, pa.timestamp("ns", tz="UTC")),
        "end_ts": pa.array(np.arange(n) * 10**9, pa.timestamp("ns", tz="UTC")),
        "severity": prio,
        "priority": prio,
    }
    for j, f in enumerate(FEATURES):
        cols[f] = feats[:, j]
    cols["duration_s"] = rng.uniform(0.5, 15.0, n)  # clip cost varies 66..150 MB
    return pa.table(cols)


def objective(table, cfg):
    types = table.column("scenario_type").to_numpy(zero_copy_only=False)
    w = type_balanced_weights(types, table.column("priority").to_numpy())
    return FacilityLocation(feature_matrix(table), types, w, cfg.sigma)


def brute_force(fl, costs, budget):
    best = 0.0
    for r in range(1, fl.n + 1):
        for subset in itertools.combinations(range(fl.n), r):
            if costs[list(subset)].sum() <= budget:
                best = max(best, fl.value(list(subset)))
    return best


def test_facility_location_matches_definition():
    rng = np.random.default_rng(1)
    t = candidates(rng, 12, ["a", "b"])
    cfg = SelectionConfig(budget_bytes=10**12)
    x = feature_matrix(t)
    types = t.column("scenario_type").to_numpy(zero_copy_only=False)
    w = t.column("priority").to_numpy()
    fl = FacilityLocation(x, types, w, cfg.sigma)
    S = [0, 3, 7]
    expect = 0.0
    for i in range(12):
        sims = [
            math.exp(-((x[i] - x[j]) ** 2).sum() / 2) if types[i] == types[j] else 0.0 for j in S
        ]
        expect += w[i] * max(sims)
    assert fl.value(S) == pytest.approx(expect, rel=1e-5)
    assert fl.value(list(range(12))) == pytest.approx(fl.value_all(), rel=1e-5)
    assert fl.value([]) == 0.0


def test_lazy_greedy_equals_plain_greedy():
    rng = np.random.default_rng(2)
    t = candidates(rng, 60, ["a", "b", "c"])
    cfg = SelectionConfig(budget_bytes=900 * MB)
    fl, costs = objective(t, cfg), clip_costs(t, cfg)
    plain, spent = [], 0
    while True:
        best, best_r = None, 0.0
        for j in range(fl.n):
            if j not in plain and spent + costs[j] <= cfg.budget_bytes:
                r = (fl.value([*plain, j]) - fl.value(plain)) / costs[j]
                if r > best_r + 1e-12:
                    best, best_r = j, r
        if best is None:
            break
        plain.append(best)
        spent += costs[best]
    assert lazy_greedy(fl, costs, cfg.budget_bytes) == plain


def test_diverse_vs_brute_force_optimum():
    ratios = []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        t = candidates(rng, 11, ["a", "b", "c"], dup_type="a")
        cfg = SelectionConfig(budget_bytes=int(rng.uniform(200, 500)) * MB)
        fl, costs = objective(t, cfg), clip_costs(t, cfg)
        opt = brute_force(fl, costs, cfg.budget_bytes)
        got = select(t, cfg, "diverse")
        assert got.cost_bytes <= cfg.budget_bytes
        assert got.objective >= BOUND * opt - 1e-9  # the worst-case guarantee
        ratios.append(got.objective / opt)
    # what it achieves in practice: far above the worst case
    assert min(ratios) > 0.8 and np.mean(ratios) > 0.95, ratios


@pytest.mark.parametrize("method", METHODS)
def test_every_method_respects_budget(method):
    rng = np.random.default_rng(3)
    t = candidates(rng, 200, ["a", "b", "c", "d"])
    cfg = SelectionConfig(budget_bytes=2_000 * MB)
    s = select(t, cfg, method)
    assert 0 < s.cost_bytes <= cfg.budget_bytes
    assert len(set(s.indices)) == len(s.indices)
    assert s.cost_bytes == clip_costs(t, cfg)[s.indices].sum()


def test_diverse_beats_baselines_when_top_is_redundant():
    rng = np.random.default_rng(4)
    t = candidates(rng, 400, ["hard_brake", "cut_in", "swerve", "dropout"], dup_type="hard_brake")
    cfg = SelectionConfig(budget_bytes=1_500 * MB)
    rows = {r["method"]: r for r in compare(t, cfg).to_pylist()}
    for m in METHODS[1:]:
        assert rows["diverse"]["objective"] > rows[m]["objective"], m
    assert rows["diverse"]["types_covered"] == 4
    # top_priority burns the budget on near-identical hard brakes
    assert rows["top_priority"]["types_covered"] == 1


def test_singleton_wins_when_greedy_is_myopic():
    # Classic knapsack failure of ratio-greedy: one tiny clip with a slightly better ratio
    # blocks a big clip that is worth far more.
    feats = np.array([[0.0] * len(FEATURES), [50.0] * len(FEATURES)])
    t = pa.table(
        {
            "scenario_id": ["small", "big"],
            "scenario_type": ["a", "b"],
            "severity": [0.2, 1.0],
            "priority": [0.2, 1.0],
            "duration_s": [0.0, 90.0],
            **{f: feats[:, j] for j, f in enumerate(FEATURES) if f != "duration_s"},
        }
    )
    cfg = SelectionConfig(budget_bytes=600 * MB, padding_s=5.0, bytes_per_s=6 * MB)
    # costs: small = 60 MB (ratio 0.2/60), big = 600 MB (ratio 1/600); small has the
    # better ratio, after it big no longer fits.
    costs = clip_costs(t, cfg)
    assert 0.2 / costs[0] > 1.0 / costs[1]
    fl = objective(t, cfg)
    assert lazy_greedy(fl, costs, cfg.budget_bytes) == [0]
    assert select(t, cfg, "diverse").indices == [1]


def test_type_balanced_weights():
    types = np.array(["a", "a", "a", "b"])
    prio = np.array([0.2, 0.4, 0.6, 0.9])
    w = type_balanced_weights(types, prio)
    # each type's weights sum to its mean priority, split in proportion to priority
    assert w[:3].sum() == pytest.approx(0.4) and w[3] == pytest.approx(0.9)
    assert w[2] / w[0] == pytest.approx(3.0)


def test_empty_and_unknown_method():
    t = candidates(np.random.default_rng(0), 0, ["a"])
    s = select(t, SelectionConfig(budget_bytes=MB))
    assert s.indices == [] and s.coverage == 0.0
    with pytest.raises(ValueError):
        select(t, SelectionConfig(budget_bytes=MB), "best")


def test_selection_table_rows_in_pick_order():
    rng = np.random.default_rng(5)
    t = candidates(rng, 50, ["a", "b"])
    cfg = SelectionConfig(budget_bytes=800 * MB)
    s = select(t, cfg)
    out = selection_table(t, s, cfg)
    assert out.num_rows == len(s.indices)
    assert out.column("scenario_id").to_pylist() == [f"s{i}" for i in s.indices]
    assert out.column("pick_order").to_pylist() == list(range(1, len(s.indices) + 1))
    assert sum(out.column("clip_bytes").to_pylist()) == s.cost_bytes
