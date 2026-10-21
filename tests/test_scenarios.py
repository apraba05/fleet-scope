import numpy as np
import pyarrow as pa
import pytest

from fleetscope.lake.sql import Lake, write_derived
from fleetscope.replay.fleet import FaultConfig, FaultStats, inject_faults, interleave
from fleetscope.scenarios.detectors import FEATURES, detect_trip, merge_runs, runs, severity
from fleetscope.scenarios.pipeline import evaluate, ground_truth_table, mine_scenarios
from fleetscope.scenarios.rarity import knn_novelty, score_scenarios
from fleetscope.schema import SAMPLE_HZ, SIGNALS
from fleetscope.sources.comma2k19 import load_segment
from fleetscope.sources.synthetic import FleetConfig, generate_fleet
from fleetscope.stream.local import InMemoryTopic, run_local
from fleetscope.stream.processor import ProcessorConfig
from tests.conftest import SAMPLE_SEGMENT

T0 = 1_790_000_000 * 10**9
PERIOD = 10**9 // SAMPLE_HZ


def steady(seconds: float, speed: float = 25.0, lead: float | None = None):
    """Steady cruise; ``lead`` is a constant-gap lead vehicle (or none)."""
    n = int(seconds * SAMPLE_HZ)
    cols = {s: np.zeros(n) for s in SIGNALS}
    cols["speed"][:] = speed
    cols["lat"][:] = 37.0
    cols["lon"][:] = -122.0
    cols["lead_dist"][:] = np.nan if lead is None else lead
    cols["lead_rel_speed"][:] = np.nan if lead is None else 0.0
    return T0 + np.arange(n, dtype=np.int64) * PERIOD, cols


def at(s: float) -> int:
    return int(s * SAMPLE_HZ)


def kinds(found) -> list[str]:
    return [s.scenario_type for s in found]


# --- run utilities -------------------------------------------------------------------
def test_runs_and_merge():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert runs(mask).tolist() == [[1, 3], [5, 6], [7, 10]]
    assert runs(np.zeros(0, bool)).shape == (0, 2)
    t = np.arange(10) * PERIOD
    # gap between run ends (idx 2 -> 5) is 0.15 s; (5 -> 7) is 0.1 s
    assert merge_runs(runs(mask), t, 0.1).tolist() == [[1, 3], [5, 10]]
    assert merge_runs(runs(mask), t, 0.2).tolist() == [[1, 10]]


def test_severity_scale():
    assert severity(4.0, 4.0, 8.0) == pytest.approx(0.2)
    assert severity(6.0, 4.0, 8.0) == pytest.approx(0.6)
    assert severity(20.0, 4.0, 8.0) == pytest.approx(1.0)
    assert severity(-0.3, -0.8, -0.3) == pytest.approx(1.0)  # "lower is worse" via negation


# --- detectors on hand-built signals -------------------------------------------------
def test_quiet_drive_has_no_events():
    t, c = steady(60, lead=50.0)
    assert detect_trip("v", "t", t, c) == []


def test_hard_brake_detected_with_peak_and_features():
    t, c = steady(20)
    c["accel_long"][at(10) : at(11.5)] = -6.0
    (s,) = detect_trip("v", "t", t, c)
    assert s.scenario_type == "hard_brake"
    assert s.peak == -6.0
    assert s.severity == pytest.approx(0.6)
    assert s.start_ns == t[at(10)] and s.end_ns == t[at(11.5) - 1]
    assert set(s.features) == set(FEATURES)
    assert s.features["duration_s"] == pytest.approx(1.5)


def test_short_spike_is_not_an_event():
    t, c = steady(20)
    c["accel_long"][at(10) : at(10) + 2] = -9.0  # 0.1 s: a pothole / sensor spike
    assert detect_trip("v", "t", t, c) == []


def test_swerve_is_not_also_a_sharp_turn():
    t, c = steady(20)
    c["accel_lat"][at(10) : at(11)] = 4.0
    c["accel_lat"][at(11) : at(12)] = -4.0
    assert kinds(detect_trip("v", "t", t, c)) == ["evasive_swerve"]


def test_one_sided_lateral_is_sharp_turn():
    t, c = steady(20)
    c["accel_lat"][at(10) : at(12)] = 4.0
    assert kinds(detect_trip("v", "t", t, c)) == ["sharp_turn"]


def closing_profile(c, a: int, b: int, gap0: float, closing: float):
    k = np.arange(b - a)
    c["lead_dist"][a:b] = gap0 - closing * k / SAMPLE_HZ
    c["lead_rel_speed"][a:b] = -closing


def test_low_ttc_on_existing_lead_vs_cut_in():
    # Existing lead 40 m ahead, ego closes at 10 m/s -> TTC falls below 2 s.
    t, c = steady(30, lead=40.0)
    closing_profile(c, at(10), at(13), 40.0, 10.0)
    c["lead_dist"][at(13) :] = 10.0
    assert "low_ttc" in kinds(detect_trip("v", "t", t, c))

    # No lead, then a vehicle appears 12 m ahead, 8 m/s slower: a cut-in.
    t, c = steady(30)
    closing_profile(c, at(10), at(11), 12.0, 8.0)
    found = kinds(detect_trip("v", "t", t, c))
    assert "cut_in" in found and "low_ttc" not in found


def test_close_entry_is_not_tailgating_but_choosing_to_follow_close_is():
    # A lead enters 8 m ahead at 25 m/s (0.32 s headway) and stays there for 6 s: imposed.
    t, c = steady(30)
    c["lead_dist"][at(10) : at(16)] = 8.0
    c["lead_rel_speed"][at(10) : at(16)] = 0.0
    assert "tailgating" not in kinds(detect_trip("v", "t", t, c))

    # Same lead, but it entered at a normal 50 m gap and the driver closed to 10 m.
    t, c = steady(40)
    c["lead_dist"][at(5) :] = 50.0
    c["lead_rel_speed"][at(5) :] = 0.0
    c["lead_dist"][at(10) : at(20)] = np.linspace(50.0, 10.0, at(10))
    c["lead_rel_speed"][at(10) : at(20)] = -4.0
    c["lead_dist"][at(20) : at(30)] = 10.0
    c["lead_rel_speed"][at(20) : at(30)] = 0.0
    tail = [s for s in detect_trip("v", "t", t, c) if s.scenario_type == "tailgating"]
    assert len(tail) == 1 and tail[0].peak == pytest.approx(0.4)


def test_tailgating_after_recovery_window_is_reported():
    # Close entry, and the driver then stays at 0.4 s headway for 20 s.
    t, c = steady(40)
    c["lead_dist"][at(5) : at(30)] = 10.0
    c["lead_rel_speed"][at(5) : at(30)] = 0.0
    (s,) = [s for s in detect_trip("v", "t", t, c) if s.scenario_type == "tailgating"]
    assert s.start_ns == t[at(15)]  # recovery is 10 s


def test_sensor_dropout_and_gnss_outage():
    t, c = steady(60)
    keep = np.ones(len(t), bool)
    keep[at(20) : at(22)] = False  # 2 s with no samples
    c["lat"][at(40) : at(50)] = np.nan  # 10 s without a GNSS fix
    t = t[keep]
    c = {k: v[keep] for k, v in c.items()}
    found = detect_trip("v", "t", t, c)
    assert kinds(found) == ["sensor_dropout", "gnss_outage"]
    assert found[0].peak == pytest.approx(2.05)
    # a run across a sample gap is not a contiguous event
    t2, c2 = steady(20)
    c2["accel_long"][at(5) : at(5.5)] = -6.0
    keep = np.ones(len(t2), bool)
    keep[at(5.2) : at(5.4)] = False  # 0.25 s gap: the two pieces merge but straddle it
    found = detect_trip("v", "t", t2[keep], {k: v[keep] for k, v in c2.items()})
    assert "hard_brake" not in kinds(found)


def test_no_gnss_at_all_is_not_an_outage():
    t, c = steady(30)
    c["lat"][:] = np.nan
    assert detect_trip("v", "t", t, c) == []


def test_scenario_ids_are_deterministic():
    t, c = steady(20)
    c["accel_long"][at(10) : at(11)] = -6.0
    a = detect_trip("v", "t", t, c)[0].scenario_id
    b = detect_trip("v", "t", t, c)[0].scenario_id
    assert a == b and len(a) == 16
    assert detect_trip("w", "t", t, c)[0].scenario_id != a


def test_real_comma2k19_segment_runs_clean():
    tr = load_segment(SAMPLE_SEGMENT)
    found = detect_trip(tr.vehicle_id, tr.trip_id, tr.t_ns, tr.signals)
    # This minute is calm highway driving: nothing safety-relevant should fire.
    assert [s for s in found if s.scenario_type not in ("sensor_dropout", "gnss_outage")] == []


# --- rarity --------------------------------------------------------------------------
def scenario_rows(types: list[str], feats: np.ndarray, sev: list[float]) -> pa.Table:
    cols = {
        "scenario_id": [f"s{i}" for i in range(len(types))],
        "scenario_type": types,
        "severity": sev,
    }
    for j, f in enumerate(FEATURES):
        cols[f] = feats[:, j]
    return pa.table(cols)


def test_rarity_rare_type_and_outlier_rank_first():
    rng = np.random.default_rng(0)
    n = 40
    feats = rng.normal(0, 0.1, (n, len(FEATURES)))
    feats[5] += 5.0  # an unusual hard brake
    types = ["hard_brake"] * (n - 1) + ["evasive_swerve"]
    scored = score_scenarios(scenario_rows(types, feats, [0.5] * n)).to_pylist()
    by_id = {r["scenario_id"]: r for r in scored}
    assert by_id["s39"]["type_rarity"] == 1.0
    assert by_id["s5"]["novelty"] == 1.0
    assert [r["scenario_id"] for r in scored[:2]] == ["s39", "s5"]
    assert [r["rank"] for r in scored] == list(range(1, n + 1))
    for r in scored:
        assert 0 <= r["rarity"] <= 1 and 0 <= r["priority"] <= 1
        assert r["priority"] == pytest.approx(0.5 * r["severity"] + 0.5 * r["rarity"])


def test_knn_novelty_small_groups_are_novel():
    x = np.zeros((3, 2))
    assert knn_novelty(x, np.array(["a", "a", "a"]), k=5).tolist() == [1.0, 1.0, 1.0]


def test_score_empty():
    empty = scenario_rows([], np.empty((0, len(FEATURES))), [])
    out = score_scenarios(empty)
    assert out.num_rows == 0 and "priority" in out.column_names


# --- end to end: lake -> mine -> evaluate against ground truth ------------------------
def mined_lake(tmp_path, intensity: str, faults: FaultConfig):
    tracks, gt = generate_fleet(
        FleetConfig(n_vehicles=40, trip_minutes=15, seed=5, intensity=intensity)
    )
    topic = InMemoryTopic(4)
    topic.extend(inject_faults(interleave(tracks, 20), faults, FaultStats(), seed=1))
    run_local(topic, ProcessorConfig(lake_root=str(tmp_path), durable=False))
    write_derived(tmp_path, "ground_truth", ground_truth_table(gt))
    lake = Lake(tmp_path)
    mine_scenarios(lake)
    return lake, {r["scenario_type"]: r for r in evaluate(lake).to_pylist()}


def test_detectors_recover_injected_events_under_stream_faults(tmp_path):
    lake, ev = mined_lake(tmp_path, "clear", FaultConfig.realistic())
    assert sum(r["labels"] for r in ev.values()) > 50
    for kind, r in ev.items():
        assert r["tp_labels"] == r["labels"], f"{kind}: missed {r['labels'] - r['tp_labels']}"
        assert r["tp_detections"] == r["detections"], f"{kind}: false positives"
    ranks = lake.sql("SELECT rank FROM scenarios ORDER BY priority DESC").fetchall()
    assert [r[0] for r in ranks] == list(range(1, len(ranks) + 1))


def test_borderline_events_degrade_recall_not_precision(tmp_path):
    _, ev = mined_lake(tmp_path, "borderline", FaultConfig())
    labels = sum(r["labels"] for r in ev.values())
    tp = sum(r["tp_labels"] for r in ev.values())
    assert 0.3 < tp / labels < 0.9  # thresholds sit inside the borderline ranges
    for kind, r in ev.items():
        assert r["tp_detections"] == r["detections"], f"{kind}: false positives"
