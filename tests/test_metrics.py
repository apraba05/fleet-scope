import math

import numpy as np
import pytest

from fleetscope.lake.sql import Lake
from fleetscope.metrics.driving import event_rates, fleet_hourly, fleet_summary, trip_metrics
from fleetscope.replay.fleet import FaultConfig, FaultStats, inject_faults, interleave
from fleetscope.scenarios.pipeline import mine_scenarios
from fleetscope.schema import SAMPLE_HZ, SIGNALS, Track
from fleetscope.sources.synthetic import FleetConfig, generate_fleet
from fleetscope.stream.local import InMemoryTopic, run_local
from fleetscope.stream.processor import ProcessorConfig

T0 = 1_790_000_000 * 10**9
PERIOD = 10**9 // SAMPLE_HZ


def track(vid: str, seconds: float, v: float, drop: tuple[float, float] | None = None, **sig):
    n = int(seconds * SAMPLE_HZ)
    t = T0 + np.arange(n, dtype=np.int64) * PERIOD
    s = {k: np.zeros(n) for k in SIGNALS}
    s["speed"][:] = v
    s["lat"][:], s["lon"][:] = 37.0, -122.0
    s["lead_dist"][:] = np.nan
    s["lead_rel_speed"][:] = np.nan
    for k, v in sig.items():
        s[k] = v
    keep = np.ones(n, bool)
    if drop:
        keep[int(drop[0] * SAMPLE_HZ) : int(drop[1] * SAMPLE_HZ)] = False
    return Track(vid, f"{vid}-t0", "synthetic", t[keep], {k: v[keep] for k, v in s.items()})


def lake_of(tmp_path, tracks, faults=None):
    topic = InMemoryTopic(2)
    topic.extend(inject_faults(interleave(tracks, 20), faults or FaultConfig(), FaultStats()))
    run_local(topic, ProcessorConfig(lake_root=str(tmp_path), durable=False))
    return Lake(tmp_path)


def test_trip_metrics_exact_values(tmp_path):
    n = 60 * SAMPLE_HZ
    lead = np.where(np.arange(n) < n // 2, 5.0, np.nan)  # 0.5 s headway for the first half
    lat, lon = np.full(n, 37.0), np.full(n, -122.0)
    lat[: n // 4] = lon[: n // 4] = np.nan  # no GNSS fix for the first quarter
    tracks = [
        track("a", 60, 10.0, lead_dist=lead, lead_rel_speed=np.zeros(n), lat=lat, lon=lon),
        track("b", 60, 20.0, drop=(20.0, 30.0)),  # a 10 s sensor dropout
    ]
    lake = lake_of(tmp_path, tracks)
    rows = {r["vehicle_id"]: r for r in trip_metrics(lake).to_pylist()}
    a, b = rows["a"], rows["b"]
    assert a["samples"] == n
    assert a["distance_km"] == pytest.approx(10.0 * (n - 1) / SAMPLE_HZ / 1000)
    assert a["duration_s"] == pytest.approx(60.0)
    assert a["completeness"] == pytest.approx(1.0)
    assert a["lead_present_frac"] == pytest.approx(0.5)
    assert a["close_following_frac"] == pytest.approx(0.5)
    assert a["gnss_availability"] == pytest.approx(0.75)
    assert a["accel_long_abs_p95"] == 0.0 and a["jerk_abs_p95"] == 0.0
    # the dropout adds no distance and no observed time, and lowers completeness
    assert b["samples"] == 50 * SAMPLE_HZ
    assert b["distance_km"] == pytest.approx(20.0 * (50 * SAMPLE_HZ - 2) / SAMPLE_HZ / 1000)
    assert b["observed_s"] == pytest.approx((50 * SAMPLE_HZ - 2) / SAMPLE_HZ)
    assert b["completeness"] == pytest.approx(50 / 60)
    assert b["close_following_frac"] == 0.0
    # without a scenarios table events are zero, not missing
    assert a["safety_events"] == 0 and a["safety_events_per_100km"] == 0.0


def test_quarantined_rows_are_attributed_to_trips(tmp_path):
    n = 30 * SAMPLE_HZ
    speed = np.full(n, 15.0)
    speed[100:110] = -3.0  # out of range: quarantined
    lake = lake_of(tmp_path, [track("q", 30, 15.0, speed=speed)])
    (row,) = trip_metrics(lake).to_pylist()
    assert row["quarantined_rows"] == 10
    assert row["samples"] == n - 10
    s = fleet_summary(lake)
    assert s["quarantine_rate"] == pytest.approx(10 / n)


def test_fleet_summary_weights_by_distance_not_trip(tmp_path):
    tracks = [track("short", 10, 10.0), track("long", 100, 30.0)]
    lake = lake_of(tmp_path, tracks)
    trip_metrics(lake)
    s = fleet_summary(lake)
    km = (10.0 * (10 * SAMPLE_HZ - 1) + 30.0 * (100 * SAMPLE_HZ - 1)) / SAMPLE_HZ / 1000
    assert s["vehicles"] == 2 and s["trips"] == 2
    assert s["distance_km"] == pytest.approx(km)
    assert s["driving_hours"] == pytest.approx(
        (10 * SAMPLE_HZ - 1 + 100 * SAMPLE_HZ - 1) / 20 / 3600
    )


def test_streaming_aggregates_agree_with_batch_metrics(tmp_path):
    tracks, _ = generate_fleet(FleetConfig(n_vehicles=10, trip_minutes=10, seed=2))
    lake = lake_of(tmp_path, tracks, FaultConfig.realistic())
    mine_scenarios(lake)
    tm = trip_metrics(lake)
    batch_km = sum(tm.column("distance_km").to_pylist())
    stream_km = sum(fleet_hourly(lake).column("distance_km").to_pylist())
    # different estimators (rectangle per window vs gap-aware trapezoid) and late rows
    # are excluded from windows, so equal to within a fraction of a percent
    assert math.isclose(stream_km, batch_km, rel_tol=0.01)
    rates = event_rates(lake).to_pylist()
    total_events = lake.sql("SELECT count(*) FROM scenarios").fetchall()[0][0]
    assert sum(r["events"] for r in rates) == total_events
    s = fleet_summary(lake)
    assert s["safety_events"] == sum(tm.column("safety_events").to_pylist())
    assert 0.99 < s["completeness"] <= 1.0
