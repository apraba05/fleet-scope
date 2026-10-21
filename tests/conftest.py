from __future__ import annotations

from pathlib import Path

import pytest

from fleetscope.replay.fleet import FaultConfig, FaultStats, inject_faults, interleave
from fleetscope.sources.synthetic import FleetConfig, generate_fleet
from fleetscope.stream.local import InMemoryTopic

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "data" / "comma2k19_sample"
SAMPLE_SEGMENT = SAMPLE_ROOT / "b0c9d2329ad1606b|2018-08-02--08-34-47" / "40"


@pytest.fixture(scope="session")
def small_fleet():
    return generate_fleet(FleetConfig(n_vehicles=12, trip_minutes=6, seed=11))


@pytest.fixture()
def faulty_topic(small_fleet):
    tracks, _ = small_fleet
    stats = FaultStats()
    cfg = FaultConfig(
        duplicate_rate=0.03,
        reorder_rate=0.03,
        late_rate=0.01,
        late_delay_chunks=60,
        corrupt_rate=0.005,
        invalid_value_rate=0.01,
    )
    topic = InMemoryTopic(4)
    topic.extend(
        inject_faults(interleave(tracks, 20), cfg, stats, seed=3), timestamp_ms=1_800_000_000_000
    )
    return topic, stats
