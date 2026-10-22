from collections import Counter

from fleetscope.replay import fleet
from fleetscope.replay.fleet import (
    Emission,
    FaultConfig,
    FaultStats,
    inject_faults,
    interleave,
    paced,
)
from fleetscope.schema import decode_chunk


def test_interleave_is_event_time_ordered(small_fleet):
    tracks, _ = small_fleet
    starts = [c.t_ns[0] for c in interleave(tracks[:5], 20)]
    assert starts == sorted(starts)


def test_no_faults_is_identity(small_fleet):
    tracks, _ = small_fleet
    stats = FaultStats()
    out = list(inject_faults(interleave(tracks[:3], 20), FaultConfig(), stats))
    assert len(out) == stats.chunks == stats.messages
    assert sum(len(decode_chunk(e.value)) for e in out) == stats.samples


def test_fault_accounting_matches_emitted_stream(faulty_topic):
    topic, stats = faulty_topic
    assert len(topic) == stats.messages == stats.chunks + stats.duplicates + stats.corrupt
    keys = Counter()
    decodable = 0
    for part in topic.partitions:
        for r in part:
            try:
                c = decode_chunk(r.value)
            except Exception:
                continue
            decodable += 1
            keys[(c.vehicle_id, c.trip_id, c.seq)] += 1
    assert decodable == stats.chunks + stats.duplicates
    assert sum(v - 1 for v in keys.values()) == stats.duplicates
    assert stats.late > 0 and stats.reordered > 0 and stats.invalid_rows > 0


def test_same_vehicle_always_maps_to_same_partition(faulty_topic):
    topic, _ = faulty_topic
    owner = {}
    for p, part in enumerate(topic.partitions):
        for r in part:
            try:
                v = decode_chunk(r.value).vehicle_id
            except Exception:
                continue
            assert owner.setdefault(v, p) == p


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


def test_paced_follows_event_time_and_compresses_long_gaps(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(fleet, "time", clock)
    s = 10**9
    year = 365 * 24 * 3600 * s
    ts = [0, s // 2, 2 * s, 8 * year, 8 * year + s, 8 * year + s // 2]  # last one reordered
    wall = []
    for _ in paced((Emission(b"k", b"v", t) for t in ts), speedup=10.0, max_gap_s=1.0):
        wall.append(round(clock.now, 6))
    # gaps 0.5, 1.5->1.0, 8 years->1.0, 1.0, and a backwards step counts as 0
    assert wall == [0.0, 0.05, 0.15, 0.25, 0.35, 0.35]


def test_paced_unthrottled_is_passthrough(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(fleet, "time", clock)
    items = [Emission(b"k", b"v", t) for t in (0, 10**12)]
    assert list(paced(items, speedup=0)) == items
    assert clock.now == 0.0
