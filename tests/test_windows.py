import numpy as np
import pytest

from fleetscope.stream.windows import TumblingWindows

S = 1_000_000_000
T0 = 1_700_000_000 * S  # aligned to 10 s


def _chunk(start_s, n=20, speed=10.0, hz=20):
    t = T0 + int(start_s * S) + np.arange(n, dtype=np.int64) * (S // hz)
    sp = np.full(n, speed)
    return t, sp, np.zeros(n), np.zeros(n), np.full(n, np.nan)


def _feed(w, start_s, key=("v", "t"), **kw):
    return w.update(*key, *_chunk(start_s, **kw))


def test_window_emitted_after_watermark_with_exact_aggregates():
    w = TumblingWindows(window_s=10, allowed_lateness_s=5)
    out = []
    for s in range(0, 16):  # 16 one-second chunks -> watermark = 16 - 5 = 11 s
        late, closed = _feed(w, s, speed=float(s))
        assert late == 0
        out += closed
    assert len(out) == 1
    win = out[0]
    assert win.window_start_ns == T0 and win.samples == 200
    assert win.speed_mean == pytest.approx(4.5)
    assert win.speed_max == 9.0
    assert win.distance_m == pytest.approx(sum(range(10)) * 20 * 0.05)
    assert win.lead_dist_min is None


def test_late_rows_are_excluded_and_counted():
    w = TumblingWindows(window_s=10, allowed_lateness_s=5)
    for s in range(0, 30):
        _feed(w, s)
    late, closed = _feed(w, 3)  # window [0,10) closed long ago
    assert late == 20 and closed == []


def test_within_lateness_out_of_order_rows_are_included():
    w = TumblingWindows(window_s=10, allowed_lateness_s=5)
    for s in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 3]:
        late, _ = _feed(w, s)
        assert late == 0
    _, closed = _feed(w, 15)
    assert closed[0].samples == 200


def test_watermarks_are_per_stream():
    w = TumblingWindows(window_s=10, allowed_lateness_s=5, idle_timeout_s=1e6)
    for s in range(0, 100):
        _feed(w, s, key=("fast", "t"))
    late, _ = _feed(w, 0, key=("slow", "t"))
    assert late == 0  # another vehicle's progress must not make this one late


def test_idle_streams_close_and_tombstone_prevents_reopening():
    w = TumblingWindows(window_s=10, allowed_lateness_s=5, idle_timeout_s=30)
    _feed(w, 0, key=("idle", "t"))
    closed = []
    for s in range(0, 60):
        closed += _feed(w, s, key=("busy", "t"))[1]
    assert any(c.vehicle_id == "idle" for c in closed)
    late, again = _feed(w, 1, key=("idle", "t"))
    assert late == 20 and again == []


def test_snapshot_restore_mid_stream_is_equivalent():
    a = TumblingWindows(window_s=10, allowed_lateness_s=5)
    for s in range(0, 13):
        _feed(a, s)
    b = TumblingWindows(window_s=10, allowed_lateness_s=5)
    b.restore(a.snapshot())
    ra = [_feed(a, s)[1] for s in range(13, 40)]
    rb = [_feed(b, s)[1] for s in range(13, 40)]
    assert ra == rb


def test_drain_closes_everything():
    w = TumblingWindows(window_s=10, allowed_lateness_s=5)
    for s in range(0, 25):
        _feed(w, s)
    assert [r.window_start_ns for r in w.drain()] == [T0 + 10 * S, T0 + 20 * S]
    assert len(w) == 0
