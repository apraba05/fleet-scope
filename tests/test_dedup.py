import random

from hypothesis import given, settings
from hypothesis import strategies as st

from fleetscope.stream.dedup import SeqDeduplicator, Verdict

NEW, DUP, EXP = Verdict.NEW, Verdict.DUPLICATE, Verdict.EXPIRED


def test_in_order_stream_keeps_constant_state():
    d = SeqDeduplicator(window=8)
    assert all(d.admit("v", "t", s, s) is NEW for s in range(1000))
    st_ = d.snapshot()[0]
    assert (st_.contig, st_.above) == (999, [])
    assert d.admit("v", "t", 10, 10) is DUP


def test_reordered_and_duplicated_arrivals():
    d = SeqDeduplicator(window=8)
    verdicts = [d.admit("v", "t", s, s) for s in [0, 2, 1, 2, 3, 0, 5, 4, 4]]
    assert verdicts == [NEW, NEW, NEW, DUP, NEW, DUP, NEW, NEW, DUP]
    assert d.snapshot()[0].contig == 5


def test_streams_are_independent():
    d = SeqDeduplicator()
    assert d.admit("a", "t1", 0, 0) is NEW
    assert d.admit("a", "t2", 0, 0) is NEW
    assert d.admit("b", "t1", 0, 0) is NEW
    assert d.admit("a", "t1", 0, 0) is DUP


def test_permanent_gap_is_given_up_and_late_arrival_is_ambiguous():
    d = SeqDeduplicator(window=4)
    d.admit("v", "t", 0, 0)
    for s in range(2, 10):  # seq 1 never arrives
        assert d.admit("v", "t", s, s) is NEW
    state = d.snapshot()[0]
    assert len(state.above) <= 4  # memory bounded despite the hole
    assert d.admit("v", "t", 1, 1) is EXP  # cannot prove it is new -> dead-letter
    assert d.admit("v", "t", 9, 9) is DUP


def test_snapshot_restore_is_equivalent():
    d = SeqDeduplicator(window=16)
    for s in [0, 1, 3, 4, 7]:
        d.admit("v", "t", s, s)
    r = SeqDeduplicator.restore(16, d.snapshot())
    probe = [2, 3, 5, 6, 7, 8]
    assert [d.admit("v", "t", s, s) for s in probe] == [r.admit("v", "t", s, s) for s in probe]


def test_idle_streams_are_evicted_by_event_time():
    d = SeqDeduplicator()
    d.admit("old", "t", 0, ts_ns=0)
    d.admit("new", "t", 0, ts_ns=100)
    assert d.evict_idle(now_event_ns=100, ttl_ns=50) == 1
    assert len(d) == 1


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(1, 300),
    dup_frac=st.floats(0, 1),
    max_shift=st.integers(0, 20),
    seed=st.integers(0, 10_000),
)
def test_every_seq_admitted_exactly_once_under_bounded_disorder(n, dup_frac, max_shift, seed):
    """Property: with reordering bounded by the window, output == set of inputs, once each."""
    rng = random.Random(seed)
    arrivals = list(range(n)) + [rng.randrange(n) for _ in range(int(n * dup_frac))]
    # bounded displacement: sort by seq + jitter
    arrivals.sort(key=lambda s: s + rng.uniform(0, max_shift))
    d = SeqDeduplicator(window=64)
    admitted = [s for s in arrivals if d.admit("v", "t", s, s) is NEW]
    assert sorted(admitted) == list(range(n))
