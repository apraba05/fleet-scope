"""Exactly-once chunk admission via per-stream sequence tracking.

Each stream ``(vehicle_id, trip_id)`` carries producer sequence numbers ``seq``. Instead
of remembering every seen seq, a stream stores:

* ``contig``  - every seq <= contig has been seen (normal, in-order case: O(1) state)
* ``above``   - seen seqs > contig (only non-empty while there are gaps / reordering)
* ``expired`` - seqs <= expired are no longer tracked exactly

If a gap stays open for more than ``window`` seqs (a chunk that never arrives), the
tracker gives up on that gap: ``contig`` jumps forward and ``expired`` is raised to
match. A later arrival with seq <= expired is *ambiguous* (could be the missing chunk or
a duplicate) and is returned as EXPIRED so the caller can dead-letter it rather than
silently dropping or duplicating data. Memory per stream is therefore bounded by
``window`` regardless of input.

Idle streams (finished trips) are evicted by event-time TTL.
"""

from __future__ import annotations

from enum import Enum

import msgspec


class Verdict(Enum):
    NEW = "new"
    DUPLICATE = "duplicate"
    EXPIRED = "expired"


class _Stream:
    __slots__ = ("above", "contig", "expired", "last_ts_ns")

    def __init__(
        self,
        contig: int = -1,
        expired: int = -1,
        above: set[int] | None = None,
        last_ts_ns: int = 0,
    ):
        self.contig = contig
        self.expired = expired
        self.above = above if above is not None else set()
        self.last_ts_ns = last_ts_ns


class StreamDedupState(msgspec.Struct, array_like=True):
    vehicle_id: str
    trip_id: str
    contig: int
    expired: int
    above: list[int]
    last_ts_ns: int


class SeqDeduplicator:
    def __init__(self, window: int = 4096):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._streams: dict[tuple[str, str], _Stream] = {}

    def __len__(self) -> int:
        return len(self._streams)

    def admit(self, vehicle_id: str, trip_id: str, seq: int, ts_ns: int) -> Verdict:
        key = (vehicle_id, trip_id)
        st = self._streams.get(key)
        if st is None:
            st = self._streams[key] = _Stream()
        if seq <= st.expired:
            return Verdict.EXPIRED
        if seq <= st.contig or seq in st.above:
            return Verdict.DUPLICATE
        if ts_ns > st.last_ts_ns:
            st.last_ts_ns = ts_ns
        if seq == st.contig + 1:
            st.contig = seq
            above = st.above
            while st.contig + 1 in above:
                st.contig += 1
                above.discard(st.contig)
        else:
            st.above.add(seq)
            top = max(seq, st.contig)
            if top - st.contig > self.window:
                self._give_up_gap(st, top - self.window)
        return Verdict.NEW

    @staticmethod
    def _give_up_gap(st: _Stream, new_floor: int) -> None:
        st.contig = new_floor
        st.expired = new_floor
        st.above = {s for s in st.above if s > new_floor}
        while st.contig + 1 in st.above:
            st.contig += 1
            st.above.discard(st.contig)

    def evict_idle(self, now_event_ns: int, ttl_ns: int) -> int:
        stale = [k for k, st in self._streams.items() if st.last_ts_ns < now_event_ns - ttl_ns]
        for k in stale:
            del self._streams[k]
        return len(stale)

    # --- checkpointing ------------------------------------------------------------------
    def snapshot(self) -> list[StreamDedupState]:
        return [
            StreamDedupState(v, t, st.contig, st.expired, sorted(st.above), st.last_ts_ns)
            for (v, t), st in self._streams.items()
        ]

    @classmethod
    def restore(cls, window: int, states: list[StreamDedupState]) -> SeqDeduplicator:
        d = cls(window)
        for s in states:
            d._streams[(s.vehicle_id, s.trip_id)] = _Stream(
                s.contig, s.expired, set(s.above), s.last_ts_ns
            )
        return d
