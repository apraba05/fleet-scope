"""Fleet replay: interleave many vehicle tracks in event-time order and inject faults.

The replayer is the stand-in for the vehicle uplink. Faults model what actually happens
on a cellular uplink with client retries:

* duplicates   - a retry after a lost ack re-sends an identical payload
* reordering   - chunks of one vehicle overtake each other (multi-connection upload)
* late data    - a vehicle buffers offline and uploads minutes later
* corruption   - truncated / garbage payloads
* invalid data - well-formed chunks carrying physically impossible samples

Every injected fault is counted in ``FaultStats`` so tests can assert the processor's
counters match exactly.
"""

from __future__ import annotations

import heapq
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import msgspec

from fleetscope.schema import TelemetryChunk, Track, encode_chunk


@dataclass
class FaultConfig:
    duplicate_rate: float = 0.0
    reorder_rate: float = 0.0
    reorder_max_delay: int = 3  # chunks of the same vehicle
    late_rate: float = 0.0
    late_delay_chunks: int = 120  # chunks of the same vehicle (~2 min at 1 s chunks)
    corrupt_rate: float = 0.0  # extra garbage messages per chunk
    invalid_value_rate: float = 0.0  # chunks with one impossible sample

    @classmethod
    def realistic(cls) -> FaultConfig:
        return cls(
            duplicate_rate=0.01,
            reorder_rate=0.02,
            late_rate=0.002,
            corrupt_rate=0.001,
            invalid_value_rate=0.005,
        )


@dataclass
class FaultStats:
    chunks: int = 0
    samples: int = 0
    duplicates: int = 0
    reordered: int = 0
    late: int = 0
    corrupt: int = 0
    invalid_rows: int = 0
    messages: int = 0
    per_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def expected_clean_rows(self) -> int:
        return self.samples - self.invalid_rows


@dataclass(frozen=True)
class Emission:
    key: bytes
    value: bytes
    event_ts_ns: int


def interleave(tracks: Iterable[Track], samples_per_chunk: int) -> Iterator[TelemetryChunk]:
    """Merge chunks of all tracks by chunk start time (what a live fleet would send)."""
    iters = [iter(t.chunks(samples_per_chunk)) for t in tracks]
    heap: list[tuple[int, int, TelemetryChunk]] = []
    for i, it in enumerate(iters):
        first = next(it, None)
        if first is not None:
            heap.append((first.t_ns[0], i, first))
    heapq.heapify(heap)
    while heap:
        _, i, chunk = heapq.heappop(heap)
        yield chunk
        nxt = next(iters[i], None)
        if nxt is not None:
            heapq.heappush(heap, (nxt.t_ns[0], i, nxt))


_INVALID_MUTATIONS = (
    ("speed", -5.0, "speed_out_of_range"),
    ("speed", float("nan"), "required_not_finite"),
    ("accel_long", 60.0, "accel_long_out_of_range"),
    ("yaw_rate", 9.0, "yaw_rate_out_of_range"),
    ("lat", 123.0, "lat_out_of_range"),
)


def inject_faults(
    chunks: Iterable[TelemetryChunk], cfg: FaultConfig, stats: FaultStats, seed: int = 0
) -> Iterator[Emission]:
    rng = random.Random(seed)
    # Per-vehicle hold-back queues: list of [remaining_chunks, emission]
    held: dict[str, list[list]] = defaultdict(list)
    dup_pending: list[list] = []  # [remaining_messages, emission]

    def emit(e: Emission) -> Iterator[Emission]:
        stats.messages += 1
        yield e
        # duplicates are re-sent after a few more messages (retry after ack timeout)
        for d in dup_pending:
            d[0] -= 1
        while dup_pending and dup_pending[0][0] <= 0:
            stats.messages += 1
            yield dup_pending.pop(0)[1]

    for chunk in chunks:
        stats.chunks += 1
        stats.samples += len(chunk)
        if cfg.invalid_value_rate and rng.random() < cfg.invalid_value_rate:
            col, bad, reason = rng.choice(_INVALID_MUTATIONS)
            i = rng.randrange(len(chunk))
            values = list(getattr(chunk, col))
            values[i] = bad
            chunk = msgspec.structs.replace(chunk, **{col: values})
            stats.invalid_rows += 1
            stats.per_reason[reason] += 1
        key = chunk.vehicle_id.encode()
        em = Emission(key, encode_chunk(chunk), chunk.t_ns[0])

        # release held chunks of this vehicle whose delay elapsed
        q = held[chunk.vehicle_id]
        for h in q:
            h[0] -= 1
        ready = [h for h in q if h[0] <= 0]
        held[chunk.vehicle_id] = [h for h in q if h[0] > 0]

        r = rng.random()
        if cfg.late_rate and r < cfg.late_rate:
            stats.late += 1
            held[chunk.vehicle_id].append([cfg.late_delay_chunks, em])
        elif cfg.reorder_rate and r < cfg.late_rate + cfg.reorder_rate:
            stats.reordered += 1
            held[chunk.vehicle_id].append([rng.randint(1, cfg.reorder_max_delay), em])
        else:
            yield from emit(em)
            # Only chunks sent immediately get retried; a retry of a held chunk could
            # overtake the original and blur the late/reordered accounting.
            if cfg.duplicate_rate and rng.random() < cfg.duplicate_rate:
                stats.duplicates += 1
                dup_pending.append([rng.randint(1, 5), em])
        for h in ready:
            yield from emit(h[1])
        if cfg.corrupt_rate and rng.random() < cfg.corrupt_rate:
            stats.corrupt += 1
            garbage = em.value[: rng.randrange(1, max(2, len(em.value) // 2))]
            yield from emit(Emission(key, garbage, em.event_ts_ns))

    # drain: vehicles that stopped sending still flush their buffers eventually
    for q in held.values():
        for _, e in sorted(q, key=lambda h: h[0]):
            yield from emit(e)
    for _, e in dup_pending:
        stats.messages += 1
        yield e


def paced(
    emissions: Iterable[Emission], speedup: float, max_gap_s: float = 1.0
) -> Iterator[Emission]:
    """Throttle so event time advances ``speedup`` x faster than wall time (0 = unthrottled).

    Event-time gaps longer than ``max_gap_s`` (between sources recorded years apart, or
    between trips) are compressed to ``max_gap_s`` so the replay never idles on them.
    """
    if speedup <= 0:
        yield from emissions
        return
    t_wall0 = time.monotonic()
    prev: int | None = None
    elapsed_s = 0.0  # compressed event time since the first emission
    max_gap_ns = int(max_gap_s * 1e9)
    for e in emissions:
        if prev is not None:
            elapsed_s += min(max(e.event_ts_ns - prev, 0), max_gap_ns) / 1e9
        prev = max(prev or e.event_ts_ns, e.event_ts_ns)
        lag = elapsed_s / speedup - (time.monotonic() - t_wall0)
        if lag > 0.002:
            time.sleep(lag)
        yield e
