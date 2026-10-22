"""In-memory broker + runner: the full processing path without a Kafka cluster.

Used by tests, benchmarks and ``fleetscope demo``. Partitioning (crc32 of the key) and
per-partition offsets mirror Kafka semantics closely enough to exercise ordering,
replay and exactly-once recovery logic.
"""

from __future__ import annotations

import time
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, field

from fleetscope.observability import Stats
from fleetscope.replay.fleet import Emission
from fleetscope.stream.processor import PartitionProcessor, ProcessorConfig, Record


class SimulatedCrash(RuntimeError):
    pass


@dataclass
class InMemoryTopic:
    n_partitions: int = 8
    partitions: list[list[Record]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.partitions:
            self.partitions = [[] for _ in range(self.n_partitions)]

    def partition_for(self, key: bytes) -> int:
        return zlib.crc32(key) % self.n_partitions

    def append(self, key: bytes, value: bytes, timestamp_ms: int | None = None) -> None:
        part = self.partitions[self.partition_for(key)]
        ts = timestamp_ms if timestamp_ms is not None else time.time_ns() // 1_000_000
        part.append(Record(len(part), value, ts))

    def extend(self, emissions: Iterable[Emission], timestamp_ms: int | None = None) -> None:
        for e in emissions:
            self.append(e.key, e.value, timestamp_ms)

    def __len__(self) -> int:
        return sum(len(p) for p in self.partitions)


def run_local(
    topic: InMemoryTopic,
    cfg: ProcessorConfig,
    batch_size: int = 2000,
    drain: bool = True,
    crash_after_batches: int | None = None,
    partitions: Iterable[int] | None = None,
) -> Stats:
    """Consume every partition from its checkpoint to the end. Returns merged stats.

    ``crash_after_batches`` aborts (without committing the open epoch) after that many
    batches of the first partition processed -- the in-flight epoch is lost exactly as
    it would be on a process crash.
    """
    total = Stats()
    for p in partitions if partitions is not None else range(topic.n_partitions):
        proc = PartitionProcessor(cfg, p)
        records = topic.partitions[p]
        start = proc.resume_offset or 0
        batches = [records[i : i + batch_size] for i in range(start, len(records), batch_size)]
        for i, batch in enumerate(batches):
            proc.process(batch)
            if crash_after_batches is not None and i + 1 >= crash_after_batches:
                total.update(proc.stats)
                raise SimulatedCrash(f"partition {p} after {i + 1} batches")
            if i < len(batches) - 1 and proc.should_checkpoint():
                proc.checkpoint()
        if drain:
            proc.finish()
        else:
            proc.checkpoint()
        total.update(proc.stats)
    return total
