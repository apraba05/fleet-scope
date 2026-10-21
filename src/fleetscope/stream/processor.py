"""Stateful, exactly-once processing of one Kafka partition.

Per consumed batch:

    decode -> chunk validation -> dedup (vehicle, trip, seq) -> columnar concat
           -> row validation -> {telemetry | quarantine} -> event-time windows

Outputs are buffered per *epoch* and committed together with operator state (see
``checkpoint.py``). The processor is transport-agnostic: it consumes ``Record`` objects,
so the same code runs behind the Kafka runner, the in-memory local runner, and tests.

Determinism: every decision depends only on the record sequence of the partition (row
validity uses the record's broker timestamp, not the wall clock; window closing is driven
by event time). Replaying a partition from a checkpoint therefore reproduces the same
outputs, which is what makes "delete uncommitted files + replay" exactly-once.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path

import msgspec
import numpy as np
import pyarrow as pa

from fleetscope import observability as obs
from fleetscope.schema import (
    DLQ_SCHEMA,
    SIGNALS,
    WINDOW_SCHEMA,
    TelemetryChunk,
    decode_chunk,
)
from fleetscope.stream.checkpoint import (
    Checkpoint,
    load_checkpoint,
    remove_uncommitted,
    save_checkpoint,
)
from fleetscope.stream.dedup import SeqDeduplicator, Verdict
from fleetscope.stream.sink import PartitionSink
from fleetscope.stream.validate import (
    ValidationConfig,
    reason_counts,
    reasons_to_strings,
    validate_header,
    validate_rows,
)
from fleetscope.stream.windows import TumblingWindows, WindowResult

_DICT = pa.dictionary(pa.int32(), pa.string())
_F32 = {
    "speed",
    "accel_long",
    "accel_lat",
    "yaw_rate",
    "steering_deg",
    "lead_dist",
    "lead_rel_speed",
}


@dataclass(frozen=True, slots=True)
class Record:
    offset: int
    value: bytes
    timestamp_ms: int  # broker / producer timestamp of the record


@dataclass
class ProcessorConfig:
    lake_root: str | Path
    topic: str = "telemetry.raw"
    window_s: int = 10
    allowed_lateness_s: float = 30.0
    idle_timeout_s: float = 120.0
    dedup_window: int = 4096
    dedup_ttl_s: float = 6 * 3600.0
    epoch_max_rows: int = 500_000
    epoch_max_s: float = 10.0
    durable: bool = True
    validation: ValidationConfig = field(default_factory=ValidationConfig)


class FencedError(RuntimeError):
    """Another processor committed this partition since we loaded it (we are a zombie)."""


@dataclass(frozen=True)
class DeadLetter:
    offset: int
    reason: str
    detail: str
    payload: bytes


class PartitionProcessor:
    def __init__(self, cfg: ProcessorConfig, partition: int):
        self.cfg = cfg
        self.partition = partition
        self.stats = obs.Stats()
        self.dedup = SeqDeduplicator(cfg.dedup_window)
        self.windows = TumblingWindows(cfg.window_s, cfg.allowed_lateness_s, cfg.idle_timeout_s)
        self.epochs = 0
        ckpt = load_checkpoint(cfg.lake_root, cfg.topic, partition)
        self.next_offset: int | None = None
        self._committed_offset: int | None = None
        if ckpt is not None:
            self._committed_offset = ckpt.next_offset
            self.next_offset = ckpt.next_offset
            self.dedup = SeqDeduplicator.restore(cfg.dedup_window, ckpt.dedup)
            self.windows.restore(ckpt.windows)
            self.epochs = ckpt.epochs
        removed = remove_uncommitted(cfg.lake_root, cfg.topic, partition, self.next_offset or 0)
        self.stats["recovery.orphan_files_removed"] += len(removed)
        self.sink = PartitionSink(cfg.lake_root, cfg.topic, partition, durable=cfg.durable)
        self._epoch_start: int | None = self.next_offset
        self._epoch_wall_start = time.monotonic()
        self._epoch_first_ts_ms: int | None = None
        self._next_evict_ns = 0
        self._dead_letters: list[DeadLetter] = []

    # ------------------------------------------------------------------------------------
    @property
    def resume_offset(self) -> int | None:
        """Where the transport should start reading this partition (None = no checkpoint)."""
        return self.next_offset

    @property
    def has_uncommitted(self) -> bool:
        return (
            self._epoch_start is not None
            and self.next_offset is not None
            and self._epoch_start < self.next_offset
        )

    def take_dead_letters(self) -> list[DeadLetter]:
        out, self._dead_letters = self._dead_letters, []
        return out

    # ------------------------------------------------------------------------------------
    def process(self, records: Sequence[Record]) -> None:
        if not records:
            return
        t0 = time.perf_counter()
        accepted: list[tuple[int, int, TelemetryChunk]] = []  # (offset, ts_ms, chunk)
        dlq: list[DeadLetter] = []
        n_bytes = 0
        n_dup = 0
        for r in records:
            if self.next_offset is not None and r.offset < self.next_offset:
                self.stats["records.skipped_committed"] += 1
                continue
            if self._epoch_start is None:
                self._epoch_start = r.offset
            if self._epoch_first_ts_ms is None:
                self._epoch_first_ts_ms = r.timestamp_ms
            self.next_offset = r.offset + 1
            n_bytes += len(r.value)
            try:
                chunk = decode_chunk(r.value)
            except (msgspec.DecodeError, msgspec.ValidationError) as e:
                dlq.append(DeadLetter(r.offset, "decode_error", str(e)[:200], r.value))
                continue
            reason = validate_header(chunk)
            if reason is not None:
                dlq.append(DeadLetter(r.offset, reason, "", r.value))
                continue
            verdict = self.dedup.admit(chunk.vehicle_id, chunk.trip_id, chunk.seq, chunk.t_ns[-1])
            if verdict is Verdict.DUPLICATE:
                n_dup += 1
                continue
            if verdict is Verdict.EXPIRED:
                dlq.append(
                    DeadLetter(r.offset, "dedup_window_expired", f"seq={chunk.seq}", r.value)
                )
                continue
            accepted.append((r.offset, r.timestamp_ms, chunk))

        self.stats["records.consumed"] += len(records)
        obs.RECORDS.inc(len(records))
        obs.BYTES.inc(n_bytes)
        self.stats["chunks.accepted"] += len(accepted)
        obs.CHUNKS.labels("accepted").inc(len(accepted))
        if n_dup:
            self.stats["chunks.duplicate"] += n_dup
            obs.CHUNKS.labels("duplicate").inc(n_dup)
        if dlq:
            self._dead_letter(dlq)
        if accepted:
            self._process_chunks(accepted)
        obs.INGEST_LATENCY.observe(max(0.0, time.time() - records[0].timestamp_ms / 1000.0))
        obs.BATCH_SECONDS.observe(time.perf_counter() - t0)
        obs.BUFFERED_ROWS.set(self.sink.buffered_rows)

    def _dead_letter(self, dlq: list[DeadLetter]) -> None:
        for d in dlq:
            self.stats[f"dlq.{d.reason}"] += 1
            obs.DLQ.labels(d.reason).inc()
        obs.CHUNKS.labels("dlq").inc(len(dlq))
        self._dead_letters.extend(dlq)
        self.sink.add(
            "dlq",
            pa.table(
                {
                    "kafka_partition": pa.array([self.partition] * len(dlq), pa.int32()),
                    "kafka_offset": pa.array([d.offset for d in dlq], pa.int64()),
                    "reason": pa.array([d.reason for d in dlq], pa.string()),
                    "detail": pa.array([d.detail for d in dlq], pa.string()),
                    "payload": pa.array([d.payload for d in dlq], pa.binary()),
                },
                schema=DLQ_SCHEMA,
            ),
        )

    def _process_chunks(self, accepted: list[tuple[int, int, TelemetryChunk]]) -> None:
        chunks = [c for _, _, c in accepted]
        k = len(chunks)
        lengths = np.fromiter((len(c.t_ns) for c in chunks), dtype=np.int64, count=k)
        total = int(lengths.sum())
        bounds = np.zeros(k + 1, dtype=np.int64)
        np.cumsum(lengths, out=bounds[1:])
        t_ns = np.fromiter(chain.from_iterable(c.t_ns for c in chunks), dtype=np.int64, count=total)
        cols = {
            s: np.fromiter(
                chain.from_iterable(getattr(c, s) for c in chunks), dtype=np.float64, count=total
            )
            for s in SIGNALS
        }
        chunk_starts = np.zeros(total, dtype=bool)
        chunk_starts[bounds[:-1]] = True
        row_chunk = np.repeat(np.arange(k, dtype=np.int32), lengths)
        ref_ns = np.repeat(
            np.fromiter((ts for _, ts, _ in accepted), np.int64, k) * 1_000_000, lengths
        )

        mask = validate_rows(t_ns, cols, chunk_starts, ref_ns, self.cfg.validation)
        valid = mask == 0
        n_valid = int(valid.sum())
        n_invalid = total - n_valid
        self.stats["rows.valid"] += n_valid
        obs.ROWS.labels("valid").inc(n_valid)

        offsets = np.fromiter((o for o, _, _ in accepted), np.int64, k)
        base = {
            "vehicle_id": pa.DictionaryArray.from_arrays(row_chunk, [c.vehicle_id for c in chunks]),
            "trip_id": pa.DictionaryArray.from_arrays(row_chunk, [c.trip_id for c in chunks]),
            "seq": np.repeat(np.fromiter((c.seq for c in chunks), np.int64, k), lengths),
            "ts": pa.array(t_ns, pa.timestamp("ns", tz="UTC")),
            **{s: (cols[s].astype(np.float32) if s in _F32 else cols[s]) for s in SIGNALS},
            "source": pa.DictionaryArray.from_arrays(row_chunk, [c.source for c in chunks]),
            "kafka_partition": np.full(total, self.partition, dtype=np.int32),
            "kafka_offset": np.repeat(offsets, lengths),
        }
        table = pa.table(base)
        if n_invalid == 0:
            self.sink.add("telemetry", table)
        else:
            self.stats["rows.invalid"] += n_invalid
            obs.ROWS.labels("invalid").inc(n_invalid)
            for rule, n in reason_counts(mask).items():
                self.stats[f"invalid.{rule}"] += n
                obs.INVALID.labels(rule).inc(n)
            self.sink.add("telemetry", table.filter(pa.array(valid)))
            bad = table.filter(pa.array(~valid))
            self.sink.add(
                "quarantine",
                bad.append_column("reasons", pa.array(reasons_to_strings(mask[~valid]))),
            )

        # Event-time windows, chunk by chunk in partition order (deterministic).
        closed: list[WindowResult] = []
        late = 0
        for i, c in enumerate(chunks):
            a, b = int(bounds[i]), int(bounds[i + 1])
            sel = (
                slice(a, b)
                if n_invalid == 0 or valid[a:b].all()
                else np.flatnonzero(valid[a:b]) + a
            )
            n_late, out = self.windows.update(
                c.vehicle_id,
                c.trip_id,
                t_ns[sel],
                cols["speed"][sel],
                cols["accel_long"][sel],
                cols["accel_lat"][sel],
                cols["lead_dist"][sel],
            )
            late += n_late
            closed.extend(out)
        if late:
            self.stats["rows.late"] += late
            obs.LATE_ROWS.inc(late)
        self._emit_windows(closed)

        frontier = self.windows._frontier_ns
        if frontier >= self._next_evict_ns:
            self.stats["dedup.evicted_streams"] += self.dedup.evict_idle(
                frontier, int(self.cfg.dedup_ttl_s * 1e9)
            )
            self._next_evict_ns = frontier + 60_000_000_000
        obs.OPEN_STREAMS.set(len(self.dedup))
        obs.OPEN_WINDOW_STREAMS.set(len(self.windows))

    def _emit_windows(self, closed: list[WindowResult]) -> None:
        if not closed:
            return
        self.stats["windows.emitted"] += len(closed)
        obs.WINDOWS.inc(len(closed))
        self.sink.add(
            "windows_10s",
            pa.table(
                {
                    "vehicle_id": [w.vehicle_id for w in closed],
                    "trip_id": [w.trip_id for w in closed],
                    "window_start": pa.array(
                        [w.window_start_ns for w in closed], pa.timestamp("ns", tz="UTC")
                    ),
                    "window_s": pa.array([w.window_s for w in closed], pa.int32()),
                    "samples": [w.samples for w in closed],
                    "distance_m": [w.distance_m for w in closed],
                    "speed_mean": [w.speed_mean for w in closed],
                    "speed_max": [w.speed_max for w in closed],
                    "accel_long_min": [w.accel_long_min for w in closed],
                    "accel_long_max": [w.accel_long_max for w in closed],
                    "accel_lat_abs_max": [w.accel_lat_abs_max for w in closed],
                    "lead_dist_min": pa.array([w.lead_dist_min for w in closed], pa.float64()),
                },
                schema=WINDOW_SCHEMA,
            ),
        )

    # ------------------------------------------------------------------------------------
    def should_checkpoint(self) -> bool:
        if not self.has_uncommitted:
            return False
        return (
            self.sink.buffered_rows >= self.cfg.epoch_max_rows
            or time.monotonic() - self._epoch_wall_start >= self.cfg.epoch_max_s
        )

    def drain_windows(self) -> None:
        """Close all open windows into the current epoch (bounded inputs only)."""
        self._emit_windows(self.windows.drain())

    def checkpoint(self) -> list[tuple[Path, int, int]]:
        """Commit the current epoch. No-op if nothing was consumed since the last commit."""
        if not self.has_uncommitted:
            return []
        assert self._epoch_start is not None and self.next_offset is not None
        t0 = time.perf_counter()
        # Compare-and-swap guard: refuse to commit over a checkpoint we did not write.
        # (Narrow race remains between check and rename; on object stores this would be
        # a conditional PUT.)
        on_disk = load_checkpoint(self.cfg.lake_root, self.cfg.topic, self.partition)
        if (on_disk.next_offset if on_disk else None) != self._committed_offset:
            self.sink.discard()
            raise FencedError(
                f"partition {self.partition}: checkpoint moved; another owner is active"
            )
        first_ms = self._epoch_first_ts_ms or 0
        ingest_date = time.strftime("%Y-%m-%d", time.gmtime(first_ms / 1000))
        files = self.sink.flush(self._epoch_start, self.next_offset - 1, ingest_date)
        self.epochs += 1
        save_checkpoint(
            self.cfg.lake_root,
            Checkpoint(
                topic=self.cfg.topic,
                partition=self.partition,
                next_offset=self.next_offset,
                dedup=self.dedup.snapshot(),
                windows=self.windows.snapshot(),
                epochs=self.epochs,
            ),
            durable=self.cfg.durable,
        )
        self._committed_offset = self.next_offset
        for path, _rows, nbytes in files:
            table = path.relative_to(self.cfg.lake_root).parts[0]
            self.stats[f"files.{table}"] += 1
            self.stats[f"file_bytes.{table}"] += nbytes
            obs.FILES.labels(table).inc()
            obs.FILE_BYTES.labels(table).inc(nbytes)
        self.stats["checkpoints"] += 1
        obs.CHECKPOINTS.inc()
        obs.CHECKPOINT_SECONDS.observe(time.perf_counter() - t0)
        if first_ms:
            obs.COMMIT_LATENCY.observe(max(0.0, time.time() - first_ms / 1000.0))
        self._epoch_start = self.next_offset
        self._epoch_wall_start = time.monotonic()
        self._epoch_first_ts_ms = None
        obs.BUFFERED_ROWS.set(0)
        return files
