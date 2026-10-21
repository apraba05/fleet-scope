"""Pipeline observability: Prometheus metrics + periodic health snapshots.

Every counter is mirrored into a plain per-processor ``Counter`` (``Stats``) so tests and
the dashboard can read exact values without scraping Prometheus.

Metric naming follows Prometheus conventions (``_total`` counters, ``_seconds``
histograms, base units). Label cardinality is bounded: reasons and statuses are small
fixed enums; ``partition`` only appears on the lag gauge.
"""

from __future__ import annotations

import json
import time
from collections import Counter as _Counter
from pathlib import Path

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

REGISTRY = CollectorRegistry(auto_describe=True)

RECORDS = Counter("fleetscope_records_consumed_total", "Kafka records consumed", registry=REGISTRY)
BYTES = Counter(
    "fleetscope_bytes_consumed_total", "Kafka payload bytes consumed", registry=REGISTRY
)
CHUNKS = Counter(
    "fleetscope_chunks_total", "Chunks by admission outcome", ["outcome"], registry=REGISTRY
)  # accepted | duplicate | dlq
ROWS = Counter(
    "fleetscope_rows_total", "Sample rows by validation status", ["status"], registry=REGISTRY
)
INVALID = Counter(
    "fleetscope_invalid_rows_total", "Quarantined rows by failed rule", ["rule"], registry=REGISTRY
)
DLQ = Counter(
    "fleetscope_dlq_total", "Dead-lettered records by reason", ["reason"], registry=REGISTRY
)
LATE_ROWS = Counter(
    "fleetscope_late_rows_total", "Rows arriving after their window was emitted", registry=REGISTRY
)
WINDOWS = Counter(
    "fleetscope_windows_emitted_total", "Closed event-time windows", registry=REGISTRY
)
FILES = Counter(
    "fleetscope_files_written_total", "Parquet files committed", ["table"], registry=REGISTRY
)
FILE_BYTES = Counter(
    "fleetscope_file_bytes_written_total", "Parquet bytes committed", ["table"], registry=REGISTRY
)
CHECKPOINTS = Counter("fleetscope_checkpoints_total", "Committed epochs", registry=REGISTRY)

BATCH_SECONDS = Histogram(
    "fleetscope_batch_process_seconds",
    "Processing time per consumed batch",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
    registry=REGISTRY,
)
CHECKPOINT_SECONDS = Histogram(
    "fleetscope_checkpoint_seconds",
    "Epoch commit time (flush + checkpoint)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
    registry=REGISTRY,
)
INGEST_LATENCY = Histogram(
    "fleetscope_ingest_latency_seconds",
    "Broker append time -> processed (excludes time to lake commit)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=REGISTRY,
)
COMMIT_LATENCY = Histogram(
    "fleetscope_commit_latency_seconds",
    "Broker append time of an epoch's oldest record -> epoch committed to the lake",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60),
    registry=REGISTRY,
)
CONSUMER_LAG = Gauge(
    "fleetscope_consumer_lag_records",
    "High watermark - next offset",
    ["partition"],
    registry=REGISTRY,
)
OPEN_STREAMS = Gauge("fleetscope_open_streams", "Streams with dedup state", registry=REGISTRY)
OPEN_WINDOW_STREAMS = Gauge(
    "fleetscope_open_window_streams", "Streams with open windows", registry=REGISTRY
)
BUFFERED_ROWS = Gauge(
    "fleetscope_buffered_rows", "Rows buffered in the current epoch", registry=REGISTRY
)


class Stats(_Counter):
    """Exact per-processor counters (keys like ``rows.valid`` / ``dlq.decode_error``)."""


def serve_metrics(port: int) -> None:
    start_http_server(port, registry=REGISTRY)


class HealthReporter:
    """Append a JSON line with cumulative stats every ``interval_s`` (read by the dashboard)."""

    def __init__(self, lake_root: str | Path, interval_s: float = 5.0, name: str = "processor"):
        self.path = Path(lake_root) / "_ops" / f"health-{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_s = interval_s
        self._last = 0.0

    def maybe_report(
        self, stats: _Counter, extra: dict[str, float] | None = None, force: bool = False
    ) -> None:
        now = time.time()
        if not force and now - self._last < self.interval_s:
            return
        self._last = now
        row = {"ts": now, **{k: int(v) for k, v in stats.items()}, **(extra or {})}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
