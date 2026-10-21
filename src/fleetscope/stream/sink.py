"""Hive-partitioned Parquet writer for one Kafka partition.

Layout: ``<lake>/<table>/date=YYYY-MM-DD/hour=HH/<topic>-p<P>-<start>-<end>.parquet``

* Partitioned by **event** date/hour: time-range queries prune directories, and late
  data lands in the partition it belongs to.
* Rows are sorted by (vehicle_id, ts) inside each file, so Parquet row-group min/max
  statistics let DuckDB skip row groups on vehicle or time predicates.
* zstd compression, dictionary encoding for low-cardinality strings.
* One epoch writes at most one file per (table, date, hour); file size is governed by
  the epoch length (rows / seconds), the main small-files control knob.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fleetscope.schema import DLQ_SCHEMA, QUARANTINE_SCHEMA, TELEMETRY_SCHEMA, WINDOW_SCHEMA
from fleetscope.stream.checkpoint import atomic_write, data_file_name

PARTITION_TS_COLUMN = {"telemetry": "ts", "quarantine": "ts", "windows_10s": "window_start"}
TABLE_SCHEMAS = {
    "telemetry": TELEMETRY_SCHEMA,
    "quarantine": QUARANTINE_SCHEMA,
    "dlq": DLQ_SCHEMA,
    "windows_10s": WINDOW_SCHEMA,
}


class PartitionSink:
    def __init__(self, lake_root: str | Path, topic: str, partition: int, durable: bool = True):
        self.root = Path(lake_root)
        self.topic = topic
        self.partition = partition
        self.durable = durable
        self._buffers: dict[str, list[pa.Table]] = defaultdict(list)
        self.buffered_rows = 0

    def add(self, table: str, data: pa.Table) -> None:
        if data.num_rows:
            self._buffers[table].append(data)
            self.buffered_rows += data.num_rows

    def flush(
        self, start_offset: int, end_offset: int, ingest_date: str
    ) -> list[tuple[Path, int, int]]:
        """Write buffered tables as files covering offsets [start, end].

        Returns ``(path, rows, bytes)`` per file written.
        """
        written: list[tuple[Path, int, int]] = []
        name = data_file_name(self.topic, self.partition, start_offset, end_offset)
        for table, parts in self._buffers.items():
            # The hot path builds dictionary-encoded string columns (cheap); materialise the
            # declared storage schema once per epoch.
            data = pa.concat_tables(parts).cast(TABLE_SCHEMAS[table])
            ts_col = PARTITION_TS_COLUMN.get(table)
            if ts_col is None:  # e.g. dlq: no event time, partition by ingest date
                written.append(
                    self._write(self.root / table / f"date={ingest_date}" / name, data, sort=False)
                )
                continue
            ts = data.column(ts_col)
            hour_bucket = pc.cast(ts, pa.int64()).to_numpy() // 3_600_000_000_000
            for bucket in np.unique(hour_bucket).tolist():
                part = data.filter(pa.array(hour_bucket == bucket))
                stamp = np.datetime64(int(bucket) * 3600, "s").astype("datetime64[h]").item()
                path = self.root / table / f"date={stamp:%Y-%m-%d}" / f"hour={stamp:%H}" / name
                written.append(self._write(path, part, sort=True))
        self._buffers.clear()
        self.buffered_rows = 0
        return written

    def _write(self, path: Path, data: pa.Table, sort: bool) -> tuple[Path, int, int]:
        if sort and "vehicle_id" in data.column_names:
            ts_key = "ts" if "ts" in data.column_names else "window_start"
            data = data.sort_by([("vehicle_id", "ascending"), (ts_key, "ascending")])
        sink = pa.BufferOutputStream()
        pq.write_table(
            data,
            sink,
            compression="zstd",
            row_group_size=128 * 1024,
            use_dictionary=["vehicle_id", "trip_id", "source", "reasons", "reason"],
            write_statistics=True,
        )
        buf = sink.getvalue().to_pybytes()
        atomic_write(path, buf, durable=self.durable)
        return path, data.num_rows, len(buf)

    def discard(self) -> None:
        self._buffers.clear()
        self.buffered_rows = 0
