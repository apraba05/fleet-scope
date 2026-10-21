"""SQL access to the lake through DuckDB.

Every table is exposed as a view over its *committed* Parquet files, with Hive
partition columns (``date``, ``hour``) available for pruning::

    lake = Lake("lake")
    lake.sql("SELECT vehicle_id, max(speed) FROM telemetry WHERE date = '2026-10-01' GROUP BY 1")

Derived tables written by batch jobs (``scenarios``, ``trip_metrics``, ``selection``,
``ground_truth``) are single Parquet files replaced atomically.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from fleetscope.lake.catalog import STREAMING_TABLES, committed_files
from fleetscope.schema import DLQ_SCHEMA, QUARANTINE_SCHEMA, TELEMETRY_SCHEMA, WINDOW_SCHEMA
from fleetscope.stream.checkpoint import atomic_write

DERIVED_TABLES = ("scenarios", "trip_metrics", "selection", "ground_truth")
_EMPTY = {
    "telemetry": TELEMETRY_SCHEMA,
    "quarantine": QUARANTINE_SCHEMA,
    "dlq": DLQ_SCHEMA,
    "windows_10s": WINDOW_SCHEMA,
}


def write_derived(lake_root: str | Path, name: str, table: pa.Table) -> Path:
    path = Path(lake_root) / name / f"{name}.parquet"
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    atomic_write(path, sink.getvalue().to_pybytes(), durable=False)
    return path


class Lake:
    def __init__(self, root: str | Path, threads: int | None = None):
        self.root = Path(root)
        self.con = duckdb.connect()
        if threads:
            self.con.execute(f"SET threads = {int(threads)}")
        self.refresh()

    def refresh(self) -> None:
        """(Re)bind views to the currently committed snapshot of every table."""
        for table in (*STREAMING_TABLES, *DERIVED_TABLES):
            files = committed_files(self.root, table)
            if files:
                file_list = ", ".join(f"'{f}'" for f in files)
                hive = "true" if table in STREAMING_TABLES else "false"
                self.con.execute(
                    f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet("
                    f"[{file_list}], hive_partitioning = {hive}, union_by_name = true)"
                )
            elif table in _EMPTY:
                self.con.register(f"_{table}_empty", _EMPTY[table].empty_table())
                self.con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM _{table}_empty")
            else:
                self.con.execute(f"DROP VIEW IF EXISTS {table}")

    def has(self, table: str) -> bool:
        rows = self.con.sql(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", params=[table]
        ).fetchall()
        return bool(rows[0][0])

    def sql(self, query: str, params: list | None = None) -> duckdb.DuckDBPyRelation:
        return self.con.sql(query, params=params)

    def arrow(self, query: str, params: list | None = None) -> pa.Table:
        return self.con.sql(query, params=params).to_arrow_table()

    def reader(self, query: str, batch_rows: int = 1_000_000) -> pa.RecordBatchReader:
        return self.con.sql(query).to_arrow_reader(batch_rows)

    def df(self, query: str, params: list | None = None):
        return self.con.sql(query, params=params).df()
