"""Snapshot-consistent file listing for lake tables.

Streaming tables only expose *committed* files: a file named
``<topic>-p<P>-<start>-<end>.parquet`` is visible iff ``end < next_offset`` of that
partition's checkpoint. Files from an in-flight or crashed epoch are invisible, so a
reader never observes partial epochs or the duplicates a crash-replay would otherwise
produce. Derived (batch) tables are written atomically and listed as-is.
"""

from __future__ import annotations

from pathlib import Path

from fleetscope.stream.checkpoint import FILE_RE, committed_offsets

STREAMING_TABLES = ("telemetry", "quarantine", "dlq", "windows_10s")


def committed_files(lake_root: str | Path, table: str) -> list[str]:
    root = Path(lake_root) / table
    if not root.exists():
        return []
    offsets = committed_offsets(lake_root) if table in STREAMING_TABLES else {}
    out = []
    for f in sorted(root.rglob("*.parquet")):
        if f.name.startswith("."):
            continue
        m = FILE_RE.match(f.name)
        if m is None:
            out.append(str(f))
            continue
        nxt = offsets.get((m["topic"], int(m["partition"])))
        if nxt is not None and int(m["end"]) < nxt:
            out.append(str(f))
    return out
