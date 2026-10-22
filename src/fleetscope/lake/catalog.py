"""Snapshot-consistent file listing for lake tables.

Streaming tables only expose *committed* files: a file named
``<topic>-p<P>-<start>-<end>.parquet`` is visible iff ``end < next_offset`` of that
partition's checkpoint, and a drain file ``<topic>-p<P>-<at>-drain.parquet`` iff
``at <= drained_at``. Files from an in-flight or crashed epoch are invisible, so a
reader never observes partial epochs or the duplicates a crash-replay would otherwise
produce. Derived (batch) tables are written atomically and listed as-is.
"""

from __future__ import annotations

from pathlib import Path

from fleetscope.stream.checkpoint import DRAIN_RE, FILE_RE, committed_positions

STREAMING_TABLES = ("telemetry", "quarantine", "dlq", "windows_10s")


def committed_files(lake_root: str | Path, table: str) -> list[str]:
    root = Path(lake_root) / table
    if not root.exists():
        return []
    positions = committed_positions(lake_root) if table in STREAMING_TABLES else {}
    out = []
    for f in sorted(root.rglob("*.parquet")):
        if not f.name.startswith("."):
            name = f.name
            if FILE_RE.match(name) is None and DRAIN_RE.match(name) is None:
                out.append(str(f))  # derived table file
            elif _is_committed(name, positions):
                out.append(str(f))
    return out


def _is_committed(name: str, positions: dict[tuple[str, int], tuple[int, int | None]]) -> bool:
    if m := FILE_RE.match(name):
        pos = positions.get((m["topic"], int(m["partition"])))
        return pos is not None and int(m["end"]) < pos[0]
    if m := DRAIN_RE.match(name):
        pos = positions.get((m["topic"], int(m["partition"])))
        return pos is not None and pos[1] is not None and int(m["at"]) <= pos[1]
    return False
