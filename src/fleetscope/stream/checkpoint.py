"""Per-partition checkpoints: the commit point for exactly-once output to the lake.

A checkpoint records, for one (topic, partition):

* ``next_offset``  - first offset NOT yet reflected in committed lake files
* operator state   - dedup and window state as of ``next_offset``

Commit protocol for one epoch [epoch_start, next_offset):

1. write all epoch files under their final, deterministic names
   ``<topic>-p<partition>-<epoch_start>-<epoch_end>.parquet`` (each via tmp + rename)
2. atomically replace the checkpoint (tmp + fsync + rename)   <- commit point

A data file is *committed* iff its end offset < the partition's ``next_offset``.

End of a bounded input is a *drain epoch*: it consumes no records but closes every open
window. Its files are named ``<topic>-p<partition>-<next_offset>-drain.parquet`` and the
checkpoint records ``drained_at = next_offset``; a drain file is committed iff its offset
<= ``drained_at``. Draining twice at one offset is impossible (the second finds no open
windows), so the name is unique.
Readers only see committed files (see ``fleetscope.lake.catalog``), and on restart the
processor deletes uncommitted files (start offset >= next_offset) before re-consuming
from ``next_offset``. A crash at any point therefore yields neither loss nor duplicates.

Kafka consumer-group offsets are committed too, but only as an *informational* copy
(for lag monitoring); the checkpoint in the lake is the source of truth.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import msgspec

from fleetscope.stream.dedup import StreamDedupState
from fleetscope.stream.windows import WindowSnapshot

CHECKPOINT_VERSION = 1
FILE_RE = re.compile(
    r"^(?P<topic>.+)-p(?P<partition>\d{3,})-(?P<start>\d{12})-(?P<end>\d{12})\.parquet$"
)


DRAIN_RE = re.compile(r"^(?P<topic>.+)-p(?P<partition>\d{3,})-(?P<at>\d{12})-drain\.parquet$")


class Checkpoint(msgspec.Struct, kw_only=True):
    version: int = CHECKPOINT_VERSION
    topic: str
    partition: int
    next_offset: int
    dedup: list[StreamDedupState]
    windows: WindowSnapshot
    epochs: int = 0
    drained_at: int | None = None  # next_offset of the last committed drain epoch
    created_at_ns: int = 0


_enc = msgspec.msgpack.Encoder()
_dec = msgspec.msgpack.Decoder(Checkpoint)


def checkpoint_dir(lake_root: str | Path) -> Path:
    return Path(lake_root) / "_checkpoints"


def checkpoint_path(lake_root: str | Path, topic: str, partition: int) -> Path:
    return checkpoint_dir(lake_root) / topic / f"p{partition:03d}.ckpt"


def data_file_name(topic: str, partition: int, start: int, end: int) -> str:
    return f"{topic}-p{partition:03d}-{start:012d}-{end:012d}.parquet"


def drain_file_name(topic: str, partition: int, at: int) -> str:
    return f"{topic}-p{partition:03d}-{at:012d}-drain.parquet"


def atomic_write(path: Path, data: bytes, durable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        if durable:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    if durable:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def load_checkpoint(lake_root: str | Path, topic: str, partition: int) -> Checkpoint | None:
    p = checkpoint_path(lake_root, topic, partition)
    if not p.exists():
        return None
    ckpt = _dec.decode(p.read_bytes())
    if ckpt.version != CHECKPOINT_VERSION:
        raise RuntimeError(f"{p}: unsupported checkpoint version {ckpt.version}")
    return ckpt


def save_checkpoint(lake_root: str | Path, ckpt: Checkpoint, durable: bool = True) -> None:
    ckpt.created_at_ns = time.time_ns()
    atomic_write(checkpoint_path(lake_root, ckpt.topic, ckpt.partition), _enc.encode(ckpt), durable)


def committed_positions(lake_root: str | Path) -> dict[tuple[str, int], tuple[int, int | None]]:
    """(topic, partition) -> (next_offset, drained_at) for every checkpoint in the lake."""
    out: dict[tuple[str, int], tuple[int, int | None]] = {}
    root = checkpoint_dir(lake_root)
    if not root.exists():
        return out
    for p in root.glob("*/p*.ckpt"):
        ckpt = _dec.decode(p.read_bytes())
        out[(ckpt.topic, ckpt.partition)] = (ckpt.next_offset, ckpt.drained_at)
    return out


def remove_uncommitted(
    lake_root: str | Path,
    topic: str,
    partition: int,
    next_offset: int,
    drained_at: int | None = None,
) -> list[Path]:
    """Delete data files of this partition left behind by an epoch that never committed."""
    removed = []
    for f in Path(lake_root).glob("*/**/*.parquet"):
        if f.relative_to(lake_root).parts[0].startswith("_"):
            continue
        if m := FILE_RE.match(f.name):
            stale = int(m["start"]) >= next_offset
        elif m := DRAIN_RE.match(f.name):
            stale = drained_at is None or int(m["at"]) > drained_at
        else:
            continue
        if m["topic"] == topic and int(m["partition"]) == partition and stale:
            f.unlink()
            removed.append(f)
    for f in Path(lake_root).glob(f"*/**/.{topic}-p{partition:03d}-*.tmp-*"):
        f.unlink()
        removed.append(f)
    return removed
