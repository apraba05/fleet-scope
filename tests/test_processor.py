"""Exactly-once and accounting guarantees of the partition processor."""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from fleetscope.lake.catalog import committed_files
from fleetscope.replay.fleet import FaultConfig, FaultStats, inject_faults, interleave
from fleetscope.stream import checkpoint as ckpt_mod
from fleetscope.stream import processor as proc_mod
from fleetscope.stream.local import InMemoryTopic, SimulatedCrash, run_local
from fleetscope.stream.processor import FencedError, PartitionProcessor, ProcessorConfig

SORT = {
    "telemetry": ["vehicle_id", "trip_id", "ts"],
    "quarantine": ["vehicle_id", "trip_id", "ts"],
    "windows_10s": ["vehicle_id", "trip_id", "window_start"],
    "dlq": ["kafka_partition", "kafka_offset"],
}


def read_table(lake: Path, table: str) -> pa.Table | None:
    files = committed_files(lake, table)
    if not files:
        return None
    t = pa.concat_tables([pq.read_table(f) for f in files])
    t = t.select([c for c in t.column_names if c not in ("date", "hour")])
    # NaN != NaN under Table.equals; canonicalise so equal data compares equal.
    for i, field in enumerate(t.schema):
        if pa.types.is_floating(field.type):
            col = t.column(i)
            t = t.set_column(
                i, field, pc.if_else(pc.is_nan(col), pa.scalar(-1e30, field.type), col)
            )
    return t.sort_by([(c, "ascending") for c in SORT[table]])


def snapshot(lake: Path) -> dict[str, pa.Table | None]:
    return {t: read_table(lake, t) for t in SORT}


def cfg(lake, rows=20_000, **kw):
    return ProcessorConfig(
        lake_root=lake, durable=False, epoch_max_rows=rows, epoch_max_s=1e9, **kw
    )


def test_counts_match_injected_faults_exactly(tmp_path, faulty_topic):
    topic, faults = faulty_topic
    stats = run_local(topic, cfg(tmp_path), batch_size=500)
    assert stats["records.consumed"] == faults.messages
    assert stats["chunks.duplicate"] == faults.duplicates
    assert stats["dlq.decode_error"] == faults.corrupt
    assert stats["rows.invalid"] == faults.invalid_rows
    assert stats["rows.valid"] == faults.expected_clean_rows
    for reason, n in faults.per_reason.items():
        assert stats[f"invalid.{reason}"] == n
    lake = snapshot(tmp_path)
    assert lake["telemetry"].num_rows == faults.expected_clean_rows
    assert lake["quarantine"].num_rows == faults.invalid_rows
    assert lake["dlq"].num_rows == faults.corrupt
    # no (vehicle, trip, ts) appears twice in the lake
    keys = set(
        zip(*(lake["telemetry"].column(c).to_pylist() for c in SORT["telemetry"]), strict=True)
    )
    assert len(keys) == lake["telemetry"].num_rows


def test_late_chunks_reach_the_lake_but_not_the_windows(tmp_path, faulty_topic):
    topic, faults = faulty_topic
    stats = run_local(topic, cfg(tmp_path), batch_size=500)
    assert faults.late > 0
    assert stats["rows.late"] > 0
    windows = read_table(tmp_path, "windows_10s")
    assert sum(windows.column("samples").to_pylist()) == stats["rows.valid"] - stats["rows.late"]


@pytest.mark.parametrize("crash_after", [1, 4, 9])
def test_crash_and_restart_equals_clean_run(tmp_path, faulty_topic, crash_after):
    topic, _ = faulty_topic
    assert min(len(p) for p in topic.partitions) > 100 * 9
    clean = tmp_path / "clean"
    run_local(topic, cfg(clean), batch_size=100)

    # ~2 batches per epoch, so crashes land both right after and between commits
    crashy = tmp_path / "crashy"
    for p in range(topic.n_partitions):
        with pytest.raises(SimulatedCrash):
            run_local(
                topic,
                cfg(crashy, rows=4000),
                batch_size=100,
                partitions=[p],
                crash_after_batches=crash_after,
            )
    stats = run_local(topic, cfg(crashy, rows=4000), batch_size=100)
    assert stats["records.skipped_committed"] == 0  # resumed exactly at the checkpoint
    a, b = snapshot(clean), snapshot(crashy)
    for table in SORT:
        assert a[table].equals(b[table]), table


def test_crash_between_file_write_and_checkpoint_leaves_no_visible_trace(
    tmp_path, faulty_topic, monkeypatch
):
    topic, _ = faulty_topic
    clean = tmp_path / "clean"
    run_local(topic, cfg(clean), batch_size=400)

    lake = tmp_path / "lake"
    real_save = ckpt_mod.save_checkpoint
    calls = {"n": 0}

    def flaky_save(root, ckpt, durable=True):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk went away")
        real_save(root, ckpt, durable)

    monkeypatch.setattr(proc_mod, "save_checkpoint", flaky_save)
    with pytest.raises(OSError):
        run_local(topic, cfg(lake), batch_size=400)
    # the failed epoch's files exist on disk but are invisible to readers
    on_disk = sum(1 for _ in lake.rglob("*.parquet"))
    visible = sum(len(committed_files(lake, t)) for t in SORT)
    assert on_disk > visible

    monkeypatch.setattr(proc_mod, "save_checkpoint", real_save)
    stats = run_local(topic, cfg(lake), batch_size=400)
    assert stats["recovery.orphan_files_removed"] > 0
    a, b = snapshot(clean), snapshot(lake)
    for table in SORT:
        assert a[table].equals(b[table]), table


def finish_all(lake, n_partitions):
    return [PartitionProcessor(cfg(lake), p).finish() for p in range(n_partitions)]


def test_drain_epoch_after_committed_input_matches_in_epoch_drain(tmp_path, faulty_topic):
    # A bounded run whose last epoch was already committed (e.g. Kafka consumer reached the
    # end after a time-based checkpoint) must still close its windows exactly once.
    topic, _ = faulty_topic
    clean, lake = tmp_path / "clean", tmp_path / "lake"
    run_local(topic, cfg(clean), batch_size=500)
    run_local(topic, cfg(lake), batch_size=500, drain=False)
    open_windows = read_table(lake, "windows_10s")
    assert open_windows.num_rows < read_table(clean, "windows_10s").num_rows
    files = finish_all(lake, topic.n_partitions)
    assert all(f and all("-drain.parquet" in str(p) for p, _, _ in f) for f in files)
    a, b = snapshot(clean), snapshot(lake)
    for table in SORT:
        assert a[table].equals(b[table]), table
    # draining again finds nothing to close and changes nothing
    assert finish_all(lake, topic.n_partitions) == [[]] * topic.n_partitions
    assert snapshot(lake)["windows_10s"].equals(b["windows_10s"])


def test_crash_during_drain_epoch_is_invisible_and_recovered(tmp_path, faulty_topic, monkeypatch):
    topic, _ = faulty_topic
    clean, lake = tmp_path / "clean", tmp_path / "lake"
    run_local(topic, cfg(clean), batch_size=500)
    run_local(topic, cfg(lake), batch_size=500, drain=False)
    before = read_table(lake, "windows_10s")

    def broken_save(root, ckpt, durable=True):
        raise OSError("disk went away")

    real_save = proc_mod.save_checkpoint
    monkeypatch.setattr(proc_mod, "save_checkpoint", broken_save)
    with pytest.raises(OSError):
        PartitionProcessor(cfg(lake), 0).finish()
    assert any(lake.rglob("*-drain.parquet"))  # written, but not committed
    assert read_table(lake, "windows_10s").equals(before)

    monkeypatch.setattr(proc_mod, "save_checkpoint", real_save)
    recovered = PartitionProcessor(cfg(lake), 0)
    assert recovered.stats["recovery.orphan_files_removed"] > 0
    recovered.finish()
    finish_all(lake, topic.n_partitions)
    assert snapshot(clean)["windows_10s"].equals(snapshot(lake)["windows_10s"])


def test_rerun_without_new_data_is_a_noop(tmp_path, faulty_topic):
    topic, _ = faulty_topic
    run_local(topic, cfg(tmp_path))
    before = snapshot(tmp_path)
    stats = run_local(topic, cfg(tmp_path))
    assert stats["records.consumed"] == 0
    after = snapshot(tmp_path)
    for table in SORT:
        assert before[table].equals(after[table])


def test_incremental_consumption_matches_single_pass(tmp_path, small_fleet):
    """Data arriving in two waves (consumer restarts in between) == one pass."""
    tracks, _ = small_fleet
    emissions = list(
        inject_faults(interleave(tracks, 20), FaultConfig(duplicate_rate=0.02), FaultStats())
    )
    half = len(emissions) // 2

    full = InMemoryTopic(2)
    full.extend(emissions, timestamp_ms=1_800_000_000_000)
    run_local(full, cfg(tmp_path / "one"), drain=False)

    waves = InMemoryTopic(2)
    waves.extend(emissions[:half], timestamp_ms=1_800_000_000_000)
    run_local(waves, cfg(tmp_path / "two"), drain=False)
    waves.extend(emissions[half:], timestamp_ms=1_800_000_000_000)
    run_local(waves, cfg(tmp_path / "two"), drain=False)

    a, b = snapshot(tmp_path / "one"), snapshot(tmp_path / "two")
    assert a["telemetry"].equals(b["telemetry"])
    assert a["windows_10s"].equals(b["windows_10s"])


def test_zombie_owner_is_fenced(tmp_path, faulty_topic):
    topic, _ = faulty_topic
    recs = topic.partitions[0]
    zombie = PartitionProcessor(cfg(tmp_path), 0)
    zombie.process(recs[:100])
    owner = PartitionProcessor(cfg(tmp_path), 0)  # took over after a rebalance
    owner.process(recs[:200])
    owner.checkpoint()
    with pytest.raises(FencedError):
        zombie.checkpoint()
    assert read_table(tmp_path, "telemetry").column("kafka_offset").to_pylist()[-1] <= 199


def test_files_are_hive_partitioned_by_event_hour(tmp_path, faulty_topic):
    topic, _ = faulty_topic
    run_local(topic, cfg(tmp_path))
    files = committed_files(tmp_path, "telemetry")
    assert files
    for f in files:
        parts = Path(f).parts
        assert parts[-3].startswith("date=") and parts[-2].startswith("hour=")
        t = pq.read_table(f, columns=["ts"])
        hours = {ts.hour for ts in t.column("ts").to_pylist()}
        assert hours == {int(parts[-2].split("=")[1])}
