"""Integration tests against a real Kafka broker (``make up``; ``pytest -m kafka``).

The broker address comes from ``FLEETSCOPE_BOOTSTRAP`` (default ``localhost:9092``).
Each test uses fresh, uniquely named topics, so runs never interfere.
"""

import os
import signal
import subprocess
import sys
import time
import uuid

import pytest
from confluent_kafka import OFFSET_BEGINNING, Consumer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic

from fleetscope.lake.sql import Lake
from fleetscope.replay.fleet import FaultConfig, FaultStats, inject_faults, interleave
from fleetscope.replay.producer import produce_all
from fleetscope.sources.synthetic import FleetConfig, generate_fleet
from fleetscope.stream.checkpoint import checkpoint_dir
from fleetscope.stream.local import InMemoryTopic, run_local
from fleetscope.stream.processor import ProcessorConfig, Record

pytestmark = pytest.mark.kafka
BOOTSTRAP = os.environ.get("FLEETSCOPE_BOOTSTRAP", "localhost:9092")
PARTITIONS = 6
KEY = ["vehicle_id", "trip_id", "ts"]


def create_topics(*names: str, partitions: int = PARTITIONS) -> None:
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    cfg = {"message.timestamp.type": "LogAppendTime"}
    futures = admin.create_topics([NewTopic(n, partitions, 1, config=cfg) for n in names])
    for f in futures.values():
        f.result(30)


@pytest.fixture(scope="module")
def fleet():
    tracks, _ = generate_fleet(FleetConfig(n_vehicles=30, trip_minutes=8, seed=21))
    return tracks


@pytest.fixture()
def loaded_topic(fleet):
    """A fresh topic holding the fleet with realistic faults: (topic, fault stats, emissions)."""
    topic = f"it-{uuid.uuid4().hex[:10]}"
    create_topics(topic, f"{topic}.dlq")
    stats = FaultStats()
    emissions = list(inject_faults(interleave(fleet, 20), FaultConfig.realistic(), stats, seed=5))
    report = produce_all(emissions, BOOTSTRAP, topic)
    assert report.failed == 0 and report.delivered == len(emissions)
    return topic, stats, emissions


def processor(lake, topic, group, *extra):
    cmd = [sys.executable, "-m", "fleetscope.cli", "process", "--lake", str(lake)]
    cmd += ["--bootstrap", BOOTSTRAP, "--topic", topic, "--group", group]
    cmd += ["--dlq-topic", f"{topic}.dlq", "--no-fsync", "--epoch-rows", "5000"]
    cmd += ["--epoch-seconds", "0.5", *extra]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_for_checkpoints(lake, topic, n=1, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(list((checkpoint_dir(lake) / topic).glob("p*.ckpt"))) >= n:
            return
        time.sleep(0.05)
    raise TimeoutError("no checkpoint appeared")


def finish(proc, timeout=120):
    out, err = proc.communicate(timeout=timeout)
    assert proc.returncode == 0, err[-3000:]
    return out


def reference_lake(tmp_path, topic: str) -> Lake:
    """The expected lake: the topic's exact partition logs (offsets, broker timestamps)
    read back and run through the in-process runner, which has no Kafka, no crashes and
    no rebalances."""
    c = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"ref-{uuid.uuid4().hex}",
            "enable.auto.commit": False,
            "enable.partition.eof": True,
        }
    )
    c.assign([TopicPartition(topic, p, OFFSET_BEGINNING) for p in range(PARTITIONS)])
    logs = InMemoryTopic(PARTITIONS)
    at_eof: set[int] = set()
    while len(at_eof) < PARTITIONS:
        for m in c.consume(5000, timeout=1.0):
            if m.error() is not None:
                at_eof.add(m.partition())
                continue
            assert m.offset() == len(logs.partitions[m.partition()])
            logs.partitions[m.partition()].append(Record(m.offset(), m.value(), m.timestamp()[1]))
    c.close()
    run_local(logs, ProcessorConfig(lake_root=tmp_path / "ref", durable=False))
    return Lake(tmp_path / "ref")


def assert_same_lake(lake: Lake, ref: Lake, stats: FaultStats):
    cols = "vehicle_id, trip_id, seq, ts, speed, accel_long, lat, lead_dist, source"
    got = lake.arrow(f"SELECT {cols} FROM telemetry ORDER BY {', '.join(KEY)}")
    want = ref.arrow(f"SELECT {cols} FROM telemetry ORDER BY {', '.join(KEY)}")
    assert got.num_rows == stats.expected_clean_rows
    assert got.equals(want)
    for table, n in (("quarantine", stats.invalid_rows), ("dlq", stats.corrupt)):
        assert lake.sql(f"SELECT count(*) FROM {table}").fetchall()[0][0] == n, table
    dupes = lake.sql(
        "SELECT count(*) FROM (SELECT vehicle_id, trip_id, window_start FROM windows_10s "
        "GROUP BY ALL HAVING count(*) > 1)"
    ).fetchall()[0][0]
    assert dupes == 0
    # Windows depend on each partition's event-time frontier (idle close), which is the
    # same here because the reference replays the very same partition logs.
    wcols = "vehicle_id, trip_id, window_start, samples, distance_m, speed_max, lead_dist_min"
    worder = "ORDER BY vehicle_id, trip_id, window_start"
    assert lake.arrow(f"SELECT {wcols} FROM windows_10s {worder}").equals(
        ref.arrow(f"SELECT {wcols} FROM windows_10s {worder}")
    )


def test_kill_9_and_restart_is_exactly_once(tmp_path, loaded_topic):
    topic, stats, _ = loaded_topic
    lake = tmp_path / "lake"
    p = processor(lake, topic, f"g-{topic}")
    wait_for_checkpoints(lake, topic, n=2)
    time.sleep(0.3)  # let it get into the middle of later epochs
    p.send_signal(signal.SIGKILL)
    p.wait(10)
    killed = Lake(lake).sql("SELECT count(*) FROM telemetry").fetchall()[0][0]
    assert 0 < killed < stats.expected_clean_rows, "kill landed outside the run"

    finish(processor(lake, topic, f"g-{topic}", "--exit-when-idle", "3", "--drain-on-exit"))
    assert_same_lake(Lake(lake), reference_lake(tmp_path, topic), stats)


def test_rebalance_between_two_consumers_is_exactly_once(tmp_path, loaded_topic):
    topic, stats, _ = loaded_topic
    lake = tmp_path / "lake"
    group = f"g-{topic}"
    a = processor(lake, topic, group, "--exit-when-idle", "6")
    wait_for_checkpoints(lake, topic, n=1)
    b = processor(lake, topic, group, "--exit-when-idle", "6")  # triggers a rebalance
    finish(a)
    finish(b)
    # windows still open at exit are closed by a final bounded drain
    finish(processor(lake, topic, group, "--exit-when-idle", "2", "--drain-on-exit"))
    assert_same_lake(Lake(lake), reference_lake(tmp_path, topic), stats)


def test_dead_letters_are_forwarded_to_the_dlq_topic(tmp_path, loaded_topic):
    topic, stats, _ = loaded_topic
    finish(processor(tmp_path / "lake", topic, f"g-{topic}", "--exit-when-idle", "3"))
    c = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"dlq-{topic}",
            "auto.offset.reset": "earliest",
        }
    )
    c.subscribe([f"{topic}.dlq"])
    got, deadline = [], time.monotonic() + 20
    while len(got) < stats.corrupt and time.monotonic() < deadline:
        got += [m for m in c.consume(500, timeout=0.5) if m.error() is None]
    c.close()
    assert len(got) >= stats.corrupt  # at-least-once copy of the exactly-once lake table
    assert {dict(m.headers())["reason"] for m in got} == {b"decode_error"}
