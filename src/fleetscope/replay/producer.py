"""Kafka producer for telemetry chunks.

Producer settings are chosen for an ingest path that must not lose or duplicate data
at the broker:

* ``enable.idempotence=true`` (implies ``acks=all``): broker-side dedup of producer
  retries, so network retries never create duplicates *inside Kafka*. Duplicates that
  originate on the vehicle (re-uploads) are still possible and are removed by the
  stream processor's ``(vehicle_id, trip_id, seq)`` dedup.
* ``linger.ms`` / ``batch.size``: trade a few ms of latency for large batches.
* ``compression.type=zstd``: telemetry floats compress well and broker disk/network is
  the scaling bottleneck long before producer CPU.
* ``partitioner=murmur2_random``: same key->partition mapping as the Java client, so
  JVM consumers/Kafka Streams apps co-partition with this producer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from confluent_kafka import KafkaError, Producer

from fleetscope.replay.fleet import Emission


def producer_config(bootstrap: str, **overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "bootstrap.servers": bootstrap,
        "enable.idempotence": True,
        "acks": "all",
        "linger.ms": 20,
        "batch.size": 1_000_000,
        "compression.type": "zstd",
        "partitioner": "murmur2_random",
        "queue.buffering.max.messages": 500_000,
        "queue.buffering.max.kbytes": 1_048_576,
    }
    cfg.update(overrides)
    return cfg


@dataclass
class ProduceReport:
    sent: int = 0
    delivered: int = 0
    failed: int = 0
    bytes: int = 0


def produce_all(
    emissions: Iterable[Emission], bootstrap: str, topic: str, **overrides: object
) -> ProduceReport:
    report = ProduceReport()
    producer = Producer(producer_config(bootstrap, **overrides))

    def on_delivery(err: KafkaError | None, _msg: object) -> None:
        if err is None:
            report.delivered += 1
        else:
            report.failed += 1

    for e in emissions:
        while True:
            try:
                producer.produce(topic, value=e.value, key=e.key, on_delivery=on_delivery)
                break
            except BufferError:
                producer.poll(0.05)  # local queue full: serve callbacks, then retry
        report.sent += 1
        report.bytes += len(e.value)
        if report.sent % 10_000 == 0:
            producer.poll(0)
    remaining = producer.flush(60)
    report.failed += remaining
    return report
