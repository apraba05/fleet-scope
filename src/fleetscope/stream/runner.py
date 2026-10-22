"""Kafka consumer loop around ``PartitionProcessor``.

* Manual offset management: on assignment each partition is positioned at the offset in
  its lake checkpoint (source of truth). Consumer-group commits are informational only.
* Eager rebalancing: on revoke every owned partition commits its epoch first, so the new
  owner resumes exactly where this one stopped.
* ``on_lost`` (session expired, partitions may already be owned elsewhere): state is
  dropped *without* committing; checkpoint CAS in the processor fences a zombie writer.
* Dead letters are also produced to ``<topic>.dlq`` (at-least-once) for alerting and
  replay tooling; the lake ``dlq`` table is the exactly-once record.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
from collections import defaultdict
from dataclasses import dataclass

from confluent_kafka import (
    OFFSET_BEGINNING,
    Consumer,
    KafkaError,
    KafkaException,
    Producer,
    TopicPartition,
)

from fleetscope import observability as obs
from fleetscope.observability import HealthReporter, Stats
from fleetscope.stream.processor import FencedError, PartitionProcessor, ProcessorConfig, Record

log = logging.getLogger("fleetscope.runner")


@dataclass
class RunnerConfig:
    bootstrap: str
    group_id: str = "fleetscope-lake"
    batch_size: int = 5000
    poll_timeout_s: float = 0.2
    dlq_topic: str | None = None
    exit_when_idle_s: float | None = None
    drain_on_exit: bool = False
    metrics_port: int | None = None
    health_interval_s: float = 5.0


class KafkaRunner:
    def __init__(self, cfg: RunnerConfig, pcfg: ProcessorConfig):
        self.cfg = cfg
        self.pcfg = pcfg
        self.processors: dict[int, PartitionProcessor] = {}
        self.stats = Stats()  # stats of partitions this runner no longer owns
        self._stop = False
        self._dlq_producer: Producer | None = None
        self._consumer: Consumer | None = None

    # --- rebalance callbacks --------------------------------------------------------------
    def _on_assign(self, consumer: Consumer, partitions: list[TopicPartition]) -> None:
        for tp in partitions:
            proc = PartitionProcessor(self.pcfg, tp.partition)
            self.processors[tp.partition] = proc
            tp.offset = proc.resume_offset if proc.resume_offset is not None else OFFSET_BEGINNING
            log.info("assigned p%d at offset %s", tp.partition, tp.offset)
        consumer.assign(partitions)

    def _on_revoke(self, consumer: Consumer, partitions: list[TopicPartition]) -> None:
        for tp in partitions:
            proc = self.processors.pop(tp.partition, None)
            if proc is not None and self._commit(consumer, tp.partition, proc):
                self.stats.update(proc.stats)  # (a fenced commit already merged them)
        consumer.unassign()

    def _on_lost(self, consumer: Consumer, partitions: list[TopicPartition]) -> None:
        for tp in partitions:
            proc = self.processors.pop(tp.partition, None)
            if proc is not None:
                log.warning("lost p%d; discarding uncommitted epoch", tp.partition)
                proc.sink.discard()
                self.stats.update(proc.stats)

    def _commit(
        self, consumer: Consumer, partition: int, proc: PartitionProcessor, finish: bool = False
    ) -> bool:
        """Commit the open epoch (``finish``: also close all windows). False if fenced."""
        if not (finish or proc.has_uncommitted):
            return True
        try:
            proc.finish() if finish else proc.checkpoint()
        except FencedError:
            # Another consumer owns this partition now (we were a zombie). Our epoch was
            # discarded by the processor; stop processing the partition.
            log.warning("p%d fenced by a newer owner; dropping it", partition)
            if self.processors.get(partition) is proc:
                del self.processors[partition]
            self.stats.update(proc.stats)
            self.stats["fenced"] += 1
            return False
        if proc.next_offset is not None:
            consumer.commit(
                offsets=[TopicPartition(self.pcfg.topic, partition, proc.next_offset)],
                asynchronous=True,
            )
        return True

    # --- main loop ------------------------------------------------------------------------
    def stop(self, *_: object) -> None:
        self._stop = True

    def run(self) -> Stats:
        cfg = self.cfg
        if cfg.metrics_port:
            obs.serve_metrics(cfg.metrics_port)
        consumer = Consumer(
            {
                "bootstrap.servers": cfg.bootstrap,
                "group.id": cfg.group_id,
                "enable.auto.commit": False,
                "enable.auto.offset.store": False,
                "auto.offset.reset": "earliest",
                "partition.assignment.strategy": "range",
                "fetch.max.bytes": 64 * 1024 * 1024,
                "max.partition.fetch.bytes": 8 * 1024 * 1024,
                "queued.max.messages.kbytes": 256 * 1024,
                "enable.partition.eof": True,
            }
        )
        self._consumer = consumer
        if cfg.dlq_topic:
            self._dlq_producer = Producer({"bootstrap.servers": cfg.bootstrap, "linger.ms": 50})
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        consumer.subscribe(
            [self.pcfg.topic],
            on_assign=self._on_assign,
            on_revoke=self._on_revoke,
            on_lost=self._on_lost,
        )
        health = HealthReporter(
            self.pcfg.lake_root,
            cfg.health_interval_s,
            f"processor-{socket.gethostname()}-{os.getpid()}",
        )
        eof: set[int] = set()
        last_msg = time.monotonic()
        last_lag = 0.0
        try:
            while not self._stop:
                msgs = consumer.consume(cfg.batch_size, timeout=cfg.poll_timeout_s)
                by_part: dict[int, list[Record]] = defaultdict(list)
                for m in msgs:
                    err = m.error()
                    p, off = m.partition(), m.offset()
                    if err is not None:
                        if err.code() == KafkaError._PARTITION_EOF and p is not None:
                            eof.add(p)
                            continue
                        raise KafkaException(err)
                    assert p is not None and off is not None  # always set on fetched records
                    eof.discard(p)
                    by_part[p].append(Record(off, m.value() or b"", m.timestamp()[1]))
                if by_part:
                    last_msg = time.monotonic()
                for p, recs in by_part.items():
                    proc = self.processors.get(p)
                    if proc is None:
                        continue  # revoked while the batch was in flight
                    proc.process(recs)
                    self._forward_dead_letters(p, proc)
                for p, proc in list(self.processors.items()):
                    if proc.should_checkpoint():
                        self._commit(consumer, p, proc)
                now = time.monotonic()
                if now - last_lag > 2.0:
                    last_lag = now
                    self._update_lag(consumer)
                health.maybe_report(self.merged_stats())
                if (
                    cfg.exit_when_idle_s is not None
                    and self.processors
                    and now - last_msg >= cfg.exit_when_idle_s
                    and eof >= set(self.processors)
                ):
                    log.info("idle for %.1fs at end of all partitions; exiting", now - last_msg)
                    break
        finally:
            for p, proc in list(self.processors.items()):
                self._commit(consumer, p, proc, finish=cfg.drain_on_exit)
            health.maybe_report(self.merged_stats(), force=True)
            if self._dlq_producer is not None:
                self._dlq_producer.flush(10)
            consumer.close()
        return self.merged_stats()

    def merged_stats(self) -> Stats:
        total = Stats(self.stats)
        for proc in self.processors.values():
            total.update(proc.stats)
        return total

    def _forward_dead_letters(self, partition: int, proc: PartitionProcessor) -> None:
        letters = proc.take_dead_letters()
        if not letters or self._dlq_producer is None or not self.cfg.dlq_topic:
            return
        for d in letters:
            self._dlq_producer.produce(
                self.cfg.dlq_topic,
                value=d.payload,
                headers=[
                    ("reason", d.reason.encode()),
                    ("source_partition", str(partition).encode()),
                    ("source_offset", str(d.offset).encode()),
                ],
            )
        self._dlq_producer.poll(0)

    def _update_lag(self, consumer: Consumer) -> None:
        for p, proc in self.processors.items():
            try:
                _low, high = consumer.get_watermark_offsets(
                    TopicPartition(self.pcfg.topic, p), timeout=1.0, cached=True
                )
            except KafkaException:
                continue
            if high >= 0 and proc.next_offset is not None:
                obs.CONSUMER_LAG.labels(str(p)).set(max(0, high - proc.next_offset))
