"""``fleetscope`` command line.

Typical flows::

    # everything in-process, no Kafka (CI, laptops)
    fleetscope demo --lake lake --vehicles 100 --minutes 20

    # against Kafka (see docker-compose.yml)
    fleetscope replay  --bootstrap localhost:9092 --lake lake --vehicles 500 --speedup 60
    fleetscope process --bootstrap localhost:9092 --lake lake --metrics-port 9108
    fleetscope mine --lake lake && fleetscope select --lake lake --budget-gb 5
    fleetscope sql --lake lake "SELECT scenario_type, count(*) FROM scenarios GROUP BY 1"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

from fleetscope.replay.fleet import Emission, FaultConfig, FaultStats, inject_faults, interleave
from fleetscope.schema import SAMPLE_HZ, Track


# --- sources -----------------------------------------------------------------------------
def _add_source_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("telemetry source")
    g.add_argument("--vehicles", type=int, default=50, help="synthetic vehicles (0 = none)")
    g.add_argument("--minutes", type=float, default=10.0, help="synthetic trip length")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--intensity", choices=("clear", "borderline"), default="clear")
    g.add_argument("--comma2k19", type=Path, help="comma2k19 root (any dir with segments)")
    g.add_argument("--clones", type=int, default=0, help="virtual vehicles per real segment")
    g.add_argument("--faults", choices=("none", "realistic"), default="realistic")
    g.add_argument("--chunk-samples", type=int, default=SAMPLE_HZ, help="samples per message")


def _tracks(args: argparse.Namespace) -> tuple[list[Track], list]:
    from fleetscope.sources import comma2k19
    from fleetscope.sources.synthetic import FleetConfig, generate_fleet

    tracks: list[Track] = []
    labels: list = []
    if args.vehicles > 0:
        cfg = FleetConfig(
            n_vehicles=args.vehicles,
            trip_minutes=args.minutes,
            seed=args.seed,
            intensity=args.intensity,
        )
        tracks, labels = generate_fleet(cfg)
    if args.comma2k19:
        real = comma2k19.load_route_tracks(args.comma2k19)
        if not real:
            sys.exit(f"no comma2k19 segments under {args.comma2k19}")
        tracks.extend(real)
        for i, tr in enumerate(real):
            tracks.extend(comma2k19.amplify(tr, args.clones, seed=args.seed + i))
    if not tracks:
        sys.exit("no telemetry source: use --vehicles N and/or --comma2k19 PATH")
    return tracks, labels


def _emissions(args: argparse.Namespace, stats: FaultStats) -> tuple[Iterator[Emission], list]:
    tracks, labels = _tracks(args)
    faults = FaultConfig.realistic() if args.faults == "realistic" else FaultConfig()
    chunks = interleave(tracks, args.chunk_samples)
    return inject_faults(chunks, faults, stats, seed=args.seed), labels


def _write_labels(lake_root: Path, labels: list) -> None:
    from fleetscope.lake.sql import write_derived
    from fleetscope.scenarios.pipeline import ground_truth_table

    if labels:
        write_derived(lake_root, "ground_truth", ground_truth_table(labels))


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _cell(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def _print_table(table, max_rows: int = 50) -> None:
    """Plain-text table (keeps the core install free of pandas)."""
    names = table.column_names
    rows = [[_cell(v) for v in r.values()] for r in table.slice(0, max_rows).to_pylist()]
    widths = [max([len(n)] + [len(r[i]) for r in rows]) for i, n in enumerate(names)]
    print("  ".join(n.rjust(w) for n, w in zip(names, widths, strict=True)))
    for r in rows:
        print("  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)))
    if table.num_rows > max_rows:
        print(f"... {table.num_rows - max_rows} more rows")


# --- commands ----------------------------------------------------------------------------
def cmd_replay(args: argparse.Namespace) -> None:
    from fleetscope.replay.fleet import paced
    from fleetscope.replay.producer import produce_all

    stats = FaultStats()
    emissions, labels = _emissions(args, stats)
    _write_labels(args.lake, labels)
    t0 = time.monotonic()
    report = produce_all(paced(emissions, args.speedup), args.bootstrap, args.topic)
    dt = time.monotonic() - t0
    _print(
        {
            "messages_sent": report.sent,
            "delivered": report.delivered,
            "failed": report.failed,
            "mb": round(report.bytes / 1e6, 2),
            "seconds": round(dt, 2),
            "messages_per_s": round(report.sent / dt),
            "samples": stats.samples,
            "expected_clean_rows": stats.expected_clean_rows,
            "injected": {
                "duplicates": stats.duplicates,
                "reordered": stats.reordered,
                "late": stats.late,
                "corrupt": stats.corrupt,
                "invalid_rows": stats.invalid_rows,
            },
        }
    )
    if report.failed:
        sys.exit(1)


def _processor_config(args: argparse.Namespace):
    from fleetscope.stream.processor import ProcessorConfig

    return ProcessorConfig(
        lake_root=args.lake,
        topic=args.topic,
        durable=not args.no_fsync,
        epoch_max_rows=args.epoch_rows,
        epoch_max_s=args.epoch_seconds,
    )


def cmd_process(args: argparse.Namespace) -> None:
    from fleetscope.stream.runner import KafkaRunner, RunnerConfig

    runner = KafkaRunner(
        RunnerConfig(
            bootstrap=args.bootstrap,
            group_id=args.group,
            dlq_topic=args.dlq_topic,
            exit_when_idle_s=args.exit_when_idle,
            drain_on_exit=args.drain_on_exit,
            metrics_port=args.metrics_port,
        ),
        _processor_config(args),
    )
    _print(dict(sorted(runner.run().items())))


def cmd_local(args: argparse.Namespace) -> dict:
    from fleetscope.stream.local import InMemoryTopic, run_local

    stats = FaultStats()
    emissions, labels = _emissions(args, stats)
    topic = InMemoryTopic(args.partitions)
    topic.extend(emissions)
    _write_labels(args.lake, labels)
    t0 = time.monotonic()
    out = run_local(topic, _processor_config(args))
    dt = time.monotonic() - t0
    rows = out["rows.valid"] + out["rows.invalid"]
    result = {
        "messages": len(topic),
        "seconds": round(dt, 2),
        "rows_per_s": round(rows / dt),
        "lake_rows": out["rows.valid"],
        "expected_clean_rows": stats.expected_clean_rows,
        "exact": out["rows.valid"] == stats.expected_clean_rows,
        **{k: v for k, v in sorted(out.items()) if not k.startswith("file_bytes")},
    }
    _print(result)
    return result


def cmd_mine(args: argparse.Namespace) -> None:
    from fleetscope.lake.sql import Lake
    from fleetscope.metrics.driving import fleet_summary, trip_metrics
    from fleetscope.scenarios.pipeline import mine_scenarios

    lake = Lake(args.lake)
    t0 = time.monotonic()
    scenarios = mine_scenarios(lake)
    trip_metrics(lake)
    print(f"mined {scenarios.num_rows} scenarios in {time.monotonic() - t0:.2f} s")
    _print_table(
        lake.arrow(
            "SELECT scenario_type, count(*) AS n, avg(severity) AS severity, "
            "avg(priority) AS priority FROM scenarios GROUP BY 1 ORDER BY n DESC"
        )
    )
    _print(fleet_summary(lake))


def cmd_select(args: argparse.Namespace) -> None:
    from fleetscope.lake.sql import Lake, write_derived
    from fleetscope.scenarios.selection import SelectionConfig, compare, select, selection_table

    lake = Lake(args.lake)
    if not lake.has("scenarios"):
        sys.exit("no scenarios table: run `fleetscope mine` first")
    table = lake.arrow("SELECT * FROM scenarios ORDER BY rank")
    cfg = SelectionConfig(budget_bytes=int(args.budget_gb * 1e9))
    chosen = select(table, cfg, args.method)
    write_derived(args.lake, "selection", selection_table(table, chosen, cfg))
    print(
        f"{args.method}: {len(chosen.indices)} clips, {chosen.cost_bytes / 1e9:.2f} GB "
        f"of {args.budget_gb} GB, coverage {chosen.coverage:.3f}"
    )
    _print_table(compare(table, cfg))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from fleetscope.lake.sql import Lake
    from fleetscope.scenarios.pipeline import evaluate

    ev = evaluate(Lake(args.lake), args.tolerance)
    _print_table(ev)


def cmd_sql(args: argparse.Namespace) -> None:
    from fleetscope.lake.sql import Lake

    _print_table(Lake(args.lake).arrow(args.query))


def cmd_demo(args: argparse.Namespace) -> None:
    result = cmd_local(args)
    cmd_mine(args)
    cmd_select(args)
    if args.vehicles > 0:
        cmd_evaluate(args)
    if not result["exact"]:
        sys.exit("lake row count does not match the replay's expected clean rows")


# --- parser ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fleetscope", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def lake_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--lake", type=Path, default=Path("lake"))

    def stream_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--topic", default="telemetry.raw")
        p.add_argument("--no-fsync", action="store_true", help="skip fsync (benchmarks, CI)")
        p.add_argument("--epoch-rows", type=int, default=500_000, help="commit after N rows")
        p.add_argument("--epoch-seconds", type=float, default=10.0, help="... or N seconds")

    p = sub.add_parser("replay", help="produce a (synthetic / comma2k19) fleet into Kafka")
    lake_args(p)
    _add_source_args(p)
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--topic", default="telemetry.raw")
    p.add_argument(
        "--speedup", type=float, default=0.0, help="x real time; 0 = as fast as possible"
    )
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("process", help="consume Kafka into the lake (exactly-once)")
    lake_args(p)
    stream_args(p)
    p.add_argument("--bootstrap", default="localhost:9092")
    p.add_argument("--group", default="fleetscope-lake")
    p.add_argument("--dlq-topic", default="telemetry.dlq")
    p.add_argument("--metrics-port", type=int, default=None)
    p.add_argument("--exit-when-idle", type=float, default=None, metavar="SECONDS")
    p.add_argument("--drain-on-exit", action="store_true", help="close all windows on exit")
    p.set_defaults(fn=cmd_process)

    p = sub.add_parser("local", help="replay + process in-process (no Kafka)")
    lake_args(p)
    stream_args(p)
    _add_source_args(p)
    p.add_argument("--partitions", type=int, default=8)
    p.set_defaults(fn=cmd_local)

    p = sub.add_parser("mine", help="detect + score scenarios, compute trip metrics")
    lake_args(p)
    p.set_defaults(fn=cmd_mine)

    def select_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--budget-gb", type=float, default=2.0)
        p.add_argument(
            "--method",
            default="diverse",
            choices=("diverse", "top_priority", "top_severity", "stratified", "random"),
        )

    p = sub.add_parser("select", help="choose clips under a storage budget")
    lake_args(p)
    select_args(p)
    p.set_defaults(fn=cmd_select)

    def eval_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tolerance", type=float, default=1.0, help="interval slack, seconds")

    p = sub.add_parser("evaluate", help="detector precision / recall vs synthetic labels")
    lake_args(p)
    eval_args(p)
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("sql", help="run a SQL query against the lake")
    lake_args(p)
    p.add_argument("query")
    p.set_defaults(fn=cmd_sql)

    p = sub.add_parser("demo", help="local + mine + select + evaluate")
    lake_args(p)
    stream_args(p)
    _add_source_args(p)
    p.add_argument("--partitions", type=int, default=8)
    select_args(p)
    eval_args(p)
    p.set_defaults(fn=cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
