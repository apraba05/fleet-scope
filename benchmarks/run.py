"""Reproducible benchmarks; writes a Markdown report.

    python benchmarks/run.py --out docs/benchmarks.md                       # in-process only
    python benchmarks/run.py --out docs/benchmarks.md --bootstrap localhost:9092   # + Kafka

Every number in the report is measured by this script on the machine that ran it.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from fleetscope.lake.sql import Lake, write_derived
from fleetscope.metrics.driving import trip_metrics
from fleetscope.replay.fleet import FaultConfig, FaultStats, inject_faults, interleave
from fleetscope.scenarios.pipeline import evaluate, ground_truth_table, mine_scenarios
from fleetscope.scenarios.selection import SelectionConfig, compare, select
from fleetscope.sources import comma2k19
from fleetscope.sources.synthetic import FleetConfig, generate_fleet
from fleetscope.stream.local import InMemoryTopic, run_local
from fleetscope.stream.processor import ProcessorConfig

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = 12
CHUNK_SAMPLES = 20
REPEATS = 3


def timed(fn: Callable[[], object], repeat: int = 1) -> tuple[float, object]:
    """Median wall time of ``repeat`` runs, and the last result."""
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def table(headers: list[str], rows: list[list[object]]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, bool):
            return "yes" if v else "**no**"
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 100 else f"{v:,.0f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*.parquet"))


def build_fleet(vehicles: int, minutes: int, clones: int, seed: int):
    tracks, labels = generate_fleet(
        FleetConfig(n_vehicles=vehicles, trip_minutes=minutes, seed=seed)
    )
    for i, tr in enumerate(comma2k19.load_route_tracks(ROOT / "data" / "comma2k19_sample")):
        tracks.append(tr)
        tracks.extend(comma2k19.amplify(tr, clones, seed=seed + i))
    stats = FaultStats()
    emissions = list(
        inject_faults(interleave(tracks, CHUNK_SAMPLES), FaultConfig.realistic(), stats, seed=seed)
    )
    return emissions, labels, stats


# --- sections ----------------------------------------------------------------------------
def bench_ingest(emissions, stats: FaultStats, work: Path) -> tuple[str, Path]:
    raw_mb = sum(len(e.value) for e in emissions) / 1e6
    rows = []
    lake_root = work
    for durable in (False, True):
        times = []
        for rep in range(REPEATS):  # a fresh lake each time: a used one would just resume
            lake_root = work / f"lake-{'fsync' if durable else 'nofsync'}-{rep}"
            topic = InMemoryTopic(PARTITIONS)
            topic.extend(emissions)
            cfg = ProcessorConfig(lake_root=str(lake_root), durable=durable)
            dt, out = timed(lambda t=topic, c=cfg: run_local(t, c))
            times.append(dt)
        dt = statistics.median(times)
        assert isinstance(out, dict)
        n = out["rows.valid"] + out["rows.invalid"]
        rows.append(
            [
                "on" if durable else "off",
                len(emissions),
                n,
                round(dt, 2),
                round(n / dt),
                round(raw_mb / dt, 1),
                out["rows.valid"] == stats.expected_clean_rows,
            ]
        )
    lake_mb = dir_bytes(lake_root / "telemetry") / 1e6
    text = (
        "Single process, single thread, all partitions consumed sequentially by "
        "`run_local` (the same `PartitionProcessor` the Kafka runner uses; no broker). "
        f"Median of {REPEATS} runs, each into an empty lake.\n\n"
        + table(
            ["fsync", "messages", "rows", "seconds", "rows/s", "wire MB/s", "lake == expected"],
            rows,
        )
        + f"\n\nWire payload {raw_mb:,.1f} MB (MessagePack) → telemetry Parquet {lake_mb:,.1f} MB "
        f"({raw_mb / lake_mb:.1f}× smaller, "
        f"{lake_mb * 1e6 / stats.expected_clean_rows:.1f} B/row, zstd, sorted by vehicle/ts)."
        + (
            " On macOS `fsync()` does not flush the drive cache (that needs `F_FULLFSYNC`), "
            "so the fsync column understates the cost of durable commits on Linux."
            if platform.system() == "Darwin"
            else ""
        )
    )
    return text, lake_root


def bench_kafka(emissions, stats: FaultStats, bootstrap: str, work: Path) -> str:
    from confluent_kafka.admin import AdminClient, NewTopic

    from fleetscope.replay.producer import produce_all

    topic = f"bench.{int(time.time())}"
    admin = AdminClient({"bootstrap.servers": bootstrap})
    fut = admin.create_topics(
        [NewTopic(topic, PARTITIONS, 1, config={"message.timestamp.type": "LogAppendTime"})]
    )
    fut[topic].result(30)
    dt_p, rep = timed(lambda: produce_all(emissions, bootstrap, topic))
    lake_root = work / "lake-kafka"
    idle = 3.0
    cmd = [
        sys.executable, "-m", "fleetscope.cli", "process", "--lake", str(lake_root),
        "--bootstrap", bootstrap, "--topic", topic, "--group", topic,
        "--exit-when-idle", str(idle), "--drain-on-exit", "--no-fsync",
    ]  # fmt: skip
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True)
    dt_c = time.perf_counter() - t0 - idle
    lake_rows = Lake(lake_root).sql("SELECT count(*) FROM telemetry").fetchall()[0][0]
    admin.delete_topics([topic])
    return (
        f"Local single-broker Kafka (Docker, KRaft), {PARTITIONS} partitions, one producer "
        "(idempotent, acks=all, zstd) and one consumer process. The consume time includes "
        "process start-up, group join and the final drain, minus the idle-exit wait.\n\n"
        + table(
            ["stage", "messages", "seconds", "messages/s", "rows/s", "MB/s"],
            [
                [
                    "produce",
                    rep.sent,
                    round(dt_p, 2),
                    round(rep.sent / dt_p),
                    round(stats.samples / dt_p),
                    round(rep.bytes / 1e6 / dt_p, 1),
                ],
                [
                    "consume → lake",
                    rep.sent,
                    round(dt_c, 2),
                    round(rep.sent / dt_c),
                    round(stats.expected_clean_rows / dt_c),
                    round(rep.bytes / 1e6 / dt_c, 1),
                ],
            ],
        )
        + f"\n\nLake rows {lake_rows:,} vs expected {stats.expected_clean_rows:,} "
        f"(exact: {lake_rows == stats.expected_clean_rows}); delivery failures: {rep.failed}."
    )


def bench_queries(lake: Lake, lake_root: Path) -> str:
    date, hour = lake.sql(
        "SELECT date, hour FROM telemetry GROUP BY 1, 2 ORDER BY count(*) DESC LIMIT 1"
    ).fetchall()[0]
    vehicle = lake.sql("SELECT min(vehicle_id) FROM telemetry").fetchall()[0][0]
    files = list((lake_root / "telemetry").rglob("*.parquet"))
    in_hour = [f for f in files if f"date={date}" in str(f) and f"hour={hour:02d}" in str(f)]
    queries = [
        (
            "fleet aggregate (full scan)",
            "SELECT vehicle_id, avg(speed), max(abs(accel_long)) FROM telemetry GROUP BY 1",
        ),
        (
            "one hour (Hive partition pruning)",
            f"SELECT vehicle_id, avg(speed) FROM telemetry WHERE date = '{date}' "
            f"AND hour = {hour} GROUP BY 1",
        ),
        (
            "one vehicle (row-group min/max pruning)",
            f"SELECT count(*), avg(speed) FROM telemetry WHERE vehicle_id = '{vehicle}'",
        ),
        (
            "hard-brake candidates (predicate + sort)",
            "SELECT vehicle_id, ts, accel_long FROM telemetry WHERE accel_long < -4 "
            "ORDER BY accel_long LIMIT 100",
        ),
        (
            "10 s window rollup (streaming aggregate table)",
            "SELECT date_trunc('hour', window_start::TIMESTAMPTZ), sum(distance_m) "
            "FROM windows_10s GROUP BY 1",
        ),
    ]
    rows = []
    for name, q in queries:
        dt, _ = timed(lambda q=q: lake.arrow(q), repeat=5)
        rows.append([name, round(dt * 1000, 1)])
    total = lake.sql("SELECT count(*) FROM telemetry").fetchall()[0][0]
    return (
        f"DuckDB {duckdb.__version__} over the committed files ({total:,} telemetry rows in "
        f"{len(files)} files; the pruned hour holds {len(in_hour)}). Median of 5 warm runs, "
        "results materialised as Arrow.\n\n" + table(["query", "ms"], rows)
    )


def bench_mining(lake: Lake, labels) -> str:
    dt, sc = timed(lambda: mine_scenarios(lake))
    assert isinstance(sc, pa.Table)
    rows_scanned = lake.sql("SELECT count(*) FROM telemetry").fetchall()[0][0]
    dt_m, _ = timed(lambda: trip_metrics(lake))
    write_derived(lake.root, "ground_truth", ground_truth_table(labels))
    lake.refresh()
    ev = evaluate(lake).to_pylist()
    pr = [
        [
            r["scenario_type"],
            r["labels"],
            r["detections"],
            f"{r['tp_labels'] / r['labels']:.1%}" if r["labels"] else "–",
            f"{r['tp_detections'] / r['detections']:.1%}" if r["detections"] else "–",
        ]
        for r in ev
    ]
    return (
        f"Detection + rarity scoring over {rows_scanned:,} rows: {dt:.2f} s "
        f"({rows_scanned / dt:,.0f} rows/s, {sc.num_rows} scenarios). "
        f"Trip metrics (SQL): {dt_m:.2f} s.\n\n"
        "Accuracy against the injected ground truth (synthetic fleet, `clear` intensity, "
        "realistic transport faults; a label is recalled if a detection of the same type "
        "overlaps it within 1 s):\n\n"
        + table(["type", "labels", "detections", "recall", "precision"], pr)
    )


def bench_selection(scenarios: pa.Table) -> str:
    rows = []
    for gb in (0.5, 1, 2, 5, 10):
        cmp = compare(scenarios, SelectionConfig(budget_bytes=int(gb * 1e9))).to_pylist()
        by = {r["method"]: r for r in cmp}
        rows.append([gb] + [f"{by[m]['coverage']:.3f} ({by[m]['types_covered']})" for m in by])
    methods = list(by)
    quality = (
        "Coverage f(S)/f(all) of the type-balanced facility-location objective, with the "
        "number of scenario types represented in brackets, at several upload budgets:\n\n"
        + table(["budget GB", *methods], rows)
    )
    # scaling: resample the real table with feature jitter to grow n
    rng = np.random.default_rng(0)
    feats = [c for c in scenarios.column_names if scenarios.schema.field(c).type == pa.float64()]
    scale = []
    for n in (500, 2000, 8000, 20000):
        idx = rng.integers(0, scenarios.num_rows, n)
        t = scenarios.take(pa.array(idx))
        for c in feats:
            if c in ("duration_s", "priority", "severity", "rarity"):
                continue
            col = t.column(c).to_numpy(zero_copy_only=False)
            jitter = rng.normal(0, 0.05 * (np.nanstd(col) or 1.0), n)
            t = t.set_column(t.schema.get_field_index(c), c, pa.array(col + jitter))
        budget = int(0.2 * float(np.sum(t.column("duration_s").to_numpy() + 10) * 6e6))
        dt, s = timed(lambda t=t, b=budget: select(t, SelectionConfig(budget_bytes=b)))
        scale.append([n, len(s.indices), round(dt, 3), f"{s.coverage:.3f}"])  # type: ignore[attr-defined]
    return (
        quality
        + "\n\nScaling of the diverse selector (lazy cost-benefit greedy + best singleton). "
        "Inputs are the mined scenario table resampled with 5% feature jitter; the budget "
        "is 20% of the total clip bytes.\n\n"
        + table(["scenarios", "picked", "seconds", "coverage"], scale)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/benchmarks.md"))
    ap.add_argument("--vehicles", type=int, default=200)
    ap.add_argument("--minutes", type=int, default=20)
    ap.add_argument("--clones", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bootstrap", default=None, help="also benchmark through Kafka")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="fleetscope-bench-"))
    try:
        t0 = time.perf_counter()
        emissions, labels, stats = build_fleet(args.vehicles, args.minutes, args.clones, args.seed)
        gen_s = time.perf_counter() - t0
        sections = [
            (
                "Workload",
                f"{args.vehicles} synthetic vehicles × {args.minutes} min plus the comma2k19 "
                f"sample segment and {args.clones} amplified clones, 20 Hz, "
                f"{CHUNK_SAMPLES}-sample chunks, realistic faults "
                f"({stats.duplicates:,} duplicates, {stats.reordered:,} reordered, "
                f"{stats.late:,} late, {stats.corrupt:,} corrupt, {stats.invalid_rows:,} invalid "
                f"rows): {len(emissions):,} messages, {stats.samples:,} samples "
                f"(generated in {gen_s:.1f} s).",
            )
        ]
        print("ingest ...", file=sys.stderr)
        text, lake_root = bench_ingest(emissions, stats, work)
        sections.append(("Stream processing (in-process)", text))
        if args.bootstrap:
            print("kafka ...", file=sys.stderr)
            sections.append(
                ("End-to-end through Kafka", bench_kafka(emissions, stats, args.bootstrap, work))
            )
        lake = Lake(lake_root)
        print("queries ...", file=sys.stderr)
        sections.append(("SQL on the lake", bench_queries(lake, lake_root)))
        print("mining ...", file=sys.stderr)
        sections.append(("Scenario mining", bench_mining(lake, labels)))
        lake.refresh()
        print("selection ...", file=sys.stderr)
        scenarios = lake.arrow("SELECT * FROM scenarios ORDER BY rank")
        sections.append(("Budgeted selection", bench_selection(scenarios)))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    env = (
        f"{platform.platform()} · {platform.machine()} · {os.cpu_count()} CPUs · "
        f"Python {platform.python_version()} · DuckDB {duckdb.__version__} · "
        f"PyArrow {pa.__version__} · {datetime.now(UTC):%Y-%m-%d}"
    )
    body = "\n\n".join(f"## {h}\n\n{t}" for h, t in sections)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "# Benchmarks\n\nGenerated by `make bench` (`benchmarks/run.py`); "
        f"do not edit by hand.\n\n{env}\n\n{body}\n"
    )
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
