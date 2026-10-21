"""Batch scenario mining over the lake: detect -> score -> persist -> evaluate."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from fleetscope.lake.sql import Lake, write_derived
from fleetscope.scenarios.detectors import FEATURES, DetectorConfig, Scenario, detect_trip
from fleetscope.scenarios.rarity import score_scenarios
from fleetscope.schema import SIGNALS
from fleetscope.sources.synthetic import GroundTruthEvent


@dataclass
class TripData:
    vehicle_id: str
    trip_id: str
    source: str
    t_ns: np.ndarray
    cols: dict[str, np.ndarray]


def _split(batch: pa.Table) -> list[tuple[int, int]]:
    """[start, end) row ranges of consecutive (vehicle_id, trip_id) groups."""
    n = batch.num_rows
    if n == 0:
        return []
    v, t = batch.column("vehicle_id"), batch.column("trip_id")
    change = pc.or_(
        pc.not_equal(v.slice(1), v.slice(0, n - 1)), pc.not_equal(t.slice(1), t.slice(0, n - 1))
    )
    cuts = (np.flatnonzero(change.to_numpy(zero_copy_only=False)) + 1).tolist()
    bounds = [0, *cuts, n]
    return list(itertools.pairwise(bounds))


def _trip(batch: pa.Table, a: int, b: int) -> TripData:
    part = batch.slice(a, b - a)
    return TripData(
        vehicle_id=part.column("vehicle_id")[0].as_py(),
        trip_id=part.column("trip_id")[0].as_py(),
        source=part.column("source")[0].as_py(),
        t_ns=part.column("t_ns").to_numpy(),
        cols={s: part.column(s).to_numpy(zero_copy_only=False).astype(np.float64) for s in SIGNALS},
    )


def iter_trips(lake: Lake, where: str = "", batch_rows: int = 1_000_000) -> Iterator[TripData]:
    """Stream time-ordered trips out of the lake without materialising the whole table."""
    reader = lake.reader(
        f"SELECT vehicle_id, trip_id, source, epoch_ns(ts) AS t_ns, {', '.join(SIGNALS)} "
        f"FROM telemetry {where} ORDER BY vehicle_id, trip_id, ts",
        batch_rows,
    )
    carry: pa.Table | None = None
    for rb in reader:
        batch = pa.Table.from_batches([rb])
        if carry is not None:
            batch = pa.concat_tables([carry, batch])
        groups = _split(batch)
        for a, b in groups[:-1]:
            yield _trip(batch, a, b)
        a, b = groups[-1]
        carry = batch.slice(a, b - a)
    if carry is not None and carry.num_rows:
        yield _trip(carry, 0, carry.num_rows)


def scenarios_to_table(scenarios: list[Scenario], sources: dict[str, str]) -> pa.Table:
    ts = pa.timestamp("ns", tz="UTC")
    cols: dict[str, pa.Array] = {
        "scenario_id": pa.array([s.scenario_id for s in scenarios], pa.string()),
        "vehicle_id": pa.array([s.vehicle_id for s in scenarios], pa.string()),
        "trip_id": pa.array([s.trip_id for s in scenarios], pa.string()),
        "source": pa.array([sources.get(s.trip_id, "") for s in scenarios], pa.string()),
        "scenario_type": pa.array([s.scenario_type for s in scenarios], pa.string()),
        "start_ts": pa.array([s.start_ns for s in scenarios], ts),
        "end_ts": pa.array([s.end_ns for s in scenarios], ts),
        "severity": pa.array([s.severity for s in scenarios], pa.float64()),
        "peak": pa.array([s.peak for s in scenarios], pa.float64()),
    }
    for f in FEATURES:
        cols[f] = pa.array([s.features[f] for s in scenarios], pa.float64())
    return pa.table(cols)


def mine_scenarios(lake: Lake, cfg: DetectorConfig | None = None, persist: bool = True) -> pa.Table:
    found: list[Scenario] = []
    sources: dict[str, str] = {}
    for trip in iter_trips(lake):
        sources[trip.trip_id] = trip.source
        found.extend(detect_trip(trip.vehicle_id, trip.trip_id, trip.t_ns, trip.cols, cfg))
    table = score_scenarios(scenarios_to_table(found, sources))
    if persist:
        write_derived(lake.root, "scenarios", table)
        lake.refresh()
    return table


# --- evaluation against synthetic ground truth -------------------------------------------
# Labels use the same type names as detections (synthetic never labels ``low_ttc``: a
# low TTC on an existing lead is never injected, so any such detection is a false positive).
def ground_truth_table(events: list[GroundTruthEvent]) -> pa.Table:
    ts = pa.timestamp("ns", tz="UTC")
    return pa.table(
        {
            "vehicle_id": pa.array([e.vehicle_id for e in events], pa.string()),
            "trip_id": pa.array([e.trip_id for e in events], pa.string()),
            "scenario_type": pa.array([e.scenario_type for e in events], pa.string()),
            "start_ts": pa.array([e.start_ns for e in events], ts),
            "end_ts": pa.array([e.end_ns for e in events], ts),
        }
    )


def evaluate(lake: Lake, tolerance_s: float = 1.0) -> pa.Table:
    """Event-level precision / recall per scenario type on synthetic trips.

    A detection matches a label of the corresponding type on the same trip when their
    intervals overlap (each widened by ``tolerance_s``). Precision counts detections on
    synthetic trips only (real-data trips have no labels).
    """
    if not (lake.has("ground_truth") and lake.has("scenarios")):
        raise RuntimeError("need both 'ground_truth' and 'scenarios' tables")
    tol = int(tolerance_s * 1e9)
    return lake.arrow(
        f"""
        WITH gt AS (
            SELECT *, epoch_ns(start_ts) AS s, epoch_ns(end_ts) AS e FROM ground_truth
        ), det AS (
            SELECT *, epoch_ns(start_ts) AS s, epoch_ns(end_ts) AS e
            FROM scenarios WHERE source = 'synthetic'
        ), types AS (
            SELECT scenario_type FROM gt UNION SELECT scenario_type FROM det
        ), gt_hit AS (
            SELECT g.scenario_type, g.trip_id, g.s,
                   bool_or(d.scenario_id IS NOT NULL) AS hit
            FROM gt g LEFT JOIN det d
              ON d.trip_id = g.trip_id AND d.scenario_type = g.scenario_type
             AND d.s <= g.e + {tol} AND g.s <= d.e + {tol}
            GROUP BY ALL
        ), det_hit AS (
            SELECT d.scenario_type, d.scenario_id,
                   bool_or(g.trip_id IS NOT NULL) AS hit
            FROM det d LEFT JOIN gt g
              ON d.trip_id = g.trip_id AND d.scenario_type = g.scenario_type
             AND d.s <= g.e + {tol} AND g.s <= d.e + {tol}
            GROUP BY ALL
        ), labels AS (
            SELECT scenario_type, count(*) AS n, count(*) FILTER (hit) AS tp
            FROM gt_hit GROUP BY 1
        ), dets AS (
            SELECT scenario_type, count(*) AS n, count(*) FILTER (hit) AS tp
            FROM det_hit GROUP BY 1
        )
        SELECT t.scenario_type,
               coalesce(l.n, 0) AS labels, coalesce(d.n, 0) AS detections,
               coalesce(l.tp, 0) AS tp_labels, coalesce(d.tp, 0) AS tp_detections
        FROM types t LEFT JOIN labels l USING (scenario_type)
        LEFT JOIN dets d USING (scenario_type) ORDER BY 1
        """
    )
