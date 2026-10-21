"""Driving and fleet metrics, computed in SQL over the lake.

Definitions (all per trip unless stated otherwise):

``distance_km``
    Trapezoidal integral of speed over consecutive samples. Steps longer than
    ``MAX_STEP_S`` (sensor dropouts) are excluded rather than bridged, so a gap never
    adds distance it did not observe.
``observed_s`` / ``completeness``
    Sum of the included steps, and accepted samples / samples expected at 20 Hz between
    the trip's first and last sample. Quarantined rows are counted separately.
``accel_long_abs_p95`` / ``accel_lat_abs_p95`` / ``jerk_abs_p95``
    95th percentiles of |a_long|, |a_lat| and |d a_long / dt| -- a driving-smoothness
    profile that, unlike event counts, is defined for every trip.
``close_following_frac``
    Share of samples at >= 10 m/s with a lead vehicle closer than 1 s of headway.
``gnss_availability``
    Share of samples with a GNSS fix.
``safety_events_per_100km``
    Safety-relevant scenarios (excludes data-quality types) per 100 km driven.

Fleet totals aggregate these with distance or time weighting, never by averaging
per-trip ratios (which would over-weight short trips).
"""

from __future__ import annotations

import pyarrow as pa

from fleetscope.lake.sql import Lake, write_derived
from fleetscope.scenarios.detectors import SAFETY_TYPES
from fleetscope.schema import SAMPLE_HZ

MAX_STEP_S = 0.2
CLOSE_HEADWAY_S = 1.0
FOLLOWING_MIN_SPEED = 10.0

_SAFETY_LIST = ", ".join(f"'{t}'" for t in sorted(SAFETY_TYPES))


def _trip_sql(lake: Lake) -> str:
    events = (
        f"""
        SELECT vehicle_id, trip_id,
               count(*) FILTER (scenario_type IN ({_SAFETY_LIST})) AS safety_events,
               count(*) FILTER (scenario_type NOT IN ({_SAFETY_LIST})) AS data_quality_events
        FROM scenarios GROUP BY 1, 2
        """
        if lake.has("scenarios")
        else "SELECT NULL::VARCHAR AS vehicle_id, NULL::VARCHAR AS trip_id, "
        "0 AS safety_events, 0 AS data_quality_events"
    )
    return f"""
    WITH steps AS (
        SELECT vehicle_id, trip_id, source, ts, speed, accel_long, accel_lat, lead_dist, lat,
               (epoch_ns(ts) - lag(epoch_ns(ts)) OVER w) / 1e9 AS dt,
               lag(speed) OVER w AS prev_speed,
               lag(accel_long) OVER w AS prev_accel
        FROM telemetry
        WINDOW w AS (PARTITION BY vehicle_id, trip_id ORDER BY ts)
    ), trips AS (
        SELECT vehicle_id, trip_id, any_value(source) AS source,
               min(ts) AS start_ts, max(ts) AS end_ts, count(*) AS samples,
               (epoch_ns(max(ts)) - epoch_ns(min(ts))) / 1e9 + {1 / SAMPLE_HZ} AS duration_s,
               coalesce(sum((speed + prev_speed) / 2 * dt) FILTER (dt <= {MAX_STEP_S}), 0)
                   / 1000 AS distance_km,
               coalesce(sum(dt) FILTER (dt <= {MAX_STEP_S}), 0) AS observed_s,
               avg(speed) AS speed_mean,
               max(speed) AS speed_max,
               quantile_cont(abs(accel_long), 0.95) AS accel_long_abs_p95,
               quantile_cont(abs(accel_lat), 0.95) AS accel_lat_abs_p95,
               quantile_cont(abs(accel_long - prev_accel) / dt, 0.95)
                   FILTER (dt <= {MAX_STEP_S}) AS jerk_abs_p95,
               avg((lead_dist IS NOT NULL)::INT) AS lead_present_frac,
               avg((coalesce(lead_dist / speed, 1e9) < {CLOSE_HEADWAY_S})::INT)
                   FILTER (speed >= {FOLLOWING_MIN_SPEED}) AS close_following_frac,
               avg((lat IS NOT NULL)::INT) AS gnss_availability
        FROM steps GROUP BY vehicle_id, trip_id
    ), quarantined AS (
        SELECT vehicle_id, trip_id, count(*) AS quarantined_rows FROM quarantine GROUP BY 1, 2
    ), events AS ({events})
    SELECT t.*,
           least(1.0, t.samples / (t.duration_s * {SAMPLE_HZ})) AS completeness,
           coalesce(q.quarantined_rows, 0) AS quarantined_rows,
           coalesce(e.safety_events, 0) AS safety_events,
           coalesce(e.data_quality_events, 0) AS data_quality_events,
           CASE WHEN t.distance_km > 0
                THEN coalesce(e.safety_events, 0) / t.distance_km * 100 END
               AS safety_events_per_100km
    FROM trips t
    LEFT JOIN quarantined q USING (vehicle_id, trip_id)
    LEFT JOIN events e USING (vehicle_id, trip_id)
    ORDER BY t.vehicle_id, t.trip_id
    """


def trip_metrics(lake: Lake, persist: bool = True) -> pa.Table:
    table = lake.arrow(_trip_sql(lake))
    if persist:
        write_derived(lake.root, "trip_metrics", table)
        lake.refresh()
    return table


def fleet_summary(lake: Lake) -> dict[str, float]:
    """Fleet-level KPIs. Requires ``trip_metrics`` (see :func:`trip_metrics`)."""
    row = lake.sql(
        f"""
        SELECT count(DISTINCT vehicle_id) AS vehicles,
               count(*) AS trips,
               sum(distance_km) AS distance_km,
               sum(observed_s) / 3600 AS driving_hours,
               sum(samples) AS samples,
               sum(quarantined_rows) AS quarantined_rows,
               sum(quarantined_rows) / (sum(samples) + sum(quarantined_rows)) AS quarantine_rate,
               sum(samples) / sum(duration_s * {SAMPLE_HZ}) AS completeness,
               sum(safety_events) AS safety_events,
               sum(safety_events) / nullif(sum(distance_km), 0) * 1000
                   AS safety_events_per_1000km,
               sum(close_following_frac * observed_s) / nullif(sum(observed_s), 0)
                   AS close_following_frac,
               sum(gnss_availability * samples) / sum(samples) AS gnss_availability
        FROM trip_metrics
        """
    ).fetchall()[0]
    names = [
        "vehicles",
        "trips",
        "distance_km",
        "driving_hours",
        "samples",
        "quarantined_rows",
        "quarantine_rate",
        "completeness",
        "safety_events",
        "safety_events_per_1000km",
        "close_following_frac",
        "gnss_availability",
    ]
    return {k: float(v) if v is not None else float("nan") for k, v in zip(names, row, strict=True)}


def event_rates(lake: Lake) -> pa.Table:
    """Scenarios per 1000 km by type and source (real vs synthetic driving differ)."""
    return lake.arrow(
        """
        WITH km AS (SELECT source, sum(distance_km) AS km FROM trip_metrics GROUP BY 1)
        SELECT s.source, s.scenario_type, count(*) AS events,
               count(*) / nullif(any_value(km.km), 0) * 1000 AS per_1000km,
               avg(s.severity) AS severity_mean
        FROM scenarios s JOIN km USING (source)
        GROUP BY 1, 2 ORDER BY 1, per_1000km DESC
        """
    )


def fleet_hourly(lake: Lake) -> pa.Table:
    """Fleet activity per event-time hour from the streaming 10 s window aggregates."""
    return lake.arrow(
        """
        SELECT date_trunc('hour', window_start::TIMESTAMPTZ) AS hour,
               count(DISTINCT vehicle_id) AS active_vehicles,
               sum(distance_m) / 1000 AS distance_km,
               sum(samples) AS samples,
               max(speed_max) AS speed_max,
               min(accel_long_min) AS accel_long_min
        FROM windows_10s GROUP BY 1 ORDER BY 1
        """
    )
