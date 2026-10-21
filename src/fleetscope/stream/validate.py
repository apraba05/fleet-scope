"""Chunk- and row-level validation.

Two failure granularities, routed differently:

* **Chunk-level** (undecodable payload, unsupported schema version, missing identity,
  mismatched column lengths): the whole message is unusable -> dead-letter (``dlq``).
* **Row-level** (one sample out of physical bounds, NaN in a required signal,
  non-monotonic timestamp): only that sample is quarantined (``quarantine`` table, with
  every failed rule recorded); the rest of the chunk is kept.

Row checks are vectorised over a whole batch; each rule sets one bit in a uint32 mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fleetscope.schema import (
    MAX_SAMPLES_PER_CHUNK,
    REQUIRED_SIGNALS,
    SIGNALS,
    SUPPORTED_SCHEMA_VERSIONS,
    TelemetryChunk,
)

ROW_RULES: tuple[str, ...] = (
    "ts_out_of_range",
    "ts_not_increasing",
    "required_not_finite",
    "speed_out_of_range",
    "accel_long_out_of_range",
    "accel_lat_out_of_range",
    "yaw_rate_out_of_range",
    "steering_out_of_range",
    "lat_out_of_range",
    "lon_out_of_range",
    "gnss_partial_fix",
    "lead_out_of_range",
)
_BIT = {name: np.uint32(1 << i) for i, name in enumerate(ROW_RULES)}


@dataclass(frozen=True)
class ValidationConfig:
    min_ts_ns: int = 1_420_070_400_000_000_000  # 2015-01-01T00:00:00Z
    max_future_skew_s: float = 300.0
    speed_max: float = 90.0  # m/s (~324 km/h)
    accel_long_abs_max: float = 15.0  # m/s^2 (~1.5 g)
    accel_lat_abs_max: float = 15.0
    yaw_rate_abs_max: float = 3.0  # rad/s
    steering_abs_max: float = 1000.0  # deg (steering wheel lock-to-lock < 2 turns)
    lead_dist_max: float = 300.0  # m (radar range)
    lead_rel_speed_abs_max: float = 80.0  # m/s


def validate_header(chunk: TelemetryChunk) -> str | None:
    """Return a chunk-level rejection reason, or None if the chunk is structurally sound."""
    if chunk.v not in SUPPORTED_SCHEMA_VERSIONS:
        return "unsupported_schema_version"
    if not chunk.vehicle_id or not chunk.trip_id:
        return "missing_identity"
    if chunk.seq < 0:
        return "negative_seq"
    n = len(chunk.t_ns)
    if n == 0:
        return "empty_chunk"
    if n > MAX_SAMPLES_PER_CHUNK:
        return "chunk_too_large"
    for name in SIGNALS:
        if len(getattr(chunk, name)) != n:
            return "length_mismatch"
    return None


def validate_rows(
    t_ns: np.ndarray,
    cols: dict[str, np.ndarray],
    chunk_starts: np.ndarray,
    ref_ns: np.ndarray | int,
    cfg: ValidationConfig,
) -> np.ndarray:
    """Return a uint32 reason bitmask per row (0 = valid).

    ``chunk_starts`` marks rows that begin a chunk; timestamp monotonicity is only
    checked within a chunk (ordering across chunks is the window operator's concern).
    ``ref_ns`` is the "now" for the future-timestamp rule. The processor passes each row's
    broker timestamp rather than the wall clock so that validation is deterministic when
    a partition is replayed after a crash.
    """
    n = len(t_ns)
    mask = np.zeros(n, dtype=np.uint32)

    def flag(rule: str, bad: np.ndarray) -> None:
        mask[bad] |= _BIT[rule]

    flag(
        "ts_out_of_range",
        (t_ns < cfg.min_ts_ns) | (t_ns > ref_ns + int(cfg.max_future_skew_s * 1e9)),
    )
    not_inc = np.zeros(n, dtype=bool)
    not_inc[1:] = t_ns[1:] <= t_ns[:-1]
    not_inc[chunk_starts] = False
    flag("ts_not_increasing", not_inc)

    req_bad = np.zeros(n, dtype=bool)
    for name in REQUIRED_SIGNALS:
        req_bad |= ~np.isfinite(cols[name])
    flag("required_not_finite", req_bad)

    # Range rules for required signals only fire on finite values (non-finite is already
    # its own rule). For optional signals NaN means "absent" and compares False; +-inf is
    # out of range.
    def exceeds(name: str, limit: float) -> np.ndarray:
        x = cols[name]
        return np.isfinite(x) & (np.abs(x) > limit)

    with np.errstate(invalid="ignore"):
        s = cols["speed"]
        flag("speed_out_of_range", np.isfinite(s) & ((s < 0) | (s > cfg.speed_max)))
        flag("accel_long_out_of_range", exceeds("accel_long", cfg.accel_long_abs_max))
        flag("accel_lat_out_of_range", exceeds("accel_lat", cfg.accel_lat_abs_max))
        flag("yaw_rate_out_of_range", exceeds("yaw_rate", cfg.yaw_rate_abs_max))
        flag("steering_out_of_range", exceeds("steering_deg", cfg.steering_abs_max))
        lat, lon = cols["lat"], cols["lon"]
        flag("lat_out_of_range", np.abs(lat) > 90)
        flag("lon_out_of_range", np.abs(lon) > 180)
        flag("gnss_partial_fix", np.isnan(lat) != np.isnan(lon))
        d, r = cols["lead_dist"], cols["lead_rel_speed"]
        flag(
            "lead_out_of_range",
            (d < 0) | (d > cfg.lead_dist_max) | (np.abs(r) > cfg.lead_rel_speed_abs_max),
        )
    return mask


def reasons_to_strings(mask: np.ndarray) -> list[str]:
    """Render bitmasks as '|'-joined rule names (only called for the rare invalid rows)."""
    cache: dict[int, str] = {}
    out = []
    for m in mask.tolist():
        if m not in cache:
            cache[m] = "|".join(r for r in ROW_RULES if m & int(_BIT[r]))
        out.append(cache[m])
    return out


def reason_counts(mask: np.ndarray) -> dict[str, int]:
    return {r: int(np.count_nonzero(mask & _BIT[r])) for r in ROW_RULES if np.any(mask & _BIT[r])}
