"""Vectorised detectors for safety-relevant and data-quality scenarios.

Each detector turns a per-sample predicate into time intervals (runs), with a minimum
duration and gap merging so sensor noise does not split or fake events. Thresholds are
grounded in common telematics / AV-safety practice:

==================  =======================================  ================  ========
scenario            trigger                                  min duration      extreme
==================  =======================================  ================  ========
hard_brake          a_long <= -4.0 m/s^2 (~0.4 g)             0.3 s             8 m/s^2
harsh_accel         a_long >= +3.0 m/s^2                      0.5 s             5 m/s^2
sharp_turn          |a_lat| >= 3.5 m/s^2                       0.5 s             7 m/s^2
evasive_swerve      |a_lat| >= 2.5 both directions <= 1.5 s   v >= 12 m/s       6 m/s^2
low_ttc / cut_in    TTC <= 2.0 s, v >= 8 m/s, closing > .5    0.2 s             0.5 s
tailgating          headway <= 0.8 s at v >= 10 m/s           3.0 s             0.3 s
sensor_dropout      sample gap > 0.5 s                        --                5 s
gnss_outage         no GNSS fix                               5.0 s             60 s
==================  =======================================  ================  ========

A low-TTC event is classified ``cut_in`` when the lead track appeared (or jumped >= 5 m
closer) within 1 s before the event, i.e. a vehicle entered the lane; otherwise it is
``low_ttc`` (closing on an existing lead). Close following within 10 s after a close lane
entry (a lead appearing, or jumping >= 5 m closer, at headway < 1 s) is imposed on the
driver, so it is not reported as tailgating.

Severity is mapped to [0.2, 1.0]: 0.2 at the trigger threshold, 1.0 at the "extreme"
value, linear in between, so severities are comparable across scenario types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from fleetscope.schema import SAMPLE_HZ

SCENARIO_TYPES: tuple[str, ...] = (
    "hard_brake",
    "harsh_accel",
    "sharp_turn",
    "evasive_swerve",
    "cut_in",
    "low_ttc",
    "tailgating",
    "sensor_dropout",
    "gnss_outage",
)
SAFETY_TYPES = frozenset(SCENARIO_TYPES) - {"sensor_dropout", "gnss_outage"}

FEATURES: tuple[str, ...] = (
    "duration_s",
    "speed_start",
    "speed_min",
    "speed_max",
    "accel_long_min",
    "accel_long_max",
    "accel_lat_abs_max",
    "jerk_abs_max",
    "yaw_rate_abs_max",
    "ttc_min",
    "headway_min",
)
TTC_CAP_S = 10.0
HEADWAY_CAP_S = 5.0


@dataclass(frozen=True)
class DetectorConfig:
    hard_brake: float = -4.0
    hard_brake_extreme: float = -8.0
    harsh_accel: float = 3.0
    harsh_accel_extreme: float = 5.0
    sharp_turn: float = 3.5
    sharp_turn_extreme: float = 7.0
    swerve_lat: float = 2.5
    swerve_extreme: float = 6.0
    swerve_max_gap_s: float = 1.5
    swerve_min_speed: float = 12.0
    ttc: float = 2.0
    ttc_extreme: float = 0.5
    ttc_min_speed: float = 8.0
    ttc_min_closing: float = 0.5
    cut_in_lookback_s: float = 1.0
    cut_in_jump_m: float = 5.0
    cut_in_recovery_s: float = 10.0  # close following this soon after a close entry is imposed
    imposed_entry_headway: float = 1.0  # a lead entering closer than this was imposed (s)
    headway: float = 0.8
    headway_extreme: float = 0.3
    headway_min_speed: float = 10.0
    tailgating_min_s: float = 3.0
    dropout_gap_s: float = 0.5
    dropout_extreme_s: float = 5.0
    gnss_outage_min_s: float = 5.0
    gnss_outage_extreme_s: float = 60.0
    context_s: float = 2.0  # padding around an event for feature extraction


@dataclass
class Scenario:
    scenario_id: str
    vehicle_id: str
    trip_id: str
    scenario_type: str
    start_ns: int
    end_ns: int
    severity: float
    peak: float  # the triggering signal's extreme value (units depend on type)
    features: dict[str, float] = field(default_factory=dict)


def scenario_id(vehicle_id: str, trip_id: str, kind: str, start_ns: int) -> str:
    return hashlib.sha1(f"{vehicle_id}|{trip_id}|{kind}|{start_ns}".encode()).hexdigest()[:16]


def severity(value: float, threshold: float, extreme: float) -> float:
    frac = (value - threshold) / (extreme - threshold)
    return float(0.2 + 0.8 * min(1.0, max(0.0, frac)))


# --- run utilities ---------------------------------------------------------------------
def runs(mask: np.ndarray) -> np.ndarray:
    """(k, 2) array of [start, end) index pairs of True runs."""
    if len(mask) == 0:
        return np.empty((0, 2), dtype=np.int64)
    m = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return edges.reshape(-1, 2)


def merge_runs(r: np.ndarray, t_ns: np.ndarray, max_gap_s: float) -> np.ndarray:
    if len(r) <= 1:
        return r
    out = [list(r[0])]
    for a, b in r[1:]:
        if (t_ns[a] - t_ns[out[-1][1] - 1]) / 1e9 <= max_gap_s:
            out[-1][1] = b
        else:
            out.append([a, b])
    return np.array(out, dtype=np.int64)


def min_duration(
    r: np.ndarray, t_ns: np.ndarray, min_s: float, contiguous: np.ndarray
) -> np.ndarray:
    """Keep runs lasting >= min_s that do not straddle a sample gap."""
    keep = []
    for a, b in r:
        dur = (t_ns[b - 1] - t_ns[a]) / 1e9 + 1.0 / SAMPLE_HZ
        if dur >= min_s and contiguous[a : b - 1].all():
            keep.append((a, b))
    return np.array(keep, dtype=np.int64).reshape(-1, 2)


# --- detection -------------------------------------------------------------------------
@dataclass
class _Trip:
    vehicle_id: str
    trip_id: str
    t: np.ndarray
    c: dict[str, np.ndarray]
    contiguous: np.ndarray  # contiguous[i]: sample i -> i+1 has no gap
    ttc: np.ndarray
    headway: np.ndarray
    jerk: np.ndarray


def _prepare(
    vehicle_id: str, trip_id: str, t_ns: np.ndarray, cols: dict[str, np.ndarray], cfg
) -> _Trip:
    dt = np.diff(t_ns) / 1e9
    contiguous = np.concatenate((dt <= 0.2, [True]))
    speed = cols["speed"]
    closing = -cols["lead_rel_speed"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing > cfg.ttc_min_closing, cols["lead_dist"] / closing, np.inf)
        headway = np.where(speed > 0.5, cols["lead_dist"] / speed, np.inf)
    ttc = np.where(np.isnan(ttc), np.inf, ttc)
    headway = np.where(np.isnan(headway), np.inf, headway)
    jerk = np.zeros(len(t_ns))
    if len(t_ns) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            j = np.diff(cols["accel_long"]) / np.maximum(dt, 1e-3)
        j[~contiguous[:-1]] = 0.0
        jerk[1:] = j
    return _Trip(vehicle_id, trip_id, t_ns, cols, contiguous, ttc, headway, jerk)


def _features(tr: _Trip, a: int, b: int, cfg: DetectorConfig) -> dict[str, float]:
    pad = int(cfg.context_s * SAMPLE_HZ)
    lo, hi = max(0, a - pad), min(len(tr.t), b + pad)
    c = tr.c
    return {
        "duration_s": float((tr.t[b - 1] - tr.t[a]) / 1e9 + 1.0 / SAMPLE_HZ),
        "speed_start": float(c["speed"][a]),
        "speed_min": float(c["speed"][lo:hi].min()),
        "speed_max": float(c["speed"][lo:hi].max()),
        "accel_long_min": float(c["accel_long"][lo:hi].min()),
        "accel_long_max": float(c["accel_long"][lo:hi].max()),
        "accel_lat_abs_max": float(np.abs(c["accel_lat"][lo:hi]).max()),
        "jerk_abs_max": float(np.abs(tr.jerk[lo:hi]).max()),
        "yaw_rate_abs_max": float(np.abs(c["yaw_rate"][lo:hi]).max()),
        "ttc_min": float(min(TTC_CAP_S, tr.ttc[lo:hi].min())),
        "headway_min": float(min(HEADWAY_CAP_S, tr.headway[lo:hi].min())),
    }


def _make(tr: _Trip, kind: str, a: int, b: int, sev: float, peak: float, cfg) -> Scenario:
    start = int(tr.t[a])
    return Scenario(
        scenario_id=scenario_id(tr.vehicle_id, tr.trip_id, kind, start),
        vehicle_id=tr.vehicle_id,
        trip_id=tr.trip_id,
        scenario_type=kind,
        start_ns=start,
        end_ns=int(tr.t[b - 1]),
        severity=sev,
        peak=peak,
        features=_features(tr, a, b, cfg),
    )


def _overlaps(a: int, b: int, taken: list[tuple[int, int]]) -> bool:
    return any(a < y and x < b for x, y in taken)


def detect_trip(
    vehicle_id: str,
    trip_id: str,
    t_ns: np.ndarray,
    cols: dict[str, np.ndarray],
    cfg: DetectorConfig | None = None,
) -> list[Scenario]:
    """Detect scenarios in one trip. ``cols`` must be time-ordered float arrays."""
    cfg = cfg or DetectorConfig()
    if len(t_ns) < 2:
        return []
    tr = _prepare(vehicle_id, trip_id, t_ns, cols, cfg)
    c, t = tr.c, tr.t
    out: list[Scenario] = []

    def interval_runs(mask: np.ndarray, min_s: float, merge_s: float) -> np.ndarray:
        return min_duration(merge_runs(runs(mask), t, merge_s), t, min_s, tr.contiguous)

    # longitudinal
    for a, b in interval_runs(c["accel_long"] <= cfg.hard_brake, 0.3, 0.3):
        peak = float(c["accel_long"][a:b].min())
        out.append(
            _make(
                tr,
                "hard_brake",
                a,
                b,
                severity(-peak, -cfg.hard_brake, -cfg.hard_brake_extreme),
                peak,
                cfg,
            )
        )
    for a, b in interval_runs(c["accel_long"] >= cfg.harsh_accel, 0.5, 0.3):
        peak = float(c["accel_long"][a:b].max())
        out.append(
            _make(
                tr,
                "harsh_accel",
                a,
                b,
                severity(peak, cfg.harsh_accel, cfg.harsh_accel_extreme),
                peak,
                cfg,
            )
        )

    # lateral: swerves first (a sign reversal of strong lateral accel), then sharp turns
    lat = c["accel_lat"]
    fast = c["speed"] >= cfg.swerve_min_speed
    pos = interval_runs((lat >= cfg.swerve_lat) & fast, 0.2, 0.2)
    neg = interval_runs((lat <= -cfg.swerve_lat) & fast, 0.2, 0.2)
    lobes = sorted([(int(a), int(b), 1) for a, b in pos] + [(int(a), int(b), -1) for a, b in neg])
    swerves: list[tuple[int, int]] = []
    i = 0
    while i < len(lobes) - 1:
        (a1, b1, s1), (a2, b2, s2) = lobes[i], lobes[i + 1]
        if (
            s1 != s2
            and (t[a2] - t[b1 - 1]) / 1e9 <= cfg.swerve_max_gap_s
            and tr.contiguous[a1 : b2 - 1].all()
        ):
            peak = float(np.abs(lat[a1:b2]).max())
            out.append(
                _make(
                    tr,
                    "evasive_swerve",
                    a1,
                    b2,
                    severity(peak, cfg.swerve_lat, cfg.swerve_extreme),
                    peak,
                    cfg,
                )
            )
            swerves.append((a1, b2))
            i += 2
        else:
            i += 1
    for a, b in interval_runs(np.abs(lat) >= cfg.sharp_turn, 0.5, 0.3):
        if _overlaps(a, b, swerves):
            continue
        peak = float(np.abs(lat[a:b]).max())
        out.append(
            _make(
                tr,
                "sharp_turn",
                a,
                b,
                severity(peak, cfg.sharp_turn, cfg.sharp_turn_extreme),
                peak,
                cfg,
            )
        )

    # time to collision
    ttc_mask = (tr.ttc <= cfg.ttc) & (c["speed"] >= cfg.ttc_min_speed)
    lead = c["lead_dist"]
    look = int(cfg.cut_in_lookback_s * SAMPLE_HZ)
    for a, b in interval_runs(ttc_mask, 0.2, 0.5):
        window = lead[max(0, a - look) : a + 1]
        appeared = a > 0 and bool(np.isnan(window[:-1]).any())
        steps = np.diff(window)
        steps = steps[np.isfinite(steps)]
        jumped = steps.size > 0 and float(steps.min()) <= -cfg.cut_in_jump_m
        kind = "cut_in" if (appeared or jumped) else "low_ttc"
        peak = float(tr.ttc[a:b].min())
        out.append(_make(tr, kind, a, b, severity(-peak, -cfg.ttc, -cfg.ttc_extreme), peak, cfg))

    # following distance
    hw_mask = (tr.headway <= cfg.headway) & (c["speed"] >= cfg.headway_min_speed)
    # A lead entering the lane already close (a lane entry at headway < 1 s) imposes the
    # gap on the driver: close following during the recovery period after such an entry is
    # not tailgating, whether or not TTC got low. A lead that appears at a normal headway
    # and is then closed on is the driver's choice, as is staying close after recovery.
    prev = np.concatenate(([np.nan], lead[:-1]))
    entry = ~np.isnan(lead) & (np.isnan(prev) | (lead - prev <= -cfg.cut_in_jump_m))
    entry &= tr.headway < cfg.imposed_entry_headway
    entry[0] = False
    recovery = int(cfg.cut_in_recovery_s * SAMPLE_HZ)
    for e in np.flatnonzero(entry):
        hw_mask[e : e + recovery] = False
    for a, b in interval_runs(hw_mask, cfg.tailgating_min_s, 1.0):
        peak = float(tr.headway[a:b].min())
        out.append(
            _make(
                tr,
                "tailgating",
                a,
                b,
                severity(-peak, -cfg.headway, -cfg.headway_extreme),
                peak,
                cfg,
            )
        )

    # data quality
    gaps = np.flatnonzero(np.diff(t) / 1e9 > cfg.dropout_gap_s)
    for g in gaps.tolist():
        gap_s = float((t[g + 1] - t[g]) / 1e9)
        s = Scenario(
            scenario_id=scenario_id(vehicle_id, trip_id, "sensor_dropout", int(t[g])),
            vehicle_id=vehicle_id,
            trip_id=trip_id,
            scenario_type="sensor_dropout",
            start_ns=int(t[g]),
            end_ns=int(t[g + 1]),
            severity=severity(gap_s, cfg.dropout_gap_s, cfg.dropout_extreme_s),
            peak=gap_s,
            features=_features(tr, int(g), int(g) + 2, cfg),
        )
        s.features["duration_s"] = gap_s
        out.append(s)
    no_fix = np.isnan(c["lat"])
    if not no_fix.all():  # a trip without any GNSS is a config issue, not an outage
        for a, b in min_duration(runs(no_fix), t, cfg.gnss_outage_min_s, np.ones(len(t), bool)):
            dur = float((t[b - 1] - t[a]) / 1e9)
            out.append(
                _make(
                    tr,
                    "gnss_outage",
                    a,
                    b,
                    severity(dur, cfg.gnss_outage_min_s, cfg.gnss_outage_extreme_s),
                    dur,
                    cfg,
                )
            )

    out.sort(key=lambda s: s.start_ns)
    return out
