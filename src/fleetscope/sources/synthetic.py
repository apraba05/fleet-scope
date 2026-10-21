"""Physics-based synthetic fleet with ground-truth scenario labels.

Real fleet data has no labels for "this was a near miss", so detector quality cannot be
measured on it directly. This generator produces kinematically consistent drives
(speed integrates acceleration, lateral acceleration = v * yaw_rate, radar gap
integrates relative speed) with *injected* safety-relevant scenarios whose exact time
intervals are recorded. That lets the pipeline report detector precision / recall.

Normal driving is kept inside comfort envelopes (|a_long| <= ~2 m/s^2, |a_lat| <= ~2
m/s^2, time headway >= 1 s) so that, by construction, detector hits outside injected
intervals are false positives.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from fleetscope.schema import SAMPLE_HZ, Track

SCENARIO_TYPES: tuple[str, ...] = (
    "hard_brake",
    "harsh_accel",
    "sharp_turn",
    "evasive_swerve",
    "cut_in",
    "tailgating",
    "sensor_dropout",
    "gnss_outage",
)

# Events per driving hour for an average (aggressiveness = 1) driver. Deliberately far
# above real-world rates so a demo-sized fleet contains enough of each type.
DEFAULT_RATES_PER_HOUR: dict[str, float] = {
    "hard_brake": 2.0,
    "harsh_accel": 1.5,
    "sharp_turn": 1.0,
    "evasive_swerve": 0.3,
    "cut_in": 1.0,
    "tailgating": 0.8,
    "sensor_dropout": 0.5,
    "gnss_outage": 0.4,
}

WHEELBASE_M = 2.9
STEERING_RATIO = 15.0
LONGITUDINAL_EVENTS = {"hard_brake", "harsh_accel", "cut_in"}


@dataclass(frozen=True)
class GroundTruthEvent:
    vehicle_id: str
    trip_id: str
    scenario_type: str
    start_ns: int
    end_ns: int


@dataclass
class FleetConfig:
    n_vehicles: int = 50
    trips_per_vehicle: int = 1
    trip_minutes: float = 10.0
    start_time_ns: int = 1_790_863_200_000_000_000  # 2026-10-01T14:00:00Z
    seed: int = 7
    hz: int = SAMPLE_HZ
    rates_per_hour: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RATES_PER_HOUR))
    urban_fraction: float = 0.4


@dataclass
class _Event:
    kind: str
    start: int  # sample index
    length: int  # samples
    params: dict[str, float]
    accepted: bool = True


def _piecewise(rng: np.random.Generator, n: int, mean_len: float, sampler) -> np.ndarray:
    out = np.empty(n)
    i = 0
    while i < n:
        length = max(1, int(rng.exponential(mean_len)))
        out[i : i + length] = sampler()
        i += length
    return out


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _ou(rng: np.random.Generator, n: int, theta: float, sigma: float, dt: float) -> np.ndarray:
    """Zero-mean Ornstein-Uhlenbeck noise."""
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.normal(0, sigma * math.sqrt(dt), n)
    decay = 1 - theta * dt
    for i in range(1, n):
        x[i] = x[i - 1] * decay + eps[i]
    return x


def _schedule_events(
    rng: np.random.Generator, n: int, hz: int, rates: dict[str, float], aggressiveness: float
) -> list[_Event]:
    hours = n / hz / 3600
    durations_s = {
        "hard_brake": (1.0, 2.2),
        "harsh_accel": (1.5, 3.0),
        "sharp_turn": (2.0, 4.0),
        "evasive_swerve": (1.6, 2.6),
        "cut_in": (3.0, 4.0),
        "tailgating": (6.0, 15.0),
        "sensor_dropout": (0.7, 3.0),
        "gnss_outage": (8.0, 30.0),
    }
    candidates: list[_Event] = []
    for kind, rate in rates.items():
        scale = aggressiveness if kind not in ("sensor_dropout", "gnss_outage") else 1.0
        k = rng.poisson(rate * scale * hours)
        for _ in range(k):
            lo, hi = durations_s[kind]
            length = int(rng.uniform(lo, hi) * hz)
            start = int(rng.uniform(30 * hz, max(30 * hz + 1, n - length - 10 * hz)))
            candidates.append(_Event(kind, start, length, {}))
    # Keep events at least 10 s apart so labels are unambiguous.
    candidates.sort(key=lambda e: e.start)
    kept: list[_Event] = []
    for ev in candidates:
        if ev.start + ev.length >= n:
            continue
        if kept and ev.start < kept[-1].start + kept[-1].length + 10 * hz:
            continue
        kept.append(ev)
    for ev in kept:
        if ev.kind == "hard_brake":
            ev.params = {"peak": -rng.uniform(4.8, 8.0)}
        elif ev.kind == "harsh_accel":
            ev.params = {"peak": rng.uniform(3.6, 4.6)}
        elif ev.kind == "sharp_turn":
            ev.params = {"a_lat": rng.choice([-1, 1]) * rng.uniform(4.2, 6.0)}
        elif ev.kind == "evasive_swerve":
            ev.params = {"a_lat": rng.choice([-1, 1]) * rng.uniform(3.4, 5.0)}
        elif ev.kind == "cut_in":
            ev.params = {
                "ttc0": rng.uniform(0.8, 1.7),
                "closing0": rng.uniform(4.0, 8.0),
                "brake": -rng.uniform(2.2, 3.2),
            }
        elif ev.kind == "tailgating":
            ev.params = {"headway": rng.uniform(0.3, 0.6)}
    return kept


def _segment_gradient(x: np.ndarray, breaks: np.ndarray, dt: float) -> np.ndarray:
    """np.gradient within segments separated by ``breaks`` (bool, True = new segment)."""
    out = np.full(len(x), np.nan)
    starts = np.flatnonzero(breaks | np.r_[True, np.isnan(x[:-1]) != np.isnan(x[1:])])
    bounds = [*starts.tolist(), len(x)]
    for a, b in itertools.pairwise(bounds):
        seg = x[a:b]
        if b - a >= 2 and not np.isnan(seg[0]):
            out[a:b] = np.gradient(seg, dt)
        elif b - a == 1 and not np.isnan(seg[0]):
            out[a] = 0.0
    return out


def generate_trip(
    rng: np.random.Generator,
    vehicle_id: str,
    trip_id: str,
    t0_ns: int,
    duration_s: float,
    hz: int = SAMPLE_HZ,
    rates: dict[str, float] | None = None,
    aggressiveness: float = 1.0,
    urban: bool = False,
    origin: tuple[float, float] = (37.55, -122.2),
) -> tuple[Track, list[GroundTruthEvent]]:
    rates = DEFAULT_RATES_PER_HOUR if rates is None else rates
    n = int(duration_s * hz)
    dt = 1.0 / hz

    # ---- target speed: piecewise constant cruise targets, urban drives add stops ------
    if urban:
        target = _piecewise(
            rng, n, 25 * hz, lambda: rng.choice([0.0, rng.uniform(8, 16)], p=[0.3, 0.7])
        )
    else:
        target = _piecewise(rng, n, 40 * hz, lambda: rng.uniform(24, 32))
    target[: 5 * hz] = target[5 * hz]  # no stop right at departure

    events = _schedule_events(rng, n, hz, rates, aggressiveness)
    lon_override = np.full(n, np.nan)
    lon_event = np.full(n, -1, dtype=np.int64)
    for idx, ev in enumerate(events):
        if ev.kind not in LONGITUDINAL_EVENTS:
            continue
        L = ev.length
        if ev.kind == "cut_in":
            # Ego reacts 0.5 s after the cut-in with moderate braking for ~2 s.
            react = int(0.5 * hz)
            prof = np.zeros(L)
            prof[react : react + 2 * hz] = ev.params["brake"]
            prof[:react] = np.nan  # no override during reaction time
        else:
            ramp = max(1, int(0.3 * hz))
            prof = np.full(L, ev.params["peak"])
            prof[:ramp] *= np.linspace(0.3, 1.0, ramp)
            prof[-ramp:] *= np.linspace(1.0, 0.3, ramp)
        lon_override[ev.start : ev.start + L] = prof
        lon_event[ev.start : ev.start + L] = idx

    # ---- longitudinal dynamics (scalar loop: v depends on v) --------------------------
    noise = _ou(rng, n, theta=1.5, sigma=0.25, dt=dt)
    v = float(target[0]) if not urban else 0.0
    speed = np.empty(n)
    accel = np.empty(n)
    decided: dict[int, bool] = {}
    preconditions = {"hard_brake": 8.0, "harsh_accel": 0.0, "cut_in": 10.0}
    max_speed_for = {"harsh_accel": 20.0}
    k_gain, a_max, a_min = 0.35, 1.5, -2.0
    for i in range(n):
        a = k_gain * (target[i] - v)
        a = a_max if a > a_max else (a_min if a < a_min else a)
        a += noise[i]
        e = lon_event[i]
        if e >= 0:
            if e not in decided:
                ev = events[e]
                ok = v >= preconditions[ev.kind] and v <= max_speed_for.get(ev.kind, 1e9)
                decided[e] = ok
                ev.accepted = ok
            if decided[e] and not math.isnan(lon_override[i]):
                a = lon_override[i] + 0.1 * noise[i]
        v_new = v + a * dt
        if v_new < 0.0:
            v_new = 0.0
        accel[i] = (v_new - v) / dt
        speed[i] = v = v_new

    # ---- lateral: road curvature + lane changes + lateral events ----------------------
    max_lat_curve = 1.8
    kappa = _piecewise(
        rng,
        n,
        30 * hz,
        lambda: rng.choice([0.0, rng.uniform(-1, 1) / rng.uniform(150, 1200)], p=[0.5, 0.5]),
    )
    kappa = _smooth(kappa, 4 * hz)
    v_safe = np.maximum(speed, 1.0)
    kappa = np.clip(kappa, -max_lat_curve / v_safe**2, max_lat_curve / v_safe**2)
    yaw = speed * kappa
    # Normal lane changes: a one-period sine in yaw rate over 5 s displaces ~3.5 m.
    for _ in range(rng.poisson(n / hz / 120)):
        s = int(rng.uniform(0, n - 6 * hz))
        L = 5 * hz
        amp = rng.choice([-1, 1]) * 0.8 / max(speed[s], 5.0)  # peak lateral accel 0.8 m/s^2
        yaw[s : s + L] += amp * np.sin(np.linspace(0, 2 * np.pi, L))
    for ev in events:
        sl = slice(ev.start, ev.start + ev.length)
        if ev.kind == "sharp_turn":
            ev.accepted = bool(speed[ev.start] >= 6.0)
            if ev.accepted:
                bump = np.sin(np.linspace(0, np.pi, ev.length)) ** 0.5
                yaw[sl] = ev.params["a_lat"] * bump / v_safe[sl]
        elif ev.kind == "evasive_swerve":
            ev.accepted = bool(speed[ev.start] >= 15.0)
            if ev.accepted:
                yaw[sl] = (
                    ev.params["a_lat"] * np.sin(np.linspace(0, 2 * np.pi, ev.length)) / v_safe[sl]
                )
    yaw += rng.normal(0, 0.002, n)
    accel_lat = speed * yaw
    steering = np.degrees(np.arctan(WHEELBASE_M * yaw / v_safe)) * STEERING_RATIO

    # ---- radar lead -------------------------------------------------------------------
    present = _piecewise(rng, n, 45 * hz, lambda: rng.choice([1.0, 0.0], p=[0.75, 0.25])) > 0.5
    h_driver = rng.uniform(1.3, 2.4) / max(aggressiveness, 0.5) ** 0.3
    headway = np.clip(h_driver + _ou(rng, n, theta=0.1, sigma=0.15, dt=dt), 1.0, 4.0)
    gap = headway * speed + 2.0
    breaks = np.zeros(n, dtype=bool)
    for ev in events:
        sl = slice(ev.start, ev.start + ev.length)
        if ev.kind == "tailgating":
            ev.accepted = bool(speed[sl].min() >= 12.0)
            if ev.accepted:
                ramp = 2 * hz
                h = np.full(ev.length + 2 * ramp, ev.params["headway"])
                h[:ramp] = np.linspace(headway[ev.start - ramp], ev.params["headway"], ramp)
                h[-ramp:] = np.linspace(
                    ev.params["headway"], headway[min(n - 1, ev.start + ev.length + ramp)], ramp
                )
                a, b = ev.start - ramp, min(n, ev.start + ev.length + ramp)
                gap[a:b] = h[: b - a] * speed[a:b] + 0.5
                present[a:b] = True
        elif ev.kind == "cut_in" and ev.accepted:
            closing0 = ev.params["closing0"]
            gap0 = ev.params["ttc0"] * closing0
            L = ev.length
            closing = closing0 * np.clip(1 - np.arange(L) / (0.8 * L), 0, 1)
            cut_gap = gap0 - np.cumsum(closing) * dt
            cut_gap = np.maximum(cut_gap, 2.0)
            blend_len = min(8 * hz, n - (ev.start + L))
            gap[sl] = cut_gap
            if blend_len > 0:
                tail = slice(ev.start + L, ev.start + L + blend_len)
                w = np.linspace(0, 1, blend_len)
                gap[tail] = (1 - w) * cut_gap[-1] + w * gap[tail]
            present[ev.start : ev.start + L + max(blend_len, 0)] = True
            breaks[ev.start] = True  # a different vehicle: new radar track
    gap += _ou(rng, n, theta=2.0, sigma=0.1, dt=dt)
    lead_dist = np.where(present, gap, np.nan)
    lead_rel = _segment_gradient(lead_dist, breaks, dt)
    lead_rel = np.where(np.isnan(lead_dist), np.nan, _smooth(np.nan_to_num(lead_rel), 3))

    # ---- position ---------------------------------------------------------------------
    heading = rng.uniform(0, 2 * np.pi) + np.cumsum(yaw) * dt
    x = np.cumsum(speed * np.cos(heading)) * dt
    y = np.cumsum(speed * np.sin(heading)) * dt
    lat0, lon0 = origin[0] + rng.normal(0, 0.15), origin[1] + rng.normal(0, 0.15)
    lat = lat0 + y / 111_320.0
    lon = lon0 + x / (111_320.0 * math.cos(math.radians(lat0)))

    t_ns = t0_ns + (np.arange(n, dtype=np.int64) * (1_000_000_000 // hz))
    keep = np.ones(n, dtype=bool)
    for ev in events:
        sl = slice(ev.start, ev.start + ev.length)
        if ev.kind == "gnss_outage":
            lat[sl] = np.nan
            lon[sl] = np.nan
        elif ev.kind == "sensor_dropout":
            keep[sl] = False

    gt = [
        GroundTruthEvent(
            vehicle_id,
            trip_id,
            ev.kind,
            int(t_ns[ev.start]),
            int(t_ns[min(n - 1, ev.start + ev.length - 1)]),
        )
        for ev in events
        if ev.accepted
    ]
    signals = {
        "speed": speed,
        "accel_long": accel,
        "accel_lat": accel_lat,
        "yaw_rate": yaw,
        "steering_deg": steering,
        "lat": lat,
        "lon": lon,
        "lead_dist": lead_dist,
        "lead_rel_speed": lead_rel,
    }
    track = Track(
        vehicle_id=vehicle_id,
        trip_id=trip_id,
        source="synthetic",
        t_ns=t_ns[keep],
        signals={k: v[keep] for k, v in signals.items()},
    )
    return track, gt


def generate_fleet(cfg: FleetConfig) -> tuple[list[Track], list[GroundTruthEvent]]:
    rng = np.random.default_rng(cfg.seed)
    tracks: list[Track] = []
    truth: list[GroundTruthEvent] = []
    for v in range(cfg.n_vehicles):
        vehicle_id = f"veh-{v:05d}"
        aggressiveness = float(rng.lognormal(0.0, 0.5))
        t0 = cfg.start_time_ns + int(rng.uniform(0, 3600) * 1e9)
        for k in range(cfg.trips_per_vehicle):
            track, gt = generate_trip(
                rng,
                vehicle_id,
                f"{vehicle_id}-t{k:03d}",
                t0,
                cfg.trip_minutes * 60,
                hz=cfg.hz,
                rates=cfg.rates_per_hour,
                aggressiveness=aggressiveness,
                urban=bool(rng.random() < cfg.urban_fraction),
            )
            tracks.append(track)
            truth.extend(gt)
            t0 = int(track.t_ns[-1]) + int(rng.uniform(5, 30) * 60e9)
    return tracks, truth
