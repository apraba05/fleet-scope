"""Loader for the comma2k19 dataset (https://github.com/commaai/comma2k19).

Layout::

    <chunk>/<dongle_id>|<route_start>/<segment_number>/processed_log/{CAN,IMU,GNSS}/<log>/{t,value}
    <chunk>/<dongle_id>|<route_start>/<segment_number>/global_pose/frame_*

Every ``processed_log`` signal is a pair of numpy arrays: ``t`` (seconds since device boot)
and ``value``. Each source runs on its own clock rate (CAN speed ~83 Hz, IMU ~100 Hz,
u-blox GNSS ~10 Hz, radar tracks ~100 Hz), so the loader resamples everything onto a
uniform grid and converts boot time to UTC.

Data quirks handled here (verified against the dataset, not just its README):

* ``live_gnss_*`` column 3 is documented as "utc_timestamp (s)" but is UTC
  *milliseconds*. The boot->UTC offset is the median of ``utc - t`` over all fixes.
* IMU axes are device frame ``[forward, right, down]``; we convert to ISO 8855
  (left / up positive), i.e. ``yaw_rate = -gyro_z``.
* Longitudinal acceleration is derived from CAN speed rather than the accelerometer:
  the phone mount is not aligned with the car body, so raw ``accel_x`` mixes in gravity
  from pitch. Lateral acceleration uses the kinematic relation ``a_y = v * yaw_rate``
  for the same reason.
* Radar rows are individual tracks ``[dist, left_dist, rel_speed, nan, nan, address,
  new_track]``. The lead is the nearest track inside the ego lane (|left| < 1.8 m).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from fleetscope.schema import SAMPLE_HZ, SIGNALS, Track

LANE_HALF_WIDTH_M = 1.8
GNSS_MAX_STALENESS_S = 1.0
RADAR_MAX_STALENESS_S = 0.25


class SegmentFormatError(ValueError):
    pass


def _load(seg: Path, rel: str) -> tuple[np.ndarray, np.ndarray]:
    base = seg / "processed_log" / rel
    try:
        return np.load(base / "t"), np.load(base / "value")
    except FileNotFoundError as e:
        raise SegmentFormatError(f"{seg}: missing {rel}") from e


def _boot_to_utc_offset_s(seg: Path) -> float:
    for log in ("GNSS/live_gnss_ublox", "GNSS/live_gnss_qcom"):
        try:
            t, v = _load(seg, log)
        except SegmentFormatError:
            continue
        utc = v[:, 3]
        ok = np.isfinite(utc) & (utc > 0)
        if ok.sum() == 0:
            continue
        utc_s = utc[ok] / 1000.0 if np.median(utc[ok]) > 1e11 else utc[ok]
        return float(np.median(utc_s - t[ok]))
    raise SegmentFormatError(f"{seg}: no GNSS fix with UTC time; cannot place segment in time")


def _interp_nearest_within(grid: np.ndarray, t: np.ndarray, v: np.ndarray, max_gap: float):
    """Linear interpolation, but NaN where the grid point is > max_gap from any sample."""
    out = np.interp(grid, t, v)
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest = np.minimum(np.abs(grid - t[idx - 1]), np.abs(grid - t[idx]))
    out[nearest > max_gap] = np.nan
    return out


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    padded = np.pad(x, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _lead_from_radar(
    grid: np.ndarray, t: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dist, left, rel = v[:, 0], v[:, 1], v[:, 2]
    in_lane = (
        np.isfinite(dist) & (dist > 0) & np.isfinite(left) & (np.abs(left) < LANE_HALF_WIDTH_M)
    )
    t, dist, rel = t[in_lane], dist[in_lane], rel[in_lane]
    lead_dist = np.full(len(grid), np.nan)
    lead_rel = np.full(len(grid), np.nan)
    if len(t) == 0:
        return lead_dist, lead_rel
    # Assign each track observation to its nearest grid bin, keep the closest track per bin.
    dt = grid[1] - grid[0] if len(grid) > 1 else 1.0 / SAMPLE_HZ
    bins = np.rint((t - grid[0]) / dt).astype(np.int64)
    keep = (
        (bins >= 0)
        & (bins < len(grid))
        & (np.abs(t - grid[np.clip(bins, 0, len(grid) - 1)]) <= RADAR_MAX_STALENESS_S)
    )
    bins, dist, rel = bins[keep], dist[keep], rel[keep]
    order = np.lexsort((dist, bins))
    bins, dist, rel = bins[order], dist[order], rel[order]
    first = np.unique(bins, return_index=True)[1]
    lead_dist[bins[first]] = dist[first]
    lead_rel[bins[first]] = rel[first]
    return lead_dist, lead_rel


def load_segment(seg_path: str | Path, vehicle_id: str | None = None, hz: int = SAMPLE_HZ) -> Track:
    """Load one 1-minute comma2k19 segment as a uniformly sampled Track."""
    seg = Path(seg_path)
    route = seg.parent.name  # "<dongle_id>|<route_start>"
    if "|" not in route:
        raise SegmentFormatError(f"{seg}: parent dir {route!r} is not '<dongle_id>|<start>'")
    dongle_id = route.split("|", 1)[0]

    t_speed, v_speed = _load(seg, "CAN/speed")
    t_steer, v_steer = _load(seg, "CAN/steering_angle")
    t_gyro, v_gyro = _load(seg, "IMU/gyro")
    v_speed = v_speed.reshape(len(v_speed), -1)[:, 0]

    start = max(t_speed[0], t_steer[0], t_gyro[0])
    end = min(t_speed[-1], t_steer[-1], t_gyro[-1])
    if end - start < 1.0:
        raise SegmentFormatError(f"{seg}: overlapping signal span < 1 s")
    grid = np.arange(start, end, 1.0 / hz)

    speed = np.clip(np.interp(grid, t_speed, v_speed), 0.0, None)
    # Derivative of a 0.25 s moving average; CAN speed is quantised to ~0.03 m/s.
    accel_long = np.gradient(_smooth(speed, max(1, hz // 4)), grid)
    yaw_rate = -np.interp(grid, t_gyro, v_gyro[:, 2])
    yaw_rate = _smooth(yaw_rate, max(1, hz // 10))
    accel_lat = speed * yaw_rate
    steering = np.interp(grid, t_steer, v_steer)

    lat = np.full(len(grid), np.nan)
    lon = np.full(len(grid), np.nan)
    try:
        t_gnss, v_gnss = _load(seg, "GNSS/live_gnss_ublox")
        ok = np.isfinite(v_gnss[:, 0]) & np.isfinite(v_gnss[:, 1]) & (v_gnss[:, 0] != 0)
        if ok.sum() >= 2:
            lat = _interp_nearest_within(grid, t_gnss[ok], v_gnss[ok, 0], GNSS_MAX_STALENESS_S)
            lon = _interp_nearest_within(grid, t_gnss[ok], v_gnss[ok, 1], GNSS_MAX_STALENESS_S)
    except SegmentFormatError:
        pass

    try:
        t_radar, v_radar = _load(seg, "CAN/radar")
        lead_dist, lead_rel = _lead_from_radar(grid, t_radar, v_radar)
    except SegmentFormatError:
        lead_dist = np.full(len(grid), np.nan)
        lead_rel = np.full(len(grid), np.nan)

    offset = _boot_to_utc_offset_s(seg)
    # Build the UTC grid in integer ns: (boot_s + offset) * 1e9 in float64 would only be
    # accurate to ~256 ns at epoch magnitudes and jitter the sample period.
    period_ns = 1_000_000_000 // hz
    t_ns = int(round((start + offset) * 1e9)) + np.arange(len(grid), dtype=np.int64) * period_ns

    signals = {
        "speed": speed,
        "accel_long": accel_long,
        "accel_lat": accel_lat,
        "yaw_rate": yaw_rate,
        "steering_deg": steering,
        "lat": lat,
        "lon": lon,
        "lead_dist": lead_dist,
        "lead_rel_speed": lead_rel,
    }
    assert set(signals) == set(SIGNALS)
    return Track(
        vehicle_id=vehicle_id or dongle_id,
        trip_id=f"{route}/{seg.name}",
        source="comma2k19",
        t_ns=t_ns,
        signals=signals,
    )


def discover_segments(root: str | Path) -> Iterator[Path]:
    """Yield every segment directory (one containing processed_log/) under ``root``."""
    for p in sorted(Path(root).rglob("processed_log")):
        if p.is_dir() and "|" in p.parent.parent.name:
            yield p.parent


def load_route_tracks(root: str | Path, hz: int = SAMPLE_HZ) -> list[Track]:
    return [load_segment(seg, hz=hz) for seg in discover_segments(root)]


def amplify(
    track: Track, n_clones: int, seed: int = 0, time_spread_s: float = 3600.0
) -> list[Track]:
    """Create ``n_clones`` virtual vehicles from one real drive (fleet amplification).

    Each clone is shifted in time, has its speed profile scaled by +-5 %, and gets
    small sensor noise, so downstream state (dedup, windows) sees distinct vehicles while
    the signal statistics stay those of real driving. Clones are labelled ``source=
    "comma2k19-amplified"`` so analyses can separate them from the original drive.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_clones):
        scale = 1.0 + rng.uniform(-0.05, 0.05)
        shift_ns = int(rng.uniform(0, time_spread_s) * 1e9)
        s = {k: v.copy() for k, v in track.signals.items()}
        s["speed"] = np.clip(s["speed"] * scale + rng.normal(0, 0.05, len(track)), 0, None)
        s["accel_long"] = s["accel_long"] * scale + rng.normal(0, 0.05, len(track))
        s["accel_lat"] = s["accel_lat"] * scale + rng.normal(0, 0.05, len(track))
        s["yaw_rate"] = s["yaw_rate"] + rng.normal(0, 0.002, len(track))
        out.append(
            Track(
                vehicle_id=f"c2k-{i:05d}",
                trip_id=f"{track.trip_id}#clone{i}",
                source="comma2k19-amplified",
                t_ns=track.t_ns + shift_ns,
                signals=s,
            )
        )
    return out
