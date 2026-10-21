"""Event-time tumbling-window aggregation with per-stream watermarks.

Watermarks are tracked per stream ``(vehicle_id, trip_id)``, not per partition: each
vehicle has its own clock and upload delay, so one vehicle's backlog must not hold back
(or prematurely close) another vehicle's windows.

* watermark(stream) = max event time seen - allowed_lateness
* a window [start, start + W) is emitted once watermark >= start + W
* rows whose window was already emitted are *late*: they are still written to the raw
  lake (the source of truth) but excluded from the streaming aggregate and counted.
  Batch jobs recompute exact aggregates from the lake when needed.

Streams that stop producing (trip ended, vehicle offline) are closed when the
partition's event-time frontier passes them by ``idle_timeout``. A tombstone keeps the
closed stream's max event time (for ``tombstone_ttl``) so data that trickles in after
closing is classified as late instead of re-opening already-emitted windows. Closing is
triggered by
event-time progress (not wall clock), so window output is a deterministic function of
partition order -- required for exactly-once output across crash/replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import msgspec
import numpy as np

from fleetscope.schema import SAMPLE_HZ


class WindowAcc(msgspec.Struct, array_like=True):
    start_ns: int
    samples: int = 0
    speed_sum: float = 0.0
    speed_max: float = float("-inf")
    accel_long_min: float = float("inf")
    accel_long_max: float = float("-inf")
    accel_lat_abs_max: float = 0.0
    lead_dist_min: float = float("inf")


class StreamWindowState(msgspec.Struct, array_like=True):
    vehicle_id: str
    trip_id: str
    max_ts_ns: int
    closed_until_ns: int
    windows: list[WindowAcc]


class WindowSnapshot(msgspec.Struct, array_like=True):
    frontier_ns: int
    next_idle_check_ns: int
    streams: list[StreamWindowState]
    tombstones: list[tuple[str, str, int]]


@dataclass(frozen=True)
class WindowResult:
    vehicle_id: str
    trip_id: str
    window_start_ns: int
    window_s: int
    samples: int
    distance_m: float
    speed_mean: float
    speed_max: float
    accel_long_min: float
    accel_long_max: float
    accel_lat_abs_max: float
    lead_dist_min: float | None


class _Stream:
    __slots__ = ("closed_until_ns", "max_ts_ns", "windows")

    def __init__(
        self, max_ts_ns: int, closed_until_ns: int = 0, windows: dict[int, WindowAcc] | None = None
    ):
        self.max_ts_ns = max_ts_ns
        self.closed_until_ns = closed_until_ns  # windows starting before this were emitted
        self.windows = windows if windows is not None else {}


class TumblingWindows:
    def __init__(
        self,
        window_s: int = 10,
        allowed_lateness_s: float = 30.0,
        idle_timeout_s: float = 120.0,
        tombstone_ttl_s: float = 6 * 3600.0,
        sample_hz: int = SAMPLE_HZ,
    ):
        self.window_ns = int(window_s * 1e9)
        self.window_s = window_s
        self.lateness_ns = int(allowed_lateness_s * 1e9)
        self.idle_ns = int(idle_timeout_s * 1e9)
        self.tombstone_ttl_ns = int(tombstone_ttl_s * 1e9)
        self.dt = 1.0 / sample_hz
        self._streams: dict[tuple[str, str], _Stream] = {}
        self._tombstones: dict[tuple[str, str], int] = {}  # closed stream -> its max_ts_ns
        self._frontier_ns = 0  # max event time across all streams of this partition
        self._next_idle_check_ns = 0

    def __len__(self) -> int:
        return len(self._streams)

    def _window_end(self, ts_ns: int) -> int:
        return ts_ns - ts_ns % self.window_ns + self.window_ns

    def _result(self, key: tuple[str, str], w: WindowAcc) -> WindowResult:
        return WindowResult(
            vehicle_id=key[0],
            trip_id=key[1],
            window_start_ns=w.start_ns,
            window_s=self.window_s,
            samples=w.samples,
            distance_m=w.speed_sum * self.dt,
            speed_mean=w.speed_sum / w.samples,
            speed_max=w.speed_max,
            accel_long_min=w.accel_long_min,
            accel_long_max=w.accel_long_max,
            accel_lat_abs_max=w.accel_lat_abs_max,
            lead_dist_min=None if w.lead_dist_min == float("inf") else w.lead_dist_min,
        )

    def update(
        self,
        vehicle_id: str,
        trip_id: str,
        t_ns: np.ndarray,
        speed: np.ndarray,
        accel_long: np.ndarray,
        accel_lat: np.ndarray,
        lead_dist: np.ndarray,
    ) -> tuple[int, list[WindowResult]]:
        """Add one chunk's valid rows. Returns (late_row_count, closed_windows)."""
        if len(t_ns) == 0:
            return 0, []
        key = (vehicle_id, trip_id)
        st = self._streams.get(key)
        if st is None:
            tomb = self._tombstones.pop(key, None)
            if tomb is None:
                st = _Stream(max_ts_ns=int(t_ns.min()))
            else:  # re-opened after idle close: everything up to the tombstone was emitted
                st = _Stream(max_ts_ns=tomb, closed_until_ns=self._window_end(tomb))
            self._streams[key] = st
        watermark = st.max_ts_ns - self.lateness_ns
        starts = t_ns - (t_ns % self.window_ns)
        late = (starts + self.window_ns <= watermark) | (starts < st.closed_until_ns)
        n_late = int(late.sum())
        on_time = ~late
        if n_late:
            starts, t_ns = starts[on_time], t_ns[on_time]
            speed, accel_long = speed[on_time], accel_long[on_time]
            accel_lat, lead_dist = accel_lat[on_time], lead_dist[on_time]
        for ws in np.unique(starts).tolist():
            m = starts == ws if len(starts) > 1 else slice(None)
            acc = st.windows.get(ws)
            if acc is None:
                acc = st.windows[ws] = WindowAcc(start_ns=ws)
            sp = speed[m]
            acc.samples += len(sp)
            acc.speed_sum += float(sp.sum())
            acc.speed_max = max(acc.speed_max, float(sp.max()))
            al = accel_long[m]
            acc.accel_long_min = min(acc.accel_long_min, float(al.min()))
            acc.accel_long_max = max(acc.accel_long_max, float(al.max()))
            acc.accel_lat_abs_max = max(acc.accel_lat_abs_max, float(np.abs(accel_lat[m]).max()))
            ld = lead_dist[m]
            if np.any(~np.isnan(ld)):
                acc.lead_dist_min = min(acc.lead_dist_min, float(np.nanmin(ld)))
        if len(t_ns):
            st.max_ts_ns = max(st.max_ts_ns, int(t_ns.max()))
        closed = self._close_ready(key, st)
        if st.max_ts_ns > self._frontier_ns:
            self._frontier_ns = st.max_ts_ns
            if self._frontier_ns >= self._next_idle_check_ns:
                closed.extend(self._close_idle())
                self._next_idle_check_ns = self._frontier_ns + self.window_ns
        return n_late, closed

    def _close_ready(self, key: tuple[str, str], st: _Stream) -> list[WindowResult]:
        watermark = st.max_ts_ns - self.lateness_ns
        ready = [ws for ws in st.windows if ws + self.window_ns <= watermark]
        return [self._result(key, st.windows.pop(ws)) for ws in sorted(ready)]

    def _close_idle(self) -> list[WindowResult]:
        out: list[WindowResult] = []
        cutoff = self._frontier_ns - self.idle_ns
        idle = [k for k, st in self._streams.items() if st.max_ts_ns < cutoff]
        for k in idle:
            st = self._streams.pop(k)
            self._tombstones[k] = st.max_ts_ns
            out.extend(self._result(k, st.windows[ws]) for ws in sorted(st.windows))
        expire = self._frontier_ns - self.tombstone_ttl_ns
        for k in [k for k, ts in self._tombstones.items() if ts < expire]:
            del self._tombstones[k]
        return out

    def drain(self) -> list[WindowResult]:
        """Close every open window (end of a bounded input)."""
        out: list[WindowResult] = []
        for k, st in self._streams.items():
            self._tombstones[k] = st.max_ts_ns
            out.extend(self._result(k, st.windows[ws]) for ws in sorted(st.windows))
        self._streams.clear()
        return out

    # --- checkpointing ------------------------------------------------------------------
    def snapshot(self) -> WindowSnapshot:
        return WindowSnapshot(
            frontier_ns=self._frontier_ns,
            next_idle_check_ns=self._next_idle_check_ns,
            streams=[
                StreamWindowState(
                    v,
                    t,
                    st.max_ts_ns,
                    st.closed_until_ns,
                    # copies: a snapshot must not alias live accumulators
                    [msgspec.structs.replace(st.windows[ws]) for ws in sorted(st.windows)],
                )
                for (v, t), st in self._streams.items()
            ],
            tombstones=[(v, t, ts) for (v, t), ts in self._tombstones.items()],
        )

    def restore(self, snap: WindowSnapshot) -> None:
        self._frontier_ns = snap.frontier_ns
        self._next_idle_check_ns = snap.next_idle_check_ns
        self._streams = {
            (s.vehicle_id, s.trip_id): _Stream(
                s.max_ts_ns,
                s.closed_until_ns,
                {w.start_ns: msgspec.structs.replace(w) for w in s.windows},
            )
            for s in snap.streams
        }
        self._tombstones = {(v, t): ts for v, t, ts in snap.tombstones}
