"""Wire and storage schemas for fleet telemetry.

Wire format
-----------
One Kafka message carries one ``TelemetryChunk``: a short, columnar run of samples
(default 1 s at 20 Hz) from a single vehicle trip. Vehicles batch samples before upload
in practice, and batching amortises per-message overhead (headers, keys, protocol
framing) across many samples. Messages are keyed by ``vehicle_id`` so that all chunks
of a vehicle land on the same partition and keep producer order.

``seq`` is the producer-assigned chunk sequence number within a trip. Together with
``(vehicle_id, trip_id)`` it is the idempotency key used for deduplication.

Coordinate conventions (ISO 8855 vehicle frame): x forward, y left, z up.
``accel_lat`` > 0 means acceleration to the left, ``yaw_rate`` > 0 is a left turn.

Optional signals (GNSS, radar lead) use NaN for "not available" so every column has
the same length as ``t_ns``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import msgspec
import numpy as np
import pyarrow as pa

SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
SAMPLE_HZ = 20
MAX_SAMPLES_PER_CHUNK = 1024

# Float signal columns, in wire/storage order.
SIGNALS: tuple[str, ...] = (
    "speed",  # m/s, >= 0
    "accel_long",  # m/s^2, + forward
    "accel_lat",  # m/s^2, + left
    "yaw_rate",  # rad/s, + left turn
    "steering_deg",  # deg, steering wheel angle
    "lat",  # deg WGS84, NaN when no fix
    "lon",  # deg WGS84, NaN when no fix
    "lead_dist",  # m to in-lane lead vehicle, NaN when none
    "lead_rel_speed",  # m/s, lead speed minus ego speed (negative = closing)
)
REQUIRED_SIGNALS: tuple[str, ...] = ("speed", "accel_long", "accel_lat", "yaw_rate", "steering_deg")
OPTIONAL_SIGNALS: tuple[str, ...] = ("lat", "lon", "lead_dist", "lead_rel_speed")


class TelemetryChunk(msgspec.Struct, frozen=True, kw_only=True):
    """A columnar run of samples for one vehicle trip (the unit of transport)."""

    v: int
    vehicle_id: str
    trip_id: str
    seq: int
    source: str
    t_ns: list[int]
    speed: list[float]
    accel_long: list[float]
    accel_lat: list[float]
    yaw_rate: list[float]
    steering_deg: list[float]
    lat: list[float]
    lon: list[float]
    lead_dist: list[float]
    lead_rel_speed: list[float]

    def __len__(self) -> int:
        return len(self.t_ns)


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(TelemetryChunk)


def encode_chunk(chunk: TelemetryChunk) -> bytes:
    return _encoder.encode(chunk)


def decode_chunk(payload: bytes) -> TelemetryChunk:
    """Decode and type-check a chunk. Raises ``msgspec.DecodeError`` / ``ValidationError``."""
    return _decoder.decode(payload)


@dataclass
class Track:
    """A full, uniformly-sampled vehicle trip held as numpy columns.

    Sources (comma2k19, synthetic) produce Tracks; the replayer slices them into chunks.
    """

    vehicle_id: str
    trip_id: str
    source: str
    t_ns: np.ndarray  # int64, UTC epoch nanoseconds, strictly increasing
    signals: dict[str, np.ndarray] = field(default_factory=dict)  # float64 columns

    def __post_init__(self) -> None:
        n = len(self.t_ns)
        missing = [s for s in SIGNALS if s not in self.signals]
        if missing:
            raise ValueError(f"track {self.trip_id} missing signals: {missing}")
        for name in SIGNALS:
            if len(self.signals[name]) != n:
                raise ValueError(f"signal {name} has {len(self.signals[name])} rows, expected {n}")

    def __len__(self) -> int:
        return len(self.t_ns)

    @property
    def duration_s(self) -> float:
        return float(self.t_ns[-1] - self.t_ns[0]) / 1e9 if len(self) else 0.0

    def chunks(self, samples_per_chunk: int = SAMPLE_HZ) -> list[TelemetryChunk]:
        out = []
        for seq, start in enumerate(range(0, len(self), samples_per_chunk)):
            sl = slice(start, start + samples_per_chunk)
            out.append(
                TelemetryChunk(
                    v=SCHEMA_VERSION,
                    vehicle_id=self.vehicle_id,
                    trip_id=self.trip_id,
                    seq=seq,
                    source=self.source,
                    t_ns=self.t_ns[sl].tolist(),
                    **{name: self.signals[name][sl].tolist() for name in SIGNALS},
                )
            )
        return out


# --------------------------------------------------------------------------------------
# Lake (Parquet) schemas
# --------------------------------------------------------------------------------------

_SIGNAL_TYPES = {
    "speed": pa.float32(),
    "accel_long": pa.float32(),
    "accel_lat": pa.float32(),
    "yaw_rate": pa.float32(),
    "steering_deg": pa.float32(),
    "lat": pa.float64(),  # float32 loses ~1 m of precision at these magnitudes
    "lon": pa.float64(),
    "lead_dist": pa.float32(),
    "lead_rel_speed": pa.float32(),
}

TELEMETRY_SCHEMA = pa.schema(
    [
        pa.field("vehicle_id", pa.string(), nullable=False),
        pa.field("trip_id", pa.string(), nullable=False),
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("ts", pa.timestamp("ns", tz="UTC"), nullable=False),
        *[pa.field(name, _SIGNAL_TYPES[name]) for name in SIGNALS],
        pa.field("source", pa.string(), nullable=False),
        pa.field("kafka_partition", pa.int32(), nullable=False),
        pa.field("kafka_offset", pa.int64(), nullable=False),
    ]
)

QUARANTINE_SCHEMA = TELEMETRY_SCHEMA.append(pa.field("reasons", pa.string(), nullable=False))

DLQ_SCHEMA = pa.schema(
    [
        pa.field("kafka_partition", pa.int32(), nullable=False),
        pa.field("kafka_offset", pa.int64(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("detail", pa.string()),
        pa.field("payload", pa.binary()),
    ]
)

WINDOW_SCHEMA = pa.schema(
    [
        pa.field("vehicle_id", pa.string(), nullable=False),
        pa.field("trip_id", pa.string(), nullable=False),
        pa.field("window_start", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("window_s", pa.int32(), nullable=False),
        pa.field("samples", pa.int64(), nullable=False),
        pa.field("distance_m", pa.float64(), nullable=False),
        pa.field("speed_mean", pa.float64(), nullable=False),
        pa.field("speed_max", pa.float64(), nullable=False),
        pa.field("accel_long_min", pa.float64(), nullable=False),
        pa.field("accel_long_max", pa.float64(), nullable=False),
        pa.field("accel_lat_abs_max", pa.float64(), nullable=False),
        pa.field("lead_dist_min", pa.float64()),
    ]
)
