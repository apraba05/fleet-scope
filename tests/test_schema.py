import msgspec
import numpy as np
import pytest

from fleetscope.schema import SIGNALS, TelemetryChunk, Track, decode_chunk, encode_chunk


def _track(n=45):
    t = 1_700_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 50_000_000
    sig = {s: np.linspace(0, 1, n) for s in SIGNALS}
    sig["lat"][3] = np.nan
    return Track("veh-1", "trip-1", "test", t, sig)


def test_chunking_covers_track_in_order():
    tr = _track()
    chunks = tr.chunks(20)
    assert [c.seq for c in chunks] == [0, 1, 2]
    assert [len(c) for c in chunks] == [20, 20, 5]
    assert sum((c.t_ns for c in chunks), []) == tr.t_ns.tolist()


def test_roundtrip_preserves_values_including_nan():
    c = _track().chunks(20)[0]
    back = decode_chunk(encode_chunk(c))
    assert back.vehicle_id == c.vehicle_id and back.seq == c.seq
    assert np.isnan(back.lat[3])
    np.testing.assert_array_equal(back.speed, c.speed)


def test_decode_rejects_garbage_and_wrong_types():
    payload = encode_chunk(_track().chunks(20)[0])
    with pytest.raises(msgspec.DecodeError):
        decode_chunk(payload[: len(payload) // 2])
    bad = msgspec.msgpack.encode({"v": 1, "vehicle_id": 5})
    with pytest.raises(msgspec.ValidationError):
        decode_chunk(bad)


def test_track_rejects_ragged_signals():
    t = np.arange(5, dtype=np.int64)
    sig = {s: np.zeros(5) for s in SIGNALS}
    sig["speed"] = np.zeros(4)
    with pytest.raises(ValueError, match="speed"):
        Track("v", "t", "s", t, sig)


def test_chunk_struct_is_immutable():
    c = _track().chunks(20)[0]
    assert isinstance(c, TelemetryChunk)
    with pytest.raises(AttributeError):
        c.seq = 5  # type: ignore[misc]
