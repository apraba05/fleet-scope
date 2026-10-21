import msgspec
import numpy as np
import pytest

from fleetscope.schema import SIGNALS, Track
from fleetscope.stream.validate import (
    ValidationConfig,
    reason_counts,
    reasons_to_strings,
    validate_header,
    validate_rows,
)

T0 = 1_700_000_000_000_000_000
CFG = ValidationConfig()


def _cols(n=10):
    t = T0 + np.arange(n, dtype=np.int64) * 50_000_000
    cols = {s: np.zeros(n) for s in SIGNALS}
    cols["speed"][:] = 20.0
    cols["lat"][:] = 37.5
    cols["lon"][:] = -122.0
    cols["lead_dist"][:] = np.nan
    cols["lead_rel_speed"][:] = np.nan
    return t, cols


def _run(t, cols, ref=T0):
    starts = np.zeros(len(t), dtype=bool)
    starts[0] = True
    return validate_rows(t, cols, starts, ref, CFG)


def test_clean_rows_pass():
    t, cols = _cols()
    assert not _run(t, cols).any()


@pytest.mark.parametrize(
    ("col", "value", "rule"),
    [
        ("speed", -1.0, "speed_out_of_range"),
        ("speed", 120.0, "speed_out_of_range"),
        ("speed", np.nan, "required_not_finite"),
        ("accel_long", np.inf, "required_not_finite"),
        ("accel_long", -20.0, "accel_long_out_of_range"),
        ("accel_lat", 16.0, "accel_lat_out_of_range"),
        ("yaw_rate", 4.0, "yaw_rate_out_of_range"),
        ("steering_deg", 2000.0, "steering_out_of_range"),
        ("lat", 95.0, "lat_out_of_range"),
        ("lon", -190.0, "lon_out_of_range"),
        ("lead_dist", -2.0, "lead_out_of_range"),
    ],
)
def test_each_rule_flags_only_the_bad_row(col, value, rule):
    t, cols = _cols()
    cols[col][4] = value
    mask = _run(t, cols)
    assert np.flatnonzero(mask).tolist() == [4]
    assert reasons_to_strings(mask[4:5]) == [rule]


def test_optional_signals_may_be_missing_but_not_half_missing():
    t, cols = _cols()
    cols["lat"][:] = np.nan
    cols["lon"][:] = np.nan
    assert not _run(t, cols).any()
    cols["lon"][2] = -122.0
    assert reasons_to_strings(_run(t, cols)[2:3]) == ["gnss_partial_fix"]


def test_timestamp_rules_use_reference_time_and_chunk_boundaries():
    t, cols = _cols()
    t[5] = t[4]  # duplicate timestamp within a chunk
    t[9] = T0 + int(3600e9)  # an hour in the future of the broker timestamp
    mask = _run(t, cols)
    assert reason_counts(mask) == {"ts_not_increasing": 1, "ts_out_of_range": 1}
    starts = np.zeros(len(t), dtype=bool)
    starts[[0, 5]] = True  # row 5 starts a new chunk: no monotonicity check across chunks
    assert reason_counts(validate_rows(t, cols, starts, T0, CFG)) == {"ts_out_of_range": 1}


def test_multiple_failures_are_all_recorded():
    t, cols = _cols()
    cols["speed"][1] = -3
    cols["yaw_rate"][1] = 10
    assert reasons_to_strings(_run(t, cols)[1:2]) == ["speed_out_of_range|yaw_rate_out_of_range"]


def _chunk(**over):
    n = 4
    tr = Track("v", "t", "s", T0 + np.arange(n, dtype=np.int64), {s: np.zeros(n) for s in SIGNALS})
    return msgspec.structs.replace(tr.chunks(4)[0], **over)


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({}, None),
        ({"v": 99}, "unsupported_schema_version"),
        ({"vehicle_id": ""}, "missing_identity"),
        ({"seq": -1}, "negative_seq"),
        ({"t_ns": [], "speed": []}, "empty_chunk"),
        ({"speed": [1.0]}, "length_mismatch"),
    ],
)
def test_header_validation(over, reason):
    assert validate_header(_chunk(**over)) == reason
