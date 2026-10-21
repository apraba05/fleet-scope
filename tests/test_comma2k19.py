import datetime as dt

import numpy as np
import pytest

from fleetscope.sources.comma2k19 import (
    SegmentFormatError,
    amplify,
    discover_segments,
    load_segment,
)

from .conftest import SAMPLE_ROOT, SAMPLE_SEGMENT


@pytest.fixture(scope="module")
def track():
    return load_segment(SAMPLE_SEGMENT)


def test_discovers_the_bundled_segment():
    segs = list(discover_segments(SAMPLE_ROOT))
    assert segs == [SAMPLE_SEGMENT]


def test_uniform_20hz_grid_over_one_minute(track):
    assert track.vehicle_id == "b0c9d2329ad1606b"
    assert track.trip_id == "b0c9d2329ad1606b|2018-08-02--08-34-47/40"
    assert 1190 <= len(track) <= 1200
    np.testing.assert_allclose(np.diff(track.t_ns), 50_000_000, atol=2)


def test_boot_time_is_converted_to_utc_using_gnss_milliseconds(track):
    # Route started 08:34:47 local (PDT, UTC-7); segment 40 starts ~40 min later.
    start = dt.datetime.fromtimestamp(track.t_ns[0] / 1e9, dt.UTC)
    assert start.date() == dt.date(2018, 8, 2)
    assert (start.hour, start.minute) == (16, 14)


def test_signals_are_physically_plausible(track):
    s = track.signals
    assert 7.5 < s["speed"].min() and s["speed"].max() < 21  # highway 280 traffic
    assert np.abs(s["accel_long"]).max() < 4
    assert np.abs(s["accel_lat"]).max() < 2
    assert np.nanmean(s["lat"]) == pytest.approx(37.72, abs=0.02)
    assert np.nanmean(s["lon"]) == pytest.approx(-122.47, abs=0.02)
    lead = s["lead_dist"][~np.isnan(s["lead_dist"])]
    assert len(lead) > 0.5 * len(track) and lead.min() > 2


def test_accel_is_consistent_with_speed(track):
    s = track.signals
    integrated = s["speed"][0] + np.cumsum(s["accel_long"]) / 20.0
    assert np.abs(integrated - s["speed"]).max() < 1.0


def test_amplify_creates_distinct_shifted_vehicles(track):
    clones = amplify(track, 5, seed=1)
    assert len({c.vehicle_id for c in clones}) == 5
    assert all(c.source == "comma2k19-amplified" for c in clones)
    assert len({int(c.t_ns[0]) for c in clones}) == 5
    for c in clones:
        assert np.corrcoef(c.signals["speed"], track.signals["speed"])[0, 1] > 0.99


def test_rejects_non_segment_dirs(tmp_path):
    seg = tmp_path / "not_a_route" / "0"
    (seg / "processed_log").mkdir(parents=True)
    with pytest.raises(SegmentFormatError):
        load_segment(seg)
