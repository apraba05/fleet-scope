import numpy as np

from fleetscope.sources.synthetic import FleetConfig, generate_fleet


def _outside_events(track, gt, pad_s=3.0):
    mask = np.ones(len(track), dtype=bool)
    for e in gt:
        if e.trip_id == track.trip_id:
            mask &= ~((track.t_ns >= e.start_ns - pad_s * 1e9) & (track.t_ns <= e.end_ns + 8e9))
    return mask


def test_deterministic_for_a_seed():
    a, ga = generate_fleet(FleetConfig(n_vehicles=3, trip_minutes=2, seed=5))
    b, gb = generate_fleet(FleetConfig(n_vehicles=3, trip_minutes=2, seed=5))
    assert ga == gb
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x.signals["speed"], y.signals["speed"])


def test_kinematic_consistency(small_fleet):
    tracks, _ = small_fleet
    for tr in tracks:
        s = tr.signals
        dt = np.diff(tr.t_ns) / 1e9
        contiguous = np.isclose(dt, 0.05)
        dv = np.diff(s["speed"])[contiguous]
        np.testing.assert_allclose(dv, s["accel_long"][1:][contiguous] * 0.05, atol=1e-9)
        np.testing.assert_allclose(s["accel_lat"], s["speed"] * s["yaw_rate"], atol=1e-9)
        assert s["speed"].min() >= 0


def test_normal_driving_stays_in_comfort_envelope(small_fleet):
    tracks, gt = small_fleet
    for tr in tracks:
        m = _outside_events(tr, gt)
        s = tr.signals
        assert np.abs(s["accel_long"][m]).max() < 3.0
        assert np.abs(s["accel_lat"][m]).max() < 3.0
        v = s["speed"][m]
        headway = s["lead_dist"][m] / np.maximum(v, 0.1)
        fast = (v > 10) & ~np.isnan(headway)
        assert headway[fast].min() > 0.9


def test_ground_truth_lies_inside_trips_and_dropouts_remove_samples():
    tracks, gt = generate_fleet(
        FleetConfig(n_vehicles=4, trip_minutes=20, seed=2, rates_per_hour={"sensor_dropout": 6.0})
    )
    by_trip = {t.trip_id: t for t in tracks}
    assert gt, "expected some dropouts at 6/h over 80 min"
    for e in gt:
        tr = by_trip[e.trip_id]
        assert tr.t_ns[0] <= e.start_ns < e.end_ns <= tr.t_ns[-1]
        gaps = np.diff(tr.t_ns)
        inside = (tr.t_ns[:-1] < e.start_ns) & (tr.t_ns[1:] > e.end_ns)
        assert inside.any() and gaps[inside].max() > 0.5e9


def test_cut_in_vehicle_fits_in_front_of_ego():
    # A vehicle can only cut in ahead of ego if there is room for it: behind an existing
    # lead the new gap must be at least a car length (+ clearance) shorter. Regression for
    # labels whose "cut-in" was physically impossible and looked like a 2.5 m range jump.
    tracks, gt = generate_fleet(FleetConfig(n_vehicles=200, trip_minutes=20, seed=7))
    by_trip = {t.trip_id: t for t in tracks}
    cut_ins = [e for e in gt if e.scenario_type == "cut_in"]
    assert len(cut_ins) > 30
    behind_lead = 0
    for e in cut_ins:
        tr = by_trip[e.trip_id]
        i = int(np.searchsorted(tr.t_ns, e.start_ns))
        before, at = tr.signals["lead_dist"][i - 1], tr.signals["lead_dist"][i]
        assert at >= 1.5
        if not np.isnan(before):
            behind_lead += 1
            assert before - at >= 6.0, (e, before, at)
    assert behind_lead > 0  # the constraint was actually exercised
