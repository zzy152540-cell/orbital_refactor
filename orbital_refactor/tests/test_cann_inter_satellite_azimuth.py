import numpy as np
from experiments.cann_inter_satellite_azimuth import (
    _cann_tracker, _circular_kalman, _difference, _gated_pll,
    _weighted_circular_mean,
)

def test_weighted_circular_mean_handles_wrap_boundary():
    value=_weighted_circular_mean(np.deg2rad(359),np.deg2rad(1),1,1)
    assert abs(_difference(value,0.0))<1e-12


def test_circular_trackers_cross_wrap_without_large_jump():
    rate = np.full(6, np.deg2rad(1.0))
    hint = np.full(6, np.nan)
    available = np.zeros(6, dtype=bool)
    for tracker in (_gated_pll, _circular_kalman):
        phase = tracker(np.deg2rad(358.0), rate, 1.0, hint, available, np.deg2rad(3.0))
        error = _difference(phase[-1], np.deg2rad(3.0))
        assert abs(error) < 1e-12


def test_zero_bias_gain_preserves_original_cann_tracker():
    times = np.arange(0.0, 8.0, 2.0)
    rate = np.full(times.size, np.deg2rad(0.1))
    hint = np.full(times.size, np.nan)
    available = np.zeros(times.size, dtype=bool)
    original, original_quality = _cann_tracker(
        times, 0.0, rate, hint, available, np.deg2rad(3.0),
    )
    explicit, explicit_quality = _cann_tracker(
        times, 0.0, rate, hint, available, np.deg2rad(3.0),
        rate_bias_gain=0.0,
    )
    np.testing.assert_array_equal(explicit, original)
    np.testing.assert_array_equal(explicit_quality, original_quality)
