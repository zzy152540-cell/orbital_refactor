import numpy as np

from experiments.single_satellite_cann_comparison import (
    _radar_range_rate_line_cann,
)
from experiments.single_satellite_radar_cann_feedback import (
    run_single_satellite_radar_cann_feedback,
)


def test_radar_dual_line_cann_reanchors_after_long_outage():
    timestamps = np.arange(0.0, 62.0, 2.0)
    measurement = np.column_stack((
        100_000.0 + 10.0 * timestamps,
        np.full(timestamps.size, 10.0),
    ))
    available = np.ones(timestamps.size, dtype=bool)
    available[2:-1] = False
    filtered, diagnostics = _radar_range_rate_line_cann(
        timestamps, measurement, available,
    )
    assert diagnostics[-1]["radar_reanchored"]
    assert np.allclose(filtered[-1], measurement[-1], atol=1e-6)


def test_radar_dual_line_cann_substitutes_only_impulsive_component():
    timestamps = np.arange(0.0, 8.0, 2.0)
    measurement = np.column_stack((
        100_000.0 + 10.0 * timestamps,
        np.full(timestamps.size, 10.0),
    ))
    measurement[2, 0] += 3_000.0
    filtered, diagnostics = _radar_range_rate_line_cann(
        timestamps, measurement, np.ones(timestamps.size, dtype=bool),
    )
    assert diagnostics[2]["range_substituted"]
    assert not diagnostics[2]["range_rate_substituted"]
    assert abs(filtered[2, 0] - measurement[2, 0]) > 1_000.0
    assert filtered[2, 1] == measurement[2, 1]


def test_radar_cann_feedback_keeps_baseline_explicit():
    result = run_single_satellite_radar_cann_feedback(
        duration=12.0, dt=2.0, outage_start=4.0, outage_end=8.0,
    )
    assert not result["baseline"]["summary"]["radar_cann_preprocess"]
    assert result["processed"]["summary"]["radar_cann_preprocess"]
    assert "position_rmse_change_m" in result["summary"]
