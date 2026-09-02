import numpy as np

from experiments.single_satellite_cann_comparison import (
    _hybrid_ring_line_ir_azimuth,
    _line_cann_ir_elevation,
)
from experiments.single_satellite_cann_measurement_feedback import (
    run_single_satellite_cann_measurement_feedback,
)


def test_measurement_feedback_keeps_baseline_explicit():
    result = run_single_satellite_cann_measurement_feedback(
        duration=12.0, dt=2.0, outage_start=4.0, outage_end=8.0,
    )
    assert not result["baseline"]["summary"]["adaptive_cann_preprocess_ir"]
    assert result["processed"]["summary"]["adaptive_cann_preprocess_ir"]
    assert result["hybrid"]["summary"]["hybrid_cann_preprocess_ir"]
    assert "position_rmse_change_m" in result["summary"]
    assert "hybrid_position_rmse_change_m" in result["summary"]


def test_infrared_elevation_line_cann_stays_bounded_and_reports_diagnostics():
    timestamps = np.arange(0.0, 10.0, 2.0)
    hint = np.deg2rad(np.array([5.0, 5.2, 5.35, 5.5, 5.7]))
    available = np.array([True, True, False, True, True])
    elevation, diagnostics = _line_cann_ir_elevation(
        timestamps, hint, available,
    )
    assert np.all(np.isfinite(elevation))
    assert np.all(np.abs(elevation) <= 0.5 * np.pi)
    assert "elevation_bump_concentration" in diagnostics[-1]
    assert np.max(np.abs(elevation[available] - hint[available])) < np.deg2rad(0.1)


def test_infrared_elevation_line_cann_reanchors_after_long_outage():
    timestamps = np.arange(0.0, 62.0, 2.0)
    hint = np.deg2rad(np.linspace(5.0, 6.0, timestamps.size))
    available = np.ones(timestamps.size, dtype=bool)
    available[2:-2] = False
    elevation, diagnostics = _line_cann_ir_elevation(
        timestamps, hint, available,
    )
    assert diagnostics[-2]["elevation_recovery_pending"]
    assert diagnostics[-1]["elevation_reanchored"]
    assert abs(elevation[-1] - hint[-1]) < 1e-12


def test_infrared_recovery_confirmation_rejects_two_opposite_faults():
    timestamps = np.arange(0.0, 70.0, 2.0)
    hint = np.deg2rad(5.0 + 0.01 * timestamps)
    available = np.ones(timestamps.size, dtype=bool)
    available[2:30] = False
    hint[30] += np.deg2rad(5.0)
    hint[31] -= np.deg2rad(5.0)
    elevation, elevation_diagnostics = _line_cann_ir_elevation(
        timestamps, hint, available,
    )
    phase, azimuth_diagnostics = _hybrid_ring_line_ir_azimuth(
        timestamps, hint, np.zeros_like(timestamps), available,
    )
    assert all(elevation_diagnostics[index]["recovery_pending"]
               for index in (30, 31, 32))
    assert all(azimuth_diagnostics[index]["recovery_pending"]
               for index in (30, 31, 32))
    assert elevation_diagnostics[33]["elevation_reanchored"]
    assert azimuth_diagnostics[33]["azimuth_reanchored"]
    assert abs(elevation[33] - hint[33]) < 1e-12
    assert abs(phase[33] - hint[33]) < 1e-6
