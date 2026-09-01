import numpy as np

from experiments.single_satellite_cann_comparison import _optical_uv_plane_cann
from experiments.single_satellite_optical_cann_feedback import (
    run_single_satellite_optical_cann_feedback,
)


def test_optical_plane_cann_substitutes_impulse_in_stable_motion():
    timestamps = np.arange(0.0, 10.0, 2.0)
    measurement = np.column_stack((
        0.1 + 0.001 * timestamps, 2.0 - 0.002 * timestamps,
    ))
    measurement[3] += [0.2, 0.4]
    filtered, diagnostics = _optical_uv_plane_cann(
        timestamps, measurement, np.ones(timestamps.size, dtype=bool),
    )
    assert diagnostics[3]["u_substituted"]
    assert diagnostics[3]["v_substituted"]
    assert np.linalg.norm(filtered[3] - measurement[3]) > 0.1


def test_optical_feedback_keeps_baseline_explicit():
    result = run_single_satellite_optical_cann_feedback(
        optical_fault_mode=None, duration=12.0, dt=2.0,
        outage_start=4.0, outage_end=8.0,
    )
    assert not result["baseline"]["summary"]["optical_cann_preprocess"]
    assert result["processed"]["summary"]["optical_cann_preprocess"]
