import numpy as np

from experiments.cann_phase_projection_feedback import (
    run_cann_phase_projection_feedback, run_cann_phase_projection_gain_sweep,
)


def test_phase_projection_only_changes_outage_samples():
    result = run_cann_phase_projection_feedback(
        duration=20.0, dt=2.0, outage_start=8.0, outage_end=12.0,
    )
    assert np.all(result["baseline_error_m"][~result["outage"]]
                  == result["projected_error_m"][~result["outage"]])
    assert result["summary"]["cue_count"] > 0
    assert np.isfinite(result["summary"]["projected_outage_rmse_m"])


def test_zero_gain_sweep_matches_original_filter():
    result = run_cann_phase_projection_gain_sweep(
        gains=(0.0, 0.01), duration=20.0, dt=2.0,
        outage_start=8.0, outage_end=12.0,
    )
    zero = result["rows"][0]
    assert np.isclose(zero["outage_improvement_m"], 0.0, atol=1e-8)
    assert np.isclose(zero["maximum_projection_change_m"], 0.0, atol=1e-8)
