import numpy as np

from experiments.cann_phase_calibration import run_cann_phase_calibration


def test_short_phase_calibration_reports_finite_delay_fit():
    result = run_cann_phase_calibration(duration=0.2, burn_in=0.1)
    assert len(result["cases"]) == 22
    assert np.isfinite(result["delay_fit"]["delay_s"])
    assert 0.0 <= result["delay_fit"]["r_squared"] <= 1.0
    assert all(np.isfinite(case.rmse_deg) for case in result["cases"])
