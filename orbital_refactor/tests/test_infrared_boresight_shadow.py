from experiments.infrared_boresight_shadow import run_infrared_boresight_shadow


def test_boresight_shadow_compares_truth_and_strictly_lagged_modes():
    report = run_infrared_boresight_shadow(
        seeds=(0,), duration=4.0, dt=2.0, calibration_samples=12,
    )
    assert [run.mode for run in report.runs] == [
        "exact_center_truth_tracking_diagnostic", "lagged_posterior_prediction",
    ]
    assert all(0.0 <= run.outage_valid_fraction <= 1.0 for run in report.runs)
    assert report.runs[1].outage_boresight_mean_error_deg > 0.0


def test_delayed_outage_is_causally_identical_before_start():
    report = run_infrared_boresight_shadow(
        seeds=(0,), duration=4.0, dt=2.0, calibration_samples=12,
        optical_outage_start=2.0,
    )
    assert all(run.pre_outage_state_max_abs_difference == 0.0
               for run in report.runs)
