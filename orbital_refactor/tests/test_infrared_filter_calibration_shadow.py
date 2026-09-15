from experiments.infrared_filter_calibration_shadow import (
    run_infrared_filter_calibration_shadow,
)


def test_filter_calibration_shadow_preserves_three_explicit_arms():
    report = run_infrared_filter_calibration_shadow(
        duration=2.0, dt=2.0, calibration_samples=12,
    )
    assert report.default_filter_unchanged
    assert [arm.name for arm in report.arms] == [
        "fixed_reported_covariance",
        "empirical_covariance",
        "bias_corrected_empirical_covariance",
    ]
    assert all(arm.position_rmse_m > 0.0 for arm in report.arms)
