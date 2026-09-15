from experiments.radar_optical_filter_calibration_shadow import (
    run_radar_optical_filter_calibration_shadow,
)


def test_radar_optical_filter_shadow_has_paired_explicit_arms():
    report = run_radar_optical_filter_calibration_shadow(
        duration=2.0, dt=2.0, calibration_samples=8,
    )
    assert report.paired_raw_measurements
    assert report.default_filter_unchanged
    assert len(report.arms) == 6
    for modality in ("OPTICAL", "RADAR"):
        assert [arm.name for arm in report.arms if arm.modality == modality] == [
            "fixed_reported_covariance",
            "empirical_covariance",
            "bias_corrected_empirical_covariance",
        ]
    assert all(arm.position_rmse_m > 0.0 for arm in report.arms)
