import numpy as np

from experiments.radar_optical_error_budget_audit import (
    OpticalErrorProfile,
    RadarErrorProfile,
    run_radar_optical_error_budget_audit,
)


def test_radar_optical_error_budget_reports_detection_bias_and_nis():
    report = run_radar_optical_error_budget_audit(
        monte_carlo_samples=8,
        optical_profiles=(
            OpticalErrorProfile("baseline"),
            OpticalErrorProfile("biased", fixed_pixel_bias_x=0.2),
        ),
        radar_profiles=(
            RadarErrorProfile("baseline"),
            RadarErrorProfile("biased", range_bias_m=2.0),
        ),
    )
    assert len(report.records) == 16
    assert {record.modality for record in report.records} == {"OPTICAL", "RADAR"}
    assert all(0.0 <= record.detection_fraction <= 1.0 for record in report.records)
    assert all(np.isfinite(record.mean_nis) for record in report.records)
    optical_bias = [
        record for record in report.records
        if record.modality == "OPTICAL" and record.profile == "biased"
    ]
    radar_bias = [
        record for record in report.records
        if record.modality == "RADAR" and record.profile == "biased"
    ]
    assert any(abs(record.component_0_bias) > 1.0e-5 for record in optical_bias)
    assert any(abs(record.component_0_bias) > 1.0 for record in radar_bias)


def test_radar_optical_error_budget_requires_covariance_sample_count():
    with np.testing.assert_raises(ValueError):
        run_radar_optical_error_budget_audit(monte_carlo_samples=1)
