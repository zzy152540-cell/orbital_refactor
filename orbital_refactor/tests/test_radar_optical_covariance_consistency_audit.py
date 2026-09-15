from experiments.radar_optical_covariance_consistency_audit import (
    run_radar_optical_covariance_consistency_audit,
)


def test_radar_optical_consistency_audit_uses_held_out_samples():
    report = run_radar_optical_covariance_consistency_audit(
        calibration_samples=8, validation_samples=8,
    )
    assert report.calibration_seed != report.validation_seed
    assert len(report.records) == 4
    assert {record.modality for record in report.records} == {"RADAR", "OPTICAL"}
    assert all(record.validation_sample_count == 8 for record in report.records)
    assert all(record.fixed_mean_nis >= 0.0 for record in report.records)
    assert all(record.empirical_mean_nis >= 0.0 for record in report.records)
    assert all(record.bias_corrected_mean_nis >= 0.0 for record in report.records)
