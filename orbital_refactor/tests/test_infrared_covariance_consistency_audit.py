from experiments.infrared_covariance_consistency_audit import (
    run_infrared_covariance_consistency_audit,
)


def test_covariance_consistency_audit_uses_independent_seeds():
    report = run_infrared_covariance_consistency_audit(
        calibration_samples=8, validation_samples=8,
        focal_lengths_pixels=(600.0,), peak_snrs=(1000.0,),
    )
    assert report.calibration_seed != report.validation_seed
    assert len(report.records) == 2
    assert all(record.validation_sample_count == 8 for record in report.records)
    assert all(record.fixed_mean_nis >= 0.0 for record in report.records)
    assert all(record.empirical_mean_nis >= 0.0 for record in report.records)
    assert all(record.bias_corrected_mean_nis >= 0.0
               for record in report.records)
