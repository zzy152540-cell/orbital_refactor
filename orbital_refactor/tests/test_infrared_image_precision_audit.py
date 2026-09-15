from experiments.infrared_image_precision_audit import (
    run_infrared_image_precision_audit,
)


def test_small_image_precision_audit_reports_accuracy_and_coverage():
    report = run_infrared_image_precision_audit(
        duration=4.0, dt=2.0, focal_lengths_pixels=(300.0,),
        psf_sigmas_pixels=(1.2,), peak_snrs=(100.0,),
        monte_carlo_samples=8,
    )
    assert len(report.records) == 1
    record = report.records[0]
    assert 0.0 <= record.detection_fraction <= 1.0
    assert record.centroid_rmse_pixels > 0.0
    assert record.angular_rmse_deg > 0.0
    assert 0.0 <= record.fixed_boresight_fov_coverage_fraction <= 1.0
