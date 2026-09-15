from experiments.radar_optical_calibration_generalization_audit import (
    run_radar_optical_calibration_generalization_audit,
)


def test_generalization_audit_crosses_profiles_positions_and_snr():
    report = run_radar_optical_calibration_generalization_audit(
        calibration_samples=8, validation_samples=8,
        validation_profiles=("moderate_combined",),
    )
    assert report.calibration_seed != report.validation_seed
    assert len(report.records) == 8
    assert {record.modality for record in report.records} == {"OPTICAL", "RADAR"}
    assert any("center_low_snr" == record.operating_point
               for record in report.records)
    assert any("high_snr" in record.operating_point
               and ("off_axis" in record.operating_point
                    or "offset" in record.operating_point)
               for record in report.records)
    assert all(record.validation_sample_count == 8 for record in report.records)
    assert all(record.calibration_expected_applicable for record in report.records)
