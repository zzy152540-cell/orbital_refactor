from experiments.single_multi_measurement_source_shadow import (
    build_moderate_calibrated_measurement_source,
    run_single_multi_measurement_source_shadow,
)


def test_single_multi_measurement_source_shadow_has_four_paired_arms():
    report = run_single_multi_measurement_source_shadow(
        duration=2.0, dt=2.0, seed=0,
    )
    assert report.shared_truth_and_initialization
    assert report.production_default_unchanged
    assert report.single_observer_raan_deg == 20.0
    assert report.single_all_modalities_observable
    assert report.empirical_calibration_modalities == ()
    assert isinstance(report.raw_frontend_nis_alignment_ready, bool)
    assert [(arm.scope, arm.source) for arm in report.arms] == [
        ("single", "analytic"), ("single", "raw_frontend"),
        ("walker", "analytic"), ("walker", "raw_frontend"),
    ]
    assert all(arm.position_rmse_m > 0.0 for arm in report.arms)
    single = [arm for arm in report.arms if arm.scope == "single"]
    assert all(arm.valid_radar_count > 0 for arm in single)
    assert all(arm.valid_infrared_count > 0 for arm in single)
    assert all(arm.valid_optical_count > 0 for arm in single)
    walker = [arm for arm in report.arms if arm.scope == "walker"]
    assert all(arm.valid_radar_count > 0 for arm in walker)
    assert all(arm.valid_infrared_count > 0 for arm in walker)
    assert all(arm.valid_optical_count > 0 for arm in walker)


def test_moderate_calibrated_source_matches_existing_work_point():
    sensors, calibrations = build_moderate_calibrated_measurement_source(
        calibration_seed=7, calibration_samples=2,
    )
    assert sensors.optical.focal_length_x_pixels == 400.0
    assert sensors.infrared.focal_length_x_pixels == 600.0
    assert set(calibrations) == {"RADAR", "INFRARED", "OPTICAL"}
    assert all(table.apply_bias_correction for table in calibrations.values())
