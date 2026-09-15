import numpy as np

from adapters.infrared_covariance_calibration import (
    InfraredCovarianceCalibrationEntry,
    InfraredCovarianceCalibrationTable,
)
from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_frame_to_observation_message,
    render_infrared_point_source_frame,
)


def test_empirical_covariance_is_opt_in_and_selected_by_field():
    center_covariance = np.diag([1.0e-8, 2.0e-8])
    off_axis_covariance = np.diag([3.0e-8, 4.0e-8])
    table = InfraredCovarianceCalibrationTable(entries=(
        InfraredCovarianceCalibrationEntry(
            50.0, 100.0, "center", center_covariance, np.zeros(2)
        ),
        InfraredCovarianceCalibrationEntry(
            50.0, 100.0, "off_axis", off_axis_covariance,
            np.array([1.0e-4, -2.0e-4]),
        ),
    ), profile="test", off_axis_radius_pixels=4.0)
    config = InfraredCameraConfig(
        width=64, height=64, focal_length_x_pixels=50.0,
        focal_length_y_pixels=50.0, source_peak=100.0,
        read_noise_sigma=0.0,
    )
    observer = np.zeros(6)
    target = np.array([1000.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    frame = render_infrared_point_source_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), config=config,
    )
    default = infrared_frame_to_observation_message(frame, config=config)
    calibrated = infrared_frame_to_observation_message(
        frame, config=config, covariance_calibration=table,
    )
    assert default.metadata["covariance_source"] == "reported_centroid_sigma"
    assert calibrated.metadata["covariance_source"] == "infrared_empirical_table"
    np.testing.assert_allclose(calibrated.covariance, off_axis_covariance)
    assert not np.array_equal(default.covariance, calibrated.covariance)


def test_bias_correction_is_explicit_and_does_not_change_covariance():
    covariance = np.diag([1.0e-8, 2.0e-8])
    bias = np.array([1.0e-4, -2.0e-4])
    entry = InfraredCovarianceCalibrationEntry(
        50.0, 100.0, "off_axis", covariance, bias,
    )
    uncorrected_table = InfraredCovarianceCalibrationTable(
        entries=(entry,), profile="test", apply_bias_correction=False,
    )
    corrected_table = InfraredCovarianceCalibrationTable(
        entries=(entry,), profile="test", apply_bias_correction=True,
    )
    config = InfraredCameraConfig(
        width=64, height=64, focal_length_x_pixels=50.0,
        focal_length_y_pixels=50.0, source_peak=100.0,
        read_noise_sigma=0.0,
    )
    frame = render_infrared_point_source_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=np.zeros(6),
        target_state=np.array([1000.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), config=config,
    )
    uncorrected = infrared_frame_to_observation_message(
        frame, config=config, covariance_calibration=uncorrected_table,
    )
    corrected = infrared_frame_to_observation_message(
        frame, config=config, covariance_calibration=corrected_table,
    )
    np.testing.assert_allclose(
        corrected.measurement, uncorrected.measurement - bias,
    )
    np.testing.assert_allclose(corrected.covariance, uncorrected.covariance)
    assert corrected.metadata["infrared_bias_correction_applied"]
