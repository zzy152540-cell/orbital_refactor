import numpy as np

from adapters.frontend_covariance_calibration import (
    FrontendCovarianceCalibrationEntry,
    FrontendCovarianceCalibrationTable,
)
from adapters.optical_image_adapter import (
    OpticalCameraConfig,
    optical_frame_to_observation_message,
    render_optical_point_source_frame,
)
from adapters.radar_range_doppler_adapter import (
    RadarRangeDopplerConfig,
    radar_frame_to_observation_message,
    render_radar_range_doppler_frame,
)
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)


def test_optical_empirical_covariance_and_bias_correction_are_opt_in():
    covariance = np.diag([1.0e-7, 2.0e-7])
    bias = np.array([1.0e-4, -2.0e-4])
    entry = FrontendCovarianceCalibrationEntry(
        "OPTICAL", "off_axis_low_snr", 100.0, covariance, bias,
    )
    table = FrontendCovarianceCalibrationTable(
        (entry,), "OPTICAL", "test", apply_bias_correction=True,
    )
    config = OpticalCameraConfig(read_noise_sigma=0.0)
    observer = np.zeros(6)
    target = np.array([1000.0, 100.0, 0.0, 0.0, 0.0, 0.0])
    frame = render_optical_point_source_frame(
        timestamp=0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), config=config,
    )
    default = optical_frame_to_observation_message(frame, config=config)
    calibrated = optical_frame_to_observation_message(
        frame, config=config, covariance_calibration=table,
    )
    np.testing.assert_allclose(calibrated.covariance, covariance)
    np.testing.assert_allclose(calibrated.measurement, default.measurement - bias)
    assert default.metadata["covariance_source"] == "reported_centroid_sigma"
    assert calibrated.metadata["covariance_source"] == "optical_empirical_table"


def test_radar_empirical_covariance_and_bias_correction_are_opt_in():
    covariance = np.diag([4.0, 1.0e-4])
    bias = np.array([1.0, -0.01])
    entry = FrontendCovarianceCalibrationEntry(
        "RADAR", "center_high_snr", 1000.0, covariance, bias,
    )
    table = FrontendCovarianceCalibrationTable(
        (entry,), "RADAR", "test", apply_bias_correction=True,
    )
    config = RadarRangeDopplerConfig(read_noise_sigma=0.0)
    observer = np.array([7e6, 0, 0, 0, 7500, 0], dtype=float)
    target = observer + np.array([1000, 100, -50, 1, -0.2, 0.1], dtype=float)
    center = np.array([
        measure_relative_range(observer, target),
        measure_relative_range_rate(observer, target),
    ])
    frame = render_radar_range_doppler_frame(
        timestamp=0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        acquisition_range_m=center[0], acquisition_range_rate_mps=center[1],
        config=config,
    )
    default = radar_frame_to_observation_message(frame, config=config)
    calibrated = radar_frame_to_observation_message(
        frame, config=config, covariance_calibration=table,
    )
    np.testing.assert_allclose(calibrated.covariance, covariance)
    np.testing.assert_allclose(calibrated.measurement, default.measurement - bias)
    assert calibrated.metadata["covariance_source"] == "radar_empirical_table"


def test_frontend_covariance_table_rejects_mixed_modalities():
    entry = FrontendCovarianceCalibrationEntry(
        "RADAR", "center", 100.0, np.eye(2), np.zeros(2),
    )
    with np.testing.assert_raises(ValueError):
        FrontendCovarianceCalibrationTable((entry,), "OPTICAL", "bad")
