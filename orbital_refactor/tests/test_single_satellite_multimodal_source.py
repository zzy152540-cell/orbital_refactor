import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import (
    MultimodalMeasurementSourceConfig,
    MultimodalSensorSimulationConfig,
)
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig
from experiments.single_satellite_cann_comparison import (
    _tracking_camera_basis_spri,
    run_single_satellite_cann_comparison,
)


def test_tracking_camera_basis_does_not_mutate_state_history_view():
    history = np.array([[1000.0, 200.0, -50.0, 1.0, 2.0, 3.0]])
    before = history.copy()
    basis = _tracking_camera_basis_spri(history[0, :3])
    np.testing.assert_array_equal(history, before)
    np.testing.assert_allclose(basis.T @ history[0, :3], [np.linalg.norm(history[0, :3]), 0.0, 0.0], atol=1.0e-10)


def test_single_satellite_default_measurement_source_remains_analytic():
    result = run_single_satellite_cann_comparison(
        duration=2.0, dt=2.0, enable_cann=False,
    )
    assert result["summary"]["measurement_source"] == "analytic"


def test_single_satellite_accepts_shared_raw_frontend_source():
    source = MultimodalMeasurementSourceConfig(
        source="raw_frontend",
        sensors=MultimodalSensorSimulationConfig(
            optical=OpticalCameraConfig(
                width=128, height=128, focal_length_x_pixels=40.0,
                focal_length_y_pixels=40.0, read_noise_sigma=0.0,
            ),
            infrared=InfraredCameraConfig(
                width=128, height=128, focal_length_x_pixels=40.0,
                focal_length_y_pixels=40.0, read_noise_sigma=0.0,
            ),
            radar=RadarRangeDopplerConfig(read_noise_sigma=0.0),
        ),
    )
    result = run_single_satellite_cann_comparison(
        duration=2.0, dt=2.0, enable_cann=False,
        measurement_source_config=source,
    )
    assert result["summary"]["measurement_source"] == "raw_frontend"
    assert np.isfinite(result["summary"]["position_rmse_m"])


def test_raw_frontend_rejects_simultaneous_measurement_preprocessing():
    source = MultimodalMeasurementSourceConfig(source="raw_frontend")
    with np.testing.assert_raises(ValueError):
        run_single_satellite_cann_comparison(
            duration=2.0, dt=2.0, enable_cann=False,
            optical_cann_preprocess=True,
            measurement_source_config=source,
        )
