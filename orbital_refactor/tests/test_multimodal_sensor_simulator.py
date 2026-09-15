import numpy as np

from adapters.infrared_image_adapter import InfraredCameraConfig
from adapters.multimodal_sensor_simulator import (
    MeasurementSource,
    MultimodalMeasurementSourceConfig,
    MultimodalSensorSimulationConfig,
    observation_from_message,
    single_spri_observation_from_message,
    select_single_epoch_observations,
    simulate_multimodal_sensor_epoch,
    simulate_multimodal_sensor_history,
)
from adapters.optical_image_adapter import OpticalCameraConfig
from adapters.radar_range_doppler_adapter import RadarRangeDopplerConfig


def _epoch():
    observer = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    target = observer + np.array([1000.0, 100.0, -50.0, 1.0, -0.2, 0.1])
    config = MultimodalSensorSimulationConfig(
        optical=OpticalCameraConfig(read_noise_sigma=0.0),
        infrared=InfraredCameraConfig(read_noise_sigma=0.0),
        radar=RadarRangeDopplerConfig(read_noise_sigma=0.0),
    )
    return simulate_multimodal_sensor_epoch(
        timestamp=2.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), config=config,
        rng_by_modality={
            modality: np.random.default_rng(seed)
            for modality, seed in (("RADAR", 1), ("INFRARED", 2), ("OPTICAL", 3))
        },
    )


def test_canonical_epoch_generates_three_physical_modalities():
    epoch = _epoch()
    assert tuple(message.modality for message in epoch.messages) == (
        "RADAR", "INFRARED", "OPTICAL",
    )
    assert set(epoch.raw_frames) == {"RADAR", "INFRARED", "OPTICAL"}
    assert all(message.measurement.shape == (2,) for message in epoch.messages)
    assert all(message.covariance.shape == (2, 2) for message in epoch.messages)


def test_single_and_swarm_wrappers_preserve_measurement_contract_exactly():
    for message in _epoch().messages:
        observation = observation_from_message(message)
        assert observation.modality == message.modality
        assert observation.frame == message.frame
        assert observation.valid_flag == message.valid_flag
        assert observation.confidence == message.confidence
        np.testing.assert_array_equal(observation.measurement, message.measurement)
        np.testing.assert_array_equal(observation.covariance, message.covariance)
        assert observation.metadata["observation_id"] == message.information_id


def test_single_spri_adapter_converts_body_camera_contract_without_filter_change():
    messages = {message.modality: message for message in _epoch().messages}
    infrared = single_spri_observation_from_message(messages["INFRARED"])
    optical = single_spri_observation_from_message(messages["OPTICAL"])
    radar = single_spri_observation_from_message(messages["RADAR"])
    np.testing.assert_allclose(
        infrared.measurement,
        [np.arctan2(100.0, 1000.0), np.arctan2(-50.0, np.hypot(1000.0, 100.0))],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(optical.measurement, [-20.0, -2.0], atol=1.0e-3)
    assert not optical.valid_flag
    assert infrared.frame == optical.frame == radar.frame == "SPRI"
    assert np.all(np.linalg.eigvalsh(infrared.covariance) >= 0.0)


def test_modality_random_streams_are_independent():
    first = _epoch()
    second = _epoch()
    for left, right in zip(first.messages, second.messages):
        np.testing.assert_array_equal(left.measurement, right.measurement)
        np.testing.assert_array_equal(left.covariance, right.covariance)


def test_camera_modalities_accept_independent_mounting_attitudes():
    observer = np.zeros(6)
    target = np.array([1000.0, 100.0, -50.0, 0.0, 0.0, 0.0])
    epoch = simulate_multimodal_sensor_epoch(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz_by_modality={
            "OPTICAL": np.array([1.0, 0.0, 0.0, 0.0]),
            "INFRARED": np.array([1.0, 0.0, 0.0, 0.0]),
        },
    )
    assert len(epoch.messages) == 3


def test_missing_camera_attitude_is_rejected_explicitly():
    with np.testing.assert_raises(ValueError):
        simulate_multimodal_sensor_epoch(
            timestamp=0.0, observer_id="a", target_id="b",
            observer_state=np.zeros(6),
            target_state=np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            quaternion_i2b_wxyz_by_modality={
                "OPTICAL": np.array([1.0, 0.0, 0.0, 0.0]),
            },
        )


def test_analytic_source_returns_original_objects_without_rendering():
    originals = tuple(observation_from_message(message) for message in _epoch().messages)
    selected = select_single_epoch_observations(
        source_config=MultimodalMeasurementSourceConfig(source="analytic"),
        analytic_observations=originals,
    )
    assert selected == originals
    assert all(left is right for left, right in zip(selected, originals))


def test_raw_source_requires_explicit_arguments_and_uses_shared_contract():
    config = MultimodalMeasurementSourceConfig(
        source=MeasurementSource.RAW_FRONTEND,
        sensors=MultimodalSensorSimulationConfig(
            optical=OpticalCameraConfig(read_noise_sigma=0.0),
            infrared=InfraredCameraConfig(read_noise_sigma=0.0),
            radar=RadarRangeDopplerConfig(read_noise_sigma=0.0),
        ),
    )
    with np.testing.assert_raises(ValueError):
        select_single_epoch_observations(
            source_config=config, analytic_observations=(),
        )
    observer = np.zeros(6)
    selected = select_single_epoch_observations(
        source_config=config, analytic_observations=(),
        raw_frontend_arguments={
            "timestamp": 0.0, "observer_id": "a", "target_id": "b",
            "observer_state": observer,
            "target_state": np.array([1000.0, 100.0, -50.0, 0.0, 0.0, 0.0]),
            "quaternion_i2b_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        },
    )
    assert tuple(item.modality for item in selected) == (
        "RADAR", "INFRARED", "OPTICAL",
    )
    assert all(item.source_type == "RAW_SENSOR" for item in selected)


def test_unknown_measurement_source_is_rejected():
    with np.testing.assert_raises(ValueError):
        MultimodalMeasurementSourceConfig(source="implicit_guess")


def test_measurement_source_normalizes_and_validates_calibration_keys():
    marker = object()
    config = MultimodalMeasurementSourceConfig(
        covariance_calibration_by_modality={"infrared": marker},
    )
    assert config.covariance_calibration_by_modality == {"INFRARED": marker}
    with np.testing.assert_raises(ValueError):
        MultimodalMeasurementSourceConfig(
            covariance_calibration_by_modality={"unknown": marker},
        )


def test_shared_epoch_forwards_opt_in_calibration_to_frontend():
    class Calibration:
        apply_bias_correction = False

        @staticmethod
        def applicability(*, config, peak_snr):
            del config, peak_snr
            return True, "test_applicable"

        @staticmethod
        def lookup(*, coordinate_xy, principal_xy, peak_snr):
            del coordinate_xy, principal_xy, peak_snr
            return np.eye(2) * 3.0e-6, np.zeros(2), {"profile": "test"}

    observer = np.zeros(6)
    target = np.array([1000.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    epoch = simulate_multimodal_sensor_epoch(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        covariance_calibration_by_modality={"OPTICAL": Calibration()},
    )
    optical = next(item for item in epoch.messages if item.modality == "OPTICAL")
    np.testing.assert_array_equal(optical.covariance, np.eye(2) * 3.0e-6)
    assert optical.metadata["covariance_source"] == "optical_empirical_table"


def test_history_pairs_single_and_swarm_interfaces_and_visibility():
    timestamps = np.array([0.0, 2.0, 4.0])
    observer = np.zeros((3, 6))
    target = np.tile(
        np.array([1000.0, 100.0, -50.0, 0.0, 0.0, 0.0]), (3, 1),
    )
    attitudes = {
        modality: np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1))
        for modality in ("OPTICAL", "INFRARED")
    }
    history = simulate_multimodal_sensor_history(
        timestamps=timestamps, observer_id="a", target_id="b",
        observer_state_history=observer, target_state_history=target,
        quaternion_i2b_wxyz_history_by_modality=attitudes,
        valid_history_by_modality={
            "OPTICAL": np.array([True, False, True]),
        },
        random_seed=12,
    )
    assert len(history.messages) == len(history.observations) == 9
    optical = [item for item in history.messages if item.modality == "OPTICAL"]
    assert [item.valid_flag for item in optical] == [True, False, True]
    for message, observation in zip(history.messages, history.observations):
        np.testing.assert_array_equal(message.measurement, observation.measurement)
        np.testing.assert_array_equal(message.covariance, observation.covariance)
        assert message.valid_flag == observation.valid_flag


def test_history_validates_state_attitude_and_visibility_shapes():
    with np.testing.assert_raises(ValueError):
        simulate_multimodal_sensor_history(
            timestamps=[0.0, 2.0], observer_id="a", target_id="b",
            observer_state_history=np.zeros((1, 6)),
            target_state_history=np.zeros((2, 6)),
            quaternion_i2b_wxyz_history_by_modality={},
        )
