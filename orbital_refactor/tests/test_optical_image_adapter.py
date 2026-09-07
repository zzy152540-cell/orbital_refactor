import numpy as np

from adapters.optical_image_adapter import (
    OpticalCameraConfig,
    optical_frame_to_observation_message,
    render_optical_point_source_frame,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology


IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


def _states(relative_position=(1000.0, 100.0, -50.0)):
    observer = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    target = observer.copy()
    target[:3] += np.asarray(relative_position, dtype=float)
    return observer, target


def test_noiseless_image_centroid_recovers_normalized_uv():
    config = OpticalCameraConfig(
        width=64, height=64, focal_length_x_pixels=50.0,
        focal_length_y_pixels=50.0, read_noise_sigma=0.0,
    )
    observer, target = _states()
    frame = render_optical_point_source_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=IDENTITY_QUATERNION,
        config=config,
    )
    message = optical_frame_to_observation_message(frame, config=config)

    assert frame.image.shape == (64, 64)
    assert message.valid_flag
    assert 0.0 <= message.metadata["frontend_quality_score"] <= 1.0
    assert message.metadata["peak_snr"] > 0.0
    np.testing.assert_allclose(message.measurement, [0.1, -0.05], atol=2e-6)
    assert message.metadata["raw_source_type"] == "POINT_SOURCE_IMAGE"


def test_target_outside_image_produces_invalid_observation():
    config = OpticalCameraConfig(
        width=32, height=32, focal_length_x_pixels=50.0,
        focal_length_y_pixels=50.0, read_noise_sigma=0.0,
    )
    observer, target = _states((1000.0, 1000.0, 0.0))
    frame = render_optical_point_source_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=IDENTITY_QUATERNION, config=config,
    )
    message = optical_frame_to_observation_message(frame, config=config)

    assert not frame.target_in_frame
    assert not message.valid_flag


def test_noisy_centroid_is_repeatable_and_nearly_unbiased():
    config = OpticalCameraConfig(
        width=48, height=48, focal_length_x_pixels=60.0,
        focal_length_y_pixels=60.0, source_peak=500.0,
        read_noise_sigma=2.0,
    )
    observer, target = _states((1000.0, 70.0, -30.0))
    estimates = []
    for seed in range(40):
        frame = render_optical_point_source_frame(
            timestamp=0.0, observer_id="a", target_id="b",
            observer_state=observer, target_state=target,
            quaternion_i2b_wxyz=IDENTITY_QUATERNION,
            config=config, rng=np.random.default_rng(seed),
        )
        message = optical_frame_to_observation_message(frame, config=config)
        assert message.valid_flag
        estimates.append(message.measurement)
    np.testing.assert_allclose(
        np.mean(estimates, axis=0), [0.07, -0.03], atol=3e-4,
    )


def test_image_derived_optical_message_runs_through_network_schmidt():
    config = OpticalCameraConfig(
        width=64, height=64, focal_length_x_pixels=50.0,
        focal_length_y_pixels=50.0, read_noise_sigma=0.0,
    )
    observer, target = _states()
    frame = render_optical_point_source_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=IDENTITY_QUATERNION, config=config,
    )
    message = optical_frame_to_observation_message(frame, config=config)
    history = run_network_schmidt_filter(
        timestamps=np.array([0.0, 2.0]),
        initial_state_by_node={"a": observer, "b": target},
        initial_covariance_by_node={"a": np.eye(6), "b": np.eye(6)},
        topology=chain_topology(["a", "b"]),
        observation_messages=[message], process_noise_acceleration=0.0,
    )

    assert np.isfinite(history.active_state_history_by_node["a"]).all()
    assert set(history.nis_history_by_node["a"][0]) == {
        message.information_id,
    }
