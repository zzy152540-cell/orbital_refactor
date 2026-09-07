import numpy as np

from adapters.infrared_image_adapter import (
    InfraredCameraConfig,
    infrared_frame_to_observation_message,
    render_infrared_point_source_frame,
)
from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology
from orbital_core.measurements import measure_relative_az_el


IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


def _states(relative=(1000.0, 100.0, -50.0)):
    observer = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    target = observer.copy()
    target[:3] += np.asarray(relative)
    return observer, target


def test_noiseless_infrared_image_recovers_body_az_el():
    config = InfraredCameraConfig(
        width=64, height=64, focal_length_x_pixels=50.0,
        focal_length_y_pixels=50.0, read_noise_sigma=0.0,
    )
    observer, target = _states()
    frame = render_infrared_point_source_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=IDENTITY_QUATERNION, config=config,
    )
    message = infrared_frame_to_observation_message(frame, config=config)
    expected = measure_relative_az_el(
        observer, target, frame="BODY",
        quaternion_i2b_wxyz=IDENTITY_QUATERNION,
    )
    assert message.valid_flag
    np.testing.assert_allclose(message.measurement, expected, atol=2e-6)
    assert np.linalg.eigvalsh(message.covariance).min() > 0.0


def test_infrared_target_behind_detector_is_invalid():
    observer, target = _states((-1000.0, 0.0, 0.0))
    frame = render_infrared_point_source_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=IDENTITY_QUATERNION,
    )
    message = infrared_frame_to_observation_message(frame)
    assert not frame.target_in_frame
    assert not message.valid_flag


def test_image_derived_infrared_message_runs_through_network_schmidt():
    config = InfraredCameraConfig(
        width=64, height=64, focal_length_x_pixels=50.0,
        focal_length_y_pixels=50.0, read_noise_sigma=0.0,
    )
    observer, target = _states()
    frame = render_infrared_point_source_frame(
        timestamp=0.0, observer_id="a", target_id="b",
        observer_state=observer, target_state=target,
        quaternion_i2b_wxyz=IDENTITY_QUATERNION, config=config,
    )
    message = infrared_frame_to_observation_message(frame, config=config)
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
