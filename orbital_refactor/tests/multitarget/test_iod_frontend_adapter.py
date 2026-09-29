import numpy as np

from adapters.multimodal_sensor_simulator import simulate_multimodal_sensor_epoch
from orbital_core.coordinates import dcm_to_quat_wxyz
from orbital_core.coordinates import build_rtn_quaternion
from tracking import (
    iod_observations_from_messages,
    tracking_observation_from_message,
)


def _camera_basis(boresight):
    forward = np.asarray(boresight, dtype=float)
    forward = forward / np.linalg.norm(forward)
    lateral = np.cross([0.0, 0.0, 1.0], forward)
    lateral = lateral / np.linalg.norm(lateral)
    vertical = np.cross(forward, lateral)
    return np.column_stack((forward, lateral, vertical))


def test_raw_three_modality_messages_convert_to_radar_and_j2000_lines():
    observer = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    target = observer + np.array([1000.0, 2000.0, 3000.0, 1.0, -2.0, 0.5])
    truth_line = (target[:3] - observer[:3]) / np.linalg.norm(target[:3] - observer[:3])
    quaternion = dcm_to_quat_wxyz(_camera_basis(target[:3] - observer[:3]).T)
    epoch = simulate_multimodal_sensor_epoch(
        timestamp=4.0,
        observer_id="observer-1",
        target_id="target-1",
        observer_state=observer,
        target_state=target,
        quaternion_i2b_wxyz=quaternion,
        rng_by_modality={
            name: np.random.default_rng(index)
            for index, name in enumerate(("RADAR", "INFRARED", "OPTICAL"))
        },
    )

    values = iod_observations_from_messages(
        epoch.messages,
        observer_state_by_epoch={("observer-1", 4.0): observer},
    )

    assert [item.modality for item in values] == ["RADAR", "LOS", "LOS"]
    for item in values[1:]:
        assert np.dot(item.measurement, truth_line) > 0.999
        assert item.metadata["lineOfSightFrame"] == "J2000_ECI"

    tracking = [
        tracking_observation_from_message(
            message, q_eci2pri=build_rtn_quaternion(observer),
        )
        for message in epoch.messages
    ]
    assert {item.modality for item in tracking} == {
        "RADAR", "INFRARED", "OPTICAL",
    }
    assert all(item.frame == "SPRI" for item in tracking)
    assert all(np.all(np.isfinite(item.measurement)) for item in tracking)
