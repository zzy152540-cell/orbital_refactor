import numpy as np

from cooperative.multi_neighbor_schmidt import (
    initialize_multi_neighbor_schmidt,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_update,
)
from cooperative.schmidt_event_replay import replay_schmidt_events
from interfaces.data_objects import ObservationMessage
from orbital_core.measurements import measure_relative_range


def _observation(timestamp, active, neighbor, suffix):
    return ObservationMessage(
        message_id=f"range-{suffix}", observer_id="a", target_id="b",
        timestamp=timestamp, modality="RANGE",
        measurement=np.array([measure_relative_range(active, neighbor) + 0.5]),
        covariance=np.array([[1.0]]),
    )


def test_event_replay_matches_manual_predict_update_sequence():
    active = np.array([7e6, 0, 0, 0, 7500, 0.0])
    neighbor = active + np.array([1000, 50, 0, 0, 0, 0.0])
    checkpoint = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="a", active_state=active,
        active_covariance=np.eye(6) * 4.0,
        neighbor_state_by_id={"b": neighbor},
        neighbor_covariance_by_id={"b": np.eye(6) * 9.0},
    )
    first = _observation(0.0, active, neighbor, "0")
    manually = multi_neighbor_schmidt_update(checkpoint, first).state
    manually = multi_neighbor_schmidt_predict(
        manually, 1.0, process_noise_acceleration=0.0
    )
    second = _observation(
        1.0, manually.active_state, manually.neighbor_state_by_id["b"], "1"
    )
    manually = multi_neighbor_schmidt_update(manually, second).state

    replay = replay_schmidt_events(
        checkpoint, current_timestamp=1.0, observations=[second, first],
        process_noise_acceleration=0.0,
    )
    assert np.allclose(replay.state.active_state, manually.active_state)
    assert np.allclose(replay.state.joint_covariance, manually.joint_covariance)
    assert replay.replayed_information_ids == ("range-0", "range-1")


def test_replay_skips_information_already_present_at_checkpoint():
    active = np.array([7e6, 0, 0, 0, 7500, 0.0])
    neighbor = active + np.array([1000, 0, 0, 0, 0, 0.0])
    state = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="a", active_state=active,
        active_covariance=np.eye(6), neighbor_state_by_id={"b": neighbor},
        neighbor_covariance_by_id={"b": np.eye(6)},
    )
    observation = _observation(0.0, active, neighbor, "same")
    updated = multi_neighbor_schmidt_update(state, observation).state
    replay = replay_schmidt_events(
        updated, current_timestamp=0.0, observations=[observation]
    )
    assert replay.replayed_information_ids == ()
    assert np.allclose(replay.state.joint_covariance, updated.joint_covariance)
