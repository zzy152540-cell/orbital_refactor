import numpy as np

from cooperative.multi_neighbor_schmidt import initialize_multi_neighbor_schmidt
from cooperative.schmidt_transport_replay import replay_transport_event_bundle
from interfaces.data_objects import CovarianceTransportEvent, ObservationMessage
from orbital_core.measurements import measure_relative_range


def test_remote_event_is_applied_before_same_timestamp_relative_observation():
    active = np.array([7e6, 0, 0, 0, 7500, 0.0])
    neighbor = active + np.array([1000, 0, 0, 0, 0, 0.0])
    state = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="a", active_state=active,
        active_covariance=np.eye(6) * 4,
        neighbor_state_by_id={"b": neighbor},
        neighbor_covariance_by_id={"b": np.eye(6) * 9},
    )
    updated_neighbor = neighbor + np.array([5, 0, 0, 0, 0, 0])
    event = CovarianceTransportEvent(
        timestamp=0.0, state_estimate=updated_neighbor,
        error_transition=np.eye(6) * 0.9,
        independent_process_noise=np.eye(6) * 0.1,
        information_ids=("remote-update",),
    )
    observation = ObservationMessage(
        message_id="relative-after-remote", observer_id="a", target_id="b",
        timestamp=0.0, modality="RANGE",
        measurement=np.array([measure_relative_range(active, updated_neighbor)]),
        covariance=np.array([[1.0]]),
    )
    replayed = replay_transport_event_bundle(
        state, neighbor_id="b", events=[event], observations=[observation],
        current_timestamp=0.0,
    )
    assert np.allclose(replayed.neighbor_state_by_id["b"], updated_neighbor)
    assert observation.information_id in replayed.information_ids
    assert replayed.transport_information_ids == ("remote-update",)


def test_duplicate_and_stale_remote_events_are_idempotent():
    active = np.array([7e6, 0, 0, 0, 7500, 0.0])
    neighbor = active + np.array([1000, 0, 0, 0, 0, 0.0])
    state = initialize_multi_neighbor_schmidt(
        timestamp=1.0, active_node_id="a", active_state=active,
        active_covariance=np.eye(6), neighbor_state_by_id={"b": neighbor},
        neighbor_covariance_by_id={"b": np.eye(6)},
    )
    stale = CovarianceTransportEvent(
        timestamp=0.0, state_estimate=neighbor + 100.0,
        error_transition=np.eye(6) * 0.1,
        independent_process_noise=np.eye(6), information_ids=("stale",),
    )
    current = CovarianceTransportEvent(
        timestamp=1.0, state_estimate=neighbor + 1.0,
        error_transition=np.eye(6) * 0.9,
        independent_process_noise=np.eye(6) * 0.1,
        information_ids=("current",),
    )
    once = replay_transport_event_bundle(
        state, neighbor_id="b", events=[stale, current], observations=[],
        current_timestamp=1.0,
    )
    twice = replay_transport_event_bundle(
        once, neighbor_id="b", events=[current], observations=[],
        current_timestamp=1.0,
    )
    assert np.allclose(twice.joint_covariance, once.joint_covariance)
    assert np.allclose(twice.neighbor_state_by_id["b"], once.neighbor_state_by_id["b"])
    assert "stale" not in twice.transport_information_ids
