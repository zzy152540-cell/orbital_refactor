import numpy as np

from cooperative.exact_transport_protocol import build_exact_transport_state_message
from cooperative.multi_neighbor_replay_coordinator import MultiNeighborReplayCoordinator
from cooperative.multi_neighbor_schmidt import initialize_multi_neighbor_schmidt
from interfaces.data_objects import CovarianceTransportEvent, ObservationMessage
from orbital_core.measurements import measure_relative_range


def _case():
    active = np.array([7e6, 0, 0, 0, 7500, 0.0])
    neighbors = {
        "left": active + np.array([-1000, 20, 0, 0, 0, 0.0]),
        "right": active + np.array([1200, -30, 0, 0, 0, 0.0]),
    }
    covariance = np.eye(6) * 4.0
    state = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="center", active_state=active,
        active_covariance=covariance,
        neighbor_state_by_id=neighbors,
        neighbor_covariance_by_id={key: covariance for key in neighbors},
    )
    messages = {}
    for index, neighbor_id in enumerate(("left", "right"), start=1):
        transition = np.eye(6); transition[:3, :3] *= 0.7 + 0.05 * index
        noise = np.eye(6) * (0.1 * index)
        updated = neighbors[neighbor_id] + np.array([index, -index, 0, 0, 0, 0])
        event = CovarianceTransportEvent(
            timestamp=0.0, state_estimate=updated,
            error_transition=transition,
            independent_process_noise=noise,
            information_ids=(f"{neighbor_id}:absolute:0",),
        )
        messages[neighbor_id] = build_exact_transport_state_message(
            source_node_id=neighbor_id, timestamp=0.0,
            reference_timestamp=0.0, reference_state=neighbors[neighbor_id],
            reference_covariance=covariance, updated_state=updated,
            error_transition=transition, independent_process_noise=noise,
            lineage_id=f"{neighbor_id}:0", transport_events=(event,),
        )
    observation = ObservationMessage(
        message_id="center-left-range", observer_id="center", target_id="left",
        timestamp=0.0, modality="RANGE",
        measurement=np.array([measure_relative_range(active, neighbors["left"]) + 0.5]),
        covariance=np.array([[1.0]]),
    )
    return state, messages, observation


def _run(order):
    state, messages, observation = _case()
    coordinator = MultiNeighborReplayCoordinator(state, process_noise_acceleration=0.0)
    for item in order:
        if item == "observation":
            coordinator.apply_observation(observation)
        else:
            result = coordinator.apply_state_message(
                messages[item], expected_lineage_id=f"{item}:0"
            )
            assert result.accepted
    return coordinator.state


def test_two_neighbor_events_share_one_order_independent_node_timeline():
    first = _run(("observation", "right", "left"))
    second = _run(("left", "right", "observation"))
    assert np.allclose(first.active_state, second.active_state)
    assert np.allclose(first.joint_covariance, second.joint_covariance)
    assert first.transport_information_ids == (
        "left:absolute:0", "right:absolute:0"
    )
    assert first.information_ids == ("center-left-range",)


def test_duplicate_message_is_idempotent_and_conflict_is_rejected():
    state, messages, _ = _case()
    coordinator = MultiNeighborReplayCoordinator(state)
    first = coordinator.apply_state_message(messages["left"])
    covariance = coordinator.state.joint_covariance.copy()
    duplicate = coordinator.apply_state_message(messages["left"])
    assert first.accepted and duplicate.accepted
    assert duplicate.replayed_event_count == 0
    assert np.allclose(coordinator.state.joint_covariance, covariance)


def test_event_bundle_must_reproduce_advertised_endpoint():
    from dataclasses import replace

    state, messages, _ = _case()
    coordinator = MultiNeighborReplayCoordinator(state)
    message = messages["left"]
    bad_event = replace(
        message.transport_events[0],
        independent_process_noise=np.eye(6) * 50.0,
    )
    tampered = replace(message, transport_events=(bad_event,))
    result = coordinator.apply_state_message(tampered)
    assert not result.accepted
    assert result.reason == "event_bundle_endpoint_mismatch"
