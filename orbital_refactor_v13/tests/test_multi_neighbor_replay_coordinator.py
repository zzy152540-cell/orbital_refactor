import numpy as np

from cooperative.exact_transport_protocol import build_exact_transport_state_message
from cooperative.exact_transport_accumulator import ExactTransportAccumulator
from cooperative.multi_neighbor_replay_coordinator import MultiNeighborReplayCoordinator
from cooperative.multi_neighbor_schmidt import initialize_multi_neighbor_schmidt
from interfaces.data_objects import CovarianceTransportEvent, ObservationMessage
from orbital_core.measurements import measure_relative_range
from orbital_core.dynamics import numerical_jacobian_discrete, rk4_step_absolute


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


def test_last_ack_checkpoint_is_pinned_beyond_fixed_lag_window():
    state, messages, _ = _case()
    first_message = messages["left"]
    coordinator = MultiNeighborReplayCoordinator(
        state, process_noise_acceleration=0.0, history_window=1.0
    )
    assert coordinator.apply_state_message(first_message).accepted
    accumulator = ExactTransportAccumulator(
        source_node_id="left", lineage_id="left:0",
        reference_timestamp=0.0,
        reference_state=first_message.state_estimate,
        reference_covariance=first_message.covariance,
    )
    remote_state = first_message.state_estimate.copy()
    for timestamp in (1.0, 2.0, 3.0, 4.0):
        transition = numerical_jacobian_discrete(
            lambda value: rk4_step_absolute(value, 1.0), remote_state
        )
        remote_state = rk4_step_absolute(remote_state, 1.0)
        accumulator.append(
            timestamp=timestamp, updated_state=remote_state,
            error_transition=transition,
            independent_process_noise=np.zeros((6, 6)),
            information_ids=(f"left:prediction:{int(timestamp)}",),
            event_error_transition=np.eye(6),
            event_process_noise=np.zeros((6, 6)),
        )
        coordinator.advance(timestamp)
    assert 0.0 not in coordinator.checkpoint_timestamps
    assert coordinator.oldest_pinned_timestamp == 0.0
    recovered = coordinator.apply_state_message(accumulator.build_message())
    assert recovered.accepted
    assert coordinator.oldest_pinned_timestamp == 4.0


def test_pinned_age_limit_requires_explicit_new_lineage_resynchronization():
    state, messages, _ = _case()
    coordinator = MultiNeighborReplayCoordinator(
        state, process_noise_acceleration=0.0,
        history_window=1.0, max_pinned_age=2.0,
    )
    assert coordinator.apply_state_message(messages["left"]).accepted
    for timestamp in (1.0, 2.0, 3.0):
        coordinator.advance(timestamp)
    assert coordinator.resynchronization_requirements[("left", "left:0")] == "max_pinned_age_exceeded"
    assert coordinator.apply_state_message(messages["left"]).reason == "resync_required"

    baseline = coordinator.establish_resynchronized_link(
        neighbor_id="left", lineage_id="left:resync:1"
    )
    transition = np.eye(6) * 0.95; noise = np.eye(6) * 0.01
    updated = baseline.state_estimate + np.array([0.5, 0, 0, 0, 0, 0])
    event = CovarianceTransportEvent(
        timestamp=3.0, state_estimate=updated,
        error_transition=transition, independent_process_noise=noise,
        information_ids=("left:resync-update",),
    )
    message = build_exact_transport_state_message(
        source_node_id="left", timestamp=3.0,
        reference_timestamp=baseline.timestamp,
        reference_state=baseline.state_estimate,
        reference_covariance=baseline.covariance,
        updated_state=updated, error_transition=transition,
        independent_process_noise=noise, lineage_id=baseline.lineage_id,
        transport_events=(event,),
    )
    assert coordinator.apply_state_message(
        message, expected_lineage_id="left:resync:1"
    ).accepted


def test_retained_event_limit_marks_link_for_resynchronization():
    state, messages, observation = _case()
    coordinator = MultiNeighborReplayCoordinator(
        state, max_retained_events=1
    )
    assert coordinator.apply_state_message(messages["left"]).accepted
    coordinator.apply_observation(observation)
    requirement = coordinator.resynchronization_requirements
    assert requirement[("left", "left:0")] == "max_retained_events_exceeded"
