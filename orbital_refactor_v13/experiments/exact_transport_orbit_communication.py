from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.exact_transport_protocol import (
    apply_exact_transport_state_message,
    build_exact_transport_state_message,
)
from cooperative.exact_transport_accumulator import ExactTransportAccumulator
from cooperative.message_transport import MessageChannel, TypedMessageBuffer
from cooperative.multi_neighbor_schmidt import (
    initialize_multi_neighbor_schmidt,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_update,
)
from cooperative.schmidt_event_replay import replay_schmidt_events
from cooperative.schmidt_transport_replay import replay_transport_event_bundle
from interfaces.data_objects import ObservationMessage
from orbital_core.constants import R_EARTH
from orbital_core.dynamics import make_process_noise, numerical_jacobian_discrete, rk4_step_absolute
from orbital_core.orbit_elements import keplerian_to_eci
from orbital_core.measurements import measure_relative_range

Array = np.ndarray


@dataclass(frozen=True)
class OrbitCommunicationSummary:
    scenario: str
    attempted_messages: int
    delivered_messages: int
    accepted_messages: int
    acceptance_rate: float
    mean_neighbor_position_disagreement: float
    maximum_covariance_disagreement: float
    minimum_joint_eigenvalue: float
    rejection_counts: dict[str, int]
    online_position_rmse: float
    online_velocity_rmse: float
    mean_nees: float
    mean_nis: float
    transmitted_payload_bytes: int
    ack_messages: int
    peak_history_snapshots_per_link: int


def run_exact_transport_orbit_communication_validation(
    *, seeds: int = 30, epochs: int = 10, dt: float = 2.0,
    absolute_sigma: float = 2.0, process_noise_acceleration: float = 1e-8,
) -> dict[str, OrbitCommunicationSummary]:
    """Exercise exact-transport provenance over a three-satellite orbit chain."""
    if seeds < 1 or epochs < 2:
        raise ValueError("seeds must be positive and epochs must be at least two.")
    scenarios = {
        "ideal": (0.0, 0.0, False, False, None),
        "one_epoch_delay": (0.0, dt, False, False, None),
        "one_epoch_delay_rollback": (0.0, dt, False, True, None),
        "two_epoch_delay_rollback": (0.0, 2.0 * dt, False, True, None),
        "two_epoch_delay_window_one": (0.0, 2.0 * dt, False, True, 1),
        "loss_20_percent": (0.2, 0.0, False, False, None),
        "loss_20_percent_ack": (0.2, 0.0, True, False, None),
        "delay_loss_ack_rollback": (0.2, dt, True, True, None),
    }
    totals = {name: _empty_totals() for name in scenarios}
    for seed in range(seeds):
        for name, (loss, delay, use_ack, use_rollback, history_window) in scenarios.items():
            _run_case(
                totals[name], seed=seed, scenario=name, packet_loss=loss,
                delay=delay, epochs=epochs, dt=dt,
                absolute_sigma=absolute_sigma,
                process_noise_acceleration=process_noise_acceleration,
                use_ack=use_ack,
                use_rollback=use_rollback,
                history_window_epochs=history_window,
            )
    result = {}
    for name, values in totals.items():
        accepted = values["accepted"]
        result[name] = OrbitCommunicationSummary(
            scenario=name, attempted_messages=values["attempted"],
            delivered_messages=values["delivered"], accepted_messages=accepted,
            acceptance_rate=accepted / values["attempted"],
            mean_neighbor_position_disagreement=(values["position_sum"] / max(accepted, 1)),
            maximum_covariance_disagreement=values["covariance_max"],
            minimum_joint_eigenvalue=values["minimum_eigenvalue"],
            rejection_counts=dict(values["rejections"]),
            online_position_rmse=float(np.sqrt(values["position_error_sq"] / values["state_component_count"])),
            online_velocity_rmse=float(np.sqrt(values["velocity_error_sq"] / values["state_component_count"])),
            mean_nees=values["nees_sum"] / values["state_sample_count"],
            mean_nis=values["nis_sum"] / values["nis_count"],
            transmitted_payload_bytes=(values["attempted"] * 156 + values["transport_event_count"] * 78) * 8,
            ack_messages=accepted,
            peak_history_snapshots_per_link=values["peak_history"],
        )
    return result


def _run_case(totals, *, seed, scenario, packet_loss, delay, epochs, dt,
              absolute_sigma, process_noise_acceleration, use_ack, use_rollback,
              history_window_epochs):
    rng = np.random.default_rng(20260820 + 101 * seed)
    base = keplerian_to_eci(R_EARTH + 700e3, 0.001, np.deg2rad(23.0), 0.0, 0.0, 0.0)
    node_ids = ("sat_01", "sat_02", "sat_03")
    offsets = (-1200.0, 0.0, 1300.0)
    truth = {node: base + np.array([offsets[i], 0, 0, 0, 0, 0]) for i, node in enumerate(node_ids)}
    prior_covariance = np.diag([10.0, 10.0, 10.0, 0.02, 0.02, 0.02]) ** 2
    sender_state = {
        node: truth[node] + rng.multivariate_normal(np.zeros(6), prior_covariance)
        for node in node_ids
    }
    sender_covariance = {node: prior_covariance.copy() for node in node_ids}
    edges = (("sat_01", "sat_02"), ("sat_02", "sat_01"),
             ("sat_02", "sat_03"), ("sat_03", "sat_02"))
    receiver = {
        edge: initialize_multi_neighbor_schmidt(
            timestamp=0.0, active_node_id=edge[0], active_state=sender_state[edge[0]],
            active_covariance=sender_covariance[edge[0]],
            neighbor_state_by_id={edge[1]: sender_state[edge[1]]},
            neighbor_covariance_by_id={edge[1]: sender_covariance[edge[1]]},
        ) for edge in edges
    }
    channels = {
        edge: MessageChannel(
            packet_loss_rate={edge[1]: packet_loss},
            delay_by_source={edge[1]: delay}, random_seed=seed * 17 + index,
        ) for index, edge in enumerate(edges)
    }
    buffers = {edge: TypedMessageBuffer() for edge in edges}
    checkpoints = {edge: receiver[edge] for edge in edges}
    rollback_history = {edge: {0.0: receiver[edge]} for edge in edges}
    observation_journal = {edge: [] for edge in edges}
    accumulators = {
        edge: ExactTransportAccumulator(
            source_node_id=edge[1], lineage_id=f"{edge[1]}:initial",
            reference_timestamp=0.0, reference_state=sender_state[edge[1]],
            reference_covariance=sender_covariance[edge[1]],
        ) for edge in edges
    }
    h = np.zeros((3, 6)); h[:, :3] = np.eye(3)
    measurement_covariance = np.eye(3) * absolute_sigma**2

    def process_available(current_timestamp):
        for edge in edges:
            for message in buffers[edge].pop_available(current_timestamp):
                totals["delivered"] += 1
                application_state = checkpoints[edge] if use_ack else receiver[edge]
                if use_rollback and not use_ack:
                    application_state = rollback_history[edge].get(float(message.reference_timestamp))
                    if application_state is None:
                        totals["rejections"]["history_unavailable"] = totals["rejections"].get("history_unavailable", 0) + 1
                        continue
                if (use_ack and float(message.reference_timestamp) < float(message.timestamp)
                        and not message.transport_events):
                    has_interleaved_observation = any(
                        float(message.reference_timestamp) <= float(observation.timestamp) < float(message.timestamp)
                        for observation in observation_journal[edge]
                    )
                    if has_interleaved_observation:
                        totals["rejections"]["interleaved_event_history_required"] = (
                            totals["rejections"].get("interleaved_event_history_required", 0) + 1
                        )
                        continue
                outcome = apply_exact_transport_state_message(
                    application_state, message,
                    expected_lineage_id=f"{edge[1]}:initial",
                )
                if outcome.accepted:
                    totals["accepted"] += 1
                    used_event_bundle = use_ack and bool(message.transport_events)
                    applied_state = outcome.state
                    acknowledged_state = outcome.state
                    if used_event_bundle:
                        acknowledged_state = replay_transport_event_bundle(
                            application_state, neighbor_id=edge[1],
                            events=message.transport_events,
                            observations=observation_journal[edge],
                            current_timestamp=float(message.timestamp),
                            process_noise_acceleration=process_noise_acceleration,
                        )
                        applied_state = acknowledged_state
                        if float(message.timestamp) < current_timestamp:
                            applied_state = replay_transport_event_bundle(
                                acknowledged_state, neighbor_id=edge[1], events=(),
                                observations=observation_journal[edge],
                                current_timestamp=current_timestamp,
                                process_noise_acceleration=process_noise_acceleration,
                            )
                    elif use_rollback and float(message.timestamp) < current_timestamp:
                        replay_times = sorted(
                            saved_time for saved_time in rollback_history[edge]
                            if float(message.timestamp) < saved_time <= current_timestamp
                        )
                        for replay_time in replay_times:
                            replay = replay_schmidt_events(
                                applied_state, current_timestamp=replay_time,
                                observations=observation_journal[edge],
                                process_noise_acceleration=process_noise_acceleration,
                            )
                            applied_state = replay.state
                            rollback_history[edge][replay_time] = applied_state
                    receiver[edge] = applied_state
                    if use_ack:
                        checkpoints[edge] = acknowledged_state
                        accumulators[edge].acknowledge(message)
                    totals["position_sum"] += float(np.linalg.norm(
                        outcome.state.neighbor_state_by_id[edge[1]][:3]
                        - message.state_estimate[:3]
                    ))
                    totals["covariance_max"] = max(
                        totals["covariance_max"], float(np.max(np.abs(
                            outcome.state.neighbor_covariance(edge[1]) - message.covariance
                        ))),
                    )
                else:
                    totals["rejections"][outcome.reason] = totals["rejections"].get(outcome.reason, 0) + 1
            totals["minimum_eigenvalue"] = min(
                totals["minimum_eigenvalue"],
                float(np.linalg.eigvalsh(receiver[edge].joint_covariance).min()),
            )

    for epoch in range(epochs):
        timestamp = epoch * dt
        prediction_transition = {node: np.eye(6) for node in node_ids}
        prediction_noise = {node: np.zeros((6, 6)) for node in node_ids}
        if epoch > 0:
            for node in node_ids:
                truth[node] = rk4_step_absolute(truth[node], dt)
                reference_mean = sender_state[node]
                transition = numerical_jacobian_discrete(
                    lambda value: rk4_step_absolute(value, dt), reference_mean
                )
                sender_state[node] = rk4_step_absolute(reference_mean, dt)
                sender_covariance[node] = (
                    transition @ sender_covariance[node] @ transition.T
                    + make_process_noise(dt, process_noise_acceleration)
                )
                prediction_transition[node] = transition
                prediction_noise[node] = make_process_noise(dt, process_noise_acceleration)
            receiver = {
                edge: multi_neighbor_schmidt_predict(
                    value, timestamp,
                    process_noise_acceleration=process_noise_acceleration,
                ) for edge, value in receiver.items()
            }
            for edge in edges:
                rollback_history[edge][timestamp] = receiver[edge]
                if history_window_epochs is not None:
                    oldest = timestamp - history_window_epochs * dt
                    rollback_history[edge] = {
                        saved_time: saved_state
                        for saved_time, saved_state in rollback_history[edge].items()
                        if saved_time >= oldest
                    }
                totals["peak_history"] = max(
                    totals["peak_history"], len(rollback_history[edge])
                )
        combined_delay_loss = use_ack and use_rollback and delay > 0.0
        if combined_delay_loss:
            process_available(timestamp)
        references = {
            node: (sender_state[node].copy(), sender_covariance[node].copy())
            for node in node_ids
        }
        for node in node_ids:
            innovation_covariance = h @ sender_covariance[node] @ h.T + measurement_covariance
            gain = sender_covariance[node] @ h.T @ np.linalg.inv(innovation_covariance)
            transition = np.eye(6) - gain @ h
            noise = gain @ measurement_covariance @ gain.T
            measurement = truth[node][:3] + rng.normal(0.0, absolute_sigma, 3)
            sender_state[node] += gain @ (measurement - h @ sender_state[node])
            sender_covariance[node] = transition @ sender_covariance[node] @ transition.T + noise
            for receiver_id, source_id in edges:
                if source_id != node:
                    continue
                reference_mean, reference_covariance = references[node]
                if use_ack:
                    combined_transition = transition @ prediction_transition[node]
                    combined_noise = (
                        transition @ prediction_noise[node] @ transition.T + noise
                    )
                    accumulators[(receiver_id, source_id)].append(
                        timestamp=timestamp, updated_state=sender_state[node],
                        error_transition=combined_transition,
                        independent_process_noise=combined_noise,
                        information_ids=(f"{node}:absolute:{epoch}",),
                        event_error_transition=transition,
                        event_process_noise=noise,
                    )
                    message = accumulators[(receiver_id, source_id)].build_message()
                else:
                    message = build_exact_transport_state_message(
                        source_node_id=node, timestamp=timestamp,
                        reference_timestamp=timestamp,
                        reference_state=reference_mean,
                        reference_covariance=reference_covariance,
                        updated_state=sender_state[node], error_transition=transition,
                        independent_process_noise=noise,
                        lineage_id=f"{node}:initial",
                        information_ids=(f"{node}:absolute:{epoch}",),
                    )
                totals["attempted"] += 1
                totals["transport_event_count"] += len(message.transport_events)
                transmitted = channels[(receiver_id, source_id)].transmit(message)
                if transmitted is not None:
                    buffers[(receiver_id, source_id)].push(transmitted)
        if not combined_delay_loss:
            process_available(timestamp)
        for edge in edges:
            observer, target = edge
            observation = ObservationMessage(
                message_id=f"{scenario}:{seed}:{observer}->{target}:range:{epoch}",
                observer_id=observer, target_id=target, timestamp=timestamp,
                modality="RANGE",
                measurement=np.array([
                    measure_relative_range(truth[observer], truth[target])
                    + rng.normal(0.0, absolute_sigma)
                ]),
                covariance=np.array([[absolute_sigma**2]]),
            )
            observation_journal[edge].append(observation)
            relative_update = multi_neighbor_schmidt_update(receiver[edge], observation)
            receiver[edge] = relative_update.state
            error = receiver[edge].active_state - truth[observer]
            totals["position_error_sq"] += float(error[:3] @ error[:3])
            totals["velocity_error_sq"] += float(error[3:] @ error[3:])
            totals["state_component_count"] += 3
            totals["state_sample_count"] += 1
            totals["nees_sum"] += float(
                error @ np.linalg.pinv(receiver[edge].active_covariance) @ error
            )
            totals["nis_sum"] += relative_update.nis
            totals["nis_count"] += 1


def _empty_totals():
    return {"attempted": 0, "delivered": 0, "accepted": 0,
            "position_sum": 0.0, "covariance_max": 0.0,
            "minimum_eigenvalue": float("inf"), "rejections": {},
            "position_error_sq": 0.0, "velocity_error_sq": 0.0,
            "state_component_count": 0, "state_sample_count": 0,
            "nees_sum": 0.0, "nis_sum": 0.0, "nis_count": 0,
            "peak_history": 1, "transport_event_count": 0}
