from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from cooperative.multi_neighbor_schmidt import (
    MultiNeighborSchmidtState,
    initialize_multi_neighbor_schmidt,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_update,
)
from cooperative.topology import NetworkTopology
from cooperative.schmidt_refresh import (
    exact_transport_eligibility,
    refresh_consider_neighbor,
)
from cooperative.multi_neighbor_replay_coordinator import (
    MultiNeighborReplayCoordinator,
    ReplayPerformanceStats,
)
from orbital_core.dynamics import make_process_noise, numerical_jacobian_discrete, rk4_step_absolute
from interfaces.data_objects import ObservationMessage, StateMessage

Array = np.ndarray


@dataclass(frozen=True)
class NetworkSchmidtHistory:
    timestamps: Array
    active_state_history_by_node: dict[str, Array]
    active_covariance_history_by_node: dict[str, Array]
    active_cross_covariance_history_by_node: dict[str, dict[str, Array]]
    joint_covariance_history_by_node: dict[str, Array]
    nis_history_by_node: dict[str, list[dict[str, float]]]
    local_dimension_by_node: dict[str, int]
    refresh_diagnostics: dict[str, int]
    refresh_diagnostic_records: tuple[dict[str, object], ...]
    replay_performance_by_node: dict[str, ReplayPerformanceStats]

    @property
    def node_ids(self) -> list[str]:
        return list(self.active_state_history_by_node)


def run_network_schmidt_filter(
    *,
    timestamps: Array,
    initial_state_by_node: Mapping[str, Array],
    initial_covariance_by_node: Mapping[str, Array],
    topology: NetworkTopology,
    observation_messages: Iterable[ObservationMessage],
    observation_usage: str = "observer_only",
    process_noise_acceleration: float = 1e-4,
    consider_refresh_mode: str = "propagate_only",
    state_messages_by_receiver: Mapping[str, Iterable[StateMessage]] | None = None,
    replay_history_window: float | None = None,
    expected_lineage_by_link: Mapping[tuple[str, str], str] | None = None,
    max_pinned_age: float | None = None,
    max_retained_events: int | None = None,
) -> NetworkSchmidtHistory:
    """Run one local multi-neighbor Schmidt filter at every topology node.

    Each node owns a different local augmented covariance. Consider-state
    means are propagated but never corrected inside that node's filter. The
    default routes each physical observation only to its actual observer.
    """

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0 or (times.size > 1 and not np.all(np.diff(times) > 0.0)):
        raise ValueError("timestamps must be nonempty and strictly increasing.")
    if observation_usage not in {"observer_only", "both_endpoints"}:
        raise ValueError(
            "observation_usage must be 'observer_only' or 'both_endpoints'."
        )
    if consider_refresh_mode not in {"propagate_only", "safe_rescale", "zero_cross", "exact_if_compatible", "exact_transport_event_replay"}:
        raise ValueError("Unsupported network consider_refresh_mode.")
    if state_messages_by_receiver and consider_refresh_mode != "exact_transport_event_replay":
        raise ValueError("State messages require exact_transport_event_replay mode.")
    node_ids = topology.node_ids
    states = _initial_values(initial_state_by_node, node_ids, (6,), "state")
    covariances = _initial_values(
        initial_covariance_by_node, node_ids, (6, 6), "covariance"
    )
    local_states = {
        node_id: initialize_multi_neighbor_schmidt(
            timestamp=float(times[0]),
            active_node_id=node_id,
            active_state=states[node_id],
            active_covariance=covariances[node_id],
            neighbor_state_by_id={
                neighbor_id: states[neighbor_id]
                for neighbor_id in topology.neighbors(node_id)
            },
            neighbor_covariance_by_id={
                neighbor_id: covariances[neighbor_id]
                for neighbor_id in topology.neighbors(node_id)
            },
        )
        for node_id in node_ids
    }
    coordinators = (
        {
            node_id: MultiNeighborReplayCoordinator(
                local_states[node_id],
                process_noise_acceleration=process_noise_acceleration,
                history_window=replay_history_window,
                max_pinned_age=max_pinned_age,
                max_retained_events=max_retained_events,
            )
            for node_id in node_ids
        }
        if consider_refresh_mode == "exact_transport_event_replay"
        else {}
    )
    pending_state_messages = _prepare_state_messages(
        state_messages_by_receiver or {}, node_ids, topology
    )
    observations_by_time_and_owner = _route_observations(
        observation_messages,
        times=times,
        topology=topology,
        observation_usage=observation_usage,
    )
    active_states = {
        node_id: np.zeros((times.size, 6), dtype=float) for node_id in node_ids
    }
    active_covariances = {
        node_id: np.zeros((times.size, 6, 6), dtype=float) for node_id in node_ids
    }
    joint_covariances = {
        node_id: np.zeros(
            (times.size, local_states[node_id].dimension, local_states[node_id].dimension),
            dtype=float,
        )
        for node_id in node_ids
    }
    active_cross_covariances = {
        node_id: {
            neighbor_id: np.zeros((times.size, 6, 6), dtype=float)
            for neighbor_id in topology.neighbors(node_id)
        }
        for node_id in node_ids
    }
    nis_history = {node_id: [] for node_id in node_ids}
    refresh_diagnostics = {"accepted": 0, "reference_covariance_mismatch": 0,
                           "reference_mean_mismatch": 0}
    refresh_diagnostic_records = []
    previous_active = {
        node_id: (local_states[node_id].active_state.copy(),
                  local_states[node_id].active_covariance)
        for node_id in node_ids
    }

    for index, timestamp in enumerate(times):
        if consider_refresh_mode == "exact_transport_event_replay":
            if index > 0:
                for coordinator in coordinators.values():
                    coordinator.advance(float(timestamp))
            for node_id in node_ids:
                remaining = []
                available = []
                for message in pending_state_messages[node_id]:
                    arrival = message.timestamp if message.arrival_timestamp is None else message.arrival_timestamp
                    if float(arrival) <= float(timestamp):
                        available.append((
                            message,
                            (expected_lineage_by_link or {}).get(
                                (node_id, str(message.source_node_id))
                            ),
                        ))
                    else:
                        remaining.append(message)
                outcomes = coordinators[node_id].apply_state_messages(tuple(available))
                for (message, _), outcome in zip(available, outcomes):
                    key = "accepted" if outcome.accepted else outcome.reason
                    refresh_diagnostics[key] = refresh_diagnostics.get(key, 0) + 1
                    checkpoints = coordinators[node_id].checkpoint_timestamps
                    refresh_diagnostic_records.append({
                        "receiver_id": node_id,
                        "source_id": str(message.source_node_id),
                        "current_timestamp": float(timestamp),
                        "message_timestamp": float(message.timestamp),
                        "reference_timestamp": message.reference_timestamp,
                        "arrival_timestamp": message.arrival_timestamp,
                        "lineage_id": message.lineage_id,
                        "information_ids": "|".join(message.information_ids),
                        "transport_event_count": len(message.transport_events),
                        "accepted": outcome.accepted,
                        "reason": outcome.reason,
                        "checkpoint_count": len(checkpoints),
                        "oldest_checkpoint": min(checkpoints) if checkpoints else None,
                        "newest_checkpoint": max(checkpoints) if checkpoints else None,
                        "pinned_checkpoint_count": coordinators[node_id].pinned_checkpoint_count,
                        "oldest_pinned_timestamp": coordinators[node_id].oldest_pinned_timestamp,
                        "resync_required_count": len(
                            coordinators[node_id].resynchronization_requirements
                        ),
                        **message.metadata,
                    })
                pending_state_messages[node_id] = remaining
                local_states[node_id] = coordinators[node_id].state
        elif index > 0:
            local_states = {
                node_id: multi_neighbor_schmidt_predict(
                    local_states[node_id], float(timestamp),
                    process_noise_acceleration=process_noise_acceleration,
                )
                for node_id in node_ids
            }
            if consider_refresh_mode != "propagate_only":
                snapshots = {
                    node_id: (local_states[node_id].active_state.copy(),
                              local_states[node_id].active_covariance)
                    for node_id in node_ids
                }
                for node_id in node_ids:
                    for neighbor_id in local_states[node_id].neighbor_ids:
                        mean, covariance = snapshots[neighbor_id]
                        if consider_refresh_mode == "exact_if_compatible":
                            reference_mean, reference_covariance = previous_active[neighbor_id]
                            # Compare against the receiver block before this epoch's
                            # prediction, reconstructed by applying the same model.
                            dt_epoch = float(timestamp) - float(times[index - 1])
                            transition = numerical_jacobian_discrete(
                                lambda value: rk4_step_absolute(value, dt_epoch),
                                reference_mean,
                            )
                            noise = make_process_noise(dt_epoch, process_noise_acceleration)
                            predicted_reference_covariance = transition @ reference_covariance @ transition.T + noise
                            eligible, reason = exact_transport_eligibility(
                                local_states[node_id], neighbor_id=neighbor_id,
                                reference_covariance=predicted_reference_covariance,
                                reference_mean=rk4_step_absolute(reference_mean, dt_epoch),
                            )
                            if eligible:
                                # It is already at the exactly transported value;
                                # reapplying would transport twice. Acceptance is
                                # recorded as a provenance/compatibility assertion.
                                refresh_diagnostics["accepted"] += 1
                            else:
                                refresh_diagnostics[reason] += 1
                        else:
                            local_states[node_id] = refresh_consider_neighbor(
                                local_states[node_id], neighbor_id=neighbor_id,
                                neighbor_state=mean, neighbor_covariance=covariance,
                                mode=consider_refresh_mode,
                            )
        for node_id in node_ids:
            state = local_states[node_id]
            epoch_nis = {}
            for observation in observations_by_time_and_owner[float(timestamp)].get(
                node_id, []
            ):
                if consider_refresh_mode == "exact_transport_event_replay":
                    value = coordinators[node_id].apply_observation(observation)
                    state = coordinators[node_id].state
                    epoch_nis[observation.information_id] = value
                else:
                    update = multi_neighbor_schmidt_update(state, observation)
                    state = update.state
                    epoch_nis[observation.information_id] = update.nis
            local_states[node_id] = state
            active_states[node_id][index] = state.active_state
            active_covariances[node_id][index] = state.active_covariance
            joint_covariances[node_id][index] = state.joint_covariance
            for neighbor_id in state.neighbor_ids:
                active_cross_covariances[node_id][neighbor_id][index] = (
                    state.active_cross_covariance(neighbor_id)
                )
            nis_history[node_id].append(epoch_nis)
        previous_active = {
            node_id: (local_states[node_id].active_state.copy(),
                      local_states[node_id].active_covariance)
            for node_id in node_ids
        }

    return NetworkSchmidtHistory(
        timestamps=times.copy(),
        active_state_history_by_node=active_states,
        active_covariance_history_by_node=active_covariances,
        active_cross_covariance_history_by_node=active_cross_covariances,
        joint_covariance_history_by_node=joint_covariances,
        nis_history_by_node=nis_history,
        local_dimension_by_node={
            node_id: local_states[node_id].dimension for node_id in node_ids
        },
        refresh_diagnostics=refresh_diagnostics,
        refresh_diagnostic_records=tuple(refresh_diagnostic_records),
        replay_performance_by_node={
            node_id: replace(coordinator.performance)
            for node_id, coordinator in coordinators.items()
        },
    )


def _route_observations(
    observations: Iterable[ObservationMessage],
    *,
    times: Array,
    topology: NetworkTopology,
    observation_usage: str,
) -> dict[float, dict[str, list[ObservationMessage]]]:
    result = {float(timestamp): {} for timestamp in times}
    seen_message_ids: set[str] = set()
    for observation in observations:
        timestamp = float(observation.timestamp)
        if timestamp not in result:
            raise ValueError("Observation timestamp is not in timestamps.")
        observer, target = str(observation.observer_id), str(observation.target_id)
        if target not in topology.neighbors(observer):
            raise ValueError("Observation endpoints must share a topology edge.")
        if observation.message_id in seen_message_ids:
            raise ValueError("Observation message_id values must be globally unique.")
        seen_message_ids.add(observation.message_id)
        owners = (
            (observer, target)
            if observation_usage == "both_endpoints"
            else (observer,)
        )
        for owner in owners:
            result[timestamp].setdefault(owner, []).append(observation)
    for per_owner in result.values():
        for messages in per_owner.values():
            messages.sort(key=lambda item: item.information_id)
    return result


def _initial_values(
    values: Mapping[str, Array],
    node_ids: tuple[str, ...],
    shape: tuple[int, ...],
    name: str,
) -> dict[str, Array]:
    if set(values) != set(node_ids):
        raise ValueError(f"Initial {name} keys must match topology node IDs.")
    result = {}
    for node_id, value in values.items():
        array = np.asarray(value, dtype=float)
        if array.shape != shape:
            raise ValueError(f"Initial {name} for {node_id} must have shape {shape}.")
        result[node_id] = array.copy()
    return result


def _prepare_state_messages(
    messages_by_receiver: Mapping[str, Iterable[StateMessage]],
    node_ids: tuple[str, ...], topology: NetworkTopology,
) -> dict[str, list[StateMessage]]:
    unknown = set(messages_by_receiver) - set(node_ids)
    if unknown:
        raise ValueError("State-message receiver IDs must belong to the topology.")
    result = {node_id: [] for node_id in node_ids}
    for receiver_id, messages in messages_by_receiver.items():
        for message in messages:
            if str(message.source_node_id) not in topology.neighbors(receiver_id):
                raise ValueError("State-message source must be a receiver neighbor.")
            result[receiver_id].append(message)
        result[receiver_id].sort(key=lambda item: (
            float(item.timestamp if item.arrival_timestamp is None else item.arrival_timestamp),
            float(item.timestamp), str(item.source_node_id),
        ))
    return result
