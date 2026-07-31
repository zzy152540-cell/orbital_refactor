from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from cooperative.cooperative_update import update_local_state
from cooperative.message_transport import MessageChannel, TypedMessageBuffer
from cooperative.temporal_alignment import align_state_message, propagate_state_covariance
from cooperative.topology import NetworkTopology
from interfaces.data_objects import ObservationMessage, StateMessage, TargetEstimate
from orbital_core.quality import quality_score_from_covariance

Array = np.ndarray


@dataclass(frozen=True)
class RecursiveCommunicationStats:
    attempted_state_count: int
    received_state_count: int
    dropped_state_count: int
    attempted_observation_count: int
    received_observation_count: int
    dropped_observation_count: int
    deferred_observation_count: int
    pending_state_count: int
    pending_observation_count: int


@dataclass(frozen=True)
class RecursiveCooperativeHistory:
    timestamps: Array
    predicted_state_history_by_node: dict[str, Array]
    predicted_covariance_history_by_node: dict[str, Array]
    posterior_state_history_by_node: dict[str, Array]
    posterior_covariance_history_by_node: dict[str, Array]
    used_observation_ids_by_node: dict[str, list[list[str]]]
    nis_history_by_node: dict[str, list[dict[str, float]]]
    replayed_from_index_by_node: dict[str, list[int | None]]
    communication_stats: RecursiveCommunicationStats

    @property
    def node_ids(self) -> list[str]:
        return list(self.posterior_state_history_by_node)


@dataclass(frozen=True)
class _RegisteredObservation:
    observation: ObservationMessage
    neighbor_state: StateMessage


def run_recursive_distributed_cooperative_filter(
    *,
    timestamps: Array,
    initial_state_by_node: Mapping[str, Array],
    initial_covariance_by_node: Mapping[str, Array],
    topology: NetworkTopology,
    observation_messages: Iterable[ObservationMessage],
    state_channel: MessageChannel | None = None,
    observation_channel: MessageChannel | None = None,
    process_noise_acceleration: float = 1e-4,
    gate_enable: bool = False,
    gate_threshold: float = np.inf,
    gate_mode: str = "soft",
    soft_scale: float = 20.0,
    observation_usage: str = "both_endpoints",
    broadcast_state_history_by_node: Mapping[str, Array] | None = None,
    broadcast_covariance_history_by_node: Mapping[str, Array] | None = None,
) -> RecursiveCooperativeHistory:
    """Run a recursive V14 distributed local-state filter.

    State messages contain each node's predicted local state for the current
    epoch. A late observation is registered at its measurement epoch and that
    node's local prediction/update sequence is replayed through the current
    epoch. Previously transmitted neighbor messages remain immutable, matching
    what was actually available over the distributed network.
    """

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0 or (times.size > 1 and not np.all(np.diff(times) > 0.0)):
        raise ValueError("timestamps must be nonempty and strictly increasing.")
    if process_noise_acceleration < 0.0:
        raise ValueError("process_noise_acceleration cannot be negative.")
    if observation_usage not in {"observer_only", "both_endpoints"}:
        raise ValueError(
            "observation_usage must be 'observer_only' or 'both_endpoints'."
        )
    node_ids = topology.node_ids
    initial_states = _initial_values(initial_state_by_node, node_ids, (6,), "state")
    initial_covariances = _initial_values(
        initial_covariance_by_node, node_ids, (6, 6), "covariance"
    )
    broadcast_states, broadcast_covariances = _optional_broadcast_histories(
        state_histories=broadcast_state_history_by_node,
        covariance_histories=broadcast_covariance_history_by_node,
        node_ids=node_ids,
        sample_count=times.size,
    )
    observations_by_time = _group_observations(
        observation_messages,
        times=times,
        topology=topology,
    )
    timestamp_to_index = {
        float(timestamp): index for index, timestamp in enumerate(times)
    }
    state_channel = state_channel or MessageChannel()
    observation_channel = observation_channel or MessageChannel()

    predicted_states = {
        node_id: np.zeros((times.size, 6), dtype=float) for node_id in node_ids
    }
    predicted_covariances = {
        node_id: np.zeros((times.size, 6, 6), dtype=float) for node_id in node_ids
    }
    posterior_states = {
        node_id: np.zeros((times.size, 6), dtype=float) for node_id in node_ids
    }
    posterior_covariances = {
        node_id: np.zeros((times.size, 6, 6), dtype=float) for node_id in node_ids
    }
    registered: dict[str, dict[int, list[_RegisteredObservation]]] = {
        node_id: {} for node_id in node_ids
    }
    state_buffers = {
        node_id: TypedMessageBuffer[StateMessage]() for node_id in node_ids
    }
    observation_buffers = {
        node_id: TypedMessageBuffer[ObservationMessage]() for node_id in node_ids
    }
    state_archive: dict[str, dict[str, list[StateMessage]]] = {
        node_id: {} for node_id in node_ids
    }
    used_history = {
        node_id: [[] for _ in range(times.size)] for node_id in node_ids
    }
    nis_history = {
        node_id: [{} for _ in range(times.size)] for node_id in node_ids
    }
    replay_history = {node_id: [] for node_id in node_ids}
    attempted_state = received_state = dropped_state = 0
    attempted_observation = received_observation = dropped_observation = 0
    deferred_observation = 0

    for index, timestamp in enumerate(times):
        current_time = float(timestamp)
        for node_id in node_ids:
            if index == 0:
                predicted_states[node_id][index] = initial_states[node_id]
                predicted_covariances[node_id][index] = initial_covariances[node_id]
            else:
                state, covariance = propagate_state_covariance(
                    posterior_states[node_id][index - 1],
                    posterior_covariances[node_id][index - 1],
                    float(times[index] - times[index - 1]),
                    process_noise_acceleration=process_noise_acceleration,
                )
                predicted_states[node_id][index] = state
                predicted_covariances[node_id][index] = covariance
            posterior_states[node_id][index] = predicted_states[node_id][index]
            posterior_covariances[node_id][index] = predicted_covariances[node_id][index]

        for source_id in node_ids:
            message_state = (
                predicted_states[source_id][index]
                if broadcast_states is None
                else broadcast_states[source_id][index]
            )
            message_covariance = (
                predicted_covariances[source_id][index]
                if broadcast_covariances is None
                else broadcast_covariances[source_id][index]
            )
            state_message = StateMessage(
                source_node_id=source_id,
                target_node_id=source_id,
                timestamp=current_time,
                state_estimate=message_state.copy(),
                covariance=message_covariance.copy(),
                quality_score=quality_score_from_covariance(
                    message_covariance
                ),
            )
            for destination_id in topology.neighbors(source_id):
                attempted_state += 1
                delivered = state_channel.transmit(state_message)
                if delivered is None:
                    dropped_state += 1
                else:
                    state_buffers[destination_id].push(delivered)

        for observation in observations_by_time[current_time]:
            local_copy = replace(
                observation,
                source_timestamp=current_time,
                arrival_timestamp=current_time,
            )
            observation_buffers[observation.observer_id].push(local_copy)
            if observation_usage == "both_endpoints":
                attempted_observation += 1
                delivered = observation_channel.transmit(observation)
                if delivered is None:
                    dropped_observation += 1
                else:
                    observation_buffers[observation.target_id].push(delivered)

        for node_id in node_ids:
            available_states = state_buffers[node_id].pop_available(current_time)
            received_state += len(available_states)
            for message in available_states:
                archive = state_archive[node_id].setdefault(
                    message.target_node_id, []
                )
                archive.append(message)
                archive.sort(key=_state_source_timestamp)

        for node_id in node_ids:
            earliest_replay: int | None = None
            available_observations = observation_buffers[node_id].pop_available(
                current_time
            )
            deferred: list[ObservationMessage] = []
            registered_ids = {
                item.observation.information_id
                for per_epoch in registered[node_id].values()
                for item in per_epoch
            }
            for observation in available_observations:
                if observation.information_id in registered_ids:
                    continue
                counterpart_id = (
                    observation.target_id
                    if observation.observer_id == node_id
                    else observation.observer_id
                )
                neighbor = _state_at_or_before(
                    state_archive[node_id].get(counterpart_id, []),
                    float(observation.timestamp),
                )
                if neighbor is None:
                    deferred.append(observation)
                    deferred_observation += 1
                    continue
                observation_index = timestamp_to_index[float(observation.timestamp)]
                registered[node_id].setdefault(observation_index, []).append(
                    _RegisteredObservation(observation, neighbor)
                )
                registered[node_id][observation_index].sort(
                    key=lambda item: item.observation.information_id
                )
                registered_ids.add(observation.information_id)
                received_observation += 1
                earliest_replay = (
                    observation_index
                    if earliest_replay is None
                    else min(earliest_replay, observation_index)
                )
            for observation in deferred:
                observation_buffers[node_id].push(observation)

            if earliest_replay is not None:
                _replay_node_history(
                    node_id=node_id,
                    through_index=index,
                    timestamps=times,
                    initial_state=initial_states[node_id],
                    initial_covariance=initial_covariances[node_id],
                    registered=registered[node_id],
                    predicted_states=predicted_states[node_id],
                    predicted_covariances=predicted_covariances[node_id],
                    posterior_states=posterior_states[node_id],
                    posterior_covariances=posterior_covariances[node_id],
                    used_history=used_history[node_id],
                    nis_history=nis_history[node_id],
                    process_noise_acceleration=process_noise_acceleration,
                    gate_enable=gate_enable,
                    gate_threshold=gate_threshold,
                    gate_mode=gate_mode,
                    soft_scale=soft_scale,
                )
            replay_history[node_id].append(earliest_replay)

    return RecursiveCooperativeHistory(
        timestamps=times.copy(),
        predicted_state_history_by_node=predicted_states,
        predicted_covariance_history_by_node=predicted_covariances,
        posterior_state_history_by_node=posterior_states,
        posterior_covariance_history_by_node=posterior_covariances,
        used_observation_ids_by_node=used_history,
        nis_history_by_node=nis_history,
        replayed_from_index_by_node=replay_history,
        communication_stats=RecursiveCommunicationStats(
            attempted_state_count=attempted_state,
            received_state_count=received_state,
            dropped_state_count=dropped_state,
            attempted_observation_count=attempted_observation,
            received_observation_count=received_observation,
            dropped_observation_count=dropped_observation,
            deferred_observation_count=deferred_observation,
            pending_state_count=sum(len(buffer) for buffer in state_buffers.values()),
            pending_observation_count=sum(
                len(buffer) for buffer in observation_buffers.values()
            ),
        ),
    )


def _replay_node_history(
    *,
    node_id: str,
    through_index: int,
    timestamps: Array,
    initial_state: Array,
    initial_covariance: Array,
    registered: dict[int, list[_RegisteredObservation]],
    predicted_states: Array,
    predicted_covariances: Array,
    posterior_states: Array,
    posterior_covariances: Array,
    used_history: list[list[str]],
    nis_history: list[dict[str, float]],
    process_noise_acceleration: float,
    gate_enable: bool,
    gate_threshold: float,
    gate_mode: str,
    soft_scale: float,
) -> None:
    information_ids: tuple[str, ...] = ()
    for index in range(through_index + 1):
        timestamp = float(timestamps[index])
        if index == 0:
            state = initial_state.copy()
            covariance = initial_covariance.copy()
        else:
            state, covariance = propagate_state_covariance(
                posterior_states[index - 1],
                posterior_covariances[index - 1],
                float(timestamps[index] - timestamps[index - 1]),
                process_noise_acceleration=process_noise_acceleration,
            )
        predicted_states[index] = state
        predicted_covariances[index] = covariance
        estimate = TargetEstimate(
            estimator_node_id=node_id,
            target_node_id=node_id,
            timestamp=timestamp,
            state_estimate=state,
            covariance=covariance,
            quality_score=quality_score_from_covariance(covariance),
            information_ids=information_ids,
        )
        epoch_used: list[str] = []
        epoch_nis: dict[str, float] = {}
        for item in registered.get(index, []):
            neighbor = align_state_message(
                item.neighbor_state,
                timestamp,
                process_noise_acceleration=process_noise_acceleration,
            )
            update = update_local_state(
                local_estimate=estimate,
                neighbor_state=neighbor,
                observation=item.observation,
                gate_enable=gate_enable,
                gate_threshold=gate_threshold,
                gate_mode=gate_mode,
                soft_scale=soft_scale,
            )
            estimate = update.estimate
            information_ids = estimate.information_ids
            epoch_used.append(item.observation.information_id)
            epoch_nis[item.observation.information_id] = update.nis
        posterior_states[index] = estimate.state_estimate
        posterior_covariances[index] = estimate.covariance
        used_history[index] = epoch_used
        nis_history[index] = epoch_nis


def _group_observations(
    observations: Iterable[ObservationMessage],
    *,
    times: Array,
    topology: NetworkTopology,
) -> dict[float, list[ObservationMessage]]:
    result = {float(timestamp): [] for timestamp in times}
    seen: set[str] = set()
    for observation in observations:
        timestamp = float(observation.timestamp)
        if timestamp not in result:
            raise ValueError(f"Observation timestamp {timestamp} is not in timestamps.")
        observer = str(observation.observer_id)
        target = str(observation.target_id)
        if observer not in topology.node_ids or target not in topology.neighbors(observer):
            raise ValueError("Observation endpoints must share a topology edge.")
        if observation.message_id in seen:
            raise ValueError("Observation message_id values must be globally unique.")
        seen.add(observation.message_id)
        result[timestamp].append(observation)
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


def _optional_broadcast_histories(
    *,
    state_histories: Mapping[str, Array] | None,
    covariance_histories: Mapping[str, Array] | None,
    node_ids: tuple[str, ...],
    sample_count: int,
) -> tuple[dict[str, Array] | None, dict[str, Array] | None]:
    if (state_histories is None) != (covariance_histories is None):
        raise ValueError(
            "Broadcast state and covariance histories must be provided together."
        )
    if state_histories is None or covariance_histories is None:
        return None, None
    if set(state_histories) != set(node_ids) or set(covariance_histories) != set(node_ids):
        raise ValueError("Broadcast history keys must match topology node IDs.")
    states = {}
    covariances = {}
    for node_id in node_ids:
        state_values = np.asarray(state_histories[node_id], dtype=float)
        covariance_values = np.asarray(covariance_histories[node_id], dtype=float)
        if state_values.shape != (sample_count, 6):
            raise ValueError("Broadcast state histories must have shape (N, 6).")
        if covariance_values.shape != (sample_count, 6, 6):
            raise ValueError("Broadcast covariance histories must have shape (N, 6, 6).")
        states[node_id] = state_values.copy()
        covariances[node_id] = covariance_values.copy()
    return states, covariances


def _state_at_or_before(
    messages: list[StateMessage],
    timestamp: float,
) -> StateMessage | None:
    candidates = [
        message
        for message in messages
        if _state_source_timestamp(message) <= float(timestamp) + 1e-12
    ]
    return max(candidates, key=_state_source_timestamp) if candidates else None


def _state_source_timestamp(message: StateMessage) -> float:
    return float(
        message.timestamp
        if message.source_timestamp is None
        else message.source_timestamp
    )
