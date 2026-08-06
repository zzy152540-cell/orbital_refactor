from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from cooperative.message_transport import MessageChannel, TypedMessageBuffer
from cooperative.temporal_alignment import apply_delayed_cooperative_update
from cooperative.topology import NetworkTopology
from interfaces.data_objects import ObservationMessage, StateMessage, TargetEstimate
from orbital_core.quality import quality_score_from_covariance

Array = np.ndarray


@dataclass(frozen=True)
class V14CommunicationStats:
    attempted_state_count: int
    received_state_count: int
    dropped_state_count: int
    attempted_observation_count: int
    received_observation_count: int
    dropped_observation_count: int
    duplicate_observation_count: int
    pending_state_count: int
    pending_observation_count: int


@dataclass(frozen=True)
class DistributedCooperativeHistory:
    timestamps: Array
    state_history_by_node: dict[str, Array]
    covariance_history_by_node: dict[str, Array]
    used_observation_ids_by_node: dict[str, list[list[str]]]
    nis_history_by_node: dict[str, list[dict[str, float]]]
    received_state_sources_by_node: dict[str, list[list[str]]]
    oosm_delay_history_by_node: dict[str, list[dict[str, float]]]
    communication_stats: V14CommunicationStats

    @property
    def node_ids(self) -> list[str]:
        return list(self.state_history_by_node)


def run_distributed_cooperative_history(
    *,
    timestamps: Array,
    state_history_by_node: Mapping[str, Array],
    covariance_history_by_node: Mapping[str, Array],
    topology: NetworkTopology,
    observation_messages: Iterable[ObservationMessage],
    state_channel: MessageChannel | None = None,
    observation_channel: MessageChannel | None = None,
    gate_enable: bool = False,
    gate_threshold: float = np.inf,
    gate_mode: str = "soft",
    soft_scale: float = 20.0,
    process_noise_acceleration: float = 1e-4,
) -> DistributedCooperativeHistory:
    """Run V14 local cooperative updates with communicated observations.

    Each directed observation is available locally to its observer and is sent
    once to its target. Both endpoints may therefore update their own state,
    but no state belonging to different satellites is fused by CI.
    """

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0 or not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be nonempty and strictly increasing.")
    node_ids = topology.node_ids
    states = _validate_histories(state_history_by_node, node_ids, times.size, False)
    covariances = _validate_histories(
        covariance_history_by_node, node_ids, times.size, True
    )
    observations_by_time = _group_observations(
        observation_messages, times=times, node_ids=set(node_ids), topology=topology
    )
    state_channel = state_channel or MessageChannel()
    observation_channel = observation_channel or MessageChannel()

    state_buffers = {
        node_id: TypedMessageBuffer[StateMessage]() for node_id in node_ids
    }
    observation_buffers = {
        node_id: TypedMessageBuffer[ObservationMessage]() for node_id in node_ids
    }
    neighbor_state_archive: dict[str, dict[str, list[StateMessage]]] = {
        node_id: {} for node_id in node_ids
    }
    used_ids: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    output_states = {
        node_id: np.zeros((times.size, 6), dtype=float) for node_id in node_ids
    }
    output_covariances = {
        node_id: np.zeros((times.size, 6, 6), dtype=float) for node_id in node_ids
    }
    used_history = {node_id: [] for node_id in node_ids}
    nis_history = {node_id: [] for node_id in node_ids}
    received_state_history = {node_id: [] for node_id in node_ids}
    oosm_delay_history = {node_id: [] for node_id in node_ids}
    timestamp_to_index = {
        float(timestamp): index for index, timestamp in enumerate(times)
    }
    attempted_state = received_state = dropped_state = 0
    attempted_observation = received_observation = dropped_observation = 0
    duplicate_observation = 0

    for index, timestamp in enumerate(times):
        current_time = float(timestamp)
        local_estimates = {
            node_id: TargetEstimate(
                estimator_node_id=node_id,
                target_node_id=node_id,
                timestamp=current_time,
                state_estimate=states[node_id][index].copy(),
                covariance=covariances[node_id][index].copy(),
                quality_score=quality_score_from_covariance(
                    covariances[node_id][index]
                ),
                information_ids=tuple(sorted(used_ids[node_id])),
            )
            for node_id in node_ids
        }

        for source_id in node_ids:
            message = StateMessage(
                source_node_id=source_id,
                target_node_id=source_id,
                timestamp=current_time,
                state_estimate=states[source_id][index].copy(),
                covariance=covariances[source_id][index].copy(),
                quality_score=local_estimates[source_id].quality_score,
            )
            for destination_id in topology.neighbors(source_id):
                attempted_state += 1
                delivered = state_channel.transmit(message)
                if delivered is None:
                    dropped_state += 1
                else:
                    state_buffers[destination_id].push(delivered)

        for observation in observations_by_time[current_time]:
            local_copy = replace(
                observation,
                source_timestamp=float(observation.timestamp),
                arrival_timestamp=float(observation.timestamp),
            )
            if not observation_buffers[observation.observer_id].push(local_copy):
                duplicate_observation += 1
            attempted_observation += 1
            delivered = observation_channel.transmit(observation)
            if delivered is None:
                dropped_observation += 1
            elif not observation_buffers[observation.target_id].push(delivered):
                duplicate_observation += 1

        for node_id in node_ids:
            available_states = state_buffers[node_id].pop_available(current_time)
            received_state += len(available_states)
            for message in available_states:
                archive = neighbor_state_archive[node_id].setdefault(
                    message.target_node_id, []
                )
                archive.append(message)
                archive.sort(key=_state_source_timestamp)
            received_state_history[node_id].append(
                sorted({message.source_node_id for message in available_states})
            )

            available_observations = observation_buffers[node_id].pop_available(
                current_time
            )
            deferred: list[ObservationMessage] = []
            epoch_used: list[str] = []
            epoch_nis: dict[str, float] = {}
            epoch_oosm_delay: dict[str, float] = {}
            estimate = local_estimates[node_id]
            for observation in available_observations:
                if observation.information_id in used_ids[node_id]:
                    duplicate_observation += 1
                    continue
                counterpart_id = (
                    observation.target_id
                    if observation.observer_id == node_id
                    else observation.observer_id
                )
                neighbor = _state_at_or_before(
                    neighbor_state_archive[node_id].get(counterpart_id, []),
                    float(observation.timestamp),
                )
                if neighbor is None:
                    deferred.append(observation)
                    continue
                observation_index = timestamp_to_index[float(observation.timestamp)]
                local_at_observation = (
                    estimate
                    if np.isclose(float(observation.timestamp), current_time)
                    else TargetEstimate(
                        estimator_node_id=node_id,
                        target_node_id=node_id,
                        timestamp=float(observation.timestamp),
                        state_estimate=states[node_id][observation_index].copy(),
                        covariance=covariances[node_id][observation_index].copy(),
                        quality_score=quality_score_from_covariance(
                            covariances[node_id][observation_index]
                        ),
                        information_ids=tuple(sorted(used_ids[node_id])),
                    )
                )
                delayed_result = apply_delayed_cooperative_update(
                    local_estimate_at_observation=local_at_observation,
                    neighbor_state=neighbor,
                    observation=observation,
                    output_timestamp=current_time,
                    process_noise_acceleration=process_noise_acceleration,
                    gate_enable=gate_enable,
                    gate_threshold=gate_threshold,
                    gate_mode=gate_mode,
                    soft_scale=soft_scale,
                )
                result = delayed_result.measurement_update
                estimate = delayed_result.estimate
                used_ids[node_id].add(observation.information_id)
                epoch_used.append(observation.information_id)
                epoch_nis[observation.information_id] = result.nis
                epoch_oosm_delay[observation.information_id] = (
                    delayed_result.posterior_propagation_dt
                )
                received_observation += 1
            for observation in deferred:
                observation_buffers[node_id].push(observation)

            output_states[node_id][index] = estimate.state_estimate
            output_covariances[node_id][index] = estimate.covariance
            used_history[node_id].append(epoch_used)
            nis_history[node_id].append(epoch_nis)
            oosm_delay_history[node_id].append(epoch_oosm_delay)

    return DistributedCooperativeHistory(
        timestamps=times.copy(),
        state_history_by_node=output_states,
        covariance_history_by_node=output_covariances,
        used_observation_ids_by_node=used_history,
        nis_history_by_node=nis_history,
        received_state_sources_by_node=received_state_history,
        oosm_delay_history_by_node=oosm_delay_history,
        communication_stats=V14CommunicationStats(
            attempted_state_count=attempted_state,
            received_state_count=received_state,
            dropped_state_count=dropped_state,
            attempted_observation_count=attempted_observation,
            received_observation_count=received_observation,
            dropped_observation_count=dropped_observation,
            duplicate_observation_count=duplicate_observation,
            pending_state_count=sum(len(buffer) for buffer in state_buffers.values()),
            pending_observation_count=sum(
                len(buffer) for buffer in observation_buffers.values()
            ),
        ),
    )


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


def _group_observations(
    observations: Iterable[ObservationMessage],
    *,
    times: Array,
    node_ids: set[str],
    topology: NetworkTopology,
) -> dict[float, list[ObservationMessage]]:
    result = {float(timestamp): [] for timestamp in times}
    seen_ids: set[str] = set()
    for observation in observations:
        timestamp = float(observation.timestamp)
        if timestamp not in result:
            raise ValueError(f"Observation timestamp {timestamp} is not in timestamps.")
        observer = str(observation.observer_id)
        target = str(observation.target_id)
        if observer not in node_ids or target not in node_ids or observer == target:
            raise ValueError("Observation endpoints must be distinct topology nodes.")
        if target not in topology.neighbors(observer):
            raise ValueError("Observation endpoints must share a topology edge.")
        if observation.message_id in seen_ids:
            raise ValueError("Observation message_id values must be globally unique.")
        seen_ids.add(observation.message_id)
        result[timestamp].append(observation)
    return result


def _validate_histories(
    histories: Mapping[str, Array],
    node_ids: tuple[str, ...],
    sample_count: int,
    covariance: bool,
) -> dict[str, Array]:
    if set(histories) != set(node_ids):
        raise ValueError("History keys must match topology node IDs.")
    expected = (sample_count, 6, 6) if covariance else (sample_count, 6)
    result = {}
    for node_id, history in histories.items():
        values = np.asarray(history, dtype=float)
        if values.shape != expected:
            raise ValueError(f"History for {node_id} must have shape {expected}.")
        result[node_id] = values.copy()
    return result
