from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from cooperative.consensus_runner import CommunicationStats
from cooperative.topology import NetworkTopology
from interfaces.data_objects import (
    AbsolutePositionObservation,
    FleetStateMessage,
    InterSatelliteObservation,
)
from orbital_core.ci_fusion import ci_fuse_posteriors
from orbital_core.fleet_centralized_ekf import FleetCentralizedEKF

Array = np.ndarray


@dataclass(frozen=True)
class DistributedFleetCIHistory:
    timestamps: Array
    node_ids: tuple[str, ...]
    local_stacked_state_history_by_node: dict[str, Array]
    local_stacked_covariance_history_by_node: dict[str, Array]
    physical_state_history_by_node: dict[str, Array]
    node_weight_history_by_node: dict[str, list[dict[str, float]]]
    nis_history_by_node: dict[str, list[dict[str, float]]]
    communication_stats: CommunicationStats


def run_distributed_fleet_state_ci(
    *,
    timestamps: Array,
    initial_state_by_node: Mapping[str, Array],
    initial_covariance_by_node: Mapping[str, Array],
    topology: NetworkTopology,
    inter_satellite_observations: Iterable[InterSatelliteObservation],
    absolute_position_observations: Iterable[AbsolutePositionObservation] = (),
    node_ids: Sequence[str] | None = None,
    process_noise_acceleration: float = 1e-4,
    ci_objective: str = "trace",
    ci_grid_points: int = 31,
    packet_loss_rate_by_node: Mapping[str, float] | None = None,
    delay_by_node: Mapping[str, float] | None = None,
    align_delayed_messages: bool = True,
    random_seed: int = 42,
    frame_by_modality: Mapping[str, str] | None = None,
) -> DistributedFleetCIHistory:
    """Distributed baseline where every node estimates the same fleet state X."""

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps cannot be empty.")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    ordered_nodes = (
        tuple(str(node_id) for node_id in node_ids)
        if node_ids is not None
        else tuple(str(node_id) for node_id in initial_state_by_node)
    )
    if set(ordered_nodes) != set(topology.node_ids):
        raise ValueError("node_ids must match topology node IDs.")
    filter_obj = FleetCentralizedEKF(
        ordered_nodes,
        process_noise_acceleration=process_noise_acceleration,
    )
    initial_state = filter_obj.stack_states(initial_state_by_node)
    initial_covariance = filter_obj.stack_covariances(initial_covariance_by_node)
    states = {node_id: initial_state.copy() for node_id in ordered_nodes}
    covariances = {node_id: initial_covariance.copy() for node_id in ordered_nodes}

    relative_by_time_and_source = _group_relative_observations(
        inter_satellite_observations, times
    )
    absolute_by_time_and_owner = _group_absolute_observations(
        absolute_position_observations, times
    )
    dimension = filter_obj.state_dimension
    state_history = {
        node_id: np.zeros((times.size, dimension), dtype=float)
        for node_id in ordered_nodes
    }
    covariance_history = {
        node_id: np.zeros((times.size, dimension, dimension), dtype=float)
        for node_id in ordered_nodes
    }
    weights = {node_id: [] for node_id in ordered_nodes}
    nis_history = {node_id: [] for node_id in ordered_nodes}
    buffers: dict[str, list[FleetStateMessage]] = {
        node_id: [] for node_id in ordered_nodes
    }
    loss_rates = {str(key): float(value) for key, value in (packet_loss_rate_by_node or {}).items()}
    delays = {str(key): float(value) for key, value in (delay_by_node or {}).items()}
    for node_id, loss_rate in loss_rates.items():
        if not 0.0 <= loss_rate <= 1.0:
            raise ValueError(f"Packet loss rate for {node_id} must be in [0, 1].")
    rng = np.random.default_rng(random_seed)
    attempted_count = 0
    received_count = 0
    dropped_count = 0
    delay_sum = 0.0

    for index, timestamp in enumerate(times):
        if index > 0:
            dt = float(timestamp - times[index - 1])
            for node_id in ordered_nodes:
                states[node_id], covariances[node_id] = filter_obj.predict(
                    states[node_id], covariances[node_id], dt
                )

        for node_id in ordered_nodes:
            relative = relative_by_time_and_source[float(timestamp)].get(node_id, [])
            states[node_id], covariances[node_id], relative_diagnostics = filter_obj.update(
                states[node_id],
                covariances[node_id],
                relative,
                frame_by_modality=frame_by_modality,
            )
            absolute = absolute_by_time_and_owner[float(timestamp)].get(node_id, [])
            states[node_id], covariances[node_id], absolute_diagnostics = (
                filter_obj.update_absolute_positions(
                    states[node_id],
                    covariances[node_id],
                    absolute,
                )
            )
            nis_history[node_id].append(
                {
                    **relative_diagnostics.nis_by_observation,
                    **absolute_diagnostics.nis_by_observation,
                }
            )

        for source_node_id in ordered_nodes:
            message = _make_message(
                source_node_id=source_node_id,
                timestamp=float(timestamp),
                node_ids=ordered_nodes,
                state=states[source_node_id],
                covariance=covariances[source_node_id],
                delay=delays.get(source_node_id, 0.0),
            )
            for destination in topology.neighbors(source_node_id):
                attempted_count += 1
                if rng.random() < loss_rates.get(source_node_id, 0.0):
                    dropped_count += 1
                    continue
                buffers[destination].append(message)

        next_states: dict[str, Array] = {}
        next_covariances: dict[str, Array] = {}
        for node_id in ordered_nodes:
            available, pending = _pop_available(buffers[node_id], float(timestamp))
            buffers[node_id] = pending
            latest = _latest_by_source(available)
            received_count += len(latest)
            delay_sum += sum(
                max(float(message.arrival_timestamp or timestamp) - message.timestamp, 0.0)
                for message in latest.values()
            )
            posteriors = [(node_id, states[node_id], covariances[node_id])]
            for source_node_id, message in latest.items():
                _validate_message(message, ordered_nodes, dimension)
                message_state = message.state_estimate
                message_covariance = message.covariance
                source_time = (
                    message.source_timestamp
                    if message.source_timestamp is not None
                    else message.timestamp
                )
                if align_delayed_messages and timestamp > source_time:
                    message_state, message_covariance = filter_obj.predict(
                        message_state,
                        message_covariance,
                        float(timestamp - source_time),
                    )
                posteriors.append(
                    (source_node_id, message_state, message_covariance)
                )
            fusion = ci_fuse_posteriors(
                posteriors,
                objective=ci_objective,
                grid_points=ci_grid_points,
            )
            next_states[node_id] = fusion.state
            next_covariances[node_id] = fusion.covariance
            weights[node_id].append(dict(fusion.weights))
        states = next_states
        covariances = next_covariances

        for node_id in ordered_nodes:
            state_history[node_id][index] = states[node_id]
            covariance_history[node_id][index] = covariances[node_id]

    physical_history = {
        node_id: state_history[node_id][:, filter_obj.state_slice(node_id)].copy()
        for node_id in ordered_nodes
    }
    return DistributedFleetCIHistory(
        timestamps=times.copy(),
        node_ids=ordered_nodes,
        local_stacked_state_history_by_node=state_history,
        local_stacked_covariance_history_by_node=covariance_history,
        physical_state_history_by_node=physical_history,
        node_weight_history_by_node=weights,
        nis_history_by_node=nis_history,
        communication_stats=CommunicationStats(
            attempted_report_count=attempted_count,
            received_report_count=received_count,
            dropped_report_count=dropped_count,
            pending_report_count=sum(len(buffer) for buffer in buffers.values()),
            average_delay=delay_sum / received_count if received_count else 0.0,
            packet_loss_rate=dropped_count / attempted_count if attempted_count else 0.0,
        ),
    )


def _group_relative_observations(observations, times):
    result = {float(timestamp): {} for timestamp in times}
    for observation in observations:
        timestamp = float(observation.timestamp)
        if timestamp not in result:
            raise ValueError(f"Observation timestamp {timestamp} is not in runtime timestamps.")
        result[timestamp].setdefault(str(observation.source_node_id), []).append(observation)
    return result


def _group_absolute_observations(observations, times):
    result = {float(timestamp): {} for timestamp in times}
    for observation in observations:
        timestamp = float(observation.timestamp)
        if timestamp not in result:
            raise ValueError(
                f"Absolute observation timestamp {timestamp} is not in runtime timestamps."
            )
        result[timestamp].setdefault(str(observation.satellite_id), []).append(observation)
    return result


def _make_message(*, source_node_id, timestamp, node_ids, state, covariance, delay):
    if delay < 0.0:
        raise ValueError("Communication delay cannot be negative.")
    return FleetStateMessage(
        source_node_id=source_node_id,
        timestamp=timestamp,
        node_ids=tuple(node_ids),
        state_estimate=np.asarray(state, dtype=float).copy(),
        covariance=np.asarray(covariance, dtype=float).copy(),
        quality_score=float(1.0 / (1.0 + np.trace(covariance))),
        valid_flag=True,
        source_timestamp=timestamp,
        arrival_timestamp=timestamp + delay,
    )


def _pop_available(messages, timestamp):
    available = [
        message
        for message in messages
        if message.arrival_timestamp is None or message.arrival_timestamp <= timestamp
    ]
    pending = [
        message
        for message in messages
        if message.arrival_timestamp is not None and message.arrival_timestamp > timestamp
    ]
    return available, pending


def _latest_by_source(messages):
    latest = {}
    for message in messages:
        current = latest.get(message.source_node_id)
        if current is None or message.timestamp > current.timestamp:
            latest[message.source_node_id] = message
    return latest


def _validate_message(message, node_ids, dimension):
    if tuple(message.node_ids) != tuple(node_ids):
        raise ValueError("Fleet-State CI requires identical node ordering.")
    if np.asarray(message.state_estimate).shape != (dimension,):
        raise ValueError("Fleet state message has incompatible state dimension.")
    if np.asarray(message.covariance).shape != (dimension, dimension):
        raise ValueError("Fleet state message has incompatible covariance dimension.")
