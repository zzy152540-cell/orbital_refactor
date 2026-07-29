from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from interfaces.data_objects import AbsolutePositionObservation, InterSatelliteObservation
from orbital_core.fleet_centralized_ekf import FleetCentralizedEKF

Array = np.ndarray


@dataclass(frozen=True)
class FleetCentralizedHistory:
    timestamps: Array
    node_ids: tuple[str, ...]
    stacked_state_history: Array
    stacked_covariance_history: Array
    state_history_by_node: dict[str, Array]
    covariance_history_by_node: dict[str, Array]
    nis_history: list[dict[str, float]]


def run_fleet_centralized_filter(
    *,
    timestamps: Array,
    initial_state_by_node: Mapping[str, Array],
    initial_covariance_by_node: Mapping[str, Array],
    inter_satellite_observations: Iterable[InterSatelliteObservation],
    absolute_position_observations: Iterable[AbsolutePositionObservation] = (),
    node_ids: Sequence[str] | None = None,
    process_noise_acceleration: float = 1e-4,
    frame_by_modality: Mapping[str, str] | None = None,
) -> FleetCentralizedHistory:
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
    filter_obj = FleetCentralizedEKF(
        ordered_nodes,
        process_noise_acceleration=process_noise_acceleration,
    )
    state = filter_obj.stack_states(initial_state_by_node)
    covariance = filter_obj.stack_covariances(initial_covariance_by_node)
    observations_by_time: dict[float, list[InterSatelliteObservation]] = {
        float(timestamp): [] for timestamp in times
    }
    for observation in inter_satellite_observations:
        timestamp = float(observation.timestamp)
        if timestamp not in observations_by_time:
            raise ValueError(f"Observation timestamp {timestamp} is not in runtime timestamps.")
        observations_by_time[timestamp].append(observation)
    absolute_by_time: dict[float, list[AbsolutePositionObservation]] = {
        float(timestamp): [] for timestamp in times
    }
    for observation in absolute_position_observations:
        timestamp = float(observation.timestamp)
        if timestamp not in absolute_by_time:
            raise ValueError(
                f"Absolute observation timestamp {timestamp} is not in runtime timestamps."
            )
        absolute_by_time[timestamp].append(observation)

    dimension = filter_obj.state_dimension
    state_history = np.zeros((times.size, dimension), dtype=float)
    covariance_history = np.zeros((times.size, dimension, dimension), dtype=float)
    nis_history: list[dict[str, float]] = []
    for index, timestamp in enumerate(times):
        if index > 0:
            state, covariance = filter_obj.predict(
                state, covariance, float(timestamp - times[index - 1])
            )
        state, covariance, diagnostics = filter_obj.update(
            state,
            covariance,
            observations_by_time[float(timestamp)],
            frame_by_modality=frame_by_modality,
        )
        state, covariance, absolute_diagnostics = filter_obj.update_absolute_positions(
            state,
            covariance,
            absolute_by_time[float(timestamp)],
        )
        state_history[index] = state
        covariance_history[index] = covariance
        nis_history.append(
            {
                **diagnostics.nis_by_observation,
                **absolute_diagnostics.nis_by_observation,
            }
        )

    state_by_node: dict[str, Array] = {}
    covariance_by_node: dict[str, Array] = {}
    for node_id in filter_obj.node_ids:
        block = filter_obj.state_slice(node_id)
        state_by_node[node_id] = state_history[:, block].copy()
        covariance_by_node[node_id] = covariance_history[:, block, block].copy()
    return FleetCentralizedHistory(
        timestamps=times.copy(),
        node_ids=filter_obj.node_ids,
        stacked_state_history=state_history,
        stacked_covariance_history=covariance_history,
        state_history_by_node=state_by_node,
        covariance_history_by_node=covariance_by_node,
        nis_history=nis_history,
    )
