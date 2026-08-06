from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cooperative.multi_node_ci import CooperativeFusionHistory, fuse_local_histories
from cooperative.communication_channel import CommunicationChannel
from cooperative.delay_channel import DelayChannel
from cooperative.multi_node_runner import extract_fused_local_history, run_multi_node_histories
from interfaces.data_objects import InitialState, ModuleInput, Observation
from orbital_core.dynamics import make_process_noise
from orbital_core.metrics import compute_rmse
from scenarios.multi_satellite_scenario import CooperativeScenario

Array = np.ndarray


@dataclass(frozen=True)
class CooperativeMetrics:
    local_position_rmse: dict[str, float]
    local_velocity_rmse: dict[str, float]
    cooperative_position_rmse: float
    cooperative_velocity_rmse: float
    best_local_position_rmse: float
    best_local_velocity_rmse: float
    position_improvement_over_best: float
    velocity_improvement_over_best: float


@dataclass(frozen=True)
class CooperativePipelineResult:
    module_inputs: dict[str, ModuleInput]
    local_histories: dict[str, Any]
    local_absolute_state_history_by_node: dict[str, Array]
    cooperative_history: CooperativeFusionHistory
    metrics: CooperativeMetrics


def build_module_inputs(
    *,
    scenario: CooperativeScenario,
    observations_by_node: dict[str, list[Observation]],
    initial_error_by_node: dict[str, Array] | None = None,
    initial_covariance: Array | None = None,
    architecture: str = "federated_ci",
    process_noise_acceleration_std: float = 1e-4,
    reset_feedback: bool = True,
    ci_objective: str = "trace",
    ci_grid_points: int = 31,
    modality_config_by_node: dict[str, dict[str, dict[str, object]]] | None = None,
) -> dict[str, ModuleInput]:
    """Build one independent single-node input for each observer.

    The initial state of each node is expressed in that observer's relative ECI
    frame. No filter object or mutable state is shared across nodes.
    """
    node_ids = list(scenario.observer_trajectories)
    if set(observations_by_node) != set(node_ids):
        raise ValueError("observations_by_node keys must match scenario observer IDs.")

    initial_covariance = (
        np.diag([100.0, 100.0, 100.0, 0.2, 0.2, 0.2]) ** 2
        if initial_covariance is None
        else np.asarray(initial_covariance, dtype=float).reshape(6, 6)
    )
    errors = initial_error_by_node or {}
    modality_configs = modality_config_by_node or {}
    timestamps = scenario.timestamps
    if len(timestamps) < 2:
        raise ValueError("At least two timestamps are required to run the filters.")
    nominal_dt = float(np.median(np.diff(timestamps)))
    process_noise = make_process_noise(nominal_dt, process_noise_acceleration_std)

    result: dict[str, ModuleInput] = {}
    for node_id in node_ids:
        error = np.asarray(errors.get(node_id, np.zeros(6)), dtype=float).reshape(6)
        relative_truth = scenario.relative_state_eci_by_node[node_id]
        initial_estimate = relative_truth[0] + error
        observer = scenario.observer_trajectories[node_id]
        result[node_id] = ModuleInput(
            initial_state=InitialState(
                target_id=scenario.target_id,
                timestamp=float(timestamps[0]),
                state_estimate=initial_estimate,
                covariance=initial_covariance.copy(),
            ),
            sensor_measurements=list(observations_by_node[node_id]),
            config={
                "runtime": {
                    "timestamps": timestamps.copy(),
                    "chief_state_history_eci": observer.state_history_eci.copy(),
                    "q_eci2pri_history": observer.q_eci2pri_history.copy(),
                    "node_id": node_id,
                },
                "filter": {
                    "architecture": architecture,
                    "process_noise": process_noise.copy(),
                    "reset_feedback": bool(reset_feedback),
                    "ci_objective": ci_objective,
                    "ci_grid_points": int(ci_grid_points),
                },
                "modalities": modality_configs.get(node_id, {}),
            },
        )
    return result


def run_cooperative_pipeline(
    *,
    scenario: CooperativeScenario,
    observations_by_node: dict[str, list[Observation]],
    initial_error_by_node: dict[str, Array] | None = None,
    initial_covariance: Array | None = None,
    architecture: str = "federated_ci",
    process_noise_acceleration_std: float = 1e-4,
    reset_feedback: bool = True,
    ci_objective: str = "trace",
    ci_grid_points: int = 31,
    modality_config_by_node: dict[str, dict[str, dict[str, object]]] | None = None,
    node_validity_by_node: dict[str, Array] | None = None,
    communication_channel: CommunicationChannel | None = None,
    delay_channel: DelayChannel | None = None,
    age_aware: bool = False,
    age_penalty: float = 1e-2,
) -> CooperativePipelineResult:
    """Run all local filters, transform to target absolute ECI, and fuse by CI."""
    module_inputs = build_module_inputs(
        scenario=scenario,
        observations_by_node=observations_by_node,
        initial_error_by_node=initial_error_by_node,
        initial_covariance=initial_covariance,
        architecture=architecture,
        process_noise_acceleration_std=process_noise_acceleration_std,
        reset_feedback=reset_feedback,
        ci_objective=ci_objective,
        ci_grid_points=ci_grid_points,
        modality_config_by_node=modality_config_by_node,
    )
    run_result = run_multi_node_histories(module_inputs)

    relative_states: dict[str, Array] = {}
    covariances: dict[str, Array] = {}
    observer_states: dict[str, Array] = {}
    local_absolute: dict[str, Array] = {}
    validity = _resolve_node_validity(
        histories=run_result.histories,
        sample_count=len(scenario.timestamps),
        requested=node_validity_by_node,
    )
    for node_id, history in run_result.histories.items():
        state_history, covariance_history = extract_fused_local_history(history)
        relative_states[node_id] = np.asarray(state_history, dtype=float)
        covariances[node_id] = np.asarray(covariance_history, dtype=float)
        observer_history = scenario.observer_trajectories[node_id].state_history_eci
        observer_states[node_id] = observer_history
        local_absolute[node_id] = relative_states[node_id] + observer_history

    cooperative = fuse_local_histories(
        timestamps=scenario.timestamps,
        relative_state_history_by_node=relative_states,
        covariance_history_by_node=covariances,
        observer_state_history_by_node=observer_states,
        target_id=scenario.target_id,
        validity_history_by_node=validity,
        communication_channel=communication_channel,
        delay_channel=delay_channel,
        objective=ci_objective,
        grid_points=ci_grid_points,
        age_aware=age_aware,
        age_penalty=age_penalty,
    )
    metrics = evaluate_cooperative_result(
        truth_state_history_eci=scenario.target_trajectory.state_history_eci,
        local_absolute_state_history_by_node=local_absolute,
        cooperative_state_history_eci=cooperative.state_history_eci,
    )
    return CooperativePipelineResult(
        module_inputs=module_inputs,
        local_histories=run_result.histories,
        local_absolute_state_history_by_node=local_absolute,
        cooperative_history=cooperative,
        metrics=metrics,
    )


def evaluate_cooperative_result(
    *,
    truth_state_history_eci: Array,
    local_absolute_state_history_by_node: dict[str, Array],
    cooperative_state_history_eci: Array,
) -> CooperativeMetrics:
    truth = np.asarray(truth_state_history_eci, dtype=float)
    cooperative = np.asarray(cooperative_state_history_eci, dtype=float)
    if truth.shape != cooperative.shape or truth.ndim != 2 or truth.shape[1] != 6:
        raise ValueError("Truth and cooperative histories must both have shape (N, 6).")
    local_position: dict[str, float] = {}
    local_velocity: dict[str, float] = {}
    for node_id, states in local_absolute_state_history_by_node.items():
        states = np.asarray(states, dtype=float)
        if states.shape != truth.shape:
            raise ValueError(f"Local history for {node_id} has an incompatible shape.")
        local_position[node_id] = compute_rmse(states[:, :3] - truth[:, :3])
        local_velocity[node_id] = compute_rmse(states[:, 3:] - truth[:, 3:])

    cooperative_position = compute_rmse(cooperative[:, :3] - truth[:, :3])
    cooperative_velocity = compute_rmse(cooperative[:, 3:] - truth[:, 3:])
    best_position = min(local_position.values())
    best_velocity = min(local_velocity.values())
    return CooperativeMetrics(
        local_position_rmse=local_position,
        local_velocity_rmse=local_velocity,
        cooperative_position_rmse=cooperative_position,
        cooperative_velocity_rmse=cooperative_velocity,
        best_local_position_rmse=best_position,
        best_local_velocity_rmse=best_velocity,
        position_improvement_over_best=_improvement(best_position, cooperative_position),
        velocity_improvement_over_best=_improvement(best_velocity, cooperative_velocity),
    )


def _resolve_node_validity(
    *,
    histories: dict[str, Any],
    sample_count: int,
    requested: dict[str, Array] | None,
) -> dict[str, Array]:
    result: dict[str, Array] = {}
    for node_id, history in histories.items():
        if requested is not None and node_id in requested:
            flags = np.asarray(requested[node_id], dtype=bool).reshape(-1)
            if flags.shape != (sample_count,):
                raise ValueError(f"Validity history for {node_id} must have shape (N,).")
            result[node_id] = flags.copy()
            continue
        measurement_flags = getattr(history, "measurement_valid_history", {})
        if measurement_flags:
            stacked = np.vstack([np.asarray(flags, dtype=bool) for flags in measurement_flags.values()])
            flags = np.any(stacked, axis=0)
            flags[0] = True
        else:
            flags = np.ones(sample_count, dtype=bool)
        result[node_id] = flags
    return result


def _improvement(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        return 0.0
    return float((reference - candidate) / reference * 100.0)
