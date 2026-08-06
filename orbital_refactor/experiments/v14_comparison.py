from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from cooperative.fleet_state_ci_runner import run_distributed_fleet_state_ci
from cooperative.recursive_cooperative_runner import (
    RecursiveCooperativeHistory,
    run_recursive_distributed_cooperative_filter,
)
from cooperative.temporal_alignment import propagate_state_covariance
from cooperative.topology import NetworkTopology, fully_connected_topology
from interfaces.data_objects import InterSatelliteObservation, ObservationMessage
from orbital_core.constants import R_EARTH
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.metrics import compute_nees_history, compute_rmse
from orbital_core.orbit_elements import keplerian_to_eci
from pipelines.fleet_centralized import run_fleet_centralized_filter
from scenarios.fleet_scenario import FleetScenario, generate_fleet_scenario

Array = np.ndarray


@dataclass(frozen=True)
class V14ComparisonCase:
    scenario: FleetScenario
    initial_state_by_node: dict[str, Array]
    initial_covariance_by_node: dict[str, Array]
    topology: NetworkTopology
    inter_satellite_observations: list[InterSatelliteObservation]
    observation_messages: list[ObservationMessage]
    frame_by_modality: dict[str, str]
    observation_strategy: str
    observation_usage: str


@dataclass(frozen=True)
class NodeMetrics:
    position_rmse: float
    velocity_rmse: float
    mean_nees: float


@dataclass(frozen=True)
class AlgorithmMetrics:
    by_node: dict[str, NodeMetrics]
    fleet_position_rmse: float
    fleet_velocity_rmse: float
    mean_nees: float
    mean_nis: float | None
    communication: dict[str, float | int]


@dataclass(frozen=True)
class V14ComparisonResult:
    case: V14ComparisonCase
    metrics_by_algorithm: dict[str, AlgorithmMetrics]
    estimate_history_by_algorithm: dict[str, dict[str, Array]]
    covariance_history_by_algorithm: dict[str, dict[str, Array]]
    v14_history: RecursiveCooperativeHistory


def build_v14_comparison_case(
    *,
    duration: float = 20.0,
    dt: float = 2.0,
    range_sigma: float = 2.0,
    range_rate_sigma: float = 0.02,
    angle_sigma: float = np.deg2rad(0.05),
    random_seed: int = 20260731,
    observation_strategy: str = "shared",
) -> V14ComparisonCase:
    if duration < 0.0 or dt <= 0.0:
        raise ValueError("duration must be nonnegative and dt must be positive.")
    if min(range_sigma, range_rate_sigma, angle_sigma) <= 0.0:
        raise ValueError("Measurement sigmas must be positive.")
    usage_by_strategy = {
        "single_endpoint": "observer_only",
        "shared": "both_endpoints",
        "independent_reciprocal": "observer_only",
    }
    if observation_strategy not in usage_by_strategy:
        raise ValueError(
            "observation_strategy must be 'single_endpoint', 'shared', or "
            "'independent_reciprocal'."
        )
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    base = keplerian_to_eci(
        R_EARTH + 700e3,
        0.001,
        np.deg2rad(23.0),
        0.0,
        0.0,
        0.0,
    )
    truth_initials = {
        "sat_01": base + np.array([0.0, -40.0, 0.0, 0.02, 0.0, 0.0]),
        "sat_02": base + np.array([50.0, 10.0, 20.0, 0.0, -0.02, 0.01]),
        "sat_03": base + np.array([-30.0, 35.0, -10.0, -0.01, 0.02, 0.0]),
    }
    scenario = generate_fleet_scenario(
        timestamps=timestamps,
        initial_state_by_node=truth_initials,
    )
    initial_errors = {
        "sat_01": np.array([25.0, -15.0, 10.0, 0.02, -0.01, 0.01]),
        "sat_02": np.array([-20.0, 18.0, -12.0, -0.02, 0.01, -0.01]),
        "sat_03": np.array([10.0, -12.0, 15.0, 0.01, -0.02, 0.01]),
    }
    initial_states = {
        node_id: truth_initials[node_id] + initial_errors[node_id]
        for node_id in scenario.node_ids
    }
    initial_covariances = {
        node_id: np.diag([50.0, 50.0, 50.0, 0.1, 0.1, 0.1]) ** 2
        for node_id in scenario.node_ids
    }
    topology = fully_connected_topology(list(scenario.node_ids))
    inter_satellite, messages = _build_observations(
        scenario,
        topology,
        range_sigma=range_sigma,
        range_rate_sigma=range_rate_sigma,
        angle_sigma=angle_sigma,
        random_seed=random_seed,
        reciprocal=observation_strategy == "independent_reciprocal",
    )
    return V14ComparisonCase(
        scenario=scenario,
        initial_state_by_node=initial_states,
        initial_covariance_by_node=initial_covariances,
        topology=topology,
        inter_satellite_observations=inter_satellite,
        observation_messages=messages,
        frame_by_modality={"AZ_EL": "RTN"},
        observation_strategy=observation_strategy,
        observation_usage=usage_by_strategy[observation_strategy],
    )


def run_v14_comparison(
    case: V14ComparisonCase,
    *,
    process_noise_acceleration: float = 1e-8,
    fleet_ci_grid_points: int = 7,
) -> V14ComparisonResult:
    scenario = case.scenario
    independent_states, independent_covariances = _run_independent(
        timestamps=scenario.timestamps,
        initial_states=case.initial_state_by_node,
        initial_covariances=case.initial_covariance_by_node,
        process_noise_acceleration=process_noise_acceleration,
    )
    v14 = run_recursive_distributed_cooperative_filter(
        timestamps=scenario.timestamps,
        initial_state_by_node=case.initial_state_by_node,
        initial_covariance_by_node=case.initial_covariance_by_node,
        topology=case.topology,
        observation_messages=case.observation_messages,
        process_noise_acceleration=process_noise_acceleration,
        observation_usage=case.observation_usage,
    )
    centralized = run_fleet_centralized_filter(
        timestamps=scenario.timestamps,
        initial_state_by_node=case.initial_state_by_node,
        initial_covariance_by_node=case.initial_covariance_by_node,
        inter_satellite_observations=case.inter_satellite_observations,
        node_ids=scenario.node_ids,
        process_noise_acceleration=process_noise_acceleration,
        frame_by_modality=case.frame_by_modality,
    )
    fleet_ci = run_distributed_fleet_state_ci(
        timestamps=scenario.timestamps,
        initial_state_by_node=case.initial_state_by_node,
        initial_covariance_by_node=case.initial_covariance_by_node,
        topology=case.topology,
        inter_satellite_observations=case.inter_satellite_observations,
        node_ids=scenario.node_ids,
        process_noise_acceleration=process_noise_acceleration,
        ci_grid_points=fleet_ci_grid_points,
        frame_by_modality=case.frame_by_modality,
    )
    fleet_ci_covariances = _fleet_ci_physical_covariances(fleet_ci)
    estimates = {
        "independent": independent_states,
        "v14_distributed": v14.posterior_state_history_by_node,
        "centralized_fleet_ekf": centralized.state_history_by_node,
        "distributed_fleet_state_ci": fleet_ci.physical_state_history_by_node,
    }
    covariances = {
        "independent": independent_covariances,
        "v14_distributed": v14.posterior_covariance_history_by_node,
        "centralized_fleet_ekf": centralized.covariance_history_by_node,
        "distributed_fleet_state_ci": fleet_ci_covariances,
    }
    nis = {
        "independent": None,
        "v14_distributed": _nested_nis(v14.nis_history_by_node),
        "centralized_fleet_ekf": _flat_nis(centralized.nis_history),
        "distributed_fleet_state_ci": _nested_nis(fleet_ci.nis_history_by_node),
    }
    communications = {
        "independent": {},
        "v14_distributed": asdict(v14.communication_stats),
        "centralized_fleet_ekf": {},
        "distributed_fleet_state_ci": asdict(fleet_ci.communication_stats),
    }
    metrics = {
        algorithm: _evaluate(
            truth=scenario.truth_state_history_by_node,
            estimates=algorithm_estimates,
            covariances=covariances[algorithm],
            mean_nis=nis[algorithm],
            communication=communications[algorithm],
        )
        for algorithm, algorithm_estimates in estimates.items()
    }
    return V14ComparisonResult(
        case=case,
        metrics_by_algorithm=metrics,
        estimate_history_by_algorithm=estimates,
        covariance_history_by_algorithm=covariances,
        v14_history=v14,
    )


def export_v14_comparison(
    result: V14ComparisonResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "v14_comparison_summary.json"
    csv_path = output / "v14_comparison_metrics.csv"
    payload = {
        algorithm: {
            **asdict(metrics),
            "by_node": {
                node_id: asdict(node_metrics)
                for node_id, node_metrics in metrics.by_node.items()
            },
        }
        for algorithm, metrics in result.metrics_by_algorithm.items()
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "algorithm",
                "node_id",
                "position_rmse",
                "velocity_rmse",
                "mean_nees",
                "mean_nis",
            ],
        )
        writer.writeheader()
        for algorithm, metrics in result.metrics_by_algorithm.items():
            for node_id, node_metrics in metrics.by_node.items():
                writer.writerow(
                    {
                        "algorithm": algorithm,
                        "node_id": node_id,
                        **asdict(node_metrics),
                        "mean_nis": metrics.mean_nis,
                    }
                )
    return {"json": json_path, "csv": csv_path}


def _build_observations(
    scenario: FleetScenario,
    topology: NetworkTopology,
    *,
    range_sigma: float,
    range_rate_sigma: float,
    angle_sigma: float,
    random_seed: int,
    reciprocal: bool,
) -> tuple[list[InterSatelliteObservation], list[ObservationMessage]]:
    rng = np.random.default_rng(random_seed)
    observations: list[InterSatelliteObservation] = []
    messages: list[ObservationMessage] = []
    directed_edges = [
        (source, target)
        for source in topology.node_ids
        for target in topology.neighbors(source)
        if reciprocal or source < target
    ]
    for source, target in directed_edges:
            for index, timestamp in enumerate(scenario.timestamps):
                state_i = scenario.trajectories[source].state_history_eci[index]
                state_j = scenario.trajectories[target].state_history_eci[index]
                values = {
                    "RANGE": (
                        np.array([
                            measure_relative_range(state_i, state_j)
                            + rng.normal(0.0, range_sigma)
                        ]),
                        np.array([[range_sigma**2]]),
                        "ECI",
                    ),
                    "RANGE_RATE": (
                        np.array([
                            measure_relative_range_rate(state_i, state_j)
                            + rng.normal(0.0, range_rate_sigma)
                        ]),
                        np.array([[range_rate_sigma**2]]),
                        "ECI",
                    ),
                    "AZ_EL": (
                        measure_relative_az_el(
                            state_i,
                            state_j,
                            frame="RTN",
                            noise=rng.normal(0.0, angle_sigma, size=2),
                        ),
                        np.eye(2) * angle_sigma**2,
                        "RTN",
                    ),
                }
                for modality, (measurement, covariance, frame) in values.items():
                    message_id = (
                        f"{source}->{target}:{modality}:{float(timestamp):.9f}"
                    )
                    physical_id = f"physical:{message_id}"
                    observations.append(
                        InterSatelliteObservation(
                            timestamp=float(timestamp),
                            source_node_id=source,
                            target_node_id=target,
                            modality=modality,
                            measurement=measurement.copy(),
                            covariance=covariance.copy(),
                            confidence=1.0,
                            valid_flag=True,
                            metadata={"frame": frame, "message_id": message_id},
                        )
                    )
                    messages.append(
                        ObservationMessage(
                            message_id=message_id,
                            observer_id=source,
                            target_id=target,
                            timestamp=float(timestamp),
                            modality=modality,
                            measurement=measurement.copy(),
                            covariance=covariance.copy(),
                            frame=frame,
                            physical_observation_id=physical_id,
                        )
                    )
    return observations, messages


def _run_independent(
    *,
    timestamps: Array,
    initial_states: Mapping[str, Array],
    initial_covariances: Mapping[str, Array],
    process_noise_acceleration: float,
) -> tuple[dict[str, Array], dict[str, Array]]:
    states = {}
    covariances = {}
    for node_id in initial_states:
        state_history = np.zeros((len(timestamps), 6), dtype=float)
        covariance_history = np.zeros((len(timestamps), 6, 6), dtype=float)
        state_history[0] = initial_states[node_id]
        covariance_history[0] = initial_covariances[node_id]
        for index in range(1, len(timestamps)):
            state_history[index], covariance_history[index] = (
                propagate_state_covariance(
                    state_history[index - 1],
                    covariance_history[index - 1],
                    float(timestamps[index] - timestamps[index - 1]),
                    process_noise_acceleration=process_noise_acceleration,
                )
            )
        states[node_id] = state_history
        covariances[node_id] = covariance_history
    return states, covariances


def _evaluate(
    *,
    truth: Mapping[str, Array],
    estimates: Mapping[str, Array],
    covariances: Mapping[str, Array],
    mean_nis: float | None,
    communication: dict[str, float | int],
) -> AlgorithmMetrics:
    by_node = {}
    all_position_errors = []
    all_velocity_errors = []
    all_nees = []
    for node_id in truth:
        error = estimates[node_id] - truth[node_id]
        nees = compute_nees_history(
            estimates[node_id],
            truth[node_id],
            covariances[node_id],
        )
        by_node[node_id] = NodeMetrics(
            position_rmse=compute_rmse(error[:, :3]),
            velocity_rmse=compute_rmse(error[:, 3:]),
            mean_nees=float(np.mean(nees)),
        )
        all_position_errors.append(error[:, :3])
        all_velocity_errors.append(error[:, 3:])
        all_nees.append(nees)
    return AlgorithmMetrics(
        by_node=by_node,
        fleet_position_rmse=compute_rmse(np.vstack(all_position_errors)),
        fleet_velocity_rmse=compute_rmse(np.vstack(all_velocity_errors)),
        mean_nees=float(np.mean(np.concatenate(all_nees))),
        mean_nis=mean_nis,
        communication=communication,
    )


def _fleet_ci_physical_covariances(history) -> dict[str, Array]:
    result = {}
    for node_index, node_id in enumerate(history.node_ids):
        block = slice(6 * node_index, 6 * (node_index + 1))
        result[node_id] = history.local_stacked_covariance_history_by_node[
            node_id
        ][:, block, block].copy()
    return result


def _flat_nis(history: list[dict[str, float]]) -> float | None:
    values = [value for epoch in history for value in epoch.values()]
    return float(np.mean(values)) if values else None


def _nested_nis(
    history_by_node: Mapping[str, list[dict[str, float]]],
) -> float | None:
    values = [
        value
        for history in history_by_node.values()
        for epoch in history
        for value in epoch.values()
    ]
    return float(np.mean(values)) if values else None
