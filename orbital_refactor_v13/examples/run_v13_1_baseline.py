"""v13.1 three-satellite centralized/distributed baseline comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cooperative.fleet_filter_runner import run_fleet_filter
from cooperative.topology import fully_connected_topology
from interfaces.data_objects import InterSatelliteObservation
from orbital_core.constants import R_EARTH
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.metrics import compute_rmse
from orbital_core.orbit_elements import keplerian_to_eci
from pipelines.fleet_centralized import run_fleet_centralized_filter
from scenarios.fleet_scenario import generate_fleet_scenario


def build_case(
    *,
    range_sigma=2.0,
    range_rate_sigma=0.01,
    angle_sigma=np.deg2rad(0.02),
    angle_frame="RTN",
    include_range=True,
    include_range_rate=True,
    include_angles=True,
    random_seed=20260729,
):
    timestamps = np.arange(0.0, 121.0, 2.0)
    base = keplerian_to_eci(
        R_EARTH + 700e3, 0.001, np.deg2rad(23.0), 0.0, 0.0, 0.0
    )
    truth_initials = {
        "sat_01": base + np.array([0.0, -40.0, 0.0, 0.02, 0.0, 0.0]),
        "sat_02": base + np.array([50.0, 10.0, 20.0, 0.0, -0.02, 0.01]),
        "sat_03": base + np.array([-30.0, 35.0, -10.0, -0.01, 0.02, 0.0]),
    }
    scenario = generate_fleet_scenario(
        timestamps=timestamps, initial_state_by_node=truth_initials
    )
    initial_errors = {
        "sat_01": np.array([25.0, -15.0, 10.0, 0.02, -0.01, 0.01]),
        "sat_02": np.array([-20.0, 18.0, -12.0, -0.02, 0.01, -0.01]),
        "sat_03": np.array([10.0, -12.0, 15.0, 0.01, -0.02, 0.01]),
    }
    estimates = {
        node_id: truth_initials[node_id] + initial_errors[node_id]
        for node_id in scenario.node_ids
    }
    covariances = {
        node_id: np.diag([50.0, 50.0, 50.0, 0.1, 0.1, 0.1]) ** 2
        for node_id in scenario.node_ids
    }
    topology = fully_connected_topology(list(scenario.node_ids))
    observations = build_observations(
        scenario,
        topology,
        range_sigma=range_sigma,
        range_rate_sigma=range_rate_sigma,
        angle_sigma=angle_sigma,
        angle_frame=angle_frame,
        include_range=include_range,
        include_range_rate=include_range_rate,
        include_angles=include_angles,
        random_seed=random_seed,
    )
    return scenario, estimates, covariances, topology, observations


def build_observations(
    scenario,
    topology,
    *,
    range_sigma=2.0,
    range_rate_sigma=0.01,
    angle_sigma=np.deg2rad(0.02),
    angle_frame="RTN",
    include_range=True,
    include_range_rate=True,
    include_angles=True,
    random_seed=20260729,
):
    if not any((include_range, include_range_rate, include_angles)):
        raise ValueError("At least one inter-satellite modality must be enabled.")
    if range_sigma <= 0.0 or range_rate_sigma <= 0.0 or angle_sigma <= 0.0:
        raise ValueError("Measurement noise sigmas must be positive.")
    rng = np.random.default_rng(random_seed)
    observations = []
    for source in topology.node_ids:
        for target in topology.neighbors(source):
            for index, timestamp in enumerate(scenario.timestamps):
                state_i = scenario.trajectories[source].state_history_eci[index]
                state_j = scenario.trajectories[target].state_history_eci[index]
                if include_range:
                    observations.append(
                        InterSatelliteObservation(
                            timestamp=float(timestamp),
                            source_node_id=source,
                            target_node_id=target,
                            modality="RANGE",
                            measurement=np.array([
                                measure_relative_range(state_i, state_j)
                                + rng.normal(0.0, range_sigma)
                            ]),
                            covariance=np.array([[range_sigma**2]]),
                            confidence=1.0,
                            valid_flag=True,
                        )
                    )
                if include_range_rate:
                    observations.append(
                        InterSatelliteObservation(
                            timestamp=float(timestamp),
                            source_node_id=source,
                            target_node_id=target,
                            modality="RANGE_RATE",
                            measurement=np.array([
                                measure_relative_range_rate(state_i, state_j)
                                + rng.normal(0.0, range_rate_sigma)
                            ]),
                            covariance=np.array([[range_rate_sigma**2]]),
                            confidence=1.0,
                            valid_flag=True,
                        )
                    )
                if include_angles:
                    observations.append(
                        InterSatelliteObservation(
                            timestamp=float(timestamp),
                            source_node_id=source,
                            target_node_id=target,
                            modality="AZ_EL",
                            measurement=measure_relative_az_el(
                                state_i,
                                state_j,
                                frame=angle_frame,
                                noise=rng.normal(0.0, angle_sigma, size=2),
                            ),
                            covariance=np.eye(2) * angle_sigma**2,
                            confidence=1.0,
                            valid_flag=True,
                            metadata={"frame": angle_frame},
                        )
                    )
    return observations


def main():
    scenario, initial, covariance, topology, observations = build_case()
    centralized = run_fleet_centralized_filter(
        timestamps=scenario.timestamps,
        initial_state_by_node=initial,
        initial_covariance_by_node=covariance,
        inter_satellite_observations=observations,
        node_ids=scenario.node_ids,
        process_noise_acceleration=1e-8,
        frame_by_modality={"AZ_EL": "RTN"},
    )
    distributed = run_fleet_filter(
        timestamps=scenario.timestamps,
        initial_state_by_node=initial,
        initial_covariance_by_node=covariance,
        topology=topology,
        inter_satellite_observations=observations,
        process_noise_acceleration_std=1e-8,
        enable_state_consensus=False,
        inter_satellite_frame_by_modality={"AZ_EL": "RTN"},
    )

    print("v13.1 three-satellite baseline")
    print("=" * 54)
    print_metrics(
        "Centralized 18-state EKF",
        scenario.truth_state_history_by_node,
        centralized.state_history_by_node,
    )
    print_metrics(
        "Distributed local EKF with neighbor covariance",
        scenario.truth_state_history_by_node,
        distributed.state_history_by_node,
    )


def print_metrics(label, truth_by_node, estimate_by_node):
    print(f"\n{label}:")
    for node_id in truth_by_node:
        error = estimate_by_node[node_id] - truth_by_node[node_id]
        print(
            f"  {node_id}: position={compute_rmse(error[:, :3]):.3f} m, "
            f"velocity={compute_rmse(error[:, 3:]):.6f} m/s"
        )


if __name__ == "__main__":
    main()
