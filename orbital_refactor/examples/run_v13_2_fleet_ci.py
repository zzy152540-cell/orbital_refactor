"""v13.2 four-way three-satellite acceptance comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cooperative.fleet_filter_runner import run_fleet_filter
from cooperative.fleet_state_ci_runner import run_distributed_fleet_state_ci
from examples.run_v13_1_baseline import build_case
from interfaces.data_objects import AbsolutePositionObservation
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.metrics import compute_nees_history, compute_rmse
from pipelines.fleet_centralized import run_fleet_centralized_filter


def main():
    scenario, initial, covariance, topology, observations = build_case()
    anchors = build_anchor_observations(
        scenario,
        interval=5,
        position_sigma=2.0,
        random_seed=20260730,
    )
    propagation = {
        node_id: propagate_absolute_orbit(initial[node_id], scenario.timestamps)
        for node_id in scenario.node_ids
    }
    centralized = run_fleet_centralized_filter(
        timestamps=scenario.timestamps,
        initial_state_by_node=initial,
        initial_covariance_by_node=covariance,
        inter_satellite_observations=observations,
        absolute_position_observations=anchors,
        node_ids=scenario.node_ids,
        process_noise_acceleration=1e-8,
        frame_by_modality={"AZ_EL": "RTN"},
    )
    distributed_local = run_fleet_filter(
        timestamps=scenario.timestamps,
        initial_state_by_node=initial,
        initial_covariance_by_node=covariance,
        topology=topology,
        inter_satellite_observations=observations,
        process_noise_acceleration_std=1e-8,
        enable_state_consensus=False,
        inter_satellite_frame_by_modality={"AZ_EL": "RTN"},
    )
    distributed_ci = run_distributed_fleet_state_ci(
        timestamps=scenario.timestamps,
        initial_state_by_node=initial,
        initial_covariance_by_node=covariance,
        topology=topology,
        inter_satellite_observations=observations,
        absolute_position_observations=anchors,
        node_ids=scenario.node_ids,
        process_noise_acceleration=1e-8,
        ci_grid_points=15,
        frame_by_modality={"AZ_EL": "RTN"},
    )

    truth = scenario.truth_state_history_by_node
    print("v13.2 three-satellite acceptance comparison")
    print("=" * 64)
    print_rmse("A. Dynamics propagation", truth, propagation)
    print_rmse("B. Centralized 18-state EKF + anchor", truth, centralized.state_history_by_node)
    print_rmse(
        "C. Per-satellite 6-state distributed EKF",
        truth,
        distributed_local.state_history_by_node,
    )
    print_rmse(
        "D. Distributed 18-state Fleet-State CI + anchor",
        truth,
        distributed_ci.physical_state_history_by_node,
    )
    print("\nMean per-satellite NEES:")
    print(f"  B centralized: {mean_node_nees(truth, centralized.covariance_history_by_node, centralized.state_history_by_node):.3f}")
    distributed_covariance = {
        node_id: distributed_ci.local_stacked_covariance_history_by_node[node_id][
            :, 6 * index:6 * (index + 1), 6 * index:6 * (index + 1)
        ]
        for index, node_id in enumerate(scenario.node_ids)
    }
    print(
        "  D Fleet-CI:   "
        f"{mean_node_nees(truth, distributed_covariance, distributed_ci.physical_state_history_by_node):.3f}"
    )


def build_anchor_observations(
    scenario,
    interval,
    *,
    position_sigma,
    random_seed,
):
    if interval < 1:
        raise ValueError("Anchor interval must be at least one sample.")
    if position_sigma <= 0.0:
        raise ValueError("Anchor position sigma must be positive.")
    rng = np.random.default_rng(random_seed)
    observations = []
    truth = scenario.trajectories["sat_01"].state_history_eci
    for index in range(0, len(scenario.timestamps), interval):
        observations.append(
            AbsolutePositionObservation(
                timestamp=float(scenario.timestamps[index]),
                satellite_id="sat_01",
                measurement_eci=(
                    truth[index, :3]
                    + rng.normal(0.0, position_sigma, size=3)
                ),
                covariance=np.eye(3) * position_sigma**2,
                confidence=1.0,
                valid_flag=True,
                source_type="GNSS",
            )
        )
    return observations


def print_rmse(label, truth, estimates):
    print(f"\n{label}:")
    for node_id in truth:
        error = estimates[node_id] - truth[node_id]
        print(
            f"  {node_id}: position={compute_rmse(error[:, :3]):.3f} m, "
            f"velocity={compute_rmse(error[:, 3:]):.6f} m/s"
        )


def mean_node_nees(truth, covariances, estimates):
    values = [
        compute_nees_history(estimates[node_id], truth[node_id], covariances[node_id])
        for node_id in truth
    ]
    return float(np.mean(np.concatenate(values)))


if __name__ == "__main__":
    main()
