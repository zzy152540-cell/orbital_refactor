"""v13.4 BODY-angle attitude-coupling comparison."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interfaces.attitude_data_objects import AttitudeEstimate
from interfaces.data_objects import InterSatelliteObservation
from orbital_core.attitude import (
    attitude_error_angle_deg,
    quat_multiply_wxyz,
    small_angle_quaternion_wxyz,
)
from orbital_core.attitude_filter import AttitudeGyroBiasMEKF
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.metrics import compute_nees_history, compute_rmse
from pipelines.fleet_centralized import (
    FleetCentralizedHistory,
    run_fleet_centralized_filter,
)
from scenarios.attitude_scenario import (
    AttitudeTruthTrajectory,
    generate_attitude_truth,
    simulate_gyro_observations,
    simulate_star_tracker_observations,
)
from examples.run_v13_1_baseline import build_case

Array = np.ndarray


@dataclass(frozen=True)
class CouplingMetrics:
    position_rmse: float
    velocity_rmse: float
    mean_angle_nis: float
    mean_orbit_nees: float


@dataclass(frozen=True)
class AttitudeCouplingComparison:
    truth_attitude: CouplingMetrics
    mekf_with_covariance: CouplingMetrics
    mekf_without_covariance: CouplingMetrics
    mean_attitude_error_deg: float
    histories: dict[str, FleetCentralizedHistory]


def run_comparison(
    *,
    duration: float = 120.0,
    step: float = 2.0,
    random_seed: int = 20260729,
) -> AttitudeCouplingComparison:
    """Run truth-attitude and MEKF-attitude BODY-angle comparisons."""

    scenario, initial, covariance, topology, range_observations = build_case(
        include_angles=False,
        random_seed=random_seed,
    )
    sample_count = int(round(duration / step)) + 1
    if sample_count < 2 or sample_count > scenario.timestamps.size:
        raise ValueError("duration and step must select at least two available samples.")
    selected_times = np.arange(sample_count, dtype=float) * step
    if not np.allclose(scenario.timestamps[:sample_count], selected_times):
        raise ValueError("Requested comparison grid does not match the baseline grid.")
    scenario = _truncate_scenario(scenario, sample_count)
    initial = {
        node_id: (
            scenario.trajectories[node_id].state_history_eci[0]
            + 0.05
            * (
                initial[node_id]
                - scenario.trajectories[node_id].state_history_eci[0]
            )
        )
        for node_id in scenario.node_ids
    }
    range_observations = [
        observation
        for observation in range_observations
        if observation.timestamp <= selected_times[-1]
    ]

    attitude_truth, truth_estimates, mekf_estimates = _build_attitude_histories(
        scenario.node_ids,
        selected_times,
        random_seed=random_seed,
    )
    body_observations = _build_body_angle_observations(
        scenario,
        topology,
        attitude_truth,
        random_seed=random_seed + 1000,
    )
    observations = [*range_observations, *body_observations]
    mekf_without_covariance = [
        _without_attitude_covariance(estimate) for estimate in mekf_estimates
    ]
    attitude_sets = {
        "truth_attitude": truth_estimates,
        "mekf_with_covariance": mekf_estimates,
        "mekf_without_covariance": mekf_without_covariance,
    }
    histories = {
        label: run_fleet_centralized_filter(
            timestamps=selected_times,
            initial_state_by_node=initial,
            initial_covariance_by_node=covariance,
            inter_satellite_observations=observations,
            attitude_estimates=estimates,
            node_ids=scenario.node_ids,
            process_noise_acceleration=1.0e-8,
            frame_by_modality={"AZ_EL": "BODY"},
        )
        for label, estimates in attitude_sets.items()
    }
    metrics = {
        label: _summarize(history, scenario.truth_state_history_by_node)
        for label, history in histories.items()
    }
    attitude_errors = [
        attitude_error_angle_deg(
            estimate.quaternion_i2b_wxyz,
            attitude_truth[estimate.satellite_id].quaternion_i2b_wxyz[index],
        )
        for index in range(sample_count)
        for estimate in mekf_estimates
        if estimate.timestamp == selected_times[index]
    ]
    return AttitudeCouplingComparison(
        truth_attitude=metrics["truth_attitude"],
        mekf_with_covariance=metrics["mekf_with_covariance"],
        mekf_without_covariance=metrics["mekf_without_covariance"],
        mean_attitude_error_deg=float(np.mean(attitude_errors)),
        histories=histories,
    )


def _build_attitude_histories(node_ids, timestamps, *, random_seed):
    inertia = np.diag([120.0, 100.0, 80.0])
    truth_by_node: dict[str, AttitudeTruthTrajectory] = {}
    truth_estimates: list[AttitudeEstimate] = []
    mekf_estimates: list[AttitudeEstimate] = []
    for node_index, node_id in enumerate(node_ids):
        initial_rotation = np.deg2rad(
            [0.4 * node_index, -0.3 * node_index, 1.0 * node_index]
        )
        truth = generate_attitude_truth(
            satellite_id=node_id,
            timestamps=timestamps,
            initial_quaternion_i2b_wxyz=small_angle_quaternion_wxyz(
                initial_rotation
            ),
            initial_angular_velocity_body=np.deg2rad(
                [0.01, -0.015, 0.02]
            ),
            inertia=inertia,
        )
        truth_by_node[node_id] = truth
        gyro, _ = simulate_gyro_observations(
            truth,
            white_noise_std=np.deg2rad(0.002),
            bias_random_walk_std=np.deg2rad(1.0e-5),
            initial_bias=np.deg2rad([0.001, -0.0015, 0.0005]),
            random_seed=random_seed + 10 * node_index,
        )
        stars = simulate_star_tracker_observations(
            truth,
            update_interval=5,
            small_angle_noise_std=np.deg2rad(0.05),
            random_seed=random_seed + 10 * node_index + 1,
        )
        star_by_time = {observation.timestamp: observation for observation in stars}
        initial_error = small_angle_quaternion_wxyz(
            np.deg2rad([0.7, -0.5, 0.8])
        )
        filter_obj = AttitudeGyroBiasMEKF(
            satellite_id=node_id,
            quaternion_i2b_wxyz=quat_multiply_wxyz(
                initial_error,
                truth.quaternion_i2b_wxyz[0],
            ),
            angular_velocity_body=truth.angular_velocity_body[0],
            gyro_bias=np.zeros(3),
            covariance=np.diag(
                [
                    *([np.deg2rad(1.0) ** 2] * 3),
                    *([np.deg2rad(0.02) ** 2] * 3),
                    *([np.deg2rad(0.005) ** 2] * 3),
                ]
            ),
            inertia=inertia,
            angular_acceleration_noise_std=np.deg2rad(2.0e-4),
            gyro_bias_random_walk_std=np.deg2rad(1.0e-5),
        )
        for index, timestamp in enumerate(timestamps):
            if index > 0:
                filter_obj.predict(float(timestamp - timestamps[index - 1]))
            filter_obj.update_gyro(
                gyro[index].angular_rate_body,
                gyro[index].covariance,
            )
            if float(timestamp) in star_by_time:
                star = star_by_time[float(timestamp)]
                filter_obj.update_star_tracker(
                    star.quaternion_i2b_wxyz,
                    star.covariance_small_angle,
                )
            mekf_estimates.append(filter_obj.estimate(float(timestamp)))
            truth_estimates.append(
                AttitudeEstimate(
                    timestamp=float(timestamp),
                    satellite_id=node_id,
                    quaternion_i2b_wxyz=truth.quaternion_i2b_wxyz[index],
                    angular_velocity_body=truth.angular_velocity_body[index],
                    gyro_bias=np.zeros(3),
                    error_covariance=np.zeros((9, 9)),
                )
            )
    return truth_by_node, truth_estimates, mekf_estimates


def _build_body_angle_observations(
    scenario,
    topology,
    attitude_truth,
    *,
    random_seed,
):
    rng = np.random.default_rng(random_seed)
    angle_sigma = np.deg2rad(0.02)
    observations = []
    for source in topology.node_ids:
        for target in topology.neighbors(source):
            for index, timestamp in enumerate(scenario.timestamps):
                state_i = scenario.trajectories[source].state_history_eci[index]
                state_j = scenario.trajectories[target].state_history_eci[index]
                quaternion = attitude_truth[source].quaternion_i2b_wxyz[index]
                observations.append(
                    InterSatelliteObservation(
                        timestamp=float(timestamp),
                        source_node_id=source,
                        target_node_id=target,
                        modality="AZ_EL",
                        measurement=measure_relative_az_el(
                            state_i,
                            state_j,
                            frame="BODY",
                            noise=rng.normal(0.0, angle_sigma, size=2),
                            quaternion_i2b_wxyz=quaternion,
                        ),
                        covariance=np.eye(2) * angle_sigma**2,
                        confidence=1.0,
                        valid_flag=True,
                        metadata={"frame": "BODY"},
                    )
                )
    return observations


def _without_attitude_covariance(estimate):
    return AttitudeEstimate(
        timestamp=estimate.timestamp,
        satellite_id=estimate.satellite_id,
        quaternion_i2b_wxyz=estimate.quaternion_i2b_wxyz,
        angular_velocity_body=estimate.angular_velocity_body,
        gyro_bias=estimate.gyro_bias,
        error_covariance=np.zeros((9, 9)),
    )


def _summarize(history, truth_by_node):
    position_errors = []
    velocity_errors = []
    nees = []
    for node_id in history.node_ids:
        estimate = history.state_history_by_node[node_id]
        truth = truth_by_node[node_id]
        error = estimate - truth
        position_errors.append(error[:, :3])
        velocity_errors.append(error[:, 3:])
        nees.extend(
            compute_nees_history(
                estimate,
                truth,
                history.covariance_history_by_node[node_id],
            )
        )
    angle_nis = [
        value
        for epoch in history.nis_history
        for label, value in epoch.items()
        if label.endswith(":AZ_EL")
    ]
    return CouplingMetrics(
        position_rmse=compute_rmse(np.vstack(position_errors)),
        velocity_rmse=compute_rmse(np.vstack(velocity_errors)),
        mean_angle_nis=float(np.mean(angle_nis)),
        mean_orbit_nees=float(np.mean(nees)),
    )


def _truncate_scenario(scenario, sample_count):
    from scenarios.fleet_scenario import FleetScenario, FleetTrajectory

    return FleetScenario(
        timestamps=scenario.timestamps[:sample_count].copy(),
        trajectories={
            node_id: FleetTrajectory(
                satellite_id=node_id,
                timestamps=trajectory.timestamps[:sample_count].copy(),
                state_history_eci=trajectory.state_history_eci[:sample_count].copy(),
                q_eci2rtn_history=trajectory.q_eci2rtn_history[:sample_count].copy(),
            )
            for node_id, trajectory in scenario.trajectories.items()
        },
    )


def main():
    comparison = run_comparison()
    print("v13.4 attitude-aware BODY-angle comparison")
    print("=" * 64)
    print(f"Mean MEKF attitude error: {comparison.mean_attitude_error_deg:.6f} deg")
    for label, metrics in (
        ("Truth attitude", comparison.truth_attitude),
        ("MEKF + attitude covariance", comparison.mekf_with_covariance),
        ("MEKF without attitude covariance", comparison.mekf_without_covariance),
    ):
        print(
            f"{label:34s} "
            f"pos={metrics.position_rmse:9.4f} m  "
            f"vel={metrics.velocity_rmse:10.6f} m/s  "
            f"angle NIS={metrics.mean_angle_nis:9.4f}  "
            f"orbit NEES={metrics.mean_orbit_nees:9.4f}"
        )


if __name__ == "__main__":
    main()
