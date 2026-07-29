"""End-to-end multi-satellite cooperative estimation example.

The example generates one target and three observer trajectories, creates
heterogeneous local observations, runs one independent federated CI filter per
observer, converts all local relative estimates to target absolute ECI, and
performs epoch-wise multi-node covariance intersection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.synthetic_measurement_adapter import (
    create_infrared_observations,
    create_nn_state_observations,
    create_radar_observations,
    visibility_flags,
)
from cooperative.multi_sat_pipeline import run_cooperative_pipeline
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario
from visualization.plot_cooperative_results import plot_cooperative_results


def build_demo_case():
    timestamps = np.arange(0.0, 301.0, 1.0)
    altitude = 700e3
    inclination = np.deg2rad(23.0)
    target_initial = keplerian_to_eci(
        R_EARTH + altitude, 0.001, inclination, 0.0, 0.0, 0.0
    )
    observer_initials = {
        "sat_01": keplerian_to_eci(
            R_EARTH + altitude, 0.001, inclination, 0.0, 0.0, np.deg2rad(0.05)
        ),
        "sat_02": keplerian_to_eci(
            R_EARTH + altitude + 1000.0, 0.001, inclination,
            np.deg2rad(0.02), 0.0, np.deg2rad(-0.05)
        ),
        "sat_03": keplerian_to_eci(
            R_EARTH + altitude - 1000.0, 0.001, inclination,
            np.deg2rad(-0.02), 0.0, np.deg2rad(0.08)
        ),
    }
    scenario = generate_cooperative_scenario(
        timestamps=timestamps,
        target_id="target_01",
        target_initial_state_eci=target_initial,
        observer_initial_states_eci=observer_initials,
    )

    modality_by_node = {
        "sat_01": ("ir", "nn"),
        "sat_02": ("ir", "rad"),
        "sat_03": ("rad", "nn"),
    }
    observations_by_node = {}
    for index, (node_id, modalities) in enumerate(modality_by_node.items(), start=1):
        rel_eci = scenario.relative_state_eci_by_node[node_id]
        rel_spri = scenario.relative_state_spri_by_node[node_id]
        rng = np.random.default_rng(100 + index)
        valid = visibility_flags(
            relative_position_spri=rel_spri[:, :3], max_range=2.0e6
        )
        observations = []
        if "ir" in modalities:
            observations += create_infrared_observations(
                timestamps=timestamps,
                relative_position_spri=rel_spri[:, :3],
                covariance=np.diag(np.deg2rad([0.02, 0.02])) ** 2,
                observer_id=node_id,
                target_id=scenario.target_id,
                rng=rng,
                valid_flags=valid,
            )
        if "rad" in modalities:
            observations += create_radar_observations(
                timestamps=timestamps,
                relative_position_spri=rel_spri[:, :3],
                relative_velocity_spri=rel_spri[:, 3:],
                covariance=np.diag([20.0, 0.05]) ** 2,
                observer_id=node_id,
                target_id=scenario.target_id,
                rng=rng,
                valid_flags=valid,
            )
        if "nn" in modalities:
            observations += create_nn_state_observations(
                timestamps=timestamps,
                relative_state_eci=rel_eci,
                covariance=np.diag([30.0, 30.0, 30.0, 0.1, 0.1, 0.1]) ** 2,
                observer_id=node_id,
                target_id=scenario.target_id,
                rng=rng,
                valid_flags=valid,
            )
        observations_by_node[node_id] = observations

    initial_errors = {
        "sat_01": np.array([80.0, -50.0, 40.0, 0.08, -0.04, 0.03]),
        "sat_02": np.array([-60.0, 70.0, -30.0, -0.06, 0.05, -0.02]),
        "sat_03": np.array([45.0, 35.0, 65.0, 0.04, 0.03, -0.05]),
    }
    modality_config = {
        "sat_01": {"nn": {"nn_meas_frame": "eci", "nn_use_pseudo_velocity": True}},
        "sat_02": {},
        "sat_03": {"nn": {"nn_meas_frame": "eci", "nn_use_pseudo_velocity": True}},
    }
    return scenario, observations_by_node, initial_errors, modality_config


def main() -> None:
    scenario, observations, initial_errors, modality_config = build_demo_case()
    result = run_cooperative_pipeline(
        scenario=scenario,
        observations_by_node=observations,
        initial_error_by_node=initial_errors,
        initial_covariance=np.diag([150.0, 150.0, 150.0, 0.3, 0.3, 0.3]) ** 2,
        architecture="federated_ci",
        process_noise_acceleration_std=1e-4,
        reset_feedback=True,
        ci_grid_points=31,
        modality_config_by_node=modality_config,
    )

    metrics = result.metrics
    print("Local RMSE:")
    for node_id in result.local_histories:
        print(
            f"  {node_id}: position={metrics.local_position_rmse[node_id]:.3f} m, "
            f"velocity={metrics.local_velocity_rmse[node_id]:.6f} m/s"
        )
    print(
        f"Cooperative CI: position={metrics.cooperative_position_rmse:.3f} m, "
        f"velocity={metrics.cooperative_velocity_rmse:.6f} m/s"
    )
    print(
        f"Improvement over best local: position={metrics.position_improvement_over_best:.2f}%, "
        f"velocity={metrics.velocity_improvement_over_best:.2f}%"
    )
    print("Final node weights:", result.cooperative_history.node_weight_history[-1])


    plot_cooperative_results(
        scenario=scenario,
        result=result,
        save_dir=None,
        show=True,
    )


if __name__ == "__main__":
    main()
