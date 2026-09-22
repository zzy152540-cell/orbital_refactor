"""Run six analytic multimodal filters on the supplied 31-satellite data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adapters.project_trajectory_dataset import load_project_trajectory_dataset
from adapters.synthetic_measurement_adapter import (
    create_infrared_observations,
    create_optical_observations,
    create_radar_observations,
)
from exporters.trajectory_time_series import export_absolute_state_histories
from orbital_core.coordinates import build_rtn_quaternion_history, state_history_eci_to_spri
from orbital_core.dynamics import make_process_noise, rk4_step_rel
from orbital_core.filters import LocalDynamicsEKF
from pipelines.federated_ci import run_federated_ci_filter


def run_filter_smoke(
    background_path, functional_path, output_directory, *, duration=None,
    seed=0, modalities=("opt", "ir", "rad"), process_acceleration_std=1.0e-4,
    targets=None, target_scope="functional",
    maximum_dynamics_velocity_residual_mps=0.1,
    minimum_cross_track_fraction=0.05,
):
    dataset = load_project_trajectory_dataset(background_path, functional_path)
    count = len(dataset.timestamps) if duration is None else min(len(dataset.timestamps), int(duration) + 1)
    times = dataset.timestamps[:count]
    residuals_by_node = dataset.maximum_dynamics_velocity_residuals()
    nonmaneuver_nodes = dataset.nonmaneuver_nodes(
        maximum_velocity_residual_mps=maximum_dynamics_velocity_residual_mps,
    )
    selected_modalities = tuple(str(item).lower() for item in modalities)
    if not selected_modalities or not set(selected_modalities) <= {"opt", "ir", "rad"}:
        raise ValueError("modalities must be a nonempty subset of opt, ir, rad.")
    if targets is not None:
        selected_targets = tuple(str(item) for item in targets)
    elif target_scope == "functional":
        selected_targets = dataset.functional_nodes
    elif target_scope == "nonmaneuver_all":
        selected_targets = nonmaneuver_nodes
    else:
        raise ValueError("target_scope must be functional or nonmaneuver_all.")
    unknown_targets = set(selected_targets) - set(dataset.state_history_by_node)
    if not selected_targets or unknown_targets:
        raise ValueError(
            f"targets must be known nonempty nodes; unknown={sorted(unknown_targets)}"
        )
    inconsistent_targets = set(selected_targets) - set(nonmaneuver_nodes)
    if target_scope == "nonmaneuver_all" and inconsistent_targets:
        raise ValueError("nonmaneuver_all selected a dynamics-inconsistent target.")
    observer_candidates = tuple(
        node for node in dataset.background_nodes if node in set(nonmaneuver_nodes)
    )
    pairings = dataset.nearest_observers(
        selected_targets, candidates=observer_candidates,
        minimum_cross_track_fraction=minimum_cross_track_fraction,
    )
    estimates = {}
    summaries = []
    for pair_index, target in enumerate(selected_targets):
        observer = pairings[target]
        chief = dataset.state_history_by_node[observer][:count]
        truth_absolute = dataset.state_history_by_node[target][:count]
        truth_relative = truth_absolute - chief
        quaternions = build_rtn_quaternion_history(chief)
        relative_spri = state_history_eci_to_spri(truth_relative, quaternions)
        rng = np.random.default_rng(int(seed) + pair_index)
        covariances = {
            "opt": np.diag([2.0e-4, 2.0e-4]) ** 2,
            "ir": np.diag(np.deg2rad([0.02, 0.02])) ** 2,
            "rad": np.diag([20.0, 0.05]) ** 2,
        }
        optical = create_optical_observations(
            timestamps=times, relative_position_spri=relative_spri[:, :3],
            covariance=covariances["opt"], observer_id=observer, target_id=target,
            rng=rng, valid_flags=np.ones(count, dtype=bool),
        )
        infrared = create_infrared_observations(
            timestamps=times, relative_position_spri=relative_spri[:, :3],
            covariance=covariances["ir"], observer_id=observer, target_id=target,
            rng=rng, valid_flags=np.ones(count, dtype=bool),
        )
        radar = create_radar_observations(
            timestamps=times, relative_position_spri=relative_spri[:, :3],
            relative_velocity_spri=relative_spri[:, 3:], covariance=covariances["rad"],
            observer_id=observer, target_id=target, rng=rng,
            valid_flags=np.ones(count, dtype=bool),
        )
        observations = {
            name: values for name, values in {
                "opt": optical, "ir": infrared, "rad": radar,
            }.items() if name in selected_modalities
        }
        measurements = {
            name: np.vstack([item.measurement for item in values])
            for name, values in observations.items()
        }
        valid = {
            name: np.array([item.valid_flag for item in values], dtype=bool)
            for name, values in observations.items()
        }
        process_noise = make_process_noise(1.0, float(process_acceleration_std))
        filters = {
            name: LocalDynamicsEKF(process_noise, covariances[name], name)
            for name in selected_modalities
        }
        initial_error = np.array([50.0, -40.0, 30.0, 0.05, -0.04, 0.03])
        result = run_federated_ci_filter(
            timestamps=times, chief_state_history_eci=chief,
            q_eci2pri_history=quaternions,
            measurements_by_modality=measurements,
            valid_flags_by_modality=valid,
            local_filters=filters,
            initial_state=truth_relative[0] + initial_error,
            initial_covariance=np.diag([100.0, 100.0, 100.0, 0.2, 0.2, 0.2]) ** 2,
            reset_feedback=True, ci_grid_points=11,
            node_id=observer, target_id=target,
        )
        estimated_absolute = chief + result.fused_state_history
        estimates[target] = estimated_absolute
        error = estimated_absolute - truth_absolute
        dynamics_residual = np.vstack([
            rk4_step_rel(truth_relative[index], chief[index], 1.0)
            - truth_relative[index + 1]
            for index in range(count - 1)
        ]) if count > 1 else np.zeros((0, 6))
        maximum_dynamics_velocity_residual = (
            float(np.max(np.linalg.norm(dynamics_residual[:, 3:], axis=1)))
            if len(dynamics_residual) else 0.0
        )
        summaries.append({
            "target": target, "observer": observer,
            "initialRangeM": float(np.linalg.norm(truth_relative[0, :3])),
            "positionRmseM": float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1)))),
            "velocityRmseMps": float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1)))),
            "finalPositionErrorM": float(np.linalg.norm(error[-1, :3])),
            "maximumOneStepDynamicsVelocityResidualMps": maximum_dynamics_velocity_residual,
            "nonManeuverDynamicsConsistent": maximum_dynamics_velocity_residual <= 0.1,
        })
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    trajectory = export_absolute_state_histories(
        times, estimates, output / "estimated_trajectory.json",
        scene_id=dataset.scene_id,
        time_base_id=f"TB-20260918T040000Z-1S-{count}",
        dataset_id="ESTIMATED-31SAT-FUNCTIONAL-ANALYTIC",
        start_time_ms=dataset.start_time_ms,
        satellite_id_by_node={node: dataset.numeric_id_by_node[node] for node in estimates},
    )
    summary = {
        "sceneId": dataset.scene_id, "frameCount": count,
        "estimatedSatelliteCount": len(estimates),
        "backgroundSatelliteCount": len(dataset.background_nodes),
        "targetScope": target_scope,
        "excludedNodes": sorted(set(dataset.state_history_by_node) - set(selected_targets)),
        "dynamicsVelocityResidualThresholdMps": float(maximum_dynamics_velocity_residual_mps),
        "minimumCrossTrackFraction": float(minimum_cross_track_fraction),
        "dynamicsInconsistentNodes": sorted(
            node for node, value in residuals_by_node.items()
            if value > float(maximum_dynamics_velocity_residual_mps)
        ),
        "measurementSource": "analytic_multimodal",
        "modalities": list(selected_modalities),
        "processAccelerationStdMps2": float(process_acceleration_std),
        "pairings": pairings, "results": summaries,
        "estimatedTrajectory": str(trajectory),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("background")
    parser.add_argument("functional")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--process-acceleration-std", type=float, default=1.0e-4)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument(
        "--target-scope", choices=("functional", "nonmaneuver_all"),
        default="functional",
    )
    parser.add_argument("--maximum-dynamics-velocity-residual", type=float, default=0.1)
    parser.add_argument("--minimum-cross-track-fraction", type=float, default=0.05)
    parser.add_argument(
        "--modalities", nargs="+", choices=("opt", "ir", "rad"),
        default=("opt", "ir", "rad"),
    )
    args = parser.parse_args()
    print(json.dumps(run_filter_smoke(
        args.background, args.functional, args.output,
        duration=args.duration, seed=args.seed, modalities=args.modalities,
        process_acceleration_std=args.process_acceleration_std,
        targets=args.targets,
        target_scope=args.target_scope,
        maximum_dynamics_velocity_residual_mps=args.maximum_dynamics_velocity_residual,
        minimum_cross_track_fraction=args.minimum_cross_track_fraction,
    ), indent=2))


if __name__ == "__main__":
    main()
