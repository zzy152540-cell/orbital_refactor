"""Parameter sweep for the v13.2 three-satellite estimation baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cooperative.fleet_filter_runner import run_fleet_filter
from cooperative.fleet_state_ci_runner import run_distributed_fleet_state_ci
from examples.run_v13_1_baseline import build_case
from examples.run_v13_2_fleet_ci import build_anchor_observations
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.metrics import compute_nees_history, compute_rmse
from pipelines.fleet_centralized import run_fleet_centralized_filter


MODES = {
    "range_rate": (True, True, False),
    "angle": (False, False, True),
    "combined": (True, True, True),
}


def run_sweep(
    *,
    seed_count: int,
    angle_sigmas_deg: list[float],
    process_noises: list[float],
    modes: list[str],
    output_path: Path,
    ci_grid_points: int = 11,
) -> list[dict[str, object]]:
    if seed_count < 1:
        raise ValueError("seed_count must be at least one.")
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise ValueError(f"Unknown measurement modes: {unknown}")
    rows: list[dict[str, object]] = []
    for seed_index in range(seed_count):
        measurement_seed = 20260729 + seed_index
        anchor_seed = 20261729 + seed_index
        for mode in modes:
            include_range, include_range_rate, include_angles = MODES[mode]
            mode_angle_sigmas = angle_sigmas_deg if include_angles else [angle_sigmas_deg[0]]
            for angle_sigma_deg in mode_angle_sigmas:
                scenario, initial, covariance, topology, observations = build_case(
                    angle_sigma=np.deg2rad(angle_sigma_deg),
                    include_range=include_range,
                    include_range_rate=include_range_rate,
                    include_angles=include_angles,
                    random_seed=measurement_seed,
                )
                anchors = build_anchor_observations(
                    scenario,
                    interval=5,
                    position_sigma=2.0,
                    random_seed=anchor_seed,
                )
                truth = scenario.truth_state_history_by_node
                propagation = {
                    node_id: propagate_absolute_orbit(initial[node_id], scenario.timestamps)
                    for node_id in scenario.node_ids
                }
                baseline_position, baseline_velocity = _fleet_rmse(truth, propagation)
                for process_noise in process_noises:
                    centralized = run_fleet_centralized_filter(
                        timestamps=scenario.timestamps,
                        initial_state_by_node=initial,
                        initial_covariance_by_node=covariance,
                        inter_satellite_observations=observations,
                        absolute_position_observations=anchors,
                        node_ids=scenario.node_ids,
                        process_noise_acceleration=process_noise,
                        frame_by_modality={"AZ_EL": "RTN"},
                    )
                    local = run_fleet_filter(
                        timestamps=scenario.timestamps,
                        initial_state_by_node=initial,
                        initial_covariance_by_node=covariance,
                        topology=topology,
                        inter_satellite_observations=observations,
                        process_noise_acceleration_std=process_noise,
                        enable_state_consensus=False,
                        inter_satellite_frame_by_modality={"AZ_EL": "RTN"},
                    )
                    fleet_ci = run_distributed_fleet_state_ci(
                        timestamps=scenario.timestamps,
                        initial_state_by_node=initial,
                        initial_covariance_by_node=covariance,
                        topology=topology,
                        inter_satellite_observations=observations,
                        absolute_position_observations=anchors,
                        node_ids=scenario.node_ids,
                        process_noise_acceleration=process_noise,
                        ci_grid_points=ci_grid_points,
                        frame_by_modality={"AZ_EL": "RTN"},
                    )
                    fleet_ci_covariances = _fleet_ci_physical_covariances(
                        fleet_ci, scenario.node_ids
                    )
                    common = {
                        "seed": measurement_seed,
                        "mode": mode,
                        "angle_sigma_deg": angle_sigma_deg if include_angles else "",
                        "process_noise": process_noise,
                        "baseline_position_rmse": baseline_position,
                        "baseline_velocity_rmse": baseline_velocity,
                    }
                    rows.extend(
                        [
                            _result_row(
                                common,
                                "centralized",
                                truth,
                                centralized.state_history_by_node,
                                centralized.covariance_history_by_node,
                                _flatten_dict_history(centralized.nis_history),
                            ),
                            _result_row(
                                common,
                                "local_6d",
                                truth,
                                local.state_history_by_node,
                                local.covariance_history_by_node,
                                _local_block_nis(local.inter_satellite_nis_history_by_node),
                            ),
                            _result_row(
                                common,
                                "fleet_ci",
                                truth,
                                fleet_ci.physical_state_history_by_node,
                                fleet_ci_covariances,
                                _flatten_node_dict_history(fleet_ci.nis_history_by_node),
                            ),
                        ]
                    )
    _write_csv(output_path, rows)
    _write_csv(_summary_path(output_path), summarize_rows(rows))
    return rows


def _result_row(common, algorithm, truth, estimates, covariances, nis_values):
    position_rmse, velocity_rmse = _fleet_rmse(truth, estimates)
    nees = _mean_node_nees(truth, estimates, covariances)
    baseline_position = float(common["baseline_position_rmse"])
    baseline_velocity = float(common["baseline_velocity_rmse"])
    per_node = {}
    for node_id in truth:
        error = estimates[node_id] - truth[node_id]
        per_node[f"{node_id}_position_rmse"] = compute_rmse(error[:, :3])
        per_node[f"{node_id}_velocity_rmse"] = compute_rmse(error[:, 3:])
    return {
        **common,
        "algorithm": algorithm,
        "position_rmse": position_rmse,
        "velocity_rmse": velocity_rmse,
        "position_improvement_pct": 100.0 * (baseline_position - position_rmse) / baseline_position,
        "velocity_ratio": velocity_rmse / baseline_velocity,
        "mean_nees": nees,
        "mean_nis": float(np.mean(nis_values)) if nis_values else float("nan"),
        **per_node,
    }


def _fleet_rmse(truth, estimates):
    position = []
    velocity = []
    for node_id in truth:
        error = estimates[node_id] - truth[node_id]
        position.append(compute_rmse(error[:, :3]))
        velocity.append(compute_rmse(error[:, 3:]))
    return float(np.mean(position)), float(np.mean(velocity))


def _mean_node_nees(truth, estimates, covariances):
    histories = [
        compute_nees_history(
            estimates[node_id],
            truth[node_id],
            covariances[node_id],
        )
        for node_id in truth
    ]
    return float(np.mean(np.concatenate(histories)))


def _fleet_ci_physical_covariances(history, node_ids):
    return {
        node_id: history.local_stacked_covariance_history_by_node[node_id][
            :, 6 * index:6 * (index + 1), 6 * index:6 * (index + 1)
        ]
        for index, node_id in enumerate(node_ids)
    }


def _flatten_dict_history(history):
    return [
        float(value)
        for per_epoch in history
        for value in per_epoch.values()
        if np.isfinite(value)
    ]


def _flatten_node_dict_history(history_by_node):
    return [
        float(value)
        for history in history_by_node.values()
        for per_epoch in history
        for value in per_epoch.values()
        if np.isfinite(value)
    ]


def _local_block_nis(history_by_node):
    return [
        float(value)
        for history in history_by_node.values()
        for per_epoch in history
        for key, value in per_epoch.items()
        if key.endswith(":BLOCK") and np.isfinite(value)
    ]


def _write_csv(output_path, rows):
    if not rows:
        raise ValueError("Sweep produced no result rows.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows):
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["algorithm"],
            row["mode"],
            row["angle_sigma_deg"],
            row["process_noise"],
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (algorithm, mode, angle_sigma_deg, process_noise), group in grouped.items():
        summary = {
            "algorithm": algorithm,
            "mode": mode,
            "angle_sigma_deg": angle_sigma_deg,
            "process_noise": process_noise,
            "runs": len(group),
        }
        for field in (
            "position_rmse",
            "velocity_rmse",
            "position_improvement_pct",
            "velocity_ratio",
            "mean_nees",
            "mean_nis",
        ):
            values = np.array([float(row[field]) for row in group], dtype=float)
            summary[f"{field}_mean"] = float(np.mean(values))
            summary[f"{field}_std"] = float(np.std(values))
        summaries.append(summary)
    return summaries


def _summary_path(output_path):
    return output_path.with_name(f"{output_path.stem}_summary{output_path.suffix}")


def print_recommendations(rows):
    rows = summarize_rows(rows)
    feasible = [
        row
        for row in rows
        if float(row["velocity_ratio_mean"]) <= 1.2
        and float(row["position_improvement_pct_mean"]) >= 50.0
    ]
    candidates = feasible or rows
    ranked = sorted(
        candidates,
        key=lambda row: (
            float(row["position_rmse_mean"]),
            float(row["velocity_ratio_mean"]),
            float(row["mean_nees_mean"]),
        ),
    )
    print(f"Generated {len(rows)} aggregated configurations.")
    print(
        f"Feasible rows (position improvement >=50%, velocity ratio <=1.2): "
        f"{len(feasible)}"
    )
    print("Top candidates:")
    for row in ranked[:10]:
        print(
            f"  {row['algorithm']:11s} mode={row['mode']:10s} "
            f"angle={str(row['angle_sigma_deg']):>5s} deg "
            f"q={float(row['process_noise']):.1e} "
            f"pos={float(row['position_rmse_mean']):.3f} m "
            f"vel={float(row['velocity_rmse_mean']):.6f} m/s "
            f"NEES={float(row['mean_nees_mean']):.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--angle-sigmas-deg",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.2, 0.5],
    )
    parser.add_argument(
        "--process-noises",
        type=float,
        nargs="+",
        default=[1e-8],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(MODES),
        default=list(MODES),
    )
    parser.add_argument("--ci-grid-points", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v13_2_parameter_sweep.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = run_sweep(
        seed_count=args.seeds,
        angle_sigmas_deg=args.angle_sigmas_deg,
        process_noises=args.process_noises,
        modes=args.modes,
        output_path=args.output,
        ci_grid_points=args.ci_grid_points,
    )
    print_recommendations(rows)
    print(f"CSV: {args.output.resolve()}")
    print(f"Summary CSV: {_summary_path(args.output).resolve()}")


if __name__ == "__main__":
    main()
