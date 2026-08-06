"""Distributed Fleet-State CI Monte Carlo for v13.4 attitude coupling."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cooperative.fleet_state_ci_runner import run_distributed_fleet_state_ci
from examples.run_v13_1_baseline import build_case
from examples.run_v13_4_attitude_coupling import (
    _build_attitude_histories,
    _build_body_angle_observations,
    _truncate_scenario,
    _without_attitude_covariance,
)
from orbital_core.attitude import attitude_error_angle_deg
from orbital_core.metrics import compute_nees_history, compute_rmse

METHODS = (
    "truth_attitude",
    "mekf_with_covariance",
    "mekf_without_covariance",
    "mekf_with_covariance_local_only",
)
METRICS = (
    "position_rmse",
    "velocity_rmse",
    "mean_angle_nis",
    "mean_orbit_nees",
)


def run_distributed_comparison(
    *,
    random_seed: int,
    packet_loss_rate: float,
    delay: float,
    duration: float = 120.0,
    step: float = 2.0,
    ci_grid_points: int = 11,
):
    if not 0.0 <= packet_loss_rate <= 1.0:
        raise ValueError("packet_loss_rate must be in [0, 1].")
    if delay < 0.0:
        raise ValueError("delay cannot be negative.")
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
    attitude_sets = {
        "truth_attitude": truth_estimates,
        "mekf_with_covariance": mekf_estimates,
        "mekf_without_covariance": [
            _without_attitude_covariance(estimate) for estimate in mekf_estimates
        ],
    }
    loss_by_node = {
        node_id: packet_loss_rate for node_id in scenario.node_ids
    }
    delay_by_node = {node_id: delay for node_id in scenario.node_ids}
    histories = {
        method: run_distributed_fleet_state_ci(
            timestamps=selected_times,
            initial_state_by_node=initial,
            initial_covariance_by_node=covariance,
            topology=topology,
            inter_satellite_observations=observations,
            attitude_estimates=estimates,
            node_ids=scenario.node_ids,
            process_noise_acceleration=1.0e-8,
            ci_grid_points=ci_grid_points,
            packet_loss_rate_by_node=loss_by_node,
            delay_by_node=delay_by_node,
            random_seed=random_seed,
            frame_by_modality={"AZ_EL": "BODY"},
        )
        for method, estimates in attitude_sets.items()
    }
    histories["mekf_with_covariance_local_only"] = (
        run_distributed_fleet_state_ci(
            timestamps=selected_times,
            initial_state_by_node=initial,
            initial_covariance_by_node=covariance,
            topology=topology,
            inter_satellite_observations=observations,
            attitude_estimates=mekf_estimates,
            node_ids=scenario.node_ids,
            process_noise_acceleration=1.0e-8,
            ci_grid_points=ci_grid_points,
            packet_loss_rate_by_node=loss_by_node,
            delay_by_node=delay_by_node,
            random_seed=random_seed,
            frame_by_modality={"AZ_EL": "BODY"},
            enable_ci_fusion=False,
        )
    )
    metrics = {}
    for method, history in histories.items():
        post_metrics = _summarize_distributed(
            history,
            scenario.truth_state_history_by_node,
        )
        metrics[method] = {
            **post_metrics,
            **_summarize_pre_ci(
                history,
                scenario.truth_state_history_by_node,
                post_metrics,
            ),
        }
    attitude_errors = [
        attitude_error_angle_deg(
            estimate.quaternion_i2b_wxyz,
            attitude_truth[estimate.satellite_id].quaternion_i2b_wxyz[index],
        )
        for index, timestamp in enumerate(selected_times)
        for estimate in mekf_estimates
        if estimate.timestamp == timestamp
    ]
    return metrics, histories, float(np.mean(attitude_errors))


def run_monte_carlo(
    seeds,
    *,
    packet_loss_rates=(0.0, 0.2),
    delays=(0.0, 4.0),
    duration=120.0,
    step=2.0,
    ci_grid_points=11,
):
    raw_rows = []
    for packet_loss_rate in packet_loss_rates:
        for delay in delays:
            for seed in (int(value) for value in seeds):
                try:
                    metrics, histories, attitude_error = run_distributed_comparison(
                        random_seed=seed,
                        packet_loss_rate=float(packet_loss_rate),
                        delay=float(delay),
                        duration=duration,
                        step=step,
                        ci_grid_points=ci_grid_points,
                    )
                except Exception as exc:
                    for method in METHODS:
                        raw_rows.append(
                            _failed_row(seed, packet_loss_rate, delay, method, exc)
                        )
                    continue
                for method in METHODS:
                    stats = histories[method].communication_stats
                    raw_rows.append(
                        {
                            "seed": seed,
                            "packet_loss_rate": float(packet_loss_rate),
                            "delay": float(delay),
                            "method": method,
                            **metrics[method],
                            "mean_attitude_error_deg": attitude_error,
                            "received_report_count": stats.received_report_count,
                            "dropped_report_count": stats.dropped_report_count,
                            "pending_report_count": stats.pending_report_count,
                            "average_delay": stats.average_delay,
                            "realized_packet_loss_rate": stats.packet_loss_rate,
                            "failed": False,
                            "error": "",
                        }
                    )
    return raw_rows, _aggregate(raw_rows)


def _summarize_distributed(history, truth_by_node):
    position_errors = []
    velocity_errors = []
    nees = []
    for index, node_id in enumerate(history.node_ids):
        estimate = history.physical_state_history_by_node[node_id]
        truth = truth_by_node[node_id]
        covariance = history.local_stacked_covariance_history_by_node[node_id][
            :, 6 * index:6 * (index + 1), 6 * index:6 * (index + 1)
        ]
        error = estimate - truth
        position_errors.append(error[:, :3])
        velocity_errors.append(error[:, 3:])
        nees.extend(compute_nees_history(estimate, truth, covariance))
    angle_nis = [
        value
        for node_history in history.nis_history_by_node.values()
        for epoch in node_history
        for label, value in epoch.items()
        if label.endswith(":AZ_EL")
    ]
    return {
        "position_rmse": compute_rmse(np.vstack(position_errors)),
        "velocity_rmse": compute_rmse(np.vstack(velocity_errors)),
        "mean_angle_nis": float(np.mean(angle_nis)),
        "mean_orbit_nees": float(np.mean(nees)),
    }


def _summarize_pre_ci(history, truth_by_node, post_metrics):
    position_errors = []
    velocity_errors = []
    nees = []
    for index, node_id in enumerate(history.node_ids):
        estimate = history.pre_ci_physical_state_history_by_node[node_id]
        truth = truth_by_node[node_id]
        covariance = history.pre_ci_stacked_covariance_history_by_node[node_id][
            :, 6 * index:6 * (index + 1), 6 * index:6 * (index + 1)
        ]
        error = estimate - truth
        position_errors.append(error[:, :3])
        velocity_errors.append(error[:, 3:])
        nees.extend(compute_nees_history(estimate, truth, covariance))
    position_rmse = compute_rmse(np.vstack(position_errors))
    velocity_rmse = compute_rmse(np.vstack(velocity_errors))
    return {
        "pre_ci_position_rmse": position_rmse,
        "pre_ci_velocity_rmse": velocity_rmse,
        "pre_ci_mean_orbit_nees": float(np.mean(nees)),
        "ci_position_gain": position_rmse - post_metrics["position_rmse"],
        "ci_velocity_gain": velocity_rmse - post_metrics["velocity_rmse"],
    }


def _failed_row(seed, packet_loss_rate, delay, method, exc):
    return {
        "seed": seed,
        "packet_loss_rate": float(packet_loss_rate),
        "delay": float(delay),
        "method": method,
        **{metric: np.nan for metric in METRICS},
        "pre_ci_position_rmse": np.nan,
        "pre_ci_velocity_rmse": np.nan,
        "pre_ci_mean_orbit_nees": np.nan,
        "ci_position_gain": np.nan,
        "ci_velocity_gain": np.nan,
        "mean_attitude_error_deg": np.nan,
        "received_report_count": np.nan,
        "dropped_report_count": np.nan,
        "pending_report_count": np.nan,
        "average_delay": np.nan,
        "realized_packet_loss_rate": np.nan,
        "failed": True,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _aggregate(raw_rows):
    summary = []
    scenarios = sorted(
        {
            (row["packet_loss_rate"], row["delay"])
            for row in raw_rows
        }
    )
    for packet_loss_rate, delay in scenarios:
        scenario_rows = [
            row
            for row in raw_rows
            if row["packet_loss_rate"] == packet_loss_rate
            and row["delay"] == delay
        ]
        for method in METHODS:
            rows = [row for row in scenario_rows if row["method"] == method]
            successful = [row for row in rows if not row["failed"]]
            result = {
                "packet_loss_rate": packet_loss_rate,
                "delay": delay,
                "method": method,
                "runs": len(rows),
                "successful_runs": len(successful),
                "failure_rate": 1.0 - len(successful) / len(rows),
            }
            for metric in (
                *METRICS,
                "pre_ci_position_rmse",
                "pre_ci_velocity_rmse",
                "pre_ci_mean_orbit_nees",
                "ci_position_gain",
                "ci_velocity_gain",
                "mean_attitude_error_deg",
                "average_delay",
                "realized_packet_loss_rate",
            ):
                values = np.asarray([row[metric] for row in successful])
                result.update(_statistics(metric, values))
            summary.append(result)
        summary.append(_paired_summary(packet_loss_rate, delay, scenario_rows))
    return summary


def _statistics(name, values):
    if values.size == 0:
        return {
            f"{name}_{statistic}": np.nan
            for statistic in ("mean", "std", "p05", "p50", "p95")
        }
    return {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f"{name}_p05": float(np.percentile(values, 5.0)),
        f"{name}_p50": float(np.percentile(values, 50.0)),
        f"{name}_p95": float(np.percentile(values, 95.0)),
    }


def _paired_summary(packet_loss_rate, delay, rows):
    by_seed = {}
    for row in rows:
        if not row["failed"]:
            by_seed.setdefault(row["seed"], {})[row["method"]] = row
    pairs = [
        methods
        for methods in by_seed.values()
        if "mekf_with_covariance" in methods
        and "mekf_without_covariance" in methods
    ]
    return {
        "packet_loss_rate": packet_loss_rate,
        "delay": delay,
        "method": "paired_covariance_effect",
        "runs": len({row["seed"] for row in rows}),
        "successful_runs": len(pairs),
        "failure_rate": (
            1.0 - len(pairs) / len({row["seed"] for row in rows}) if rows else np.nan
        ),
        "position_rmse_win_rate": _paired_win_rate(pairs, "position_rmse"),
        "velocity_rmse_win_rate": _paired_win_rate(pairs, "velocity_rmse"),
        "angle_nis_consistency_win_rate": (
            float(
                np.mean(
                    [
                        abs(pair["mekf_with_covariance"]["mean_angle_nis"] - 2.0)
                        < abs(
                            pair["mekf_without_covariance"]["mean_angle_nis"]
                            - 2.0
                        )
                        for pair in pairs
                    ]
                )
            )
            if pairs
            else np.nan
        ),
    }


def _paired_win_rate(pairs, metric):
    if not pairs:
        return np.nan
    return float(
        np.mean(
            [
                pair["mekf_with_covariance"][metric]
                < pair["mekf_without_covariance"][metric]
                for pair in pairs
            ]
        )
    )


def export_results(raw_rows, summary_rows, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(f"{output.stem}_summary{output.suffix}")
    _write_csv(output, raw_rows)
    _write_csv(summary_path, summary_rows)
    return output, summary_path


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument(
        "--packet-loss-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.2],
    )
    parser.add_argument("--delays", type=float, nargs="+", default=[0.0, 4.0])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument("--ci-grid-points", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v13_4_distributed_attitude_monte_carlo.csv"),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    raw, summary = run_monte_carlo(
        range(args.base_seed, args.base_seed + args.seeds),
        packet_loss_rates=args.packet_loss_rates,
        delays=args.delays,
        duration=args.duration,
        step=args.step,
        ci_grid_points=args.ci_grid_points,
    )
    raw_path, summary_path = export_results(raw, summary, args.output)
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")
    for row in summary:
        if row["method"] == "paired_covariance_effect":
            print(
                f"loss={row['packet_loss_rate']:.2f}, "
                f"delay={row['delay']:.1f}s: "
                f"position win={row['position_rmse_win_rate']:.3f}, "
                f"velocity win={row['velocity_rmse_win_rate']:.3f}, "
                f"NIS win={row['angle_nis_consistency_win_rate']:.3f}"
            )


if __name__ == "__main__":
    main()
