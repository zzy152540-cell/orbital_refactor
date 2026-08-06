"""Independent MEKF covariance-consistency Monte Carlo validation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orbital_core.attitude import (
    attitude_error_angle_deg,
    left_attitude_error_vector,
    quat_multiply_wxyz,
    small_angle_quaternion_wxyz,
)
from orbital_core.attitude_filter import AttitudeGyroBiasMEKF
from scenarios.attitude_scenario import (
    generate_attitude_truth,
    simulate_gyro_observations,
    simulate_star_tracker_observations,
)

METRICS = (
    "attitude_error_deg",
    "attitude_nees",
    "gyro_nis",
    "star_tracker_nis",
)


def run_monte_carlo(
    seeds,
    *,
    duration: float = 120.0,
    step: float = 2.0,
    satellite_count: int = 3,
):
    if duration <= 0.0 or step <= 0.0:
        raise ValueError("duration and step must be positive.")
    if satellite_count < 1:
        raise ValueError("satellite_count must be at least one.")
    timestamps = np.arange(0.0, duration + 0.5 * step, step)
    rows = []
    failed_seeds = []
    for seed in (int(value) for value in seeds):
        try:
            rows.extend(
                _run_seed(
                    seed,
                    timestamps=timestamps,
                    satellite_count=satellite_count,
                )
            )
        except Exception as exc:
            failed_seeds.append(
                {
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows, _aggregate(rows, failed_seeds), failed_seeds


def _run_seed(seed, *, timestamps, satellite_count):
    inertia = np.diag([120.0, 100.0, 80.0])
    rows = []
    for satellite_index in range(satellite_count):
        satellite_id = f"sat_{satellite_index + 1:02d}"
        initial_rotation = np.deg2rad(
            [
                0.4 * satellite_index,
                -0.3 * satellite_index,
                1.0 * satellite_index,
            ]
        )
        truth = generate_attitude_truth(
            satellite_id=satellite_id,
            timestamps=timestamps,
            initial_quaternion_i2b_wxyz=small_angle_quaternion_wxyz(
                initial_rotation
            ),
            initial_angular_velocity_body=np.deg2rad(
                [0.01, -0.015, 0.02]
            ),
            inertia=inertia,
        )
        gyro, _ = simulate_gyro_observations(
            truth,
            white_noise_std=np.deg2rad(0.002),
            bias_random_walk_std=np.deg2rad(1.0e-5),
            initial_bias=np.deg2rad([0.001, -0.0015, 0.0005]),
            random_seed=seed + 10 * satellite_index,
        )
        stars = simulate_star_tracker_observations(
            truth,
            update_interval=5,
            small_angle_noise_std=np.deg2rad(0.05),
            random_seed=seed + 10 * satellite_index + 1,
        )
        star_by_time = {observation.timestamp: observation for observation in stars}
        initial_error = small_angle_quaternion_wxyz(
            np.deg2rad([0.7, -0.5, 0.8])
        )
        filter_obj = AttitudeGyroBiasMEKF(
            satellite_id=satellite_id,
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
            gyro_nis = filter_obj.update_gyro(
                gyro[index].angular_rate_body,
                gyro[index].covariance,
            )
            star_nis = np.nan
            if float(timestamp) in star_by_time:
                star = star_by_time[float(timestamp)]
                star_nis = filter_obj.update_star_tracker(
                    star.quaternion_i2b_wxyz,
                    star.covariance_small_angle,
                )
            estimate = filter_obj.estimate(float(timestamp))
            attitude_error = left_attitude_error_vector(
                truth.quaternion_i2b_wxyz[index],
                estimate.quaternion_i2b_wxyz,
            )
            attitude_covariance = estimate.attitude_covariance
            attitude_nees = float(
                attitude_error.T
                @ np.linalg.pinv(attitude_covariance)
                @ attitude_error
            )
            rows.append(
                {
                    "seed": seed,
                    "satellite_id": satellite_id,
                    "timestamp": float(timestamp),
                    "attitude_error_deg": attitude_error_angle_deg(
                        estimate.quaternion_i2b_wxyz,
                        truth.quaternion_i2b_wxyz[index],
                    ),
                    "attitude_nees": attitude_nees,
                    "gyro_nis": gyro_nis,
                    "star_tracker_nis": star_nis,
                }
            )
    return rows


def _aggregate(rows, failed_seeds):
    seeds = sorted({row["seed"] for row in rows})
    summary = {
        "successful_seeds": len(seeds),
        "failed_seeds": len(failed_seeds),
        "failure_rate": (
            len(failed_seeds) / (len(seeds) + len(failed_seeds))
            if seeds or failed_seeds
            else np.nan
        ),
        "samples": len(rows),
        "attitude_state_dimension": 3,
        "gyro_measurement_dimension": 3,
        "star_tracker_measurement_dimension": 3,
    }
    for metric in METRICS:
        sample_values = _finite_values(row[metric] for row in rows)
        summary.update(_statistics(f"{metric}_sample", sample_values))
        seed_means = _finite_values(
            np.mean(
                _finite_values(
                    row[metric] for row in rows if row["seed"] == seed
                )
            )
            for seed in seeds
        )
        summary.update(_statistics(f"{metric}_seed_mean", seed_means))
    summary["attitude_nees_ratio_to_dimension"] = (
        summary["attitude_nees_sample_mean"] / 3.0
    )
    summary["gyro_nis_ratio_to_dimension"] = (
        summary["gyro_nis_sample_mean"] / 3.0
    )
    summary["star_tracker_nis_ratio_to_dimension"] = (
        summary["star_tracker_nis_sample_mean"] / 3.0
    )
    return summary


def _finite_values(values):
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


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


def export_results(rows, summary, failed_seeds, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name(f"{output.stem}_summary{output.suffix}")
    failures_path = output.with_name(f"{output.stem}_failures{output.suffix}")
    _write_csv(output, rows)
    _write_csv(summary_path, [summary])
    _write_csv(failures_path, failed_seeds, default_fields=("seed", "error"))
    return output, summary_path, failures_path


def _write_csv(path, rows, *, default_fields=()):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = list(default_fields)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument("--satellites", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v13_4_attitude_consistency.csv"),
    )
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    rows, summary, failures = run_monte_carlo(
        range(args.base_seed, args.base_seed + args.seeds),
        duration=args.duration,
        step=args.step,
        satellite_count=args.satellites,
    )
    raw_path, summary_path, failures_path = export_results(
        rows,
        summary,
        failures,
        args.output,
    )
    print(f"Raw results: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Failures: {failures_path}")
    print(
        "Consistency ratios to dimension 3: "
        f"attitude NEES={summary['attitude_nees_ratio_to_dimension']:.3f}, "
        f"gyro NIS={summary['gyro_nis_ratio_to_dimension']:.3f}, "
        "star-tracker NIS="
        f"{summary['star_tracker_nis_ratio_to_dimension']:.3f}"
    )


if __name__ == "__main__":
    main()
