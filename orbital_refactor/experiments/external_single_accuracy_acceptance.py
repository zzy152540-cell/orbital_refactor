"""Paired P07 acceptance comparison for the external requirements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
)
from orbital_core.dynamics import accel_two_body_j2


@dataclass(frozen=True)
class SingleAccuracyRun:
    seed: int
    duration_seconds: float
    dt_seconds: float
    ekf_position_rmse_m: float
    federated_position_rmse_m: float
    position_improvement_percent: float
    radar_only_position_rmse_m: float
    infrared_only_position_rmse_m: float
    optical_only_position_rmse_m: float
    best_single_modality: str
    best_single_modality_position_rmse_m: float
    position_improvement_over_best_single_percent: float
    ekf_velocity_rmse_mps: float
    federated_velocity_rmse_mps: float
    velocity_improvement_percent: float
    ekf_acceleration_rmse_mps2: float
    federated_acceleration_rmse_mps2: float
    acceleration_improvement_percent: float
    early_position_improvement_percent: float
    stable_position_improvement_percent: float
    ekf_mean_nis_optical: float
    ekf_mean_nis_infrared: float
    ekf_mean_nis_radar: float
    federated_mean_nis_optical: float
    federated_mean_nis_infrared: float
    federated_mean_nis_radar: float
    federated_mean_ci_weight_optical: float
    federated_mean_ci_weight_infrared: float
    federated_mean_ci_weight_radar: float
    federated_local_position_rmse_optical_m: float
    federated_local_position_rmse_infrared_m: float
    federated_local_position_rmse_radar_m: float
    federated_local_mean_covariance_trace_optical: float
    federated_local_mean_covariance_trace_infrared: float
    federated_local_mean_covariance_trace_radar: float
    ekf_rejection_count: int
    federated_rejection_count: int
    ekf_runtime_seconds: float
    federated_runtime_seconds: float
    measurement_counts_identical: bool
    finite: bool


@dataclass(frozen=True)
class SingleAccuracyAcceptance:
    threshold_percent: float
    formal_minimum_seed_count: int
    threshold_met: bool
    formal_sample_size_met: bool
    passed: bool
    mean_position_improvement_percent: float
    mean_position_improvement_over_best_single_percent: float
    position_improvement_95_half_width_percent: float
    mean_velocity_improvement_percent: float
    mean_acceleration_improvement_percent: float
    runs: tuple[SingleAccuracyRun, ...]
    interpretation: str


def run_external_single_accuracy_acceptance(
    *, seeds=range(5), duration=120.0, dt=2.0, threshold_percent=5.0,
    formal_minimum_seed_count=20,
):
    """Compare existing centralized EKF and Federated-CI on paired data."""
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be nonempty and unique.")
    if duration <= 0.0 or dt <= 0.0 or duration < dt:
        raise ValueError("duration and dt must define at least two epochs.")
    runs = tuple(
        _run_pair(seed, duration=float(duration), dt=float(dt))
        for seed in seed_values
    )
    centralized_improvements = np.asarray([
        run.position_improvement_percent for run in runs
    ])
    improvements = np.asarray([
        run.position_improvement_over_best_single_percent for run in runs
    ])
    half_width = (
        0.0 if len(runs) == 1 else
        1.96 * float(np.std(improvements, ddof=1)) / np.sqrt(len(runs))
    )
    mean_improvement = float(np.mean(improvements))
    threshold_met = bool(mean_improvement >= threshold_percent)
    sample_size_met = bool(len(runs) >= formal_minimum_seed_count)
    passed = bool(
        all(run.finite and run.measurement_counts_identical for run in runs)
        and threshold_met and sample_size_met
    )
    return SingleAccuracyAcceptance(
        threshold_percent=float(threshold_percent),
        formal_minimum_seed_count=int(formal_minimum_seed_count),
        threshold_met=threshold_met,
        formal_sample_size_met=sample_size_met, passed=passed,
        mean_position_improvement_percent=float(np.mean(
            centralized_improvements
        )),
        mean_position_improvement_over_best_single_percent=mean_improvement,
        position_improvement_95_half_width_percent=half_width,
        mean_velocity_improvement_percent=float(np.mean([
            run.velocity_improvement_percent for run in runs
        ])),
        mean_acceleration_improvement_percent=float(np.mean([
            run.acceleration_improvement_percent for run in runs
        ])),
        runs=runs,
        interpretation=(
            "P07 pilot: centralized EKF and three-modal Federated-CI use the "
            "same deterministic truth and measurement realization for each seed."
        ),
    )


def save_external_single_accuracy_acceptance(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "summary.json"
    csv_path = output / "runs.csv"
    json_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(SingleAccuracyRun.__dataclass_fields__),
        )
        writer.writeheader()
        for run in report.runs:
            writer.writerow(asdict(run))
    return json_path, csv_path


def _run_pair(seed, *, duration, dt):
    common = {
        "duration": duration,
        "dt": dt,
        "seed": seed,
        "outage_modalities": (),
        "enable_cann": False,
        # Put the target on the positive-SPRI-depth side of the observer so
        # that the three-modal acceptance case genuinely includes optical.
        "observer_raan_deg": 15.5,
    }
    started = perf_counter()
    ekf = run_single_satellite_cann_comparison(
        **common, filter_architecture="centralized",
    )
    ekf_runtime = perf_counter() - started
    started = perf_counter()
    federated = run_single_satellite_cann_comparison(
        **common, filter_architecture="federated_ci",
    )
    federated_runtime = perf_counter() - started
    single_modality = {
        modality: run_single_satellite_cann_comparison(
            **common,
            filter_architecture="centralized",
            outage_windows={
                other: (0.0, duration)
                for other in ("opt", "ir", "rad") if other != modality
            },
        )
        for modality in ("opt", "ir", "rad")
    }

    ekf_summary = ekf["summary"]
    federated_summary = federated["summary"]
    ekf_acceleration = _acceleration_rmse(ekf)
    federated_acceleration = _acceleration_rmse(federated)
    measurement_keys = (
        "optical_valid_count", "infrared_valid_count", "radar_valid_count",
    )
    values = (
        ekf_summary["position_rmse_m"],
        federated_summary["position_rmse_m"],
        ekf_summary["velocity_rmse_mps"],
        federated_summary["velocity_rmse_mps"],
        ekf_acceleration,
        federated_acceleration,
    )
    split = max(1, int(np.ceil(len(ekf["position_error_m"]) / 3.0)))
    ekf_early = _rmse(ekf["position_error_m"][:split])
    federated_early = _rmse(federated["position_error_m"][:split])
    ekf_stable = _rmse(ekf["position_error_m"][split:])
    federated_stable = _rmse(federated["position_error_m"][split:])
    ekf_nis = _mean_nis_by_sensor(ekf)
    federated_nis = _mean_nis_by_sensor(federated)
    ci_weights = _mean_ci_weights(federated["ci_weight_history"])
    single_position_rmse = {
        modality: float(result["summary"]["position_rmse_m"])
        for modality, result in single_modality.items()
    }
    best_single_modality = min(
        single_position_rmse, key=single_position_rmse.get,
    )
    best_single_rmse = single_position_rmse[best_single_modality]
    local_rmse = {
        modality: _rmse(values)
        for modality, values in federated[
            "local_position_error_by_modality"
        ].items()
    }
    local_trace = {
        modality: float(np.mean(values))
        for modality, values in federated[
            "local_covariance_trace_by_modality"
        ].items()
    }
    return SingleAccuracyRun(
        seed=seed, duration_seconds=duration, dt_seconds=dt,
        ekf_position_rmse_m=float(values[0]),
        federated_position_rmse_m=float(values[1]),
        position_improvement_percent=_improvement(values[0], values[1]),
        radar_only_position_rmse_m=single_position_rmse["rad"],
        infrared_only_position_rmse_m=single_position_rmse["ir"],
        optical_only_position_rmse_m=single_position_rmse["opt"],
        best_single_modality=best_single_modality,
        best_single_modality_position_rmse_m=best_single_rmse,
        position_improvement_over_best_single_percent=_improvement(
            best_single_rmse, values[1],
        ),
        ekf_velocity_rmse_mps=float(values[2]),
        federated_velocity_rmse_mps=float(values[3]),
        velocity_improvement_percent=_improvement(values[2], values[3]),
        ekf_acceleration_rmse_mps2=float(values[4]),
        federated_acceleration_rmse_mps2=float(values[5]),
        acceleration_improvement_percent=_improvement(values[4], values[5]),
        early_position_improvement_percent=_improvement(
            ekf_early, federated_early,
        ),
        stable_position_improvement_percent=_improvement(
            ekf_stable, federated_stable,
        ),
        ekf_mean_nis_optical=ekf_nis["opt"],
        ekf_mean_nis_infrared=ekf_nis["ir"],
        ekf_mean_nis_radar=ekf_nis["rad"],
        federated_mean_nis_optical=federated_nis["opt"],
        federated_mean_nis_infrared=federated_nis["ir"],
        federated_mean_nis_radar=federated_nis["rad"],
        federated_mean_ci_weight_optical=ci_weights["opt"],
        federated_mean_ci_weight_infrared=ci_weights["ir"],
        federated_mean_ci_weight_radar=ci_weights["rad"],
        federated_local_position_rmse_optical_m=local_rmse["opt"],
        federated_local_position_rmse_infrared_m=local_rmse["ir"],
        federated_local_position_rmse_radar_m=local_rmse["rad"],
        federated_local_mean_covariance_trace_optical=local_trace["opt"],
        federated_local_mean_covariance_trace_infrared=local_trace["ir"],
        federated_local_mean_covariance_trace_radar=local_trace["rad"],
        ekf_rejection_count=_rejection_count(ekf),
        federated_rejection_count=_rejection_count(federated),
        ekf_runtime_seconds=float(ekf_runtime),
        federated_runtime_seconds=float(federated_runtime),
        measurement_counts_identical=all(
            ekf_summary[key] == federated_summary[key]
            for key in measurement_keys
        ),
        finite=bool(np.all(np.isfinite(values))),
    )


def _acceleration_rmse(result):
    estimated = np.asarray([
        accel_two_body_j2(state[:3])
        for state in result["estimated_state_history_eci"]
    ])
    truth = np.asarray([
        accel_two_body_j2(state[:3])
        for state in result["truth_state_history_eci"]
    ])
    return float(np.sqrt(np.mean(np.sum((estimated - truth) ** 2, axis=1))))


def _improvement(reference, candidate):
    if reference <= 0.0:
        return 0.0 if candidate == reference else float("-inf")
    return float(100.0 * (reference - candidate) / reference)


def _rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def _mean_nis_by_sensor(result):
    means = {}
    for modality in ("opt", "ir", "rad"):
        values = np.asarray(result["nis_by_modality"][modality], dtype=float)
        finite = values[np.isfinite(values)]
        means[modality] = float(np.mean(finite)) if finite.size else float("nan")
    return means


def _mean_ci_weights(history):
    samples = {modality: [] for modality in ("opt", "ir", "rad")}
    for weights in history or ():
        if not weights:
            continue
        for modality in samples:
            samples[modality].append(float(weights.get(modality, 0.0)))
    return {
        modality: float(np.mean(values)) if values else float("nan")
        for modality, values in samples.items()
    }


def _rejection_count(result):
    return int(sum(
        int(statistics.get("rejected", 0))
        for statistics in result["filter_statistics"].values()
    ))
