"""Paired P08 single-modality-loss acceptance for external requirements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from experiments.single_satellite_cann_comparison import (
    run_single_satellite_cann_comparison,
)
from orbital_core.dynamics import accel_two_body_j2


_MODALITY_NAMES = ("opt", "ir", "rad")
_VALID_COUNT_KEYS = {
    "opt": "optical_valid_count",
    "ir": "infrared_valid_count",
    "rad": "radar_valid_count",
}


@dataclass(frozen=True)
class SingleRobustnessRun:
    seed: int
    missing_modality: str
    duration_seconds: float
    dt_seconds: float
    reference_position_rmse_m: float
    dropout_position_rmse_m: float
    position_rmse_increase_percent: float
    reference_velocity_rmse_mps: float
    dropout_velocity_rmse_mps: float
    velocity_rmse_increase_percent: float
    reference_acceleration_rmse_mps2: float
    dropout_acceleration_rmse_mps2: float
    acceleration_rmse_increase_percent: float
    missing_modality_valid_count: int
    remaining_modality_valid_count: int
    dropout_optical_valid_count: int
    dropout_infrared_valid_count: int
    dropout_radar_valid_count: int
    reference_mean_ci_weight_optical: float
    reference_mean_ci_weight_infrared: float
    reference_mean_ci_weight_radar: float
    dropout_mean_ci_weight_optical: float
    dropout_mean_ci_weight_infrared: float
    dropout_mean_ci_weight_radar: float
    reference_local_position_rmse_optical_m: float
    reference_local_position_rmse_infrared_m: float
    reference_local_position_rmse_radar_m: float
    dropout_local_position_rmse_optical_m: float
    dropout_local_position_rmse_infrared_m: float
    dropout_local_position_rmse_radar_m: float
    reference_radial_rmse_m: float
    reference_transverse_rmse_m: float
    reference_normal_rmse_m: float
    dropout_radial_rmse_m: float
    dropout_transverse_rmse_m: float
    dropout_normal_rmse_m: float
    finite: bool


@dataclass(frozen=True)
class ModalityRobustnessGroup:
    missing_modality: str
    run_count: int
    mean_position_rmse_increase_percent: float
    position_increase_95_half_width_percent: float
    worst_position_rmse_increase_percent: float
    threshold_met: bool


@dataclass(frozen=True)
class SingleRobustnessAcceptance:
    measurement_source: str
    threshold_percent: float
    formal_minimum_seed_count: int
    formal_sample_size_met: bool
    passed: bool
    groups: tuple[ModalityRobustnessGroup, ...]
    runs: tuple[SingleRobustnessRun, ...]
    interpretation: str


def run_external_single_robustness_acceptance(
    *, seeds=range(5), duration=120.0, dt=2.0, threshold_percent=20.0,
    formal_minimum_seed_count=20,
    measurement_source_config=None,
    infrared_extent_enabled=False,
    infrared_effective_target_diameter_m=10.0,
    infrared_extent_fractional_sigma=0.1,
):
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be nonempty and unique.")
    if duration <= 0.0 or dt <= 0.0 or duration < dt:
        raise ValueError("duration and dt must define at least two epochs.")
    runs = []
    for seed in seed_values:
        reference = _run(
            seed, duration=duration, dt=dt,
            measurement_source_config=measurement_source_config,
            infrared_extent_enabled=infrared_extent_enabled,
            infrared_effective_target_diameter_m=(
                infrared_effective_target_diameter_m
            ),
            infrared_extent_fractional_sigma=infrared_extent_fractional_sigma,
        )
        for modality in _MODALITY_NAMES:
            dropout = _run(
                seed, duration=duration, dt=dt,
                outage_windows={modality: (0.0, duration)},
                measurement_source_config=measurement_source_config,
                infrared_extent_enabled=infrared_extent_enabled,
                infrared_effective_target_diameter_m=(
                    infrared_effective_target_diameter_m
                ),
                infrared_extent_fractional_sigma=(
                    infrared_extent_fractional_sigma
                ),
            )
            runs.append(_compare(
                seed, modality, reference, dropout,
                duration=float(duration), dt=float(dt),
            ))
    groups = tuple(
        _summarize(modality, runs, threshold_percent)
        for modality in _MODALITY_NAMES
    )
    sample_size_met = len(seed_values) >= formal_minimum_seed_count
    passed = bool(
        sample_size_met
        and all(group.threshold_met for group in groups)
        and all(run.finite for run in runs)
    )
    return SingleRobustnessAcceptance(
        measurement_source=(
            "analytic" if measurement_source_config is None
            else measurement_source_config.source.value
        ),
        threshold_percent=float(threshold_percent),
        formal_minimum_seed_count=int(formal_minimum_seed_count),
        formal_sample_size_met=sample_size_met, passed=passed,
        groups=groups, runs=tuple(runs),
        interpretation=(
            "P08 pilot: each full-duration modality outage is paired with a "
            "three-modal Federated-CI reference using the same seed. CANN is off."
        ),
    )


def save_external_single_robustness_acceptance(report, output_directory):
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
            stream, fieldnames=tuple(SingleRobustnessRun.__dataclass_fields__),
        )
        writer.writeheader()
        for run in report.runs:
            writer.writerow(asdict(run))
    return json_path, csv_path


def _run(
    seed, *, duration, dt, outage_windows=None,
    measurement_source_config=None,
    infrared_extent_enabled=False,
    infrared_effective_target_diameter_m=10.0,
    infrared_extent_fractional_sigma=0.1,
):
    return run_single_satellite_cann_comparison(
        duration=duration, dt=dt, seed=seed,
        outage_modalities=(), outage_windows=outage_windows,
        enable_cann=False, filter_architecture="federated_ci",
        observer_raan_deg=15.5,
        measurement_source_config=measurement_source_config,
        infrared_extent_enabled=infrared_extent_enabled,
        infrared_effective_target_diameter_m=(
            infrared_effective_target_diameter_m
        ),
        infrared_extent_fractional_sigma=infrared_extent_fractional_sigma,
    )


def _compare(seed, modality, reference, dropout, *, duration, dt):
    ref = reference["summary"]
    degraded = dropout["summary"]
    ref_acceleration = _acceleration_rmse(reference)
    dropout_acceleration = _acceleration_rmse(dropout)
    remaining = sum(
        degraded[_VALID_COUNT_KEYS[name]]
        for name in _MODALITY_NAMES if name != modality
    )
    reference_weights = _mean_ci_weights(reference["ci_weight_history"])
    dropout_weights = _mean_ci_weights(dropout["ci_weight_history"])
    reference_local_rmse = _local_position_rmse(reference)
    dropout_local_rmse = _local_position_rmse(dropout)
    reference_components = _spri_position_component_rmse(reference)
    dropout_components = _spri_position_component_rmse(dropout)
    values = (
        ref["position_rmse_m"], degraded["position_rmse_m"],
        ref["velocity_rmse_mps"], degraded["velocity_rmse_mps"],
        ref_acceleration, dropout_acceleration,
    )
    return SingleRobustnessRun(
        seed=seed, missing_modality=modality,
        duration_seconds=duration, dt_seconds=dt,
        reference_position_rmse_m=float(values[0]),
        dropout_position_rmse_m=float(values[1]),
        position_rmse_increase_percent=_increase(values[0], values[1]),
        reference_velocity_rmse_mps=float(values[2]),
        dropout_velocity_rmse_mps=float(values[3]),
        velocity_rmse_increase_percent=_increase(values[2], values[3]),
        reference_acceleration_rmse_mps2=float(values[4]),
        dropout_acceleration_rmse_mps2=float(values[5]),
        acceleration_rmse_increase_percent=_increase(values[4], values[5]),
        missing_modality_valid_count=int(
            degraded[_VALID_COUNT_KEYS[modality]]
        ),
        remaining_modality_valid_count=int(remaining),
        dropout_optical_valid_count=int(degraded["optical_valid_count"]),
        dropout_infrared_valid_count=int(degraded["infrared_valid_count"]),
        dropout_radar_valid_count=int(degraded["radar_valid_count"]),
        reference_mean_ci_weight_optical=reference_weights["opt"],
        reference_mean_ci_weight_infrared=reference_weights["ir"],
        reference_mean_ci_weight_radar=reference_weights["rad"],
        dropout_mean_ci_weight_optical=dropout_weights["opt"],
        dropout_mean_ci_weight_infrared=dropout_weights["ir"],
        dropout_mean_ci_weight_radar=dropout_weights["rad"],
        reference_local_position_rmse_optical_m=reference_local_rmse["opt"],
        reference_local_position_rmse_infrared_m=reference_local_rmse["ir"],
        reference_local_position_rmse_radar_m=reference_local_rmse["rad"],
        dropout_local_position_rmse_optical_m=dropout_local_rmse["opt"],
        dropout_local_position_rmse_infrared_m=dropout_local_rmse["ir"],
        dropout_local_position_rmse_radar_m=dropout_local_rmse["rad"],
        reference_radial_rmse_m=reference_components[0],
        reference_transverse_rmse_m=reference_components[1],
        reference_normal_rmse_m=reference_components[2],
        dropout_radial_rmse_m=dropout_components[0],
        dropout_transverse_rmse_m=dropout_components[1],
        dropout_normal_rmse_m=dropout_components[2],
        finite=bool(np.all(np.isfinite(values))),
    )


def _summarize(modality, runs, threshold):
    selected = [run for run in runs if run.missing_modality == modality]
    increases = np.asarray([
        run.position_rmse_increase_percent for run in selected
    ])
    half_width = (
        0.0 if len(selected) == 1 else
        1.96 * float(np.std(increases, ddof=1)) / np.sqrt(len(selected))
    )
    mean = float(np.mean(increases))
    return ModalityRobustnessGroup(
        missing_modality=modality, run_count=len(selected),
        mean_position_rmse_increase_percent=mean,
        position_increase_95_half_width_percent=half_width,
        worst_position_rmse_increase_percent=float(np.max(increases)),
        threshold_met=bool(mean <= threshold),
    )


def _mean_ci_weights(history):
    samples = {name: [] for name in _MODALITY_NAMES}
    for weights in history or ():
        if not weights:
            continue
        for name in samples:
            samples[name].append(float(weights.get(name, 0.0)))
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in samples.items()
    }


def _local_position_rmse(result):
    errors = result["local_position_error_by_modality"]
    return {
        name: float(np.sqrt(np.mean(np.asarray(errors[name]) ** 2)))
        for name in _MODALITY_NAMES
    }


def _spri_position_component_rmse(result):
    error = (
        np.asarray(result["estimated_relative_state_history_spri"])[:, :3]
        - np.asarray(result["truth_relative_state_history_spri"])[:, :3]
    )
    return tuple(float(value) for value in np.sqrt(np.mean(error**2, axis=0)))


def _acceleration_rmse(result):
    estimate = np.asarray([
        accel_two_body_j2(state[:3])
        for state in result["estimated_state_history_eci"]
    ])
    truth = np.asarray([
        accel_two_body_j2(state[:3])
        for state in result["truth_state_history_eci"]
    ])
    return float(np.sqrt(np.mean(np.sum((estimate - truth) ** 2, axis=1))))


def _increase(reference, degraded):
    if reference <= 0.0:
        return 0.0 if degraded == reference else float("inf")
    return float(100.0 * (degraded - reference) / reference)
