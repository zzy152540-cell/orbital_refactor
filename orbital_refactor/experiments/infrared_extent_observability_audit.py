"""Shadow observability audit for an infrared angular-extent measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from experiments.single_modality_observability_audit import (
    _BASE_COVARIANCE,
    _build_scenario,
    _finite_horizon_matrix,
    _measurement_jacobians,
    _numerical_rank,
    _rank_from_singular_values,
    _whiten,
)
from orbital_core.filters import LocalDynamicsEKF
from orbital_core.measurements import h_ir_spri


_COMBINATIONS = (
    ("ir",), ("ir_extent",),
    ("ir", "rad"), ("ir_extent", "rad"),
    ("opt", "ir_extent"), ("opt", "ir_extent", "rad"),
)


@dataclass(frozen=True)
class InfraredExtentObservabilityRecord:
    extent_fractional_sigma: float
    modalities: str
    measurement_dimension_per_epoch: int
    instantaneous_rank_min: int
    instantaneous_rank_max: int
    horizon_rank: int
    horizon_smallest_singular_value: float
    horizon_condition_number: float
    horizon_log_pseudo_determinant: float


@dataclass(frozen=True)
class InfraredExtentObservabilityReport:
    duration_seconds: float
    dt_seconds: float
    effective_target_diameter_m: float
    relative_range_minimum_m: float
    relative_range_maximum_m: float
    angular_extent_minimum_microrad: float
    angular_extent_maximum_microrad: float
    angular_extent_at_300px_focal_minimum_pixels: float
    angular_extent_at_300px_focal_maximum_pixels: float
    extent_measurement: str
    extent_uncertainty_interpretation: str
    baseline_unchanged: bool
    records: tuple[InfraredExtentObservabilityRecord, ...]


def run_infrared_extent_observability_audit(
    *, duration=120.0, dt=2.0, observer_raan_deg=15.5,
    effective_target_diameter_m=10.0,
    extent_fractional_sigmas=(0.01, 0.05, 0.1, 0.2, 0.3),
    state_scaling=(1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0),
):
    diameter = float(effective_target_diameter_m)
    if diameter <= 0.0:
        raise ValueError("effective_target_diameter_m must be positive.")
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    if timestamps.size < 2:
        raise ValueError("duration and dt must define at least two epochs.")
    scenario = _build_scenario(timestamps, observer_raan_deg)
    observer = scenario.observer_trajectories["sat_01"]
    relative = scenario.relative_state_eci_by_node["sat_01"]
    ranges = np.linalg.norm(relative[:, :3], axis=1)
    angular_extents = 2.0 * np.arctan(diameter / (2.0 * ranges))
    scale = np.diag(np.asarray(state_scaling, dtype=float).reshape(6))
    jacobians = _measurement_jacobians(
        relative, observer.q_eci2pri_history
    )
    jacobians["ir_extent"] = tuple(
        _fixed_step_jacobian(
            lambda value, quaternion=observer.q_eci2pri_history[index]:
                h_ir_with_log_extent(value, quaternion, diameter),
            state,
        )
        for index, state in enumerate(relative)
    )
    transition_filter = LocalDynamicsEKF(
        process_noise=np.zeros((6, 6)),
        measurement_covariance=_BASE_COVARIANCE["opt"], mode_name="opt",
    )
    transitions = tuple(
        transition_filter.discrete_transition_jacobian(
            relative[index], observer.state_history_eci[index], dt
        )
        for index in range(timestamps.size - 1)
    )

    records = []
    for sigma in map(float, extent_fractional_sigmas):
        if sigma <= 0.0:
            raise ValueError("extent_fractional_sigmas must be positive.")
        covariance = dict(_BASE_COVARIANCE)
        covariance["ir_extent"] = np.diag([
            np.deg2rad(0.05), np.deg2rad(0.05), sigma,
        ]) ** 2
        for modalities in _COMBINATIONS:
            instantaneous = [
                np.vstack([
                    _whiten(jacobians[name][index], covariance[name]) @ scale
                    for name in modalities
                ])
                for index in range(timestamps.size)
            ]
            ranks = tuple(_numerical_rank(value) for value in instantaneous)
            horizon = _finite_horizon_matrix(
                modalities, jacobians, covariance, transitions, scale
            )
            singular = np.linalg.svd(horizon, compute_uv=False)
            rank = _rank_from_singular_values(singular)
            retained = singular[:rank]
            records.append(InfraredExtentObservabilityRecord(
                extent_fractional_sigma=sigma,
                modalities="+".join(modalities),
                measurement_dimension_per_epoch=sum(
                    jacobians[name][0].shape[0] for name in modalities
                ),
                instantaneous_rank_min=min(ranks),
                instantaneous_rank_max=max(ranks),
                horizon_rank=rank,
                horizon_smallest_singular_value=(
                    float(retained[-1]) if rank else 0.0
                ),
                horizon_condition_number=(
                    float(retained[0] / retained[-1]) if rank else float("inf")
                ),
                horizon_log_pseudo_determinant=(
                    float(2.0 * np.sum(np.log(retained))) if rank else float("-inf")
                ),
            ))
    return InfraredExtentObservabilityReport(
        duration_seconds=float(duration), dt_seconds=float(dt),
        effective_target_diameter_m=diameter,
        relative_range_minimum_m=float(np.min(ranges)),
        relative_range_maximum_m=float(np.max(ranges)),
        angular_extent_minimum_microrad=float(np.min(angular_extents) * 1e6),
        angular_extent_maximum_microrad=float(np.max(angular_extents) * 1e6),
        angular_extent_at_300px_focal_minimum_pixels=float(
            np.min(angular_extents) * 300.0
        ),
        angular_extent_at_300px_focal_maximum_pixels=float(
            np.max(angular_extents) * 300.0
        ),
        extent_measurement="log(2*atan(D_eff/(2*rho)))",
        extent_uncertainty_interpretation=(
            "One-sigma fractional uncertainty combining image-width extraction "
            "and effective projected-size modelling."
        ),
        baseline_unchanged=True, records=tuple(records),
    )


def h_ir_with_log_extent(relative_state_eci, q_eci2pri, diameter):
    state = np.asarray(relative_state_eci, dtype=float).reshape(6)
    rho = float(np.linalg.norm(state[:3]))
    if rho <= 0.0:
        raise ValueError("Infrared extent requires positive target range.")
    angular_extent = 2.0 * np.arctan(float(diameter) / (2.0 * rho))
    return np.concatenate((
        h_ir_spri(state, q_eci2pri), [np.log(angular_extent)],
    ))


def save_infrared_extent_observability_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary, records = output / "summary.json", output / "records.csv"
    summary.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with records.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(InfraredExtentObservabilityRecord.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
    return summary, records


def _fixed_step_jacobian(function, state, epsilon=1e-6):
    state = np.asarray(state, dtype=float).reshape(6)
    output = np.asarray(function(state), dtype=float)
    jacobian = np.empty((output.size, state.size), dtype=float)
    for index in range(state.size):
        offset = np.zeros(6)
        offset[index] = epsilon
        jacobian[:, index] = (
            np.asarray(function(state + offset))
            - np.asarray(function(state - offset))
        ) / (2.0 * epsilon)
    return jacobian
