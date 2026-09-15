"""Finite-horizon observability audit for the single-satellite sensor modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np

from orbital_core.constants import R_EARTH
from orbital_core.filters import LocalDynamicsEKF
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


_MODALITY_COMBINATIONS = (
    ("opt",), ("ir",), ("rad",),
    ("opt", "ir"), ("opt", "rad"), ("ir", "rad"),
    ("opt", "ir", "rad"),
)
_BASE_COVARIANCE = {
    "opt": np.diag([2e-4, 2e-4]) ** 2,
    "ir": np.diag(np.deg2rad([0.05, 0.05])) ** 2,
    "rad": np.diag([30.0, 0.05]) ** 2,
}
_RANK_RELATIVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ObservabilityRecord:
    infrared_noise_scale: float
    modalities: str
    measurement_dimension_per_epoch: int
    instantaneous_rank_min: int
    instantaneous_rank_max: int
    horizon_rank: int
    horizon_smallest_singular_value: float
    horizon_largest_singular_value: float
    horizon_condition_number: float
    horizon_log_pseudo_determinant: float


@dataclass(frozen=True)
class SingleModalityObservabilityReport:
    duration_seconds: float
    dt_seconds: float
    observer_raan_deg: float
    state_scaling: tuple[float, ...]
    dynamics_linearization: str
    rank_relative_tolerance: float
    optical_infrared_row_space_overlap_mean: float
    optical_infrared_row_space_overlap_minimum: float
    records: tuple[ObservabilityRecord, ...]


def run_single_modality_observability_audit(
    *, duration=120.0, dt=2.0, observer_raan_deg=15.5,
    infrared_noise_scales=(0.25, 0.5, 1.0, 2.0),
    state_scaling=(1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0),
):
    timestamps = np.arange(0.0, duration + 0.5 * dt, dt)
    if timestamps.size < 2:
        raise ValueError("duration and dt must define at least two epochs.")
    scenario = _build_scenario(timestamps, observer_raan_deg)
    observer = scenario.observer_trajectories["sat_01"]
    relative = scenario.relative_state_eci_by_node["sat_01"]
    scale = np.diag(np.asarray(state_scaling, dtype=float).reshape(6))

    jacobians = _measurement_jacobians(
        relative, observer.q_eci2pri_history
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
    overlaps = np.asarray([
        _row_space_overlap(jacobians["opt"][index], jacobians["ir"][index])
        for index in range(timestamps.size)
    ])

    records = []
    for ir_scale in map(float, infrared_noise_scales):
        if ir_scale <= 0.0:
            raise ValueError("infrared_noise_scales must be positive.")
        covariance = dict(_BASE_COVARIANCE)
        covariance["ir"] = _BASE_COVARIANCE["ir"] * ir_scale**2
        for modalities in _MODALITY_COMBINATIONS:
            instantaneous = [
                np.vstack([
                    _whiten(jacobians[name][index], covariance[name]) @ scale
                    for name in modalities
                ])
                for index in range(timestamps.size)
            ]
            ranks = [_numerical_rank(matrix) for matrix in instantaneous]
            horizon = _finite_horizon_matrix(
                modalities, jacobians, covariance, transitions, scale
            )
            singular = np.linalg.svd(horizon, compute_uv=False)
            rank = _rank_from_singular_values(singular)
            retained = singular[:rank]
            records.append(ObservabilityRecord(
                infrared_noise_scale=ir_scale,
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
                horizon_largest_singular_value=(
                    float(retained[0]) if rank else 0.0
                ),
                horizon_condition_number=(
                    float(retained[0] / retained[-1]) if rank else float("inf")
                ),
                horizon_log_pseudo_determinant=(
                    float(2.0 * np.sum(np.log(retained))) if rank else float("-inf")
                ),
            ))
    return SingleModalityObservabilityReport(
        duration_seconds=float(duration), dt_seconds=float(dt),
        observer_raan_deg=float(observer_raan_deg),
        state_scaling=tuple(map(float, state_scaling)),
        dynamics_linearization="legacy_fixed_step_rk4_relative",
        rank_relative_tolerance=_RANK_RELATIVE_TOLERANCE,
        optical_infrared_row_space_overlap_mean=float(np.mean(overlaps)),
        optical_infrared_row_space_overlap_minimum=float(np.min(overlaps)),
        records=tuple(records),
    )


def save_single_modality_observability_audit(report, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path, records_path = output / "summary.json", output / "records.csv"
    summary_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with records_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(ObservabilityRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in report.records)
    return summary_path, records_path


def _build_scenario(timestamps, observer_raan_deg):
    target = keplerian_to_eci(
        R_EARTH + 700e3, 0.001, np.deg2rad(55.0),
        np.deg2rad(15.0), 0.0, np.deg2rad(8.0),
    )
    observer = keplerian_to_eci(
        R_EARTH + 702e3, 0.0012, np.deg2rad(54.5),
        np.deg2rad(observer_raan_deg), 0.0, np.deg2rad(7.0),
    )
    return generate_cooperative_scenario(
        timestamps=timestamps, target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci={"sat_01": observer},
    )


def _measurement_jacobians(relative_states, quaternions):
    filters = {
        name: LocalDynamicsEKF(
            process_noise=np.zeros((6, 6)),
            measurement_covariance=_BASE_COVARIANCE[name], mode_name=name,
        )
        for name in ("opt", "ir", "rad")
    }
    return {
        name: tuple(
            sensor.measurement_jacobian(state, quaternions[index])
            for index, state in enumerate(relative_states)
        )
        for name, sensor in filters.items()
    }


def _finite_horizon_matrix(modalities, jacobians, covariance, transitions, scale):
    rows = []
    transition_from_start = np.eye(6)
    for index in range(len(jacobians[modalities[0]])):
        rows.extend(
            _whiten(jacobians[name][index], covariance[name])
            @ transition_from_start @ scale
            for name in modalities
        )
        if index < len(transitions):
            transition_from_start = transitions[index] @ transition_from_start
    return np.vstack(rows)


def _whiten(jacobian, covariance):
    return np.linalg.solve(np.linalg.cholesky(covariance), jacobian)


def _row_space_overlap(left, right):
    left_basis = np.linalg.svd(left, full_matrices=False)[2]
    right_basis = np.linalg.svd(right, full_matrices=False)[2]
    cosines = np.linalg.svd(left_basis @ right_basis.T, compute_uv=False)
    return float(np.mean(cosines))


def _numerical_rank(matrix):
    return _rank_from_singular_values(np.linalg.svd(matrix, compute_uv=False))


def _rank_from_singular_values(values):
    if values.size == 0 or values[0] == 0.0:
        return 0
    return int(np.count_nonzero(
        values > values[0] * _RANK_RELATIVE_TOLERANCE
    ))
