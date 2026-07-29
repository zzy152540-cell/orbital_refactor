from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.data_objects import NodeReport
from orbital_core.inter_satellite_model import inter_satellite_jacobians
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
    wrap_angle,
)

Array = np.ndarray


@dataclass(frozen=True)
class RangeUpdateResult:
    state: Array
    covariance: Array
    innovation: float
    nis: float
    gated: bool = False
    skipped: bool = False


@dataclass(frozen=True)
class InterSatelliteBlockUpdateResult:
    state: Array
    covariance: Array
    innovation: Array
    nis: float
    modalities: tuple[str, ...]
    gated: bool = False
    skipped: bool = False


InterSatelliteScalarUpdateResult = RangeUpdateResult


def update_with_relative_range(
    *,
    state: Array,
    covariance: Array,
    neighbor_report: NodeReport,
    measured_range: float,
    range_variance: float,
    regularization: float = 1e-9,
) -> RangeUpdateResult:
    """Update one satellite absolute ECI state using an inter-satellite range.

    The neighbor state is treated as the received report's current best estimate.
    This first v13 range update keeps the measurement scalar and updates only
    the local satellite state/covariance.
    """

    local_state = np.asarray(state, dtype=float).reshape(6)
    local_covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
    neighbor_state = np.asarray(neighbor_report.state_estimate, dtype=float).reshape(6)
    neighbor_covariance = np.asarray(
        neighbor_report.covariance, dtype=float
    ).reshape(6, 6)
    if range_variance <= 0.0:
        raise ValueError("range_variance must be positive.")

    predicted_range = measure_relative_range(local_state, neighbor_state)
    if predicted_range <= 0.0:
        raise ValueError("Predicted inter-satellite range must be positive.")
    innovation = float(measured_range - predicted_range)

    los = (neighbor_state[:3] - local_state[:3]) / predicted_range
    measurement_matrix = np.zeros((1, 6), dtype=float)
    measurement_matrix[0, :3] = -los
    neighbor_measurement_matrix = -measurement_matrix
    measurement_covariance = np.array([[float(range_variance)]], dtype=float)
    effective_measurement_covariance = (
        measurement_covariance
        + neighbor_measurement_matrix
        @ neighbor_covariance
        @ neighbor_measurement_matrix.T
    )
    innovation_covariance = (
        measurement_matrix @ local_covariance @ measurement_matrix.T
        + effective_measurement_covariance
        + regularization * np.eye(1)
    )
    gain = (
        local_covariance @ measurement_matrix.T
    ) @ np.linalg.pinv(innovation_covariance)
    updated_state = local_state + (gain[:, 0] * innovation)

    identity = np.eye(6)
    residual_matrix = identity - gain @ measurement_matrix
    updated_covariance = (
        residual_matrix @ local_covariance @ residual_matrix.T
        + gain @ effective_measurement_covariance @ gain.T
    )
    nis = float(innovation**2 / innovation_covariance[0, 0])
    return RangeUpdateResult(
        state=updated_state,
        covariance=0.5 * (updated_covariance + updated_covariance.T),
        innovation=innovation,
        nis=nis,
    )


def update_with_relative_range_rate(
    *,
    state: Array,
    covariance: Array,
    neighbor_report: NodeReport,
    measured_range_rate: float,
    range_rate_variance: float,
    regularization: float = 1e-9,
) -> RangeUpdateResult:
    """Update one satellite absolute ECI state using inter-satellite range-rate."""

    local_state = np.asarray(state, dtype=float).reshape(6)
    local_covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
    neighbor_state = np.asarray(neighbor_report.state_estimate, dtype=float).reshape(6)
    neighbor_covariance = np.asarray(
        neighbor_report.covariance, dtype=float
    ).reshape(6, 6)
    if range_rate_variance <= 0.0:
        raise ValueError("range_rate_variance must be positive.")

    predicted = measure_relative_range_rate(local_state, neighbor_state)
    relative_position = neighbor_state[:3] - local_state[:3]
    relative_velocity = neighbor_state[3:] - local_state[3:]
    range_value = np.linalg.norm(relative_position)
    if range_value <= 0.0:
        raise ValueError("Predicted inter-satellite range must be positive.")
    los = relative_position / range_value
    dh_drelative_position = (relative_velocity - los * predicted) / range_value

    measurement_matrix = np.zeros((1, 6), dtype=float)
    measurement_matrix[0, :3] = -dh_drelative_position
    measurement_matrix[0, 3:] = -los
    neighbor_measurement_matrix = -measurement_matrix
    effective_variance = float(
        range_rate_variance
        + (
            neighbor_measurement_matrix
            @ neighbor_covariance
            @ neighbor_measurement_matrix.T
        )[0, 0]
    )
    return _scalar_joseph_update(
        state=local_state,
        covariance=local_covariance,
        measurement_matrix=measurement_matrix,
        innovation=float(measured_range_rate - predicted),
        measurement_variance=effective_variance,
        regularization=regularization,
    )


def update_with_inter_satellite_observation(
    *,
    state: Array,
    covariance: Array,
    neighbor_report: NodeReport,
    modality: str,
    measurement: float,
    variance: float,
    regularization: float = 1e-9,
) -> RangeUpdateResult:
    normalized = str(modality).upper()
    if normalized in {"RANGE", "INTER_SATELLITE_RANGE"}:
        return update_with_relative_range(
            state=state,
            covariance=covariance,
            neighbor_report=neighbor_report,
            measured_range=measurement,
            range_variance=variance,
            regularization=regularization,
        )
    if normalized in {"RANGE_RATE", "RANGERATE", "INTER_SATELLITE_RANGE_RATE"}:
        return update_with_relative_range_rate(
            state=state,
            covariance=covariance,
            neighbor_report=neighbor_report,
            measured_range_rate=measurement,
            range_rate_variance=variance,
            regularization=regularization,
        )
    raise ValueError(f"Unsupported inter-satellite modality: {modality}")


def update_with_inter_satellite_observation_block(
    *,
    state: Array,
    covariance: Array,
    neighbor_report: NodeReport,
    measurements_by_modality: dict[str, float | Array],
    variance_by_modality: dict[str, float] | None = None,
    covariance_by_modality: dict[str, Array] | None = None,
    frame_by_modality: dict[str, str] | None = None,
    regularization: float = 1e-9,
    gate_enable: bool = False,
    gate_threshold: float = np.inf,
    gate_mode: str = "soft",
    soft_scale: float = 20.0,
) -> InterSatelliteBlockUpdateResult:
    """Joint EKF update for one neighbor's inter-satellite observation block."""

    local_state = np.asarray(state, dtype=float).reshape(6)
    local_covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
    neighbor_state = np.asarray(neighbor_report.state_estimate, dtype=float).reshape(6)
    neighbor_covariance = np.asarray(
        neighbor_report.covariance, dtype=float
    ).reshape(6, 6)
    if not measurements_by_modality:
        raise ValueError("At least one inter-satellite measurement is required.")
    if gate_mode not in {"soft", "hard"}:
        raise ValueError("gate_mode must be 'soft' or 'hard'.")
    if soft_scale < 1.0:
        raise ValueError("soft_scale must be at least 1.0.")

    relative_position = neighbor_state[:3] - local_state[:3]
    relative_velocity = neighbor_state[3:] - local_state[3:]
    range_value = np.linalg.norm(relative_position)
    if range_value <= 0.0:
        raise ValueError("Predicted inter-satellite range must be positive.")
    los = relative_position / range_value
    predicted_range_rate = float(relative_position @ relative_velocity / range_value)
    range_rate_position_gradient = (
        relative_velocity - los * predicted_range_rate
    ) / range_value

    covariance_lookup = _resolve_covariance_lookup(
        variance_by_modality=variance_by_modality,
        covariance_by_modality=covariance_by_modality,
    )
    innovation_blocks: list[Array] = []
    jacobian_rows: list[Array] = []
    neighbor_jacobian_rows: list[Array] = []
    covariance_blocks: list[Array] = []
    modalities: list[str] = []
    for modality in sorted(measurements_by_modality):
        normalized = _normalize_modality(modality)
        measurement_covariance = np.asarray(covariance_lookup[normalized], dtype=float)
        if normalized == "RANGE":
            measurement = float(measurements_by_modality[modality])
            predicted = float(range_value)
            row = np.zeros(6, dtype=float)
            row[:3] = -los
            innovation_block = np.array([measurement - predicted], dtype=float)
            jacobian_block = row.reshape(1, 6)
        elif normalized == "RANGE_RATE":
            measurement = float(measurements_by_modality[modality])
            predicted = predicted_range_rate
            row = np.zeros(6, dtype=float)
            row[:3] = -range_rate_position_gradient
            row[3:] = -los
            innovation_block = np.array([measurement - predicted], dtype=float)
            jacobian_block = row.reshape(1, 6)
        elif normalized == "AZ_EL":
            measurement = np.asarray(measurements_by_modality[modality], dtype=float).reshape(2)
            frame = _frame_for_modality(normalized, frame_by_modality)
            predicted = measure_relative_az_el(local_state, neighbor_state, frame=frame)
            innovation_block = wrap_angle(measurement - predicted)
            jacobian_block = _az_el_jacobian(local_state, neighbor_state, frame=frame)
        else:
            raise ValueError(f"Unsupported inter-satellite modality: {modality}")
        if measurement_covariance.shape != (innovation_block.size, innovation_block.size):
            raise ValueError(f"{normalized} covariance has incompatible shape.")
        if np.min(np.linalg.eigvalsh(measurement_covariance)) <= 0.0:
            raise ValueError(f"{normalized} covariance must be positive definite.")
        innovation_blocks.append(innovation_block)
        jacobian_rows.append(jacobian_block)
        _, neighbor_jacobian = inter_satellite_jacobians(
            local_state,
            neighbor_state,
            modality=normalized,
            frame=_frame_for_modality(normalized, frame_by_modality),
        )
        neighbor_jacobian_rows.append(neighbor_jacobian)
        covariance_blocks.append(measurement_covariance)
        modalities.append(normalized)

    innovation = np.concatenate(innovation_blocks)
    measurement_matrix = np.vstack(jacobian_rows)
    neighbor_measurement_matrix = np.vstack(neighbor_jacobian_rows)
    measurement_covariance = (
        _block_diag(covariance_blocks)
        + neighbor_measurement_matrix
        @ neighbor_covariance
        @ neighbor_measurement_matrix.T
    )
    innovation_covariance = (
        measurement_matrix @ local_covariance @ measurement_matrix.T
        + measurement_covariance
        + regularization * np.eye(measurement_matrix.shape[0])
    )
    nis = float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
    gated = bool(gate_enable and np.isfinite(gate_threshold) and nis > gate_threshold)
    if gated and gate_mode == "hard":
        return InterSatelliteBlockUpdateResult(
            state=local_state.copy(),
            covariance=local_covariance.copy(),
            innovation=innovation,
            nis=nis,
            modalities=tuple(modalities),
            gated=True,
            skipped=True,
        )
    effective_covariance = (
        soft_scale * measurement_covariance
        if gated and gate_mode == "soft"
        else measurement_covariance
    )
    updated_state, updated_covariance, nis = _joseph_update(
        state=local_state,
        covariance=local_covariance,
        measurement_matrix=measurement_matrix,
        innovation=innovation,
        measurement_covariance=effective_covariance,
        regularization=regularization,
        precomputed_nis=nis,
    )
    return InterSatelliteBlockUpdateResult(
        state=updated_state,
        covariance=updated_covariance,
        innovation=innovation,
        nis=nis,
        modalities=tuple(modalities),
        gated=gated,
        skipped=False,
    )


def _scalar_joseph_update(
    *,
    state: Array,
    covariance: Array,
    measurement_matrix: Array,
    innovation: float,
    measurement_variance: float,
    regularization: float,
) -> RangeUpdateResult:
    measurement_covariance = np.array([[measurement_variance]], dtype=float)
    updated_state, updated_covariance, nis = _joseph_update(
        state=state,
        covariance=covariance,
        measurement_matrix=measurement_matrix,
        innovation=np.array([innovation], dtype=float),
        measurement_covariance=measurement_covariance,
        regularization=regularization,
    )
    return RangeUpdateResult(
        state=updated_state,
        covariance=updated_covariance,
        innovation=float(innovation),
        nis=nis,
    )


def _joseph_update(
    *,
    state: Array,
    covariance: Array,
    measurement_matrix: Array,
    innovation: Array,
    measurement_covariance: Array,
    regularization: float,
    precomputed_nis: float | None = None,
) -> tuple[Array, Array, float]:
    innovation_covariance = (
        measurement_matrix @ covariance @ measurement_matrix.T
        + measurement_covariance
        + regularization * np.eye(measurement_matrix.shape[0])
    )
    gain = (covariance @ measurement_matrix.T) @ np.linalg.pinv(innovation_covariance)
    updated_state = state + gain @ innovation
    identity = np.eye(6)
    residual_matrix = identity - gain @ measurement_matrix
    updated_covariance = (
        residual_matrix @ covariance @ residual_matrix.T
        + gain @ measurement_covariance @ gain.T
    )
    nis = (
        float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
        if precomputed_nis is None
        else float(precomputed_nis)
    )
    return updated_state, 0.5 * (updated_covariance + updated_covariance.T), nis


def _normalize_modality(modality: str) -> str:
    normalized = str(modality).upper()
    if normalized in {"RANGE", "INTER_SATELLITE_RANGE"}:
        return "RANGE"
    if normalized in {"RANGE_RATE", "RANGERATE", "INTER_SATELLITE_RANGE_RATE"}:
        return "RANGE_RATE"
    if normalized in {"AZ_EL", "AZEL", "ANGLE", "INTER_SATELLITE_AZ_EL"}:
        return "AZ_EL"
    raise ValueError(f"Unsupported inter-satellite modality: {modality}")


def _resolve_covariance_lookup(
    *,
    variance_by_modality: dict[str, float] | None,
    covariance_by_modality: dict[str, Array] | None,
) -> dict[str, Array]:
    if covariance_by_modality is not None:
        return {
            _normalize_modality(modality): np.asarray(covariance, dtype=float)
            for modality, covariance in covariance_by_modality.items()
        }
    if variance_by_modality is None:
        raise ValueError("Either covariance_by_modality or variance_by_modality is required.")
    return {
        _normalize_modality(modality): np.array([[float(variance)]], dtype=float)
        for modality, variance in variance_by_modality.items()
    }


def _az_el_jacobian(
    local_state: Array,
    neighbor_state: Array,
    frame: str,
    eps: float = 1e-6,
) -> Array:
    jacobian = np.zeros((2, 6), dtype=float)
    for index in range(6):
        step = eps * max(1.0, abs(float(local_state[index])))
        plus = local_state.copy()
        minus = local_state.copy()
        plus[index] += step
        minus[index] -= step
        diff = wrap_angle(
            measure_relative_az_el(plus, neighbor_state, frame=frame)
            - measure_relative_az_el(minus, neighbor_state, frame=frame)
        )
        jacobian[:, index] = diff / (2.0 * step)
    return jacobian


def _frame_for_modality(
    modality: str,
    frame_by_modality: dict[str, str] | None,
) -> str:
    if frame_by_modality is None:
        return "ECI"
    return str(frame_by_modality.get(modality, frame_by_modality.get("AZ_EL", "ECI")))


def _block_diag(matrices: list[Array]) -> Array:
    total = sum(matrix.shape[0] for matrix in matrices)
    result = np.zeros((total, total), dtype=float)
    offset = 0
    for matrix in matrices:
        size = matrix.shape[0]
        result[offset:offset + size, offset:offset + size] = matrix
        offset += size
    return result
