"""Unified EKF update from simultaneous radar and line-of-sight data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from tracking.association.data_contracts import UnlabeledIODObservation
from tracking.initial_orbit.data_contracts import IODObservation
from tracking.multi_observer_bearing_filter import _tangent_basis


@dataclass(frozen=True)
class UnifiedMultimodalUpdate:
    timestamp: float
    state_eci: np.ndarray
    covariance_eci: np.ndarray
    observer_ids: tuple[str, ...]
    measurement_ids: tuple[str, ...]
    modality_counts: dict[str, int]
    observation_count: int
    measurement_dimension: int
    nis: float


def unified_multi_observer_multimodal_update(
    prior,
    observations,
    *,
    timestamp: float,
    process_noise_acceleration: float = 1e-4,
) -> UnifiedMultimodalUpdate:
    """Propagate one target prior and jointly update it with RADAR and LOS.

    Each physical observation enters exactly once. RADAR contributes range and
    range rate, while optical/infrared LOS contributes two tangent-plane
    residuals. All Jacobians are evaluated at the same predicted state.
    """

    epoch = float(timestamp)
    values = tuple(observations)
    if not values:
        raise ValueError("At least one multimodal observation is required.")
    if any(
        not isinstance(item, (UnlabeledIODObservation, IODObservation))
        for item in values
    ):
        raise TypeError("Unified update requires IOD observations.")
    if any(item.modality not in {"RADAR", "LOS"} for item in values):
        raise ValueError("Unified update accepts RADAR and LOS only.")
    if any(
        not item.valid_flag or not np.isclose(item.timestamp, epoch)
        for item in values
    ):
        raise ValueError("Every observation must be valid and epoch-aligned.")

    state, covariance = _predict_prior(
        prior,
        epoch,
        process_noise_acceleration=process_noise_acceleration,
    )
    jacobians = []
    innovations = []
    covariances = []
    modality_counts: dict[str, int] = {}
    measurement_ids = []
    for item in values:
        jacobian, innovation, measurement_covariance = _measurement_block(
            item, state,
        )
        jacobians.append(jacobian)
        innovations.append(innovation)
        covariances.append(measurement_covariance)
        source_modality = str(
            item.metadata.get("sourceModality", item.modality)
        ).upper()
        modality_counts[source_modality] = (
            modality_counts.get(source_modality, 0) + 1
        )
        identifier = item.metadata.get("sourceMessageId")
        if identifier is not None:
            measurement_ids.append(str(identifier))

    measurement_matrix = np.vstack(jacobians)
    innovation = np.concatenate(innovations)
    measurement_covariance = _block_diagonal(covariances)
    innovation_covariance = (
        measurement_matrix @ covariance @ measurement_matrix.T
        + measurement_covariance
    )
    innovation_covariance = _symmetrize(innovation_covariance)
    inverse_innovation = np.linalg.pinv(innovation_covariance)
    gain = covariance @ measurement_matrix.T @ inverse_innovation
    updated_state = state + gain @ innovation
    identity = np.eye(6)
    residual_matrix = identity - gain @ measurement_matrix
    updated_covariance = _symmetrize(
        residual_matrix @ covariance @ residual_matrix.T
        + gain @ measurement_covariance @ gain.T
    )
    return UnifiedMultimodalUpdate(
        timestamp=epoch,
        state_eci=updated_state,
        covariance_eci=updated_covariance,
        observer_ids=tuple(sorted({item.observer_id for item in values})),
        measurement_ids=tuple(dict.fromkeys(measurement_ids)),
        modality_counts=dict(sorted(modality_counts.items())),
        observation_count=len(values),
        measurement_dimension=int(innovation.size),
        nis=float(innovation @ inverse_innovation @ innovation),
    )


def _measurement_block(item, state):
    relative_position = state[:3] - item.observer_state_eci[:3]
    relative_velocity = state[3:] - item.observer_state_eci[3:]
    distance = float(np.linalg.norm(relative_position))
    if distance <= 0.0:
        raise ValueError("Observer and target positions must differ.")
    line = relative_position / distance
    if item.modality == "RADAR":
        range_rate = float(line @ relative_velocity)
        predicted = np.array([distance, range_rate], dtype=float)
        jacobian = np.zeros((2, 6), dtype=float)
        jacobian[0, :3] = line
        jacobian[1, :3] = (
            relative_velocity - range_rate * line
        ) / distance
        jacobian[1, 3:] = line
        return (
            jacobian,
            np.asarray(item.measurement, dtype=float) - predicted,
            _regularize_covariance(item.covariance),
        )

    tangent = np.vstack(_tangent_basis(line))
    measured_line = np.asarray(item.measurement, dtype=float).reshape(3)
    jacobian = np.zeros((2, 6), dtype=float)
    jacobian[:, :3] = (
        tangent @ (np.eye(3) - np.outer(line, line)) / distance
    )
    line_covariance = item.metadata.get("lineOfSightCovarianceEci")
    tangent_covariance = (
        np.asarray(item.covariance, dtype=float)
        if line_covariance is None
        else tangent
        @ np.asarray(line_covariance, dtype=float).reshape(3, 3)
        @ tangent.T
    )
    return (
        jacobian,
        tangent @ (measured_line - line),
        _regularize_covariance(tangent_covariance),
    )


def _predict_prior(prior, timestamp, *, process_noise_acceleration):
    dt = float(timestamp - prior.timestamp)
    if dt <= 0.0:
        raise ValueError("Unified update timestamp must follow the prior timestamp.")
    initial_state = np.asarray(prior.state_eci, dtype=float).reshape(6)
    initial_covariance = np.asarray(prior.covariance_eci, dtype=float).reshape(6, 6)
    propagate = lambda value: rk4_step_absolute(value, dt)
    transition = numerical_jacobian_discrete(propagate, initial_state)
    state = propagate(initial_state)
    covariance = (
        transition @ initial_covariance @ transition.T
        + make_process_noise(dt, process_noise_acceleration)
    )
    return state, _symmetrize(covariance)


def _block_diagonal(matrices):
    sizes = [matrix.shape[0] for matrix in matrices]
    result = np.zeros((sum(sizes), sum(sizes)), dtype=float)
    offset = 0
    for matrix, size in zip(matrices, sizes, strict=True):
        result[offset:offset + size, offset:offset + size] = matrix
        offset += size
    return result


def _regularize_covariance(covariance):
    covariance = _symmetrize(np.asarray(covariance, dtype=float).reshape(2, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = np.finfo(float).eps * scale
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _symmetrize(matrix):
    value = np.asarray(matrix, dtype=float)
    return 0.5 * (value + value.T)
