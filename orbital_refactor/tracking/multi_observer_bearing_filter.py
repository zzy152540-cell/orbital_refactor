"""Batch EKF update from simultaneous multi-observer line-of-sight data."""

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


@dataclass(frozen=True)
class MultiObserverBearingUpdate:
    timestamp: float
    state_eci: np.ndarray
    covariance_eci: np.ndarray
    observer_ids: tuple[str, ...]
    measurement_ids: tuple[str, ...]
    modality_counts: dict[str, int]
    measurement_count: int
    nis: float


def batch_multi_observer_bearing_update(
    prior,
    observations,
    *,
    timestamp: float,
    process_noise_acceleration: float = 1e-4,
) -> MultiObserverBearingUpdate:
    """Propagate one target prior and update it with all aligned LOS data.

    The update stacks two tangent-plane residual components per observation.
    Multiple observer locations make the combined geometry sensitive to all
    three target-position components even though each individual LOS retains
    its radial nullspace.
    """

    epoch = float(timestamp)
    values = tuple(observations)
    if not values:
        raise ValueError("At least one LOS observation is required.")
    if any(
        not isinstance(item, (UnlabeledIODObservation, IODObservation))
        for item in values
    ):
        raise TypeError("Multi-observer bearing update requires IOD observations.")
    if any(item.modality != "LOS" for item in values):
        raise ValueError("Multi-observer bearing update accepts LOS only.")
    if any(not item.valid_flag or not np.isclose(item.timestamp, epoch) for item in values):
        raise ValueError("Every LOS observation must be valid and epoch-aligned.")
    observer_ids = tuple(sorted({item.observer_id for item in values}))
    if len(observer_ids) < 2:
        raise ValueError("At least two distinct observers are required.")

    state, covariance = _predict_prior(
        prior,
        epoch,
        process_noise_acceleration=process_noise_acceleration,
    )
    measurement_ids = []
    modality_counts: dict[str, int] = {}
    for item in values:
        identifier = item.metadata.get("sourceMessageId")
        if identifier is not None:
            measurement_ids.append(str(identifier))
        modality = str(item.metadata.get("sourceModality", "LOS")).upper()
        modality_counts[modality] = modality_counts.get(modality, 0) + 1

    measurement_matrix, innovation, measurement_covariance = (
        _linearized_measurement(values, state)
    )
    innovation_covariance = (
        measurement_matrix @ covariance @ measurement_matrix.T
        + measurement_covariance
    )
    innovation_covariance = 0.5 * (
        innovation_covariance + innovation_covariance.T
    )
    inverse_innovation = np.linalg.pinv(innovation_covariance)
    gain = covariance @ measurement_matrix.T @ inverse_innovation
    updated_state = state + gain @ innovation
    identity = np.eye(6)
    residual_matrix = identity - gain @ measurement_matrix
    updated_covariance = (
        residual_matrix @ covariance @ residual_matrix.T
        + gain @ measurement_covariance @ gain.T
    )
    updated_covariance = 0.5 * (
        updated_covariance + updated_covariance.T
    )
    return MultiObserverBearingUpdate(
        timestamp=epoch,
        state_eci=updated_state,
        covariance_eci=updated_covariance,
        observer_ids=observer_ids,
        measurement_ids=tuple(dict.fromkeys(measurement_ids)),
        modality_counts=dict(sorted(modality_counts.items())),
        measurement_count=len(values),
        nis=float(innovation @ inverse_innovation @ innovation),
    )


def _linearized_measurement(observations, state):
    jacobians = []
    residuals = []
    covariances = []
    for item in observations:
        relative = state[:3] - item.observer_state_eci[:3]
        distance = float(np.linalg.norm(relative))
        if distance <= 0.0:
            raise ValueError("Observer and target positions must differ.")
        predicted_line = relative / distance
        tangent = np.vstack(_tangent_basis(predicted_line))
        measured_line = np.asarray(item.measurement, dtype=float).reshape(3)
        residuals.append(tangent @ (measured_line - predicted_line))
        jacobian = np.zeros((2, 6), dtype=float)
        jacobian[:, :3] = (
            tangent @ (np.eye(3) - np.outer(predicted_line, predicted_line))
            / distance
        )
        jacobians.append(jacobian)
        line_covariance = item.metadata.get("lineOfSightCovarianceEci")
        if line_covariance is None:
            tangent_covariance = np.asarray(item.covariance, dtype=float)
        else:
            line_covariance = np.asarray(
                line_covariance, dtype=float,
            ).reshape(3, 3)
            tangent_covariance = tangent @ line_covariance @ tangent.T
        covariances.append(_regularize_covariance(tangent_covariance))
    return (
        np.vstack(jacobians),
        np.concatenate(residuals),
        _block_diagonal(covariances),
    )


def _predict_prior(prior, timestamp, *, process_noise_acceleration):
    dt = float(timestamp - prior.timestamp)
    if dt <= 0.0:
        raise ValueError("Bearing update timestamp must follow the prior timestamp.")
    initial_state = np.asarray(prior.state_eci, dtype=float).reshape(6)
    initial_covariance = np.asarray(prior.covariance_eci, dtype=float).reshape(6, 6)
    propagate = lambda state: rk4_step_absolute(state, dt)
    transition = numerical_jacobian_discrete(propagate, initial_state)
    state = propagate(initial_state)
    covariance = (
        transition @ initial_covariance @ transition.T
        + make_process_noise(dt, process_noise_acceleration)
    )
    return state, 0.5 * (covariance + covariance.T)


def _tangent_basis(line):
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(line @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(line, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(line, first)


def _block_diagonal(matrices):
    sizes = [matrix.shape[0] for matrix in matrices]
    result = np.zeros((sum(sizes), sum(sizes)), dtype=float)
    offset = 0
    for matrix, size in zip(matrices, sizes, strict=True):
        result[offset:offset + size, offset:offset + size] = matrix
        offset += size
    return result


def _regularize_covariance(covariance):
    covariance = np.asarray(covariance, dtype=float).reshape(2, 2)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = np.finfo(float).eps * scale
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T
