"""Iterated inverse-depth update for simultaneous multi-observer LOS data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tracking.multi_observer_bearing_filter import (
    MultiObserverBearingUpdate,
    _predict_prior,
    _regularize_covariance,
    _tangent_basis,
)


@dataclass(frozen=True)
class _InverseDepthChart:
    observer_position_eci: np.ndarray
    reference_line_eci: np.ndarray
    first_tangent_eci: np.ndarray
    second_tangent_eci: np.ndarray
    reference_distance_m: float

    @classmethod
    def around(cls, state_eci, observer_position_eci):
        observer = np.asarray(observer_position_eci, dtype=float).reshape(3)
        relative = np.asarray(state_eci, dtype=float).reshape(6)[:3] - observer
        distance = float(np.linalg.norm(relative))
        if distance <= 0.0:
            raise ValueError("Observer and target positions must differ.")
        line = relative / distance
        first, second = _tangent_basis(line)
        return cls(observer, line, first, second, distance)

    def from_cartesian(self, state_eci):
        state = np.asarray(state_eci, dtype=float).reshape(6)
        relative = state[:3] - self.observer_position_eci
        distance = float(np.linalg.norm(relative))
        if distance <= 0.0:
            raise ValueError("Inverse depth requires positive target range.")
        line = relative / distance
        denominator = float(line @ self.reference_line_eci)
        if denominator <= 1e-8:
            raise ValueError("Target left the local inverse-depth chart.")
        return np.concatenate((
            np.array([
                float(line @ self.first_tangent_eci) / denominator,
                float(line @ self.second_tangent_eci) / denominator,
                self.reference_distance_m / distance,
            ]),
            state[3:],
        ))

    def to_cartesian(self, inverse_depth_state):
        value = np.asarray(inverse_depth_state, dtype=float).reshape(6)
        scaled_inverse_range = float(value[2])
        if (
            scaled_inverse_range <= 0.0
            or not np.isfinite(scaled_inverse_range)
        ):
            raise ValueError("Inverse range must remain positive and finite.")
        direction = (
            self.reference_line_eci
            + value[0] * self.first_tangent_eci
            + value[1] * self.second_tangent_eci
        )
        direction /= np.linalg.norm(direction)
        return np.concatenate((
            self.observer_position_eci
            + direction * self.reference_distance_m / scaled_inverse_range,
            value[3:],
        ))


def inverse_depth_multi_observer_bearing_update(
    prior,
    observations,
    *,
    timestamp: float,
    process_noise_acceleration: float = 1e-4,
    maximum_iterations: int = 5,
) -> MultiObserverBearingUpdate:
    """Update a Cartesian orbit through a local inverse-depth LOS chart.

    The chart improves conditioning of the weak range direction without adding
    a pseudo-range measurement. Every physical LOS still enters exactly once.
    Multiple observer locations and orbital propagation supply the only range
    information.
    """

    epoch = float(timestamp)
    values = tuple(observations)
    if not values:
        raise ValueError("At least one LOS observation is required.")
    if any(item.modality != "LOS" for item in values):
        raise ValueError("Inverse-depth update accepts LOS only.")
    if any(
        not item.valid_flag or not np.isclose(item.timestamp, epoch)
        for item in values
    ):
        raise ValueError("Every LOS observation must be valid and epoch-aligned.")
    observer_ids = tuple(sorted({item.observer_id for item in values}))
    if len(observer_ids) < 2:
        raise ValueError("At least two distinct observers are required.")
    if int(maximum_iterations) < 1:
        raise ValueError("maximum_iterations must be positive.")

    predicted_state, predicted_covariance = _predict_prior(
        prior,
        epoch,
        process_noise_acceleration=process_noise_acceleration,
    )
    reference = min(values, key=lambda item: str(item.observer_id))
    chart = _InverseDepthChart.around(
        predicted_state, reference.observer_state_eci[:3],
    )
    inverse_prior = chart.from_cartesian(predicted_state)
    cartesian_to_inverse = _numerical_jacobian(
        chart.from_cartesian, predicted_state,
    )
    inverse_covariance = _regularize_matrix(
        cartesian_to_inverse
        @ predicted_covariance
        @ cartesian_to_inverse.T
    )
    prior_root = _covariance_root(inverse_covariance)
    inverse_prior_root = np.linalg.solve(prior_root, np.eye(6))

    estimate = inverse_prior.copy()
    for _iteration in range(int(maximum_iterations)):
        residuals = [inverse_prior_root @ (inverse_prior - estimate)]
        jacobians = [inverse_prior_root]
        state = chart.to_cartesian(estimate)
        state_jacobian = _numerical_jacobian(chart.to_cartesian, estimate)
        for item in values:
            measurement_jacobian, innovation, covariance = _los_block(item, state)
            root = _covariance_root(covariance)
            inverse_root = np.linalg.solve(root, np.eye(2))
            residuals.append(inverse_root @ innovation)
            jacobians.append(
                inverse_root @ measurement_jacobian @ state_jacobian
            )
        stacked_jacobian = np.vstack(jacobians)
        correction = np.linalg.lstsq(
            stacked_jacobian, np.concatenate(residuals), rcond=None,
        )[0]
        candidate = estimate + correction
        if candidate[2] <= 0.0:
            candidate[2] = 0.5 * estimate[2]
        estimate = candidate
        if np.linalg.norm(correction) <= 1e-9 * max(
            np.linalg.norm(estimate), 1.0,
        ):
            break

    updated_state = chart.to_cartesian(estimate)
    final_jacobian = _information_jacobian(
        estimate, inverse_prior_root, chart, values,
    )
    inverse_posterior_covariance = np.linalg.pinv(
        final_jacobian.T @ final_jacobian
    )
    inverse_to_cartesian = _numerical_jacobian(chart.to_cartesian, estimate)
    updated_covariance = _regularize_matrix(
        inverse_to_cartesian
        @ inverse_posterior_covariance
        @ inverse_to_cartesian.T
    )

    measurement_ids = []
    modality_counts: dict[str, int] = {}
    final_innovations = []
    for item in values:
        identifier = item.metadata.get("sourceMessageId")
        if identifier is not None:
            measurement_ids.append(str(identifier))
        modality = str(item.metadata.get("sourceModality", "LOS")).upper()
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
        _, innovation, covariance = _los_block(item, updated_state)
        final_innovations.append(
            np.linalg.solve(_covariance_root(covariance), innovation)
        )
    whitened_innovation = np.concatenate(final_innovations)
    return MultiObserverBearingUpdate(
        timestamp=epoch,
        state_eci=updated_state,
        covariance_eci=updated_covariance,
        observer_ids=observer_ids,
        measurement_ids=tuple(dict.fromkeys(measurement_ids)),
        modality_counts=dict(sorted(modality_counts.items())),
        measurement_count=len(values),
        nis=float(whitened_innovation @ whitened_innovation),
    )


def _information_jacobian(estimate, inverse_prior_root, chart, observations):
    state = chart.to_cartesian(estimate)
    state_jacobian = _numerical_jacobian(chart.to_cartesian, estimate)
    rows = [inverse_prior_root]
    for item in observations:
        measurement_jacobian, _innovation, covariance = _los_block(item, state)
        inverse_root = np.linalg.solve(
            _covariance_root(covariance), np.eye(2),
        )
        rows.append(inverse_root @ measurement_jacobian @ state_jacobian)
    return np.vstack(rows)


def _los_block(item, state):
    relative = np.asarray(state[:3]) - item.observer_state_eci[:3]
    distance = float(np.linalg.norm(relative))
    if distance <= 0.0:
        raise ValueError("Observer and target positions must differ.")
    predicted_line = relative / distance
    tangent = np.vstack(_tangent_basis(predicted_line))
    measured_line = np.asarray(item.measurement, dtype=float).reshape(3)
    jacobian = np.zeros((2, 6), dtype=float)
    jacobian[:, :3] = (
        tangent @ (np.eye(3) - np.outer(predicted_line, predicted_line))
        / distance
    )
    line_covariance = item.metadata.get("lineOfSightCovarianceEci")
    covariance = (
        np.asarray(item.covariance, dtype=float)
        if line_covariance is None
        else tangent
        @ np.asarray(line_covariance, dtype=float).reshape(3, 3)
        @ tangent.T
    )
    return (
        jacobian,
        tangent @ (measured_line - predicted_line),
        _regularize_covariance(covariance),
    )


def _numerical_jacobian(function, value):
    point = np.asarray(value, dtype=float)
    baseline = np.asarray(function(point), dtype=float)
    jacobian = np.zeros((baseline.size, point.size), dtype=float)
    for index in range(point.size):
        scale = max(abs(float(point[index])), 1.0)
        step = np.sqrt(np.finfo(float).eps) * scale
        positive = point.copy()
        negative = point.copy()
        positive[index] += step
        negative[index] -= step
        if index == 2 and negative[index] <= 0.0:
            jacobian[:, index] = (
                np.asarray(function(positive)) - baseline
            ) / step
        else:
            jacobian[:, index] = (
                np.asarray(function(positive))
                - np.asarray(function(negative))
            ) / (2.0 * step)
    return jacobian


def _covariance_root(covariance):
    return np.linalg.cholesky(_regularize_matrix(covariance))


def _regularize_matrix(matrix):
    value = np.asarray(matrix, dtype=float)
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = np.finfo(float).eps * scale
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T
