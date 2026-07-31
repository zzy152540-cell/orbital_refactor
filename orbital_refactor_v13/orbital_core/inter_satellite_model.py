from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.attitude import (
    quat_multiply_wxyz,
    quat_normalize_wxyz,
    small_angle_quaternion_wxyz,
)
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
    wrap_angle,
)

Array = np.ndarray

_ALIASES = {
    "RANGE": "RANGE",
    "INTER_SATELLITE_RANGE": "RANGE",
    "RANGE_RATE": "RANGE_RATE",
    "RANGERATE": "RANGE_RATE",
    "INTER_SATELLITE_RANGE_RATE": "RANGE_RATE",
    "AZ_EL": "AZ_EL",
    "AZEL": "AZ_EL",
    "ANGLE": "AZ_EL",
    "INTER_SATELLITE_AZ_EL": "AZ_EL",
}


@dataclass(frozen=True)
class RelativeMeasurementModel:
    """Unified directed model for inter-satellite relative measurements."""

    modality: str
    frame: str = "ECI"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "modality", normalize_inter_satellite_modality(self.modality)
        )
        object.__setattr__(self, "frame", str(self.frame).upper())

    def predict(
        self,
        observer_state: Array,
        target_state: Array,
        *,
        quaternion_i2b_wxyz: Array | None = None,
    ) -> Array:
        return predict_inter_satellite_measurement(
            observer_state,
            target_state,
            modality=self.modality,
            frame=self.frame,
            quaternion_i2b_wxyz=quaternion_i2b_wxyz,
        )

    def jacobians(
        self,
        observer_state: Array,
        target_state: Array,
        *,
        quaternion_i2b_wxyz: Array | None = None,
        eps: float = 1e-6,
    ) -> tuple[Array, Array]:
        return inter_satellite_jacobians(
            observer_state,
            target_state,
            modality=self.modality,
            frame=self.frame,
            quaternion_i2b_wxyz=quaternion_i2b_wxyz,
            eps=eps,
        )

    def residual(self, measurement: Array, prediction: Array) -> Array:
        difference = (
            np.asarray(measurement, dtype=float).reshape(-1)
            - np.asarray(prediction, dtype=float).reshape(-1)
        )
        return wrap_angle(difference) if self.modality == "AZ_EL" else difference


def normalize_inter_satellite_modality(modality: str) -> str:
    try:
        return _ALIASES[str(modality).upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported inter-satellite modality: {modality}") from exc


def predict_inter_satellite_measurement(
    state_i: Array,
    state_j: Array,
    *,
    modality: str,
    frame: str = "ECI",
    quaternion_i2b_wxyz: Array | None = None,
) -> Array:
    normalized = normalize_inter_satellite_modality(modality)
    if normalized == "RANGE":
        return np.array([measure_relative_range(state_i, state_j)], dtype=float)
    if normalized == "RANGE_RATE":
        return np.array([measure_relative_range_rate(state_i, state_j)], dtype=float)
    return measure_relative_az_el(
        state_i,
        state_j,
        frame=frame,
        quaternion_i2b_wxyz=quaternion_i2b_wxyz,
    )


def inter_satellite_jacobians(
    state_i: Array,
    state_j: Array,
    *,
    modality: str,
    frame: str = "ECI",
    quaternion_i2b_wxyz: Array | None = None,
    eps: float = 1e-6,
) -> tuple[Array, Array]:
    """Return measurement Jacobians with respect to both absolute states."""

    state_i = np.asarray(state_i, dtype=float).reshape(6)
    state_j = np.asarray(state_j, dtype=float).reshape(6)
    normalized = normalize_inter_satellite_modality(modality)

    if normalized == "RANGE":
        relative_position = state_j[:3] - state_i[:3]
        rho = np.linalg.norm(relative_position)
        if rho <= 0.0:
            raise ValueError("Inter-satellite range must be positive.")
        line_of_sight = relative_position / rho
        h_i = np.zeros((1, 6), dtype=float)
        h_i[0, :3] = -line_of_sight
        return h_i, -h_i

    if normalized == "RANGE_RATE":
        relative_position = state_j[:3] - state_i[:3]
        relative_velocity = state_j[3:] - state_i[3:]
        rho = np.linalg.norm(relative_position)
        if rho <= 0.0:
            raise ValueError("Inter-satellite range must be positive.")
        line_of_sight = relative_position / rho
        rho_dot = float(relative_position @ relative_velocity / rho)
        position_gradient = (relative_velocity - line_of_sight * rho_dot) / rho
        h_i = np.zeros((1, 6), dtype=float)
        h_i[0, :3] = -position_gradient
        h_i[0, 3:] = -line_of_sight
        return h_i, -h_i

    function = lambda left, right: predict_inter_satellite_measurement(
        left,
        right,
        modality=normalized,
        frame=frame,
        quaternion_i2b_wxyz=quaternion_i2b_wxyz,
    )
    h_i = _numerical_state_jacobian(
        lambda value: function(value, state_j), state_i, angular=True, eps=eps
    )
    h_j = _numerical_state_jacobian(
        lambda value: function(state_i, value), state_j, angular=True, eps=eps
    )
    return h_i, h_j


def body_angle_attitude_jacobian(
    state_i: Array,
    state_j: Array,
    *,
    quaternion_i2b_wxyz: Array,
    eps: float = 1e-6,
) -> Array:
    """Return the BODY az/el Jacobian with respect to left attitude error.

    The three perturbation components use the same small-angle, left-error
    convention as :class:`AttitudeGyroBiasMEKF`.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive.")
    state_i = np.asarray(state_i, dtype=float).reshape(6)
    state_j = np.asarray(state_j, dtype=float).reshape(6)
    quaternion = quat_normalize_wxyz(quaternion_i2b_wxyz)
    jacobian = np.zeros((2, 3), dtype=float)
    for index in range(3):
        perturbation = np.zeros(3, dtype=float)
        perturbation[index] = eps
        plus = quat_multiply_wxyz(
            small_angle_quaternion_wxyz(perturbation),
            quaternion,
        )
        minus = quat_multiply_wxyz(
            small_angle_quaternion_wxyz(-perturbation),
            quaternion,
        )
        predicted_plus = predict_inter_satellite_measurement(
            state_i,
            state_j,
            modality="AZ_EL",
            frame="BODY",
            quaternion_i2b_wxyz=plus,
        )
        predicted_minus = predict_inter_satellite_measurement(
            state_i,
            state_j,
            modality="AZ_EL",
            frame="BODY",
            quaternion_i2b_wxyz=minus,
        )
        jacobian[:, index] = (
            wrap_angle(predicted_plus - predicted_minus) / (2.0 * eps)
        )
    return jacobian


def body_angle_effective_covariance(
    state_i: Array,
    state_j: Array,
    *,
    quaternion_i2b_wxyz: Array,
    sensor_covariance: Array,
    attitude_covariance: Array,
    eps: float = 1e-6,
) -> Array:
    """Inflate BODY az/el covariance with MEKF attitude uncertainty."""

    sensor = _validated_covariance(
        sensor_covariance,
        dimension=2,
        name="sensor_covariance",
    )
    attitude = _validated_covariance(
        attitude_covariance,
        dimension=3,
        name="attitude_covariance",
    )
    jacobian = body_angle_attitude_jacobian(
        state_i,
        state_j,
        quaternion_i2b_wxyz=quaternion_i2b_wxyz,
        eps=eps,
    )
    effective = sensor + jacobian @ attitude @ jacobian.T
    return 0.5 * (effective + effective.T)


def _validated_covariance(
    covariance: Array,
    *,
    dimension: int,
    name: str,
) -> Array:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    symmetric = 0.5 * (matrix + matrix.T)
    if np.min(np.linalg.eigvalsh(symmetric)) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite.")
    return symmetric


def _numerical_state_jacobian(
    function,
    state: Array,
    *,
    angular: bool,
    eps: float,
) -> Array:
    state = np.asarray(state, dtype=float).reshape(6)
    output = np.asarray(function(state), dtype=float).reshape(-1)
    jacobian = np.zeros((output.size, 6), dtype=float)
    for index in range(6):
        step = eps * max(1.0, abs(float(state[index])))
        plus = state.copy()
        minus = state.copy()
        plus[index] += step
        minus[index] -= step
        difference = np.asarray(function(plus)) - np.asarray(function(minus))
        if angular:
            difference = wrap_angle(difference)
        jacobian[:, index] = difference / (2.0 * step)
    return jacobian
