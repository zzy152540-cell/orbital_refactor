from __future__ import annotations

import numpy as np

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
) -> Array:
    normalized = normalize_inter_satellite_modality(modality)
    if normalized == "RANGE":
        return np.array([measure_relative_range(state_i, state_j)], dtype=float)
    if normalized == "RANGE_RATE":
        return np.array([measure_relative_range_rate(state_i, state_j)], dtype=float)
    return measure_relative_az_el(state_i, state_j, frame=frame)


def inter_satellite_jacobians(
    state_i: Array,
    state_j: Array,
    *,
    modality: str,
    frame: str = "ECI",
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
        left, right, modality=normalized, frame=frame
    )
    h_i = _numerical_state_jacobian(
        lambda value: function(value, state_j), state_i, angular=True, eps=eps
    )
    h_j = _numerical_state_jacobian(
        lambda value: function(state_i, value), state_j, angular=True, eps=eps
    )
    return h_i, h_j


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
