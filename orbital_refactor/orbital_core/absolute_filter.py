from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.dynamics import numerical_jacobian_discrete, rk4_step_absolute

Array = np.ndarray


@dataclass
class AbsoluteOrbitEKF:
    """Absolute ECI orbit EKF prediction model for v13 satellite nodes."""

    process_noise: Array
    legacy_fixed_jacobian_step: bool = True

    def __post_init__(self) -> None:
        self.process_noise = np.asarray(self.process_noise, dtype=float).reshape(6, 6)

    def predict(
        self,
        state: Array,
        covariance: Array,
        dt: float,
    ) -> tuple[Array, Array]:
        state = np.asarray(state, dtype=float).reshape(6)
        covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        propagate = lambda value: rk4_step_absolute(value, dt)
        predicted_state = propagate(state)
        transition = (
            _fixed_step_jacobian(propagate, state)
            if self.legacy_fixed_jacobian_step
            else numerical_jacobian_discrete(propagate, state)
        )
        predicted_covariance = transition @ covariance @ transition.T + self.process_noise
        return predicted_state, 0.5 * (predicted_covariance + predicted_covariance.T)


def _fixed_step_jacobian(function, state: Array, eps: float = 1e-6) -> Array:
    state = np.asarray(state, dtype=float).reshape(-1)
    output = np.asarray(function(state), dtype=float).reshape(-1)
    jacobian = np.zeros((output.size, state.size), dtype=float)
    for index in range(state.size):
        delta = np.zeros(state.size, dtype=float)
        delta[index] = eps
        jacobian[:, index] = (
            np.asarray(function(state + delta), dtype=float)
            - np.asarray(function(state - delta), dtype=float)
        ) / (2.0 * eps)
    return jacobian
