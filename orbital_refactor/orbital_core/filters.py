from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dynamics import numerical_jacobian_discrete, rk4_step_rel
from .measurements import (
    h_ir_spri,
    h_nn_position_eci,
    h_nn_position_spri,
    h_nn_position_velocity_eci,
    h_nn_position_velocity_spri,
    h_optical_spri,
    h_radar_spri,
    measurement_residual,
    numerical_measurement_jacobian,
)
from .measurement_integrity import (
    INTEGRITY_MODE_HARD_GATE,
    INTEGRITY_MODE_LEGACY_FIXED_SOFT,
    INTEGRITY_MODE_NONE,
    MeasurementIntegrityDiagnostics,
    MeasurementIntegrityPolicy,
    evaluate_measurement_integrity,
)


Array = np.ndarray


@dataclass(frozen=True)
class UpdateDiagnostics:
    nis: float
    gated: bool
    skipped: bool
    effective_measurement_covariance: Array
    integrity: MeasurementIntegrityDiagnostics


class LocalDynamicsEKF:
    """Six-state relative-orbit EKF extracted from the legacy orbital scripts.

    The default numerical behavior intentionally matches the legacy implementation:
    pseudo-inverses, Joseph covariance update, fixed finite-difference step, and
    soft/hard NIS gating are retained.
    """

    _SUPPORTED_MODES = {"opt", "nn", "ir", "rad"}

    def __init__(
        self,
        process_noise: Array,
        measurement_covariance: Array,
        mode_name: str,
        *,
        gate_enable: bool = False,
        gate_threshold: float = np.inf,
        gate_mode: str = "soft",
        soft_scale: float = 20.0,
        regularization: float = 1e-9,
        legacy_fixed_jacobian_step: bool = True,
        nn_meas_frame: str = "eci",
        nn_use_pseudo_velocity: bool = False,
        integrity_policy: MeasurementIntegrityPolicy | None = None,
    ) -> None:
        mode = mode_name.lower()
        if mode not in self._SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode_name: {mode_name}")
        if gate_mode not in {"soft", "hard"}:
            raise ValueError("gate_mode must be 'soft' or 'hard'.")
        if soft_scale < 1.0:
            raise ValueError("soft_scale must be at least 1.0.")

        self.Q = np.asarray(process_noise, dtype=float)
        self.R = np.asarray(measurement_covariance, dtype=float)
        self.mode_name = mode
        self.gate_enable = bool(gate_enable)
        self.gate_threshold = float(gate_threshold)
        self.gate_mode = gate_mode
        self.soft_scale = float(soft_scale)
        self.regularization = float(regularization)
        self.legacy_fixed_jacobian_step = bool(legacy_fixed_jacobian_step)
        self.nn_meas_frame = nn_meas_frame.lower()
        self.nn_use_pseudo_velocity = bool(nn_use_pseudo_velocity)
        if integrity_policy is None:
            if not self.gate_enable or not np.isfinite(self.gate_threshold):
                integrity_policy = MeasurementIntegrityPolicy()
            elif self.gate_mode == "hard":
                integrity_policy = MeasurementIntegrityPolicy(
                    mode=INTEGRITY_MODE_HARD_GATE,
                    hard_gate_threshold=self.gate_threshold,
                )
            else:
                integrity_policy = MeasurementIntegrityPolicy(
                    mode=INTEGRITY_MODE_LEGACY_FIXED_SOFT,
                    inflation_threshold=self.gate_threshold,
                    fixed_covariance_scale=self.soft_scale,
                )
        self.integrity_policy = integrity_policy
        if self.mode_name == "nn" and self.nn_meas_frame not in {"eci", "spri"}:
            raise ValueError("nn_meas_frame must be 'eci' or 'spri'.")

        if self.Q.shape != (6, 6):
            raise ValueError("process_noise must have shape (6, 6).")
        if self.R.ndim != 2 or self.R.shape[0] != self.R.shape[1]:
            raise ValueError("measurement_covariance must be square.")

    def predict(
        self,
        state: Array,
        covariance: Array,
        chief_state_eci: Array,
        dt: float,
    ) -> tuple[Array, Array]:
        state = np.asarray(state, dtype=float).reshape(6)
        covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
        propagate: Callable[[Array], Array] = lambda x: rk4_step_rel(x, chief_state_eci, dt)
        predicted_state = propagate(state)
        transition = self._discrete_jacobian(propagate, state)
        predicted_covariance = transition @ covariance @ transition.T + self.Q
        return predicted_state, self._symmetrize(predicted_covariance)

    def measurement_function(self, q_eci2pri: Array) -> Callable[[Array], Array]:
        if self.mode_name == "nn":
            if self.nn_meas_frame == "eci":
                function = (
                    h_nn_position_velocity_eci
                    if self.nn_use_pseudo_velocity
                    else h_nn_position_eci
                )
            else:
                function = (
                    h_nn_position_velocity_spri
                    if self.nn_use_pseudo_velocity
                    else h_nn_position_spri
                )
            return lambda x: function(x, q_eci2pri)
        if self.mode_name == "opt":
            return lambda x: h_optical_spri(x, q_eci2pri)
        if self.mode_name == "ir":
            return lambda x: h_ir_spri(x, q_eci2pri)
        return lambda x: h_radar_spri(x, q_eci2pri)

    def update(
        self,
        predicted_state: Array,
        predicted_covariance: Array,
        measurement: Array,
        q_eci2pri: Array,
    ) -> tuple[Array, Array, UpdateDiagnostics]:
        predicted_state = np.asarray(predicted_state, dtype=float).reshape(6)
        predicted_covariance = np.asarray(predicted_covariance, dtype=float).reshape(6, 6)
        measurement = np.asarray(measurement, dtype=float)

        h = self.measurement_function(q_eci2pri)
        predicted_measurement = h(predicted_state)
        measurement_matrix = self._measurement_jacobian(h, predicted_state)
        innovation = measurement_residual(measurement, predicted_measurement, self.mode_name)

        evaluation = evaluate_measurement_integrity(
            innovation=innovation,
            predicted_measurement_covariance=(
                measurement_matrix @ predicted_covariance @ measurement_matrix.T
            ),
            measurement_covariance=self.R,
            policy=self.integrity_policy,
            regularization=self.regularization,
        )
        effective_r = evaluation.effective_measurement_covariance
        innovation_covariance = evaluation.innovation_covariance
        nis = float(evaluation.diagnostics.raw_nis)
        gated = evaluation.diagnostics.anomalous
        skipped = evaluation.skipped
        if skipped:
            return (
                predicted_state.copy(), predicted_covariance.copy(),
                UpdateDiagnostics(
                    nis, gated, skipped, effective_r, evaluation.diagnostics
                ),
            )

        kalman_gain = (
            predicted_covariance @ measurement_matrix.T
        ) @ np.linalg.pinv(innovation_covariance)
        updated_state = predicted_state + kalman_gain @ innovation

        identity = np.eye(6)
        residual_matrix = identity - kalman_gain @ measurement_matrix
        updated_covariance = (
            residual_matrix @ predicted_covariance @ residual_matrix.T
            + kalman_gain @ effective_r @ kalman_gain.T
        )
        diagnostics = UpdateDiagnostics(
            nis, gated, skipped, effective_r, evaluation.diagnostics,
        )
        return updated_state, self._symmetrize(updated_covariance), diagnostics

    def _innovation_covariance(self, h: Array, p: Array, r: Array) -> Array:
        return h @ p @ h.T + r + self.regularization * np.eye(h.shape[0])

    def _discrete_jacobian(self, f: Callable[[Array], Array], x: Array) -> Array:
        if not self.legacy_fixed_jacobian_step:
            return numerical_jacobian_discrete(f, x)
        return _fixed_step_jacobian(f, x)

    def _measurement_jacobian(self, h: Callable[[Array], Array], x: Array) -> Array:
        if not self.legacy_fixed_jacobian_step:
            return numerical_measurement_jacobian(h, x)
        return _fixed_step_jacobian(h, x)

    @staticmethod
    def _symmetrize(matrix: Array) -> Array:
        return 0.5 * (matrix + matrix.T)


# Backward-compatible alias for the single-modal script terminology.
DynamicsEKF = LocalDynamicsEKF


def _fixed_step_jacobian(
    function: Callable[[Array], Array],
    state: Array,
    eps: float = 1e-6,
) -> Array:
    state = np.asarray(state, dtype=float)
    output = np.asarray(function(state), dtype=float)
    jacobian = np.zeros((output.size, state.size), dtype=float)
    for index in range(state.size):
        delta = np.zeros(state.size)
        delta[index] = eps
        jacobian[:, index] = (
            np.asarray(function(state + delta), dtype=float)
            - np.asarray(function(state - delta), dtype=float)
        ) / (2.0 * eps)
    return jacobian
