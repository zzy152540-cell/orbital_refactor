from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .dynamics import rk4_step_rel
from .measurements import (
    h_ir_spri,
    h_nn_position_eci,
    h_nn_position_spri,
    h_nn_position_velocity_eci,
    h_nn_position_velocity_spri,
    h_optical_spri,
    h_radar_spri,
    measurement_residual,
)

Array = np.ndarray


@dataclass(frozen=True)
class CentralizedUpdateDiagnostics:
    nis_by_modality: dict[str, float]
    gated_by_modality: dict[str, bool]
    accepted_modalities: list[str]
    rejected_modalities: list[str]
    skipped_modalities: list[str]


class CentralizedDynamicsEKF:
    """Legacy-compatible centralized EKF for heterogeneous orbital measurements."""

    def __init__(
        self,
        *,
        process_noise: Array,
        measurement_covariances: Mapping[str, Array],
        gate_enable: bool = False,
        gate_thresholds: Mapping[str, float] | None = None,
        gate_mode: str = "soft",
        soft_scale: float = 20.0,
        regularization: float = 1e-9,
        legacy_fixed_jacobian_step: bool = True,
        nn_meas_frame: str = "eci",
        nn_use_pseudo_velocity: bool = False,
    ) -> None:
        self.process_noise = np.asarray(process_noise, dtype=float).reshape(6, 6)
        self.measurement_covariances = {
            str(name): np.asarray(value, dtype=float).copy()
            for name, value in measurement_covariances.items()
        }
        self.gate_enable = bool(gate_enable)
        self.gate_thresholds = dict(gate_thresholds or {})
        self.gate_mode = str(gate_mode).lower()
        self.soft_scale = float(soft_scale)
        self.regularization = float(regularization)
        self.legacy_fixed_jacobian_step = bool(legacy_fixed_jacobian_step)
        self.nn_meas_frame = str(nn_meas_frame).lower()
        self.nn_use_pseudo_velocity = bool(nn_use_pseudo_velocity)
        if self.gate_mode not in {"soft", "hard"}:
            raise ValueError("gate_mode must be 'soft' or 'hard'.")

    def predict(self, state: Array, covariance: Array, chief_state_eci: Array, dt: float) -> tuple[Array, Array]:
        state = np.asarray(state, dtype=float).reshape(6)
        covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
        f = lambda value: rk4_step_rel(value, chief_state_eci, dt)
        predicted_state = f(state)
        transition = np.zeros((6, 6), dtype=float)
        for index in range(6):
            step = 1e-6 if self.legacy_fixed_jacobian_step else 1e-6 * max(1.0, abs(float(state[index])))
            plus = state.copy(); plus[index] += step
            minus = state.copy(); minus[index] -= step
            transition[:, index] = (f(plus) - f(minus)) / (2.0 * step)
        predicted_covariance = transition @ covariance @ transition.T + self.process_noise
        return predicted_state, predicted_covariance

    def centralized_update(
        self,
        predicted_state: Array,
        predicted_covariance: Array,
        measurements: Mapping[str, Array | None],
        q_eci2pri: Array,
    ) -> tuple[Array, Array, CentralizedUpdateDiagnostics]:
        accepted: list[str] = []
        rejected: list[str] = []
        skipped: list[str] = []
        nis_by_modality: dict[str, float] = {}
        gated_by_modality: dict[str, bool] = {}
        measurement_blocks: list[Array] = []
        prediction_blocks: list[Array] = []
        jacobian_blocks: list[Array] = []
        covariance_blocks: list[Array] = []
        modality_dimensions: list[tuple[str, int]] = []

        for modality in self.measurement_covariances:
            measurement = measurements.get(modality)
            if measurement is None:
                skipped.append(modality)
                continue
            measurement = np.asarray(measurement, dtype=float).reshape(-1)
            h = self._measurement_function(modality, q_eci2pri)
            predicted_measurement = h(predicted_state)
            jacobian = self._measurement_jacobian(h, predicted_state)
            residual = measurement_residual(measurement, predicted_measurement, modality)
            base_covariance = self.measurement_covariances[modality]
            effective_covariance = base_covariance.copy()
            innovation_covariance = (
                jacobian @ predicted_covariance @ jacobian.T
                + effective_covariance
                + self.regularization * np.eye(jacobian.shape[0])
            )
            nis = float(residual.T @ np.linalg.pinv(innovation_covariance) @ residual)
            nis_by_modality[modality] = nis
            threshold = float(self.gate_thresholds.get(modality, np.inf))
            gated = False
            if self.gate_enable and np.isfinite(threshold) and nis > threshold:
                gated = True
                if self.gate_mode == "hard":
                    rejected.append(modality)
                    gated_by_modality[modality] = True
                    continue
                effective_covariance = self.soft_scale * base_covariance
            gated_by_modality[modality] = gated
            accepted.append(modality)
            measurement_blocks.append(measurement)
            prediction_blocks.append(predicted_measurement)
            jacobian_blocks.append(jacobian)
            covariance_blocks.append(effective_covariance)
            modality_dimensions.append((modality, measurement.size))

        diagnostics = CentralizedUpdateDiagnostics(
            nis_by_modality=nis_by_modality,
            gated_by_modality=gated_by_modality,
            accepted_modalities=accepted,
            rejected_modalities=rejected,
            skipped_modalities=skipped,
        )
        if not measurement_blocks:
            return predicted_state.copy(), predicted_covariance.copy(), diagnostics

        measurement_stack = np.concatenate(measurement_blocks)
        prediction_stack = np.concatenate(prediction_blocks)
        jacobian_stack = np.vstack(jacobian_blocks)
        covariance_stack = _block_diag(covariance_blocks)
        residual_stack = measurement_stack - prediction_stack
        offset = 0
        for modality, dimension in modality_dimensions:
            if modality == "ir":
                residual_stack[offset:offset + dimension] = (
                    residual_stack[offset:offset + dimension] + np.pi
                ) % (2.0 * np.pi) - np.pi
            offset += dimension

        innovation_covariance = (
            jacobian_stack @ predicted_covariance @ jacobian_stack.T
            + covariance_stack
            + self.regularization * np.eye(jacobian_stack.shape[0])
        )
        gain = (predicted_covariance @ jacobian_stack.T) @ np.linalg.pinv(innovation_covariance)
        updated_state = predicted_state + gain @ residual_stack
        identity = np.eye(6)
        residual_matrix = identity - gain @ jacobian_stack
        updated_covariance = (
            residual_matrix @ predicted_covariance @ residual_matrix.T
            + gain @ covariance_stack @ gain.T
        )
        return updated_state, updated_covariance, diagnostics

    def _measurement_function(self, modality: str, q_eci2pri: Array):
        if modality == "opt":
            return lambda state: h_optical_spri(state, q_eci2pri)
        if modality == "ir":
            return lambda state: h_ir_spri(state, q_eci2pri)
        if modality == "rad":
            return lambda state: h_radar_spri(state, q_eci2pri)
        if modality == "nn":
            if self.nn_meas_frame == "spri":
                function = h_nn_position_velocity_spri if self.nn_use_pseudo_velocity else h_nn_position_spri
            else:
                function = h_nn_position_velocity_eci if self.nn_use_pseudo_velocity else h_nn_position_eci
            return lambda state: function(state, q_eci2pri)
        raise ValueError(f"Unsupported centralized modality: {modality}")

    def _measurement_jacobian(self, function, state: Array) -> Array:
        state = np.asarray(state, dtype=float)
        output_dimension = len(function(state))
        jacobian = np.zeros((output_dimension, state.size))
        for index in range(state.size):
            step = 1e-6 if self.legacy_fixed_jacobian_step else 1e-6 * max(1.0, abs(float(state[index])))
            plus = state.copy(); plus[index] += step
            minus = state.copy(); minus[index] -= step
            jacobian[:, index] = (function(plus) - function(minus)) / (2.0 * step)
        return jacobian


def _block_diag(matrices: list[Array]) -> Array:
    total = sum(matrix.shape[0] for matrix in matrices)
    result = np.zeros((total, total), dtype=float)
    offset = 0
    for matrix in matrices:
        dimension = matrix.shape[0]
        result[offset:offset + dimension, offset:offset + dimension] = matrix
        offset += dimension
    return result
