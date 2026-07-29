from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from orbital_core.inter_satellite_model import (
    inter_satellite_jacobians,
    normalize_inter_satellite_modality,
    predict_inter_satellite_measurement,
)
from interfaces.data_objects import AbsolutePositionObservation, InterSatelliteObservation
from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from orbital_core.measurements import wrap_angle

Array = np.ndarray


@dataclass(frozen=True)
class FleetUpdateDiagnostics:
    nis_by_observation: dict[str, float]
    accepted_observations: tuple[str, ...]


class FleetCentralizedEKF:
    """Centralized EKF over X=[x_1,...,x_N] with 6-state absolute ECI blocks."""

    def __init__(
        self,
        node_ids: Sequence[str],
        *,
        process_noise_acceleration: float = 1e-4,
        regularization: float = 1e-9,
    ) -> None:
        normalized = tuple(str(node_id) for node_id in node_ids)
        if not normalized:
            raise ValueError("At least one node is required.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("node_ids must be unique.")
        if process_noise_acceleration < 0.0:
            raise ValueError("process_noise_acceleration cannot be negative.")
        self.node_ids = normalized
        self.node_index = {node_id: index for index, node_id in enumerate(normalized)}
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.regularization = float(regularization)

    @property
    def state_dimension(self) -> int:
        return 6 * len(self.node_ids)

    def state_slice(self, node_id: str) -> slice:
        try:
            index = self.node_index[str(node_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown fleet node: {node_id}") from exc
        return slice(6 * index, 6 * (index + 1))

    def stack_states(self, state_by_node: Mapping[str, Array]) -> Array:
        self._validate_node_keys(state_by_node, "state_by_node")
        return np.concatenate(
            [
                np.asarray(state_by_node[node_id], dtype=float).reshape(6)
                for node_id in self.node_ids
            ]
        )

    def stack_covariances(self, covariance_by_node: Mapping[str, Array]) -> Array:
        self._validate_node_keys(covariance_by_node, "covariance_by_node")
        return _block_diag(
            [
                np.asarray(covariance_by_node[node_id], dtype=float).reshape(6, 6)
                for node_id in self.node_ids
            ]
        )

    def split_state(self, state: Array) -> dict[str, Array]:
        vector = np.asarray(state, dtype=float).reshape(self.state_dimension)
        return {
            node_id: vector[self.state_slice(node_id)].copy()
            for node_id in self.node_ids
        }

    def predict(self, state: Array, covariance: Array, dt: float) -> tuple[Array, Array]:
        vector = np.asarray(state, dtype=float).reshape(self.state_dimension)
        matrix = np.asarray(covariance, dtype=float).reshape(
            self.state_dimension, self.state_dimension
        )
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        predicted = np.zeros_like(vector)
        transitions: list[Array] = []
        for node_id in self.node_ids:
            block = self.state_slice(node_id)
            local_state = vector[block]
            propagate = lambda value: rk4_step_absolute(value, dt)
            predicted[block] = propagate(local_state)
            transitions.append(numerical_jacobian_discrete(propagate, local_state))
        transition = _block_diag(transitions)
        process_noise = _block_diag(
            [
                make_process_noise(dt, self.process_noise_acceleration)
                for _ in self.node_ids
            ]
        )
        predicted_covariance = transition @ matrix @ transition.T + process_noise
        return predicted, _symmetrize(predicted_covariance)

    def update(
        self,
        predicted_state: Array,
        predicted_covariance: Array,
        observations: Iterable[InterSatelliteObservation],
        *,
        frame_by_modality: Mapping[str, str] | None = None,
    ) -> tuple[Array, Array, FleetUpdateDiagnostics]:
        state = np.asarray(predicted_state, dtype=float).reshape(self.state_dimension)
        covariance = np.asarray(predicted_covariance, dtype=float).reshape(
            self.state_dimension, self.state_dimension
        )
        measurement_blocks: list[Array] = []
        prediction_blocks: list[Array] = []
        jacobian_blocks: list[Array] = []
        measurement_covariances: list[Array] = []
        angular_blocks: list[tuple[int, int]] = []
        labels: list[str] = []
        offset = 0

        for observation in observations:
            if not observation.valid_flag:
                continue
            source = str(observation.source_node_id)
            target = str(observation.target_node_id)
            if source == target:
                raise ValueError("Inter-satellite observation endpoints must differ.")
            source_block = self.state_slice(source)
            target_block = self.state_slice(target)
            modality = normalize_inter_satellite_modality(observation.modality)
            frame = (
                str(frame_by_modality.get(modality, "ECI"))
                if frame_by_modality is not None
                else "ECI"
            )
            state_i = state[source_block]
            state_j = state[target_block]
            predicted = predict_inter_satellite_measurement(
                state_i, state_j, modality=modality, frame=frame
            )
            h_i, h_j = inter_satellite_jacobians(
                state_i, state_j, modality=modality, frame=frame
            )
            measurement = np.asarray(observation.measurement, dtype=float).reshape(-1)
            base_covariance = np.asarray(observation.covariance, dtype=float)
            if measurement.shape != predicted.shape:
                raise ValueError(f"{modality} measurement has incompatible dimensions.")
            if base_covariance.shape != (measurement.size, measurement.size):
                raise ValueError(f"{modality} covariance has incompatible dimensions.")
            confidence = float(np.clip(observation.confidence, 1e-6, 1.0))

            full_jacobian = np.zeros(
                (measurement.size, self.state_dimension), dtype=float
            )
            full_jacobian[:, source_block] = h_i
            full_jacobian[:, target_block] = h_j
            measurement_blocks.append(measurement)
            prediction_blocks.append(predicted)
            jacobian_blocks.append(full_jacobian)
            measurement_covariances.append(base_covariance / confidence)
            label = f"{source}->{target}:{modality}"
            labels.append(label)
            if modality == "AZ_EL":
                angular_blocks.append((offset, offset + measurement.size))
            offset += measurement.size

        if not measurement_blocks:
            return (
                state.copy(),
                covariance.copy(),
                FleetUpdateDiagnostics({}, ()),
            )

        measurement = np.concatenate(measurement_blocks)
        prediction = np.concatenate(prediction_blocks)
        jacobian = np.vstack(jacobian_blocks)
        measurement_covariance = _block_diag(measurement_covariances)
        innovation = measurement - prediction
        for start, stop in angular_blocks:
            innovation[start:stop] = wrap_angle(innovation[start:stop])

        innovation_covariance = (
            jacobian @ covariance @ jacobian.T
            + measurement_covariance
            + self.regularization * np.eye(jacobian.shape[0])
        )
        gain = covariance @ jacobian.T @ np.linalg.pinv(innovation_covariance)
        updated_state = state + gain @ innovation
        identity = np.eye(self.state_dimension)
        residual_matrix = identity - gain @ jacobian
        updated_covariance = (
            residual_matrix @ covariance @ residual_matrix.T
            + gain @ measurement_covariance @ gain.T
        )

        nis_by_observation: dict[str, float] = {}
        offset = 0
        for label, block in zip(labels, measurement_blocks, strict=True):
            dimension = block.size
            local_innovation = innovation[offset:offset + dimension]
            local_jacobian = jacobian[offset:offset + dimension]
            local_r = measurement_covariance[offset:offset + dimension, offset:offset + dimension]
            local_s = (
                local_jacobian @ covariance @ local_jacobian.T
                + local_r
                + self.regularization * np.eye(dimension)
            )
            nis_by_observation[label] = float(
                local_innovation.T @ np.linalg.pinv(local_s) @ local_innovation
            )
            offset += dimension
        return (
            updated_state,
            _symmetrize(updated_covariance),
            FleetUpdateDiagnostics(nis_by_observation, tuple(labels)),
        )

    def update_absolute_positions(
        self,
        predicted_state: Array,
        predicted_covariance: Array,
        observations: Iterable[AbsolutePositionObservation],
    ) -> tuple[Array, Array, FleetUpdateDiagnostics]:
        """Update one or more satellite position blocks using absolute ECI anchors."""

        state = np.asarray(predicted_state, dtype=float).reshape(self.state_dimension)
        covariance = np.asarray(predicted_covariance, dtype=float).reshape(
            self.state_dimension, self.state_dimension
        )
        measurement_blocks: list[Array] = []
        prediction_blocks: list[Array] = []
        jacobian_blocks: list[Array] = []
        covariance_blocks: list[Array] = []
        labels: list[str] = []
        for observation in observations:
            if not observation.valid_flag:
                continue
            block = self.state_slice(observation.satellite_id)
            measurement = np.asarray(
                observation.measurement_eci, dtype=float
            ).reshape(3)
            measurement_covariance = np.asarray(
                observation.covariance, dtype=float
            ).reshape(3, 3)
            confidence = float(np.clip(observation.confidence, 1e-6, 1.0))
            jacobian = np.zeros((3, self.state_dimension), dtype=float)
            jacobian[:, block.start:block.start + 3] = np.eye(3)
            measurement_blocks.append(measurement)
            prediction_blocks.append(state[block][:3])
            jacobian_blocks.append(jacobian)
            covariance_blocks.append(measurement_covariance / confidence)
            labels.append(f"{observation.satellite_id}:ABS_POSITION")

        if not measurement_blocks:
            return state.copy(), covariance.copy(), FleetUpdateDiagnostics({}, ())

        measurement = np.concatenate(measurement_blocks)
        prediction = np.concatenate(prediction_blocks)
        jacobian = np.vstack(jacobian_blocks)
        measurement_covariance = _block_diag(covariance_blocks)
        innovation = measurement - prediction
        innovation_covariance = (
            jacobian @ covariance @ jacobian.T
            + measurement_covariance
            + self.regularization * np.eye(jacobian.shape[0])
        )
        gain = covariance @ jacobian.T @ np.linalg.pinv(innovation_covariance)
        updated_state = state + gain @ innovation
        residual_matrix = np.eye(self.state_dimension) - gain @ jacobian
        updated_covariance = (
            residual_matrix @ covariance @ residual_matrix.T
            + gain @ measurement_covariance @ gain.T
        )
        diagnostics: dict[str, float] = {}
        offset = 0
        for label, block_measurement, block_covariance, block_jacobian in zip(
            labels,
            measurement_blocks,
            covariance_blocks,
            jacobian_blocks,
            strict=True,
        ):
            dimension = block_measurement.size
            block_innovation = innovation[offset:offset + dimension]
            block_s = (
                block_jacobian @ covariance @ block_jacobian.T
                + block_covariance
                + self.regularization * np.eye(dimension)
            )
            diagnostics[label] = float(
                block_innovation.T @ np.linalg.pinv(block_s) @ block_innovation
            )
            offset += dimension
        return (
            updated_state,
            _symmetrize(updated_covariance),
            FleetUpdateDiagnostics(diagnostics, tuple(labels)),
        )

    def _validate_node_keys(self, values: Mapping[str, Array], name: str) -> None:
        if set(values) != set(self.node_ids):
            raise ValueError(f"{name} keys must match node_ids.")


def _block_diag(matrices: Sequence[Array]) -> Array:
    total = sum(matrix.shape[0] for matrix in matrices)
    result = np.zeros((total, total), dtype=float)
    offset = 0
    for matrix in matrices:
        rows, columns = matrix.shape
        if rows != columns:
            raise ValueError("Block diagonal inputs must be square.")
        result[offset:offset + rows, offset:offset + rows] = matrix
        offset += rows
    return result


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)
