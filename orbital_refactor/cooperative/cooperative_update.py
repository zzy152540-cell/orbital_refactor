from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.data_objects import ObservationMessage, StateMessage, TargetEstimate
from orbital_core.inter_satellite_model import RelativeMeasurementModel

Array = np.ndarray


@dataclass(frozen=True)
class CooperativeUpdateResult:
    estimate: TargetEstimate
    innovation: Array
    innovation_covariance: Array
    effective_measurement_covariance: Array
    nis: float
    gated: bool
    skipped: bool
    observation_id: str
    neighbor_source_timestamp: float


def update_local_state(
    *,
    local_estimate: TargetEstimate,
    neighbor_state: StateMessage,
    observation: ObservationMessage,
    regularization: float = 1e-9,
    gate_enable: bool = False,
    gate_threshold: float = np.inf,
    gate_mode: str = "soft",
    soft_scale: float = 20.0,
    quaternion_i2b_wxyz: Array | None = None,
) -> CooperativeUpdateResult:
    """Update one node's own state from a directed relative observation.

    The updated random variable is always ``local_estimate.target_node_id``.
    The neighbor state is a nuisance variable whose uncertainty is propagated
    into ``R_eff = R + H_neighbor P_neighbor H_neighbor.T``.
    """

    _validate_semantics(local_estimate, neighbor_state, observation)
    if regularization < 0.0:
        raise ValueError("regularization cannot be negative.")
    if gate_mode not in {"soft", "hard"}:
        raise ValueError("gate_mode must be 'soft' or 'hard'.")
    if soft_scale < 1.0:
        raise ValueError("soft_scale must be at least 1.0.")

    local_state = np.asarray(local_estimate.state_estimate, dtype=float).reshape(6)
    local_covariance = _covariance(
        local_estimate.covariance, dimension=6, name="local covariance"
    )
    neighbor_vector = np.asarray(neighbor_state.state_estimate, dtype=float).reshape(6)
    neighbor_covariance = _covariance(
        neighbor_state.covariance, dimension=6, name="neighbor covariance"
    )

    model = RelativeMeasurementModel(
        modality=observation.modality,
        frame=observation.frame,
    )
    local_is_observer = str(observation.observer_id) == str(
        local_estimate.target_node_id
    )
    observer_state = local_state if local_is_observer else neighbor_vector
    target_state = neighbor_vector if local_is_observer else local_state
    predicted = model.predict(
        observer_state,
        target_state,
        quaternion_i2b_wxyz=quaternion_i2b_wxyz,
    )
    observer_jacobian, target_jacobian = model.jacobians(
        observer_state,
        target_state,
        quaternion_i2b_wxyz=quaternion_i2b_wxyz,
    )
    local_jacobian = observer_jacobian if local_is_observer else target_jacobian
    neighbor_jacobian = target_jacobian if local_is_observer else observer_jacobian
    measurement = np.asarray(observation.measurement, dtype=float).reshape(-1)
    if measurement.shape != predicted.shape:
        raise ValueError("Observation measurement has incompatible dimensions.")
    sensor_covariance = _covariance(
        observation.covariance,
        dimension=measurement.size,
        name="observation covariance",
        positive_definite=True,
    )
    confidence = float(observation.confidence)
    if not 0.0 < confidence <= 1.0:
        raise ValueError("observation confidence must be in (0, 1].")
    sensor_covariance = sensor_covariance / confidence

    innovation = model.residual(measurement, predicted)
    effective_covariance = (
        sensor_covariance
        + neighbor_jacobian @ neighbor_covariance @ neighbor_jacobian.T
    )
    innovation_covariance = (
        local_jacobian @ local_covariance @ local_jacobian.T
        + effective_covariance
        + regularization * np.eye(measurement.size)
    )
    nis = float(
        innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation
    )
    gated = bool(gate_enable and np.isfinite(gate_threshold) and nis > gate_threshold)
    if gated and gate_mode == "hard":
        return _result(
            local_estimate=local_estimate,
            observation=observation,
            neighbor_state=neighbor_state,
            innovation=innovation,
            innovation_covariance=innovation_covariance,
            effective_covariance=effective_covariance,
            nis=nis,
            gated=True,
            skipped=True,
        )

    update_covariance = (
        soft_scale * effective_covariance
        if gated and gate_mode == "soft"
        else effective_covariance
    )
    update_innovation_covariance = (
        local_jacobian @ local_covariance @ local_jacobian.T
        + update_covariance
        + regularization * np.eye(measurement.size)
    )
    gain = (
        local_covariance @ local_jacobian.T
    ) @ np.linalg.pinv(update_innovation_covariance)
    updated_state = local_state + gain @ innovation
    residual_matrix = np.eye(6) - gain @ local_jacobian
    updated_covariance = (
        residual_matrix @ local_covariance @ residual_matrix.T
        + gain @ update_covariance @ gain.T
    )
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    information_id = observation.information_id
    used_information = tuple(
        dict.fromkeys((*local_estimate.information_ids, information_id))
    )
    updated_estimate = TargetEstimate(
        estimator_node_id=local_estimate.estimator_node_id,
        target_node_id=local_estimate.target_node_id,
        timestamp=float(observation.timestamp),
        state_estimate=updated_state,
        covariance=updated_covariance,
        quality_score=float(local_estimate.quality_score),
        valid_flag=bool(local_estimate.valid_flag),
        information_ids=used_information,
    )
    return CooperativeUpdateResult(
        estimate=updated_estimate,
        innovation=innovation,
        innovation_covariance=update_innovation_covariance,
        effective_measurement_covariance=update_covariance,
        nis=nis,
        gated=gated,
        skipped=False,
        observation_id=information_id,
        neighbor_source_timestamp=_source_timestamp(neighbor_state),
    )


def _validate_semantics(
    local_estimate: TargetEstimate,
    neighbor_state: StateMessage,
    observation: ObservationMessage,
) -> None:
    local_id = str(local_estimate.target_node_id)
    neighbor_id = str(neighbor_state.target_node_id)
    if local_id == neighbor_id:
        raise ValueError("Local and neighbor states must describe different satellites.")
    endpoints = {str(observation.observer_id), str(observation.target_id)}
    if endpoints != {local_id, neighbor_id}:
        raise ValueError(
            "Observation endpoints must match the local and neighbor state targets."
        )
    if not local_estimate.valid_flag or not neighbor_state.valid_flag:
        raise ValueError("Local and neighbor state estimates must be valid.")
    if not observation.valid_flag:
        raise ValueError("Observation message must be valid.")
    if observation.information_id in local_estimate.information_ids:
        raise ValueError("Observation has already been used by this target estimate.")


def _result(
    *,
    local_estimate: TargetEstimate,
    observation: ObservationMessage,
    neighbor_state: StateMessage,
    innovation: Array,
    innovation_covariance: Array,
    effective_covariance: Array,
    nis: float,
    gated: bool,
    skipped: bool,
) -> CooperativeUpdateResult:
    return CooperativeUpdateResult(
        estimate=local_estimate,
        innovation=innovation,
        innovation_covariance=innovation_covariance,
        effective_measurement_covariance=effective_covariance,
        nis=float(nis),
        gated=gated,
        skipped=skipped,
        observation_id=observation.information_id,
        neighbor_source_timestamp=_source_timestamp(neighbor_state),
    )


def _source_timestamp(message: StateMessage) -> float:
    return float(
        message.timestamp
        if message.source_timestamp is None
        else message.source_timestamp
    )


def _covariance(
    value: Array,
    *,
    dimension: int,
    name: str,
    positive_definite: bool = False,
) -> Array:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    symmetric = 0.5 * (matrix + matrix.T)
    minimum = float(np.min(np.linalg.eigvalsh(symmetric)))
    threshold = 0.0 if positive_definite else -1e-12
    invalid = minimum <= threshold if positive_definite else minimum < threshold
    if invalid:
        qualifier = "positive definite" if positive_definite else "positive semidefinite"
        raise ValueError(f"{name} must be {qualifier}.")
    return symmetric
