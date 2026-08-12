from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from interfaces.data_objects import AbsolutePositionObservation, ObservationMessage
from orbital_core.dynamics import make_process_noise, numerical_jacobian_discrete, rk4_step_absolute
from orbital_core.inter_satellite_model import RelativeMeasurementModel
from orbital_core.measurement_integrity import (
    INTEGRITY_ACCEPTED,
    INTEGRITY_DOWNWEIGHTED,
    INTEGRITY_HARD_REJECTED,
    MeasurementIntegrityDiagnostics,
    MeasurementIntegrityPolicy,
    INTEGRITY_MODE_HARD_GATE,
    INTEGRITY_MODE_NONE,
    INTEGRITY_MODE_PROPORTIONAL_INFLATION,
    INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
    evaluate_measurement_integrity,
)

Array = np.ndarray


@dataclass(frozen=True)
class MultiNeighborSchmidtState:
    timestamp: float
    active_node_id: str
    neighbor_ids: tuple[str, ...]
    active_state: Array
    neighbor_state_by_id: dict[str, Array]
    joint_covariance: Array
    information_ids: tuple[str, ...] = ()
    transport_information_ids: tuple[str, ...] = ()

    @property
    def dimension(self) -> int:
        return 6 * (1 + len(self.neighbor_ids))

    @property
    def active_covariance(self) -> Array:
        return self.joint_covariance[:6, :6].copy()

    def neighbor_slice(self, neighbor_id: str) -> slice:
        try:
            index = self.neighbor_ids.index(str(neighbor_id))
        except ValueError as exc:
            raise KeyError(f"Unknown consider neighbor: {neighbor_id}") from exc
        return slice(6 * (index + 1), 6 * (index + 2))

    def neighbor_covariance(self, neighbor_id: str) -> Array:
        block = self.neighbor_slice(neighbor_id)
        return self.joint_covariance[block, block].copy()

    def active_cross_covariance(self, neighbor_id: str) -> Array:
        return self.joint_covariance[:6, self.neighbor_slice(neighbor_id)].copy()


@dataclass(frozen=True)
class MultiNeighborSchmidtUpdateResult:
    state: MultiNeighborSchmidtState
    innovation: Array
    innovation_covariance: Array
    nis: float
    skipped: bool = False
    raw_nis: float | None = None
    measurement_covariance_scale: float = 1.0
    prior_active_state: Array | None = None
    prior_neighbor_state: Array | None = None
    active_jacobian: Array | None = None
    neighbor_jacobian: Array | None = None
    active_correction: Array | None = None
    projected_neighbor_covariance: Array | None = None
    nominal_measurement_covariance: Array | None = None
    active_gain: Array | None = None
    prior_active_covariance: Array | None = None
    neighbor_uncertainty_inflation: float = 0.0

    @property
    def integrity(self) -> MeasurementIntegrityDiagnostics:
        status = (
            INTEGRITY_HARD_REJECTED if self.skipped
            else INTEGRITY_DOWNWEIGHTED
            if self.measurement_covariance_scale > 1.0
            else INTEGRITY_ACCEPTED
        )
        return MeasurementIntegrityDiagnostics(
            raw_nis=self.nis if self.raw_nis is None else self.raw_nis,
            processed_nis=self.nis,
            measurement_covariance_scale=self.measurement_covariance_scale,
            status=status,
        )


@dataclass(frozen=True)
class MultiNeighborSchmidtHistory:
    timestamps: Array
    active_state_history: Array
    active_covariance_history: Array
    neighbor_state_history_by_id: dict[str, Array]
    joint_covariance_history: Array
    nis_history: list[dict[str, float]]


def initialize_multi_neighbor_schmidt(
    *, timestamp: float, active_node_id: str, active_state: Array,
    active_covariance: Array, neighbor_state_by_id: Mapping[str, Array],
    neighbor_covariance_by_id: Mapping[str, Array],
) -> MultiNeighborSchmidtState:
    neighbor_ids = tuple(str(node_id) for node_id in neighbor_state_by_id)
    if set(neighbor_ids) != set(neighbor_covariance_by_id):
        raise ValueError("Neighbor state and covariance IDs must match.")
    blocks = [np.asarray(active_covariance, dtype=float).reshape(6, 6)]
    blocks.extend(np.asarray(neighbor_covariance_by_id[node_id], dtype=float).reshape(6, 6) for node_id in neighbor_ids)
    return MultiNeighborSchmidtState(
        timestamp=float(timestamp), active_node_id=str(active_node_id),
        neighbor_ids=neighbor_ids,
        active_state=np.asarray(active_state, dtype=float).reshape(6).copy(),
        neighbor_state_by_id={node_id: np.asarray(neighbor_state_by_id[node_id], dtype=float).reshape(6).copy() for node_id in neighbor_ids},
        joint_covariance=_block_diag(blocks),
    )


def multi_neighbor_schmidt_predict(
    state: MultiNeighborSchmidtState, target_timestamp: float, *,
    process_noise_acceleration: float = 1e-4,
) -> MultiNeighborSchmidtState:
    dt = float(target_timestamp) - float(state.timestamp)
    if dt <= 0.0:
        raise ValueError("target_timestamp must be later than the state timestamp.")
    states = [state.active_state] + [state.neighbor_state_by_id[node_id] for node_id in state.neighbor_ids]
    transitions = [numerical_jacobian_discrete(lambda value: rk4_step_absolute(value, dt), vector) for vector in states]
    transition = _block_diag(transitions)
    process_noise = _block_diag([make_process_noise(dt, process_noise_acceleration) for _ in states])
    covariance = transition @ state.joint_covariance @ transition.T + process_noise
    propagated = [rk4_step_absolute(vector, dt) for vector in states]
    return MultiNeighborSchmidtState(
        timestamp=float(target_timestamp), active_node_id=state.active_node_id,
        neighbor_ids=state.neighbor_ids, active_state=propagated[0],
        neighbor_state_by_id={node_id: propagated[index + 1] for index, node_id in enumerate(state.neighbor_ids)},
        joint_covariance=_symmetrize(covariance), information_ids=state.information_ids,
        transport_information_ids=state.transport_information_ids,
    )


def multi_neighbor_schmidt_update(
    state: MultiNeighborSchmidtState, observation: ObservationMessage, *,
    quaternion_i2b_wxyz: Array | None = None, regularization: float = 1e-9,
    neighbor_linearization_state: Array | None = None,
    neighbor_uncertainty_inflation: float = 0.0,
    nis_gate_threshold: float | None = None,
    nis_inflation_threshold: float | None = None,
    maximum_measurement_covariance_scale: float = 1.0,
    integrity_policy: MeasurementIntegrityPolicy | None = None,
) -> MultiNeighborSchmidtUpdateResult:
    if quaternion_i2b_wxyz is None and observation.frame.upper() == "BODY":
        quaternion_i2b_wxyz = observation.metadata.get("quaternion_i2b_wxyz")
    if not np.isclose(float(observation.timestamp), float(state.timestamp)):
        raise ValueError("Observation and Schmidt state timestamps must match.")
    if observation.information_id in state.information_ids:
        raise ValueError("Observation has already been used.")
    observer, target = str(observation.observer_id), str(observation.target_id)
    if state.active_node_id not in {observer, target}:
        raise ValueError("Observation must involve the active node.")
    neighbor_id = target if observer == state.active_node_id else observer
    if neighbor_id not in state.neighbor_ids:
        raise ValueError("Observation counterpart is not a consider neighbor.")
    active_is_observer = observer == state.active_node_id
    stored_neighbor_state = state.neighbor_state_by_id[neighbor_id]
    neighbor_state = (
        stored_neighbor_state
        if neighbor_linearization_state is None
        else np.asarray(neighbor_linearization_state, dtype=float).reshape(6)
    )
    observer_state = state.active_state if active_is_observer else neighbor_state
    target_state = neighbor_state if active_is_observer else state.active_state
    model = RelativeMeasurementModel(observation.modality, observation.frame)
    prediction = model.predict(observer_state, target_state, quaternion_i2b_wxyz=quaternion_i2b_wxyz)
    observer_jacobian, target_jacobian = model.jacobians(observer_state, target_state, quaternion_i2b_wxyz=quaternion_i2b_wxyz)
    h_active = observer_jacobian if active_is_observer else target_jacobian
    h_neighbor = target_jacobian if active_is_observer else observer_jacobian
    measurement = np.asarray(observation.measurement, dtype=float).reshape(-1)
    measurement_covariance = np.asarray(observation.covariance, dtype=float)
    if measurement_covariance.shape != (measurement.size, measurement.size):
        raise ValueError("Observation covariance has incompatible dimensions.")
    if not 0.0 < float(observation.confidence) <= 1.0:
        raise ValueError("Observation confidence must be in (0, 1].")
    measurement_covariance = measurement_covariance / float(observation.confidence)
    nominal_measurement_covariance = measurement_covariance.copy()
    projected_neighbor_covariance = (
        h_neighbor
        @ state.neighbor_covariance(neighbor_id)
        @ h_neighbor.T
    )
    if neighbor_uncertainty_inflation < 0.0:
        raise ValueError("Neighbor uncertainty inflation cannot be negative.")
    measurement_covariance = measurement_covariance + (
        float(neighbor_uncertainty_inflation)
        * projected_neighbor_covariance
    )
    innovation = model.residual(measurement, prediction)
    jacobian = np.zeros((measurement.size, state.dimension), dtype=float)
    jacobian[:, :6] = h_active
    jacobian[:, state.neighbor_slice(neighbor_id)] = h_neighbor
    predicted_measurement_covariance = jacobian @ state.joint_covariance @ jacobian.T
    policy = integrity_policy or _policy_from_legacy_integrity_parameters(
        nis_gate_threshold=nis_gate_threshold,
        nis_inflation_threshold=nis_inflation_threshold,
        maximum_measurement_covariance_scale=(
            maximum_measurement_covariance_scale
        ),
    )
    evaluation = evaluate_measurement_integrity(
        innovation=innovation,
        predicted_measurement_covariance=predicted_measurement_covariance,
        measurement_covariance=measurement_covariance,
        policy=policy, regularization=regularization,
    )
    measurement_covariance = evaluation.effective_measurement_covariance
    innovation_covariance = evaluation.innovation_covariance
    raw_nis = float(evaluation.diagnostics.raw_nis)
    nis = float(evaluation.diagnostics.processed_nis)
    covariance_scale = evaluation.diagnostics.measurement_covariance_scale
    if evaluation.skipped:
        return MultiNeighborSchmidtUpdateResult(
            state, innovation, innovation_covariance, nis, skipped=True,
            raw_nis=raw_nis,
            measurement_covariance_scale=covariance_scale,
            prior_active_state=state.active_state.copy(),
            prior_neighbor_state=stored_neighbor_state.copy(),
            active_jacobian=h_active.copy(),
            neighbor_jacobian=h_neighbor.copy(),
            active_correction=np.zeros(6),
            projected_neighbor_covariance=projected_neighbor_covariance.copy(),
            nominal_measurement_covariance=nominal_measurement_covariance.copy(),
            active_gain=np.zeros((6, measurement.size)),
            prior_active_covariance=state.active_covariance,
            neighbor_uncertainty_inflation=float(
                neighbor_uncertainty_inflation
            ),
        )
    gain = np.zeros((state.dimension, measurement.size), dtype=float)
    gain[:6] = (state.joint_covariance[:6] @ jacobian.T) @ np.linalg.pinv(innovation_covariance)
    residual = np.eye(state.dimension) - gain @ jacobian
    covariance = residual @ state.joint_covariance @ residual.T + gain @ measurement_covariance @ gain.T
    updated = replace(state, active_state=state.active_state + gain[:6] @ innovation, joint_covariance=_symmetrize(covariance), information_ids=(*state.information_ids, observation.information_id))
    return MultiNeighborSchmidtUpdateResult(
        updated, innovation, innovation_covariance, nis,
        raw_nis=raw_nis,
        measurement_covariance_scale=covariance_scale,
        prior_active_state=state.active_state.copy(),
        prior_neighbor_state=stored_neighbor_state.copy(),
        active_jacobian=h_active.copy(),
        neighbor_jacobian=h_neighbor.copy(),
        active_correction=gain[:6] @ innovation,
        projected_neighbor_covariance=projected_neighbor_covariance.copy(),
        nominal_measurement_covariance=nominal_measurement_covariance.copy(),
        active_gain=gain[:6].copy(),
        prior_active_covariance=state.active_covariance,
        neighbor_uncertainty_inflation=float(
            neighbor_uncertainty_inflation
        ),
    )


def multi_neighbor_schmidt_batch_update(
    state: MultiNeighborSchmidtState,
    observations: Iterable[ObservationMessage],
    *,
    regularization: float = 1e-9,
    integrity_policy: MeasurementIntegrityPolicy | None = None,
    neighbor_linearization_state: Array | None = None,
    neighbor_uncertainty_inflation_by_modality: Mapping[
        str, float
    ] | None = None,
) -> MultiNeighborSchmidtUpdateResult:
    """Jointly update one active-neighbor pair from one common prior.

    The first diagnostic implementation assumes independent measurement noise
    between observations and therefore constructs a block-diagonal joint
    covariance. Every prediction and Jacobian is evaluated at the same prior
    state; only one Joseph covariance update is performed.
    """

    values = tuple(observations)
    if not values:
        raise ValueError("At least one relative observation is required.")
    if len({item.information_id for item in values}) != len(values):
        raise ValueError("Batch observation information IDs must be unique.")
    if set(item.information_id for item in values) & set(state.information_ids):
        raise ValueError("A batch observation has already been used.")

    reference = values[0]
    timestamp = float(reference.timestamp)
    endpoints = (str(reference.observer_id), str(reference.target_id))
    if not np.isclose(timestamp, float(state.timestamp)):
        raise ValueError("Observation and Schmidt state timestamps must match.")
    if state.active_node_id not in endpoints:
        raise ValueError("Observations must involve the active node.")
    if any(
        not np.isclose(float(item.timestamp), timestamp)
        or (str(item.observer_id), str(item.target_id)) != endpoints
        for item in values
    ):
        raise ValueError(
            "Batch observations must share timestamp, observer, and target."
        )

    observer, target = endpoints
    active_is_observer = observer == state.active_node_id
    neighbor_id = target if active_is_observer else observer
    if neighbor_id not in state.neighbor_ids:
        raise ValueError("Observation counterpart is not a consider neighbor.")
    stored_neighbor_state = state.neighbor_state_by_id[neighbor_id]
    neighbor_state = (
        stored_neighbor_state
        if neighbor_linearization_state is None
        else np.asarray(neighbor_linearization_state, dtype=float).reshape(6)
    )
    observer_state = state.active_state if active_is_observer else neighbor_state
    target_state = neighbor_state if active_is_observer else state.active_state

    innovations = []
    jacobians = []
    active_jacobians = []
    neighbor_jacobians = []
    covariances = []
    nominal_covariances = []
    projected_neighbor_covariances = []
    inflation_factors = neighbor_uncertainty_inflation_by_modality or {}
    for observation in values:
        quaternion = (
            observation.metadata.get("quaternion_i2b_wxyz")
            if observation.frame.upper() == "BODY" else None
        )
        model = RelativeMeasurementModel(
            observation.modality, observation.frame
        )
        prediction = model.predict(
            observer_state, target_state,
            quaternion_i2b_wxyz=quaternion,
        )
        observer_jacobian, target_jacobian = model.jacobians(
            observer_state, target_state,
            quaternion_i2b_wxyz=quaternion,
        )
        measurement = np.asarray(
            observation.measurement, dtype=float
        ).reshape(-1)
        covariance = np.asarray(observation.covariance, dtype=float)
        if covariance.shape != (measurement.size, measurement.size):
            raise ValueError("Observation covariance has incompatible dimensions.")
        if not 0.0 < float(observation.confidence) <= 1.0:
            raise ValueError("Observation confidence must be in (0, 1].")
        jacobian = np.zeros((measurement.size, state.dimension), dtype=float)
        jacobian[:, :6] = (
            observer_jacobian if active_is_observer else target_jacobian
        )
        jacobian[:, state.neighbor_slice(neighbor_id)] = (
            target_jacobian if active_is_observer else observer_jacobian
        )
        innovations.append(model.residual(measurement, prediction))
        jacobians.append(jacobian)
        active_jacobians.append(jacobian[:, :6])
        neighbor_jacobians.append(
            jacobian[:, state.neighbor_slice(neighbor_id)]
        )
        factor = float(inflation_factors.get(observation.modality, 0.0))
        if factor < 0.0:
            raise ValueError("Neighbor uncertainty inflation cannot be negative.")
        neighbor_jacobian = jacobian[:, state.neighbor_slice(neighbor_id)]
        nominal_covariance = covariance / float(observation.confidence)
        projected_neighbor_covariance = (
            neighbor_jacobian
            @ state.neighbor_covariance(neighbor_id)
            @ neighbor_jacobian.T
        )
        nominal_covariances.append(nominal_covariance)
        projected_neighbor_covariances.append(projected_neighbor_covariance)
        covariances.append(
            nominal_covariance
            + factor
            * projected_neighbor_covariance
        )

    innovation = np.concatenate(innovations)
    jacobian = np.vstack(jacobians)
    measurement_covariance = _block_diag(covariances)
    predicted_measurement_covariance = (
        jacobian @ state.joint_covariance @ jacobian.T
    )
    evaluation = evaluate_measurement_integrity(
        innovation=innovation,
        predicted_measurement_covariance=predicted_measurement_covariance,
        measurement_covariance=measurement_covariance,
        policy=integrity_policy or MeasurementIntegrityPolicy(
            mode=INTEGRITY_MODE_NONE
        ),
        regularization=regularization,
    )
    effective_covariance = evaluation.effective_measurement_covariance
    innovation_covariance = evaluation.innovation_covariance
    raw_nis = float(evaluation.diagnostics.raw_nis)
    nis = float(evaluation.diagnostics.processed_nis)
    scale = evaluation.diagnostics.measurement_covariance_scale
    if evaluation.skipped:
        return MultiNeighborSchmidtUpdateResult(
            state, innovation, innovation_covariance, nis,
            skipped=True, raw_nis=raw_nis,
            measurement_covariance_scale=scale,
            prior_active_state=state.active_state.copy(),
            prior_neighbor_state=stored_neighbor_state.copy(),
            active_jacobian=np.vstack(active_jacobians),
            neighbor_jacobian=np.vstack(neighbor_jacobians),
            active_correction=np.zeros(6),
            projected_neighbor_covariance=_block_diag(
                projected_neighbor_covariances
            ),
            nominal_measurement_covariance=_block_diag(nominal_covariances),
            active_gain=np.zeros((6, innovation.size)),
            prior_active_covariance=state.active_covariance,
            neighbor_uncertainty_inflation=float(max(
                inflation_factors.get(item.modality, 0.0)
                for item in values
            )),
        )
    gain = np.zeros((state.dimension, innovation.size), dtype=float)
    gain[:6] = (
        state.joint_covariance[:6] @ jacobian.T
    ) @ np.linalg.pinv(innovation_covariance)
    residual = np.eye(state.dimension) - gain @ jacobian
    covariance = (
        residual @ state.joint_covariance @ residual.T
        + gain @ effective_covariance @ gain.T
    )
    updated = replace(
        state,
        active_state=state.active_state + gain[:6] @ innovation,
        joint_covariance=_symmetrize(covariance),
        information_ids=(
            *state.information_ids,
            *(item.information_id for item in values),
        ),
    )
    return MultiNeighborSchmidtUpdateResult(
        updated, innovation, innovation_covariance, nis,
        raw_nis=raw_nis, measurement_covariance_scale=scale,
        prior_active_state=state.active_state.copy(),
        prior_neighbor_state=stored_neighbor_state.copy(),
        active_jacobian=np.vstack(active_jacobians),
        neighbor_jacobian=np.vstack(neighbor_jacobians),
        active_correction=gain[:6] @ innovation,
        projected_neighbor_covariance=_block_diag(
            projected_neighbor_covariances
        ),
        nominal_measurement_covariance=_block_diag(nominal_covariances),
        active_gain=gain[:6].copy(),
        prior_active_covariance=state.active_covariance,
        neighbor_uncertainty_inflation=float(max(
            inflation_factors.get(item.modality, 0.0)
            for item in values
        )),
    )


def multi_neighbor_schmidt_absolute_position_update(
    state: MultiNeighborSchmidtState,
    observation: AbsolutePositionObservation,
    *, regularization: float = 1e-9,
    nis_gate_threshold: float | None = None,
    nis_inflation_threshold: float | None = None,
    maximum_measurement_covariance_scale: float = 1.0,
    integrity_policy: MeasurementIntegrityPolicy | None = None,
) -> MultiNeighborSchmidtUpdateResult:
    """Update only the active satellite from its own absolute position fix."""

    if not np.isclose(float(observation.timestamp), float(state.timestamp)):
        raise ValueError("Absolute observation and Schmidt timestamps must match.")
    if str(observation.satellite_id) != state.active_node_id:
        raise ValueError("Absolute observation must belong to the active node.")
    if not observation.valid_flag:
        raise ValueError("Absolute observation must be valid before filtering.")
    if observation.information_id in state.information_ids:
        raise ValueError("Absolute observation has already been used.")
    measurement = np.asarray(observation.measurement_eci, dtype=float).reshape(3)
    measurement_covariance = np.asarray(
        observation.covariance, dtype=float
    ).reshape(3, 3)
    if not 0.0 < float(observation.confidence) <= 1.0:
        raise ValueError("Observation confidence must be in (0, 1].")
    measurement_covariance = measurement_covariance / float(
        observation.confidence
    )
    jacobian = np.zeros((3, state.dimension), dtype=float)
    jacobian[:, :3] = np.eye(3)
    innovation = measurement - state.active_state[:3]
    predicted_measurement_covariance = (
        jacobian @ state.joint_covariance @ jacobian.T
    )
    policy = integrity_policy or _policy_from_legacy_integrity_parameters(
        nis_gate_threshold=nis_gate_threshold,
        nis_inflation_threshold=nis_inflation_threshold,
        maximum_measurement_covariance_scale=(
            maximum_measurement_covariance_scale
        ),
    )
    evaluation = evaluate_measurement_integrity(
        innovation=innovation,
        predicted_measurement_covariance=predicted_measurement_covariance,
        measurement_covariance=measurement_covariance,
        policy=policy, regularization=regularization,
    )
    measurement_covariance = evaluation.effective_measurement_covariance
    innovation_covariance = evaluation.innovation_covariance
    raw_nis = float(evaluation.diagnostics.raw_nis)
    nis = float(evaluation.diagnostics.processed_nis)
    covariance_scale = evaluation.diagnostics.measurement_covariance_scale
    if evaluation.skipped:
        return MultiNeighborSchmidtUpdateResult(
            state, innovation, innovation_covariance, nis, skipped=True,
            raw_nis=raw_nis,
            measurement_covariance_scale=covariance_scale,
        )
    gain = np.zeros((state.dimension, 3), dtype=float)
    gain[:6] = (
        state.joint_covariance[:6] @ jacobian.T
    ) @ np.linalg.pinv(innovation_covariance)
    residual = np.eye(state.dimension) - gain @ jacobian
    covariance = (
        residual @ state.joint_covariance @ residual.T
        + gain @ measurement_covariance @ gain.T
    )
    updated = replace(
        state,
        active_state=state.active_state + gain[:6] @ innovation,
        joint_covariance=_symmetrize(covariance),
        information_ids=(*state.information_ids, observation.information_id),
    )
    return MultiNeighborSchmidtUpdateResult(
        updated, innovation, innovation_covariance, nis,
        raw_nis=raw_nis,
        measurement_covariance_scale=covariance_scale,
    )


def add_consider_neighbor(
    state: MultiNeighborSchmidtState, *, neighbor_id: str, neighbor_state: Array,
    neighbor_covariance: Array, active_cross_covariance: Array | None = None,
    cross_covariance_by_neighbor: Mapping[str, Array] | None = None,
) -> MultiNeighborSchmidtState:
    neighbor_id = str(neighbor_id)
    if neighbor_id == state.active_node_id or neighbor_id in state.neighbor_ids:
        raise ValueError("Neighbor ID is already represented by this Schmidt state.")
    old_dimension = state.dimension
    covariance = np.zeros((old_dimension + 6, old_dimension + 6), dtype=float)
    covariance[:old_dimension, :old_dimension] = state.joint_covariance
    covariance[old_dimension:, old_dimension:] = np.asarray(neighbor_covariance, dtype=float).reshape(6, 6)
    active_cross = np.zeros((6, 6)) if active_cross_covariance is None else np.asarray(active_cross_covariance, dtype=float).reshape(6, 6)
    covariance[:6, old_dimension:] = active_cross
    covariance[old_dimension:, :6] = active_cross.T
    for existing_id, cross in (cross_covariance_by_neighbor or {}).items():
        block = state.neighbor_slice(existing_id)
        matrix = np.asarray(cross, dtype=float).reshape(6, 6)
        covariance[block, old_dimension:] = matrix
        covariance[old_dimension:, block] = matrix.T
    return MultiNeighborSchmidtState(
        timestamp=state.timestamp, active_node_id=state.active_node_id,
        neighbor_ids=(*state.neighbor_ids, neighbor_id), active_state=state.active_state.copy(),
        neighbor_state_by_id={**state.neighbor_state_by_id, neighbor_id: np.asarray(neighbor_state, dtype=float).reshape(6).copy()},
        joint_covariance=_symmetrize(covariance), information_ids=state.information_ids,
        transport_information_ids=state.transport_information_ids,
    )


def remove_consider_neighbor(state: MultiNeighborSchmidtState, neighbor_id: str) -> MultiNeighborSchmidtState:
    block = state.neighbor_slice(neighbor_id)
    keep = np.ones(state.dimension, dtype=bool)
    keep[block] = False
    neighbor_id = str(neighbor_id)
    return MultiNeighborSchmidtState(
        timestamp=state.timestamp, active_node_id=state.active_node_id,
        neighbor_ids=tuple(node_id for node_id in state.neighbor_ids if node_id != neighbor_id),
        active_state=state.active_state.copy(),
        neighbor_state_by_id={node_id: vector.copy() for node_id, vector in state.neighbor_state_by_id.items() if node_id != neighbor_id},
        joint_covariance=state.joint_covariance[np.ix_(keep, keep)].copy(), information_ids=state.information_ids,
        transport_information_ids=state.transport_information_ids,
    )


def run_multi_neighbor_schmidt_history(
    *, timestamps: Array, initial_state: MultiNeighborSchmidtState,
    observations: Iterable[ObservationMessage], process_noise_acceleration: float = 1e-4,
) -> MultiNeighborSchmidtHistory:
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0 or not np.isclose(times[0], initial_state.timestamp):
        raise ValueError("timestamps must start at the initial state timestamp.")
    grouped = {float(timestamp): [] for timestamp in times}
    for observation in observations:
        if float(observation.timestamp) not in grouped:
            raise ValueError("Observation timestamp is not in timestamps.")
        grouped[float(observation.timestamp)].append(observation)
    active_states = np.zeros((times.size, 6)); active_covariances = np.zeros((times.size, 6, 6))
    neighbor_states = {node_id: np.zeros((times.size, 6)) for node_id in initial_state.neighbor_ids}
    joint_covariances = np.zeros((times.size, initial_state.dimension, initial_state.dimension))
    nis_history = []; state = initial_state
    for index, timestamp in enumerate(times):
        if index > 0:
            state = multi_neighbor_schmidt_predict(state, float(timestamp), process_noise_acceleration=process_noise_acceleration)
        epoch_nis = {}
        for observation in sorted(grouped[float(timestamp)], key=lambda item: item.information_id):
            update = multi_neighbor_schmidt_update(state, observation); state = update.state
            epoch_nis[observation.information_id] = update.nis
        active_states[index] = state.active_state; active_covariances[index] = state.active_covariance
        for node_id in state.neighbor_ids: neighbor_states[node_id][index] = state.neighbor_state_by_id[node_id]
        joint_covariances[index] = state.joint_covariance; nis_history.append(epoch_nis)
    return MultiNeighborSchmidtHistory(times.copy(), active_states, active_covariances, neighbor_states, joint_covariances, nis_history)


def _policy_from_legacy_integrity_parameters(
    *, nis_gate_threshold: float | None,
    nis_inflation_threshold: float | None,
    maximum_measurement_covariance_scale: float,
) -> MeasurementIntegrityPolicy:
    if nis_inflation_threshold is not None and nis_gate_threshold is not None:
        mode = INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE
    elif nis_inflation_threshold is not None:
        mode = INTEGRITY_MODE_PROPORTIONAL_INFLATION
    elif nis_gate_threshold is not None:
        mode = INTEGRITY_MODE_HARD_GATE
    else:
        mode = INTEGRITY_MODE_NONE
    return MeasurementIntegrityPolicy(
        mode=mode,
        inflation_threshold=nis_inflation_threshold,
        maximum_covariance_scale=float(
            maximum_measurement_covariance_scale
        ),
        hard_gate_threshold=nis_gate_threshold,
    )


def _block_diag(matrices: list[Array]) -> Array:
    dimension = sum(matrix.shape[0] for matrix in matrices); result = np.zeros((dimension, dimension)); offset = 0
    for matrix in matrices:
        size = matrix.shape[0]; result[offset:offset + size, offset:offset + size] = matrix; offset += size
    return result


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)
