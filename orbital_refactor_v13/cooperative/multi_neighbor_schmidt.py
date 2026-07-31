from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

import numpy as np

from interfaces.data_objects import ObservationMessage
from orbital_core.dynamics import make_process_noise, numerical_jacobian_discrete, rk4_step_absolute
from orbital_core.inter_satellite_model import RelativeMeasurementModel

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
    )


def multi_neighbor_schmidt_update(
    state: MultiNeighborSchmidtState, observation: ObservationMessage, *,
    quaternion_i2b_wxyz: Array | None = None, regularization: float = 1e-9,
) -> MultiNeighborSchmidtUpdateResult:
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
    neighbor_state = state.neighbor_state_by_id[neighbor_id]
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
    innovation = model.residual(measurement, prediction)
    jacobian = np.zeros((measurement.size, state.dimension), dtype=float)
    jacobian[:, :6] = h_active
    jacobian[:, state.neighbor_slice(neighbor_id)] = h_neighbor
    innovation_covariance = jacobian @ state.joint_covariance @ jacobian.T + measurement_covariance + regularization * np.eye(measurement.size)
    gain = np.zeros((state.dimension, measurement.size), dtype=float)
    gain[:6] = (state.joint_covariance[:6] @ jacobian.T) @ np.linalg.pinv(innovation_covariance)
    residual = np.eye(state.dimension) - gain @ jacobian
    covariance = residual @ state.joint_covariance @ residual.T + gain @ measurement_covariance @ gain.T
    updated = replace(state, active_state=state.active_state + gain[:6] @ innovation, joint_covariance=_symmetrize(covariance), information_ids=(*state.information_ids, observation.information_id))
    nis = float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
    return MultiNeighborSchmidtUpdateResult(updated, innovation, innovation_covariance, nis)


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


def _block_diag(matrices: list[Array]) -> Array:
    dimension = sum(matrix.shape[0] for matrix in matrices); result = np.zeros((dimension, dimension)); offset = 0
    for matrix in matrices:
        size = matrix.shape[0]; result[offset:offset + size, offset:offset + size] = matrix; offset += size
    return result


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)
