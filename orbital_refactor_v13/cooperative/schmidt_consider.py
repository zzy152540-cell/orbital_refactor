from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from interfaces.data_objects import ObservationMessage
from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from orbital_core.inter_satellite_model import RelativeMeasurementModel

Array = np.ndarray


@dataclass(frozen=True)
class SchmidtState:
    timestamp: float
    active_node_id: str
    consider_node_id: str
    active_state: Array
    consider_state: Array
    active_covariance: Array
    consider_covariance: Array
    cross_covariance: Array
    information_ids: tuple[str, ...] = ()

    @property
    def augmented_covariance(self) -> Array:
        return np.block(
            [
                [self.active_covariance, self.cross_covariance],
                [self.cross_covariance.T, self.consider_covariance],
            ]
        )


@dataclass(frozen=True)
class SchmidtUpdateResult:
    state: SchmidtState
    innovation: Array
    innovation_covariance: Array
    nis: float


@dataclass(frozen=True)
class SchmidtHistory:
    timestamps: Array
    active_state_history: Array
    consider_state_history: Array
    active_covariance_history: Array
    consider_covariance_history: Array
    cross_covariance_history: Array
    nis_history: list[dict[str, float]]


def schmidt_predict(
    state: SchmidtState,
    target_timestamp: float,
    *,
    process_noise_acceleration: float = 1e-4,
) -> SchmidtState:
    dt = float(target_timestamp) - float(state.timestamp)
    if dt <= 0.0:
        raise ValueError("target_timestamp must be later than the state timestamp.")
    active_transition = numerical_jacobian_discrete(
        lambda value: rk4_step_absolute(value, dt),
        state.active_state,
    )
    consider_transition = numerical_jacobian_discrete(
        lambda value: rk4_step_absolute(value, dt),
        state.consider_state,
    )
    process_noise = make_process_noise(dt, process_noise_acceleration)
    active_covariance = (
        active_transition @ state.active_covariance @ active_transition.T
        + process_noise
    )
    consider_covariance = (
        consider_transition @ state.consider_covariance @ consider_transition.T
        + process_noise
    )
    cross_covariance = (
        active_transition @ state.cross_covariance @ consider_transition.T
    )
    return SchmidtState(
        timestamp=float(target_timestamp),
        active_node_id=state.active_node_id,
        consider_node_id=state.consider_node_id,
        active_state=rk4_step_absolute(state.active_state, dt),
        consider_state=rk4_step_absolute(state.consider_state, dt),
        active_covariance=_symmetrize(active_covariance),
        consider_covariance=_symmetrize(consider_covariance),
        cross_covariance=cross_covariance,
        information_ids=state.information_ids,
    )


def schmidt_update(
    state: SchmidtState,
    observation: ObservationMessage,
    *,
    regularization: float = 1e-9,
) -> SchmidtUpdateResult:
    if not np.isclose(float(observation.timestamp), float(state.timestamp)):
        raise ValueError("Observation and Schmidt state timestamps must match.")
    if observation.information_id in state.information_ids:
        raise ValueError("Observation has already been used by this Schmidt state.")
    endpoints = {str(observation.observer_id), str(observation.target_id)}
    if endpoints != {state.active_node_id, state.consider_node_id}:
        raise ValueError("Observation endpoints do not match Schmidt state nodes.")
    if not observation.valid_flag:
        raise ValueError("Observation must be valid.")

    active_is_observer = str(observation.observer_id) == state.active_node_id
    observer_state = state.active_state if active_is_observer else state.consider_state
    target_state = state.consider_state if active_is_observer else state.active_state
    model = RelativeMeasurementModel(observation.modality, observation.frame)
    prediction = model.predict(observer_state, target_state)
    observer_jacobian, target_jacobian = model.jacobians(
        observer_state,
        target_state,
    )
    h_active = observer_jacobian if active_is_observer else target_jacobian
    h_consider = target_jacobian if active_is_observer else observer_jacobian
    measurement = np.asarray(observation.measurement, dtype=float).reshape(-1)
    covariance = np.asarray(observation.covariance, dtype=float)
    if covariance.shape != (measurement.size, measurement.size):
        raise ValueError("Observation covariance has incompatible dimensions.")
    covariance = covariance / float(observation.confidence)
    innovation = model.residual(measurement, prediction)
    h_augmented = np.hstack([h_active, h_consider])
    augmented_covariance = state.augmented_covariance
    innovation_covariance = (
        h_augmented @ augmented_covariance @ h_augmented.T
        + covariance
        + regularization * np.eye(measurement.size)
    )
    active_cross_measurement = (
        state.active_covariance @ h_active.T
        + state.cross_covariance @ h_consider.T
    )
    active_gain = active_cross_measurement @ np.linalg.pinv(
        innovation_covariance
    )
    augmented_gain = np.vstack([active_gain, np.zeros((6, measurement.size))])
    residual = np.eye(12) - augmented_gain @ h_augmented
    updated_augmented_covariance = (
        residual @ augmented_covariance @ residual.T
        + augmented_gain @ covariance @ augmented_gain.T
    )
    updated_augmented_covariance = _symmetrize(updated_augmented_covariance)
    updated = SchmidtState(
        timestamp=state.timestamp,
        active_node_id=state.active_node_id,
        consider_node_id=state.consider_node_id,
        active_state=state.active_state + active_gain @ innovation,
        consider_state=state.consider_state.copy(),
        active_covariance=updated_augmented_covariance[:6, :6].copy(),
        consider_covariance=updated_augmented_covariance[6:, 6:].copy(),
        cross_covariance=updated_augmented_covariance[:6, 6:].copy(),
        information_ids=(*state.information_ids, observation.information_id),
    )
    nis = float(
        innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation
    )
    return SchmidtUpdateResult(updated, innovation, innovation_covariance, nis)


def run_schmidt_consider_history(
    *,
    timestamps: Array,
    active_node_id: str,
    consider_node_id: str,
    initial_active_state: Array,
    initial_consider_state: Array,
    initial_active_covariance: Array,
    initial_consider_covariance: Array,
    observations: Iterable[ObservationMessage],
    initial_cross_covariance: Array | None = None,
    process_noise_acceleration: float = 1e-4,
) -> SchmidtHistory:
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0 or (times.size > 1 and not np.all(np.diff(times) > 0.0)):
        raise ValueError("timestamps must be nonempty and strictly increasing.")
    grouped = {float(timestamp): [] for timestamp in times}
    for observation in observations:
        if float(observation.timestamp) not in grouped:
            raise ValueError("Observation timestamp is not in timestamps.")
        grouped[float(observation.timestamp)].append(observation)
    state = SchmidtState(
        timestamp=float(times[0]),
        active_node_id=str(active_node_id),
        consider_node_id=str(consider_node_id),
        active_state=np.asarray(initial_active_state, dtype=float).reshape(6),
        consider_state=np.asarray(initial_consider_state, dtype=float).reshape(6),
        active_covariance=np.asarray(initial_active_covariance, dtype=float).reshape(6, 6),
        consider_covariance=np.asarray(initial_consider_covariance, dtype=float).reshape(6, 6),
        cross_covariance=(
            np.zeros((6, 6), dtype=float)
            if initial_cross_covariance is None
            else np.asarray(initial_cross_covariance, dtype=float).reshape(6, 6)
        ),
    )
    active_states = np.zeros((times.size, 6), dtype=float)
    consider_states = np.zeros((times.size, 6), dtype=float)
    active_covariances = np.zeros((times.size, 6, 6), dtype=float)
    consider_covariances = np.zeros((times.size, 6, 6), dtype=float)
    cross_covariances = np.zeros((times.size, 6, 6), dtype=float)
    nis_history: list[dict[str, float]] = []
    for index, timestamp in enumerate(times):
        if index > 0:
            state = schmidt_predict(
                state,
                float(timestamp),
                process_noise_acceleration=process_noise_acceleration,
            )
        epoch_nis = {}
        for observation in sorted(
            grouped[float(timestamp)], key=lambda item: item.information_id
        ):
            update = schmidt_update(state, observation)
            state = update.state
            epoch_nis[observation.information_id] = update.nis
        active_states[index] = state.active_state
        consider_states[index] = state.consider_state
        active_covariances[index] = state.active_covariance
        consider_covariances[index] = state.consider_covariance
        cross_covariances[index] = state.cross_covariance
        nis_history.append(epoch_nis)
    return SchmidtHistory(
        timestamps=times.copy(),
        active_state_history=active_states,
        consider_state_history=consider_states,
        active_covariance_history=active_covariances,
        consider_covariance_history=consider_covariances,
        cross_covariance_history=cross_covariances,
        nis_history=nis_history,
    )


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)
