"""Causal fixed-lag batch estimator for radar and multi-observer LOS data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from tracking.data_contracts import TargetInitialState
from tracking.multi_observer_multimodal_filter import (
    UnifiedMultimodalUpdate,
    _measurement_block,
    unified_multi_observer_multimodal_update,
)


@dataclass(frozen=True)
class _MeasurementEpoch:
    timestamp: float
    observations: tuple


class FixedLagMultimodalWindow:
    """Jointly estimate a short Markov state window without reusing factors.

    The first state has one Gaussian prior. Consecutive states are connected
    by two-body/J2 process factors and every radar/LOS observation appears in
    exactly one epoch factor. When the window slides, the oldest epoch is
    absorbed by one sequential update and then removed before later factors
    are solved again.
    """

    def __init__(
        self,
        prior,
        *,
        maximum_epochs: int,
        process_noise_acceleration: float = 1e-4,
        maximum_iterations: int = 4,
    ):
        if int(maximum_epochs) < 2:
            raise ValueError("maximum_epochs must be at least two.")
        if int(maximum_iterations) < 1:
            raise ValueError("maximum_iterations must be positive.")
        self.maximum_epochs = int(maximum_epochs)
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.maximum_iterations = int(maximum_iterations)
        self._anchor = _copy_prior(prior)
        self._epochs: list[_MeasurementEpoch] = []
        self._information_ids = list(getattr(prior, "information_ids", ()))

    @property
    def epoch_count(self) -> int:
        return len(self._epochs)

    @property
    def anchor_timestamp(self) -> float:
        return float(self._anchor.timestamp)

    def append(self, observations, *, timestamp: float) -> None:
        epoch = float(timestamp)
        values = tuple(observations)
        if not values:
            raise ValueError("A fixed-lag epoch requires observations.")
        if epoch <= self._anchor.timestamp:
            raise ValueError("Window observations must follow the anchor prior.")
        if self._epochs and epoch <= self._epochs[-1].timestamp:
            raise ValueError("Window epochs must be strictly increasing.")
        while len(self._epochs) >= self.maximum_epochs:
            self._marginalize_oldest_epoch()
        self._epochs.append(_MeasurementEpoch(epoch, values))
        for item in values:
            identifier = item.metadata.get("sourceMessageId")
            if identifier is not None:
                self._information_ids.append(str(identifier))

    def estimate(self) -> UnifiedMultimodalUpdate:
        if not self._epochs:
            raise ValueError("The multimodal window is empty.")
        timestamps = [float(self._anchor.timestamp)] + [
            item.timestamp for item in self._epochs
        ]
        states = [np.asarray(self._anchor.state_eci, dtype=float).copy()]
        transitions = []
        process_covariances = []
        for previous, current in zip(timestamps[:-1], timestamps[1:], strict=True):
            dt = current - previous
            propagate = lambda state, step=dt: rk4_step_absolute(state, step)
            transitions.append(numerical_jacobian_discrete(propagate, states[-1]))
            process_covariances.append(
                make_process_noise(dt, self.process_noise_acceleration)
            )
            states.append(propagate(states[-1]))
        state_vector = np.concatenate(states)
        for _iteration in range(self.maximum_iterations):
            residual, jacobian = self._linearized_system(
                state_vector,
                transitions=transitions,
                process_covariances=process_covariances,
            )
            correction = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            state_vector = state_vector + correction
            if np.linalg.norm(correction) <= 1e-7 * max(
                np.linalg.norm(state_vector), 1.0,
            ):
                break
        residual, jacobian = self._linearized_system(
            state_vector,
            transitions=transitions,
            process_covariances=process_covariances,
        )
        covariance = np.linalg.pinv(jacobian.T @ jacobian)
        final_slice = slice(len(self._epochs) * 6, (len(self._epochs) + 1) * 6)
        final_covariance = covariance[final_slice, final_slice]
        final_covariance = 0.5 * (final_covariance + final_covariance.T)
        all_observations = tuple(
            item
            for epoch in self._epochs
            for item in epoch.observations
        )
        modality_counts: dict[str, int] = {}
        measurement_dimension = 0
        for item in all_observations:
            modality = str(
                item.metadata.get("sourceModality", item.modality)
            ).upper()
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            measurement_dimension += 2
        return UnifiedMultimodalUpdate(
            timestamp=self._epochs[-1].timestamp,
            state_eci=state_vector[final_slice].copy(),
            covariance_eci=final_covariance,
            observer_ids=tuple(sorted({
                item.observer_id for item in all_observations
            })),
            measurement_ids=tuple(dict.fromkeys(self._information_ids)),
            modality_counts=dict(sorted(modality_counts.items())),
            observation_count=len(all_observations),
            measurement_dimension=measurement_dimension,
            nis=float(residual @ residual),
        )

    def _linearized_system(
        self, state_vector, *, transitions, process_covariances,
    ):
        state_count = len(self._epochs) + 1
        dimension = state_count * 6
        residuals = []
        jacobians = []

        prior_root = _covariance_cholesky(self._anchor.covariance_eci)
        prior_row = np.zeros((6, dimension))
        prior_row[:, :6] = np.linalg.solve(prior_root, np.eye(6))
        residuals.append(np.linalg.solve(
            prior_root,
            state_vector[:6] - self._anchor.state_eci,
        ))
        jacobians.append(prior_row)

        for index, (transition, covariance) in enumerate(zip(
            transitions, process_covariances, strict=True,
        )):
            previous_slice = slice(index * 6, (index + 1) * 6)
            current_slice = slice((index + 1) * 6, (index + 2) * 6)
            previous_state = state_vector[previous_slice]
            current_state = state_vector[current_slice]
            dt = self._epochs[index].timestamp - (
                self._anchor.timestamp
                if index == 0 else self._epochs[index - 1].timestamp
            )
            predicted = rk4_step_absolute(previous_state, dt)
            root = _covariance_cholesky(covariance)
            inverse_root = np.linalg.solve(root, np.eye(6))
            row = np.zeros((6, dimension))
            row[:, previous_slice] = -inverse_root @ transition
            row[:, current_slice] = inverse_root
            residuals.append(inverse_root @ (current_state - predicted))
            jacobians.append(row)

        for index, epoch in enumerate(self._epochs, start=1):
            state_slice = slice(index * 6, (index + 1) * 6)
            state = state_vector[state_slice]
            for item in epoch.observations:
                measurement_jacobian, innovation, covariance = (
                    _measurement_block(item, state)
                )
                root = _covariance_cholesky(covariance)
                inverse_root = np.linalg.solve(root, np.eye(2))
                row = np.zeros((2, dimension))
                row[:, state_slice] = inverse_root @ measurement_jacobian
                residuals.append(-inverse_root @ innovation)
                jacobians.append(row)
        return np.concatenate(residuals), np.vstack(jacobians)

    def _marginalize_oldest_epoch(self) -> None:
        oldest = self._epochs.pop(0)
        update = unified_multi_observer_multimodal_update(
            self._anchor,
            oldest.observations,
            timestamp=oldest.timestamp,
            process_noise_acceleration=self.process_noise_acceleration,
        )
        self._anchor = TargetInitialState(
            target_id=self._anchor.target_id,
            track_id=self._anchor.track_id,
            timestamp=update.timestamp,
            state_eci=update.state_eci,
            covariance_eci=update.covariance_eci,
        )


def _copy_prior(prior) -> TargetInitialState:
    return TargetInitialState(
        target_id=str(prior.target_id),
        track_id=str(prior.track_id),
        timestamp=float(prior.timestamp),
        state_eci=np.asarray(prior.state_eci, dtype=float).copy(),
        covariance_eci=np.asarray(prior.covariance_eci, dtype=float).copy(),
    )


def _covariance_cholesky(covariance):
    value = np.asarray(covariance, dtype=float)
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    floor = np.finfo(float).eps * max(
        float(np.max(np.abs(eigenvalues))), 1.0,
    )
    regularized = (
        eigenvectors
        @ np.diag(np.maximum(eigenvalues, floor))
        @ eigenvectors.T
    )
    return np.linalg.cholesky(regularized)
