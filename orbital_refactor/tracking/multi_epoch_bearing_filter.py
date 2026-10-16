"""Experimental fixed-window estimator for past multi-observer LOS data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from tracking.multi_observer_bearing_filter import (
    MultiObserverBearingUpdate,
    _regularize_covariance,
    _tangent_basis,
)


@dataclass(frozen=True)
class _WhitenedLine:
    timestamp: float
    observer_state_eci: np.ndarray
    measured_line_eci: np.ndarray
    tangent: np.ndarray
    covariance_cholesky: np.ndarray
    observer_id: str
    measurement_id: str | None
    source_modality: str


class MultiEpochBearingWindow:
    """Re-linearize a short history of LOS data around one anchor state.

    This is an experimental validation baseline.  It uses only past and
    current measurements, so it remains causal, but its nonlinear solve is
    intentionally separate from the default real-time EKF path.
    """

    def __init__(
        self,
        prior,
        *,
        maximum_epochs: int,
        process_noise_acceleration: float = 1e-4,
        maximum_function_evaluations: int = 10,
    ):
        if int(maximum_epochs) < 2:
            raise ValueError("maximum_epochs must be at least two.")
        self.maximum_epochs = int(maximum_epochs)
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.maximum_function_evaluations = int(maximum_function_evaluations)
        self._set_anchor(prior)
        self._epochs: list[tuple[float, tuple[_WhitenedLine, ...]]] = []

    @property
    def epoch_count(self) -> int:
        return len(self._epochs)

    def reset(self, prior) -> None:
        self._set_anchor(prior)
        self._epochs.clear()

    def append(self, observations, *, timestamp: float, restart_prior=None) -> None:
        epoch = float(timestamp)
        if self._epochs and epoch <= self._epochs[-1][0]:
            raise ValueError("Bearing-window epochs must be strictly increasing.")
        if len(self._epochs) >= self.maximum_epochs:
            if restart_prior is None:
                raise ValueError("A full bearing window requires a restart prior.")
            self.reset(restart_prior)
        lines = tuple(_whitened_line(item) for item in observations)
        if len({item.observer_id for item in lines}) < 2:
            raise ValueError("Each bearing epoch requires at least two observers.")
        self._epochs.append((epoch, lines))

    def estimate(self) -> MultiObserverBearingUpdate:
        if not self._epochs:
            raise ValueError("The bearing window is empty.")
        scale = _covariance_square_root(self._anchor_covariance)

        def state_from_scaled(value):
            return self._anchor_state + scale @ np.asarray(value, dtype=float)

        def residual(value):
            anchor_state = state_from_scaled(value)
            parts = [np.asarray(value, dtype=float)]
            for timestamp, lines in self._epochs:
                state = _propagate(anchor_state, timestamp - self._anchor_timestamp)
                for line in lines:
                    relative = state[:3] - line.observer_state_eci[:3]
                    distance = float(np.linalg.norm(relative))
                    if distance <= 0.0:
                        raise ValueError("Observer and target positions must differ.")
                    predicted = relative / distance
                    raw = line.tangent @ (line.measured_line_eci - predicted)
                    parts.append(np.linalg.solve(line.covariance_cholesky, raw))
            return np.concatenate(parts)

        solution = least_squares(
            residual,
            np.zeros(6),
            method="trf",
            bounds=(-5.0, 5.0),
            x_scale="jac",
            max_nfev=self.maximum_function_evaluations,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        optimized_anchor = state_from_scaled(solution.x)
        final_timestamp = self._epochs[-1][0]
        elapsed = final_timestamp - self._anchor_timestamp
        final_state = _propagate(optimized_anchor, elapsed)
        scaled_covariance = np.linalg.pinv(solution.jac.T @ solution.jac)
        anchor_covariance = scale @ scaled_covariance @ scale.T
        transition = numerical_jacobian_discrete(
            lambda state: _propagate(state, elapsed), optimized_anchor,
        )
        final_covariance = (
            transition @ anchor_covariance @ transition.T
            + make_process_noise(elapsed, self.process_noise_acceleration)
        )
        final_covariance = 0.5 * (final_covariance + final_covariance.T)
        all_lines = tuple(
            line for _timestamp, lines in self._epochs for line in lines
        )
        modality_counts: dict[str, int] = {}
        for line in all_lines:
            modality_counts[line.source_modality] = (
                modality_counts.get(line.source_modality, 0) + 1
            )
        measurement_residual = residual(solution.x)[6:]
        return MultiObserverBearingUpdate(
            timestamp=final_timestamp,
            state_eci=final_state,
            covariance_eci=final_covariance,
            observer_ids=tuple(sorted({line.observer_id for line in all_lines})),
            measurement_ids=tuple(dict.fromkeys(
                line.measurement_id
                for line in all_lines
                if line.measurement_id is not None
            )),
            modality_counts=dict(sorted(modality_counts.items())),
            measurement_count=len(all_lines),
            nis=float(measurement_residual @ measurement_residual),
        )

    def _set_anchor(self, prior) -> None:
        self._anchor_timestamp = float(prior.timestamp)
        self._anchor_state = np.asarray(prior.state_eci, dtype=float).reshape(6).copy()
        self._anchor_covariance = np.asarray(
            prior.covariance_eci, dtype=float,
        ).reshape(6, 6).copy()


def _whitened_line(item) -> _WhitenedLine:
    measured = np.asarray(item.measurement, dtype=float).reshape(3)
    tangent = np.vstack(_tangent_basis(measured))
    line_covariance = item.metadata.get("lineOfSightCovarianceEci")
    if line_covariance is None:
        covariance = np.asarray(item.covariance, dtype=float)
    else:
        covariance = (
            tangent
            @ np.asarray(line_covariance, dtype=float).reshape(3, 3)
            @ tangent.T
        )
    covariance = _regularize_covariance(covariance)
    return _WhitenedLine(
        timestamp=float(item.timestamp),
        observer_state_eci=np.asarray(item.observer_state_eci, dtype=float).copy(),
        measured_line_eci=measured.copy(),
        tangent=tangent,
        covariance_cholesky=np.linalg.cholesky(covariance),
        observer_id=str(item.observer_id),
        measurement_id=(
            None if item.metadata.get("sourceMessageId") is None
            else str(item.metadata["sourceMessageId"])
        ),
        source_modality=str(
            item.metadata.get("sourceModality", "LOS")
        ).upper(),
    )


def _covariance_square_root(covariance):
    covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = np.finfo(float).eps * max(float(np.max(np.abs(eigenvalues))), 1.0)
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, floor)))


def _propagate(state, elapsed):
    if np.isclose(elapsed, 0.0):
        return np.asarray(state, dtype=float).copy()
    if elapsed < 0.0:
        raise ValueError("Bearing measurements cannot precede the anchor prior.")
    return rk4_step_absolute(state, float(elapsed))
