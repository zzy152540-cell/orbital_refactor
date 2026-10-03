from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from .data_contracts import GlobalTargetEstimate, TargetInitialState


@dataclass(frozen=True)
class ManeuverDetectionConfig:
    nis_threshold: float = 22.457744484825323
    confirmation_epochs: int = 2
    clearing_epochs: int = 3
    adaptive_process_noise_scale: float = 25.0
    process_noise_acceleration: float = 1e-4

    def __post_init__(self):
        if self.nis_threshold <= 0.0:
            raise ValueError("nis_threshold must be positive.")
        if self.confirmation_epochs < 1 or self.clearing_epochs < 1:
            raise ValueError("Maneuver confirmation and clearing epochs must be positive.")
        if self.adaptive_process_noise_scale < 1.0:
            raise ValueError("adaptive_process_noise_scale must be at least one.")
        if self.process_noise_acceleration < 0.0:
            raise ValueError("process_noise_acceleration must be non-negative.")


@dataclass(frozen=True)
class ManeuverAssessment:
    target_id: str
    timestamp: float
    normalized_innovation_squared: float
    threshold_exceeded: bool
    suspected: bool
    process_noise_scale: float
    consecutive_exceedances: int
    consecutive_clear_epochs: int


@dataclass
class _DetectionState:
    exceedances: int = 0
    clear_epochs: int = 0
    suspected: bool = False


class ManeuverDetector:
    """Detect persistent dynamic-model mismatch from absolute-state innovations."""

    def __init__(self, config: ManeuverDetectionConfig | None = None):
        self.config = config or ManeuverDetectionConfig()
        self._state_by_target: dict[str, _DetectionState] = {}

    def assess(
        self,
        prior: GlobalTargetEstimate | TargetInitialState,
        posterior: GlobalTargetEstimate,
    ) -> ManeuverAssessment:
        if prior.target_id != posterior.target_id:
            raise ValueError("Maneuver assessment requires the same target ID.")
        dt = float(posterior.timestamp - prior.timestamp)
        if dt <= 0.0:
            raise ValueError("Maneuver assessment requires a later posterior epoch.")
        propagate = lambda state: rk4_step_absolute(state, dt)
        predicted_state = propagate(prior.state_eci)
        transition = numerical_jacobian_discrete(propagate, prior.state_eci)
        predicted_covariance = (
            transition @ prior.covariance_eci @ transition.T
            + make_process_noise(dt, self.config.process_noise_acceleration)
        )
        innovation = np.asarray(posterior.state_eci) - predicted_state
        innovation_covariance = (
            predicted_covariance + np.asarray(posterior.covariance_eci)
        )
        innovation_covariance = 0.5 * (
            innovation_covariance + innovation_covariance.T
        ) + np.eye(6) * 1e-9
        try:
            nis = float(
                innovation @ np.linalg.solve(innovation_covariance, innovation)
            )
        except np.linalg.LinAlgError:
            nis = float("inf")
        exceeded = bool(nis > self.config.nis_threshold)
        state = self._state_by_target.setdefault(prior.target_id, _DetectionState())
        if exceeded:
            state.exceedances += 1
            state.clear_epochs = 0
            if state.exceedances >= self.config.confirmation_epochs:
                state.suspected = True
        else:
            state.exceedances = 0
            if state.suspected:
                state.clear_epochs += 1
                if state.clear_epochs >= self.config.clearing_epochs:
                    state.suspected = False
                    state.clear_epochs = 0
            else:
                state.clear_epochs = 0
        return ManeuverAssessment(
            target_id=prior.target_id,
            timestamp=posterior.timestamp,
            normalized_innovation_squared=nis,
            threshold_exceeded=exceeded,
            suspected=state.suspected,
            process_noise_scale=self.process_noise_scale(prior.target_id),
            consecutive_exceedances=state.exceedances,
            consecutive_clear_epochs=state.clear_epochs,
        )

    def process_noise_scale(self, target_id: str) -> float:
        state = self._state_by_target.get(str(target_id))
        return (
            self.config.adaptive_process_noise_scale
            if state is not None and state.suspected
            else 1.0
        )

