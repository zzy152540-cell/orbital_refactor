from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from orbital_core.dynamics import (
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from .data_contracts import GlobalTargetEstimate, TargetInitialState, TargetTrack


@dataclass(frozen=True)
class DuplicateTrackConfig:
    maximum_position_separation_m: float = 1000.0
    maximum_velocity_separation_mps: float = 2.0
    maximum_squared_mahalanobis: float = 22.457744484825323

    def __post_init__(self):
        if self.maximum_position_separation_m <= 0.0:
            raise ValueError("maximum_position_separation_m must be positive.")
        if self.maximum_velocity_separation_mps <= 0.0:
            raise ValueError("maximum_velocity_separation_mps must be positive.")
        if self.maximum_squared_mahalanobis <= 0.0:
            raise ValueError("maximum_squared_mahalanobis must be positive.")


@dataclass(frozen=True)
class DuplicateResolution:
    accepted_births: Mapping[str, TargetInitialState]
    aliases: Mapping[str, str]


class DuplicateTrackResolver:
    def __init__(self, config: DuplicateTrackConfig | None = None):
        self.config = config or DuplicateTrackConfig()

    def resolve(
        self,
        births: Mapping[str, TargetInitialState],
        existing_tracks: Mapping[str, TargetTrack],
    ) -> DuplicateResolution:
        accepted = {}
        aliases = {}
        active = {
            target_id: track.estimate
            for target_id, track in existing_tracks.items()
            if track.lifecycle.value != "TERMINATED"
        }
        for target_id, birth in sorted(births.items()):
            duplicate = self._best_duplicate(birth, active)
            if duplicate is None:
                accepted[target_id] = birth
                active[target_id] = birth
            else:
                aliases[target_id] = duplicate
        return DuplicateResolution(accepted_births=accepted, aliases=aliases)

    def _best_duplicate(self, birth, active):
        candidates = []
        for target_id, estimate in active.items():
            distance = _state_distance(birth, estimate)
            if distance is None:
                continue
            position, velocity, mahalanobis = distance
            if (
                position <= self.config.maximum_position_separation_m
                and velocity <= self.config.maximum_velocity_separation_mps
                and mahalanobis <= self.config.maximum_squared_mahalanobis
            ):
                candidates.append((mahalanobis, position, target_id))
        return min(candidates)[2] if candidates else None


def _state_distance(
    first: TargetInitialState,
    second: TargetInitialState | GlobalTargetEstimate,
):
    epoch = max(first.timestamp, second.timestamp)
    first_state, first_covariance = _at_epoch(first, epoch)
    second_state, second_covariance = _at_epoch(second, epoch)
    difference = first_state - second_state
    covariance = first_covariance + second_covariance + np.eye(6) * 1e-9
    try:
        mahalanobis = float(difference @ np.linalg.solve(covariance, difference))
    except np.linalg.LinAlgError:
        return None
    return (
        float(np.linalg.norm(difference[:3])),
        float(np.linalg.norm(difference[3:])),
        mahalanobis,
    )


def _at_epoch(value, epoch):
    dt = float(epoch - value.timestamp)
    if dt == 0.0:
        return np.asarray(value.state_eci), np.asarray(value.covariance_eci)
    propagate = lambda state: rk4_step_absolute(state, dt)
    transition = numerical_jacobian_discrete(propagate, value.state_eci)
    return (
        propagate(value.state_eci),
        transition @ value.covariance_eci @ transition.T,
    )
