from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brain_inspired.navigation_place_cells import (
    NavigationPlaceCellConfig,
    NavigationPlaceCellEncoder,
)

Array = np.ndarray


def _coarse_config():
    centers = tuple(np.linspace(-20_000.0, 20_000.0, 9))
    return NavigationPlaceCellConfig(
        radial_centers_m=centers, along_track_centers_m=centers,
        radial_sigma_m=2_500.0, along_track_sigma_m=2_500.0,
    )


def _fine_config():
    centers = (-250.0, -125.0, 0.0, 125.0, 250.0)
    return NavigationPlaceCellConfig(
        radial_centers_m=centers, along_track_centers_m=centers,
        radial_sigma_m=62.5, along_track_sigma_m=62.5,
    )


@dataclass(frozen=True)
class HierarchicalNavigationPlaceCellConfig:
    coarse: NavigationPlaceCellConfig = field(default_factory=_coarse_config)
    fine: NavigationPlaceCellConfig = field(default_factory=_fine_config)
    fine_enter_fraction: float = 0.70
    fine_exit_fraction: float = 0.90

    def validate(self):
        self.coarse.validate()
        self.fine.validate()
        if (not np.isfinite(self.fine_enter_fraction)
                or not np.isfinite(self.fine_exit_fraction)
                or not 0.0 < self.fine_enter_fraction
                < self.fine_exit_fraction < 1.0):
            raise ValueError(
                "Fine-scale hysteresis must satisfy 0 < enter < exit < 1."
            )

    @property
    def cell_count(self):
        return self.coarse.cell_count + self.fine.cell_count


@dataclass(frozen=True)
class HierarchicalNavigationPlaceCellOutput:
    phase: float
    radial_position: float
    along_track_position: float
    decoded_phase: float
    decoded_radial_position: float
    decoded_along_track_position: float
    coarse_decoded_rt: Array
    fine_residual_rt: Array
    fine_decoded_residual_rt: Array
    coarse_activity: Array
    fine_activity: Array
    activity: Array
    fine_scale_active: bool
    scale_transition: bool
    boundary_saturated: bool
    peak_activity: float
    normalized_entropy: float
    valid: bool


class HierarchicalNavigationPlaceCellEncoder:
    """Coarse global place code plus a fine code of its RT residual."""

    def __init__(self, config=None):
        self.config = config or HierarchicalNavigationPlaceCellConfig()
        self.config.validate()
        self._coarse = NavigationPlaceCellEncoder(self.config.coarse)
        self._fine = NavigationPlaceCellEncoder(self.config.fine)
        self._fine_active: bool | None = None

    def reset(self):
        self._fine_active = None

    def encode(self, *, phase, radial_position, along_track_position):
        coarse = self._coarse.encode(
            phase=phase, radial_position=radial_position,
            along_track_position=along_track_position,
        )
        coarse_rt = np.array([
            coarse.decoded_radial_position,
            coarse.decoded_along_track_position,
        ])
        source_rt = np.asarray(
            [radial_position, along_track_position], dtype=float,
        )
        residual = source_rt - coarse_rt
        fine = self._fine.encode(
            phase=phase, radial_position=residual[0],
            along_track_position=residual[1],
        )
        utilization = self._fine_support_utilization(residual)
        previous = self._fine_active
        if previous is None:
            self._fine_active = bool(
                utilization <= self.config.fine_enter_fraction
            )
        elif previous and utilization >= self.config.fine_exit_fraction:
            self._fine_active = False
        elif not previous and utilization <= self.config.fine_enter_fraction:
            self._fine_active = True
        transition = previous is not None and previous != self._fine_active
        fine_decoded = np.array([
            fine.decoded_radial_position,
            fine.decoded_along_track_position,
        ])
        decoded_rt = (
            coarse_rt + fine_decoded if self._fine_active else coarse_rt
        )
        activity = np.concatenate([
            0.5 * coarse.activity, 0.5 * fine.activity,
        ])
        entropy = -float(np.sum(
            activity * np.log(np.maximum(activity, 1.0e-15))
        )) / np.log(activity.size)
        return HierarchicalNavigationPlaceCellOutput(
            phase=coarse.phase, radial_position=float(radial_position),
            along_track_position=float(along_track_position),
            decoded_phase=(fine.decoded_phase if self._fine_active
                           else coarse.decoded_phase),
            decoded_radial_position=float(decoded_rt[0]),
            decoded_along_track_position=float(decoded_rt[1]),
            coarse_decoded_rt=coarse_rt, fine_residual_rt=residual.copy(),
            fine_decoded_residual_rt=fine_decoded,
            coarse_activity=coarse.activity.copy(),
            fine_activity=fine.activity.copy(), activity=activity,
            fine_scale_active=bool(self._fine_active),
            scale_transition=bool(transition),
            boundary_saturated=bool(
                coarse.boundary_saturated
                or (self._fine_active and fine.boundary_saturated)
            ),
            peak_activity=float(np.max(activity)),
            normalized_entropy=float(entropy),
            valid=bool(coarse.valid and fine.valid),
        )

    def _fine_support_utilization(self, residual):
        radial_limit = max(abs(np.asarray(
            self.config.fine.radial_centers_m, dtype=float,
        )))
        along_limit = max(abs(np.asarray(
            self.config.fine.along_track_centers_m, dtype=float,
        )))
        return float(max(
            abs(residual[0]) / radial_limit,
            abs(residual[1]) / along_limit,
        ))
