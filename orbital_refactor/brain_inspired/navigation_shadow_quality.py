from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.orbital_direction_runner import OrbitalDirectionHistory
from brain_inspired.orbital_radial_runner import OrbitalRadialHistory

Array = np.ndarray


@dataclass(frozen=True)
class NavigationShadowQualityConfig:
    direction_residual_scale: float = np.deg2rad(0.1)
    radial_residual_scale: float = 10.0
    anchor_age_scale: float = 120.0
    minimum_quality: float = 0.05

    def validate(self) -> None:
        positive = (
            self.direction_residual_scale, self.radial_residual_scale,
            self.anchor_age_scale,
        )
        if np.any(~np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("Shadow-quality scales must be finite and positive.")
        if (not np.isfinite(self.minimum_quality)
                or not 0.0 <= self.minimum_quality <= 1.0):
            raise ValueError("minimum_quality must lie in [0, 1].")


@dataclass(frozen=True)
class NavigationShadowQualityHistory:
    node_id: str
    timestamps: Array
    direction_discrepancy: Array
    radial_discrepancy: Array
    consistency_quality: Array
    freshness_quality: Array
    shadow_quality: Array
    feedback_quality: Array
    valid: Array
    boundary_saturated: Array


def build_navigation_shadow_quality(
    *, direction_by_node: dict[str, OrbitalDirectionHistory],
    radial_by_node: dict[str, OrbitalRadialHistory],
    config: NavigationShadowQualityConfig | None = None,
) -> dict[str, NavigationShadowQualityHistory]:
    """Build diagnostic quality and an explicitly one-epoch-delayed signal.

    ``shadow_quality[k]`` may inspect the posterior at epoch k and is diagnostic
    only. ``feedback_quality[k]`` equals ``shadow_quality[k-1]`` and is the only
    signal eligible for a future update at epoch k.
    """
    settings = config or NavigationShadowQualityConfig()
    settings.validate()
    if set(direction_by_node) != set(radial_by_node):
        raise ValueError("Direction and radial histories must cover identical nodes.")
    result = {}
    for node_id in direction_by_node:
        direction = direction_by_node[node_id]
        radial = radial_by_node[node_id]
        if not np.array_equal(direction.timestamps, radial.timestamps):
            raise ValueError("Direction and radial timestamps must align.")
        d_direction = np.abs(direction.prediction_residual) / (
            settings.direction_residual_scale
        )
        d_radial = np.abs(radial.prediction_residual) / settings.radial_residual_scale
        consistency = np.exp(-0.5 * (d_direction**2 + d_radial**2))
        age = np.maximum(direction.anchor_age, radial.anchor_age)
        freshness = np.exp(-age / settings.anchor_age_scale)
        valid = direction.valid & radial.valid
        saturated = radial.saturated_at_boundary
        quality = np.clip(consistency * freshness,
                          settings.minimum_quality, 1.0)
        quality = np.where(valid & ~saturated, quality, settings.minimum_quality)
        feedback = np.empty_like(quality)
        feedback[0] = 1.0
        feedback[1:] = quality[:-1]
        result[node_id] = NavigationShadowQualityHistory(
            node_id=node_id, timestamps=direction.timestamps.copy(),
            direction_discrepancy=d_direction,
            radial_discrepancy=d_radial,
            consistency_quality=consistency,
            freshness_quality=freshness,
            shadow_quality=quality, feedback_quality=feedback,
            valid=valid.copy(), boundary_saturated=saturated.copy(),
        )
    return result
