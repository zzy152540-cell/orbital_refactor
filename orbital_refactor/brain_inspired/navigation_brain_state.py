from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brain_inspired.navigation_shadow_quality import NavigationShadowQualityHistory
from brain_inspired.orbital_direction_runner import OrbitalDirectionHistory
from brain_inspired.orbital_rt_grid_runner import OrbitalRTGridHistory

Array = np.ndarray


FEATURE_NAMES = (
    "direction_sin", "direction_cos",
    "radial_position_m", "along_track_position_m",
    "direction_bump_concentration",
    "direction_residual_rad", "radial_residual_m", "along_residual_m",
    "anchor_age_s",
    "shadow_quality", "delayed_feedback_quality",
)


@dataclass(frozen=True)
class NavigationBrainStateHistory:
    """Read-only combined neural-navigation representation for one node."""

    node_id: str
    timestamps: Array
    decoded_phase: Array
    decoded_rt: Array
    feature_matrix: Array
    feature_names: tuple[str, ...]
    valid: Array
    boundary_saturated: Array
    anchor_rejected: Array
    reference_rebased: Array
    shadow_quality: Array
    delayed_feedback_quality: Array


def build_navigation_brain_states(
    *, direction_by_node: dict[str, OrbitalDirectionHistory],
    rt_by_node: dict[str, OrbitalRTGridHistory],
    quality_by_node: dict[str, NavigationShadowQualityHistory] | None = None,
) -> dict[str, NavigationBrainStateHistory]:
    """Align direction, RT, and optional quality without estimator feedback."""
    if set(direction_by_node) != set(rt_by_node):
        raise ValueError("Direction and RT histories must cover identical nodes.")
    if quality_by_node is not None and set(quality_by_node) != set(direction_by_node):
        raise ValueError("Quality histories must cover identical nodes.")
    result = {}
    for node, direction in direction_by_node.items():
        rt = rt_by_node[node]
        if not np.array_equal(direction.timestamps, rt.timestamps):
            raise ValueError("Direction and RT timestamps must align.")
        size = direction.timestamps.size
        if quality_by_node is None:
            shadow = np.ones(size)
            delayed = np.ones(size)
            quality_valid = np.ones(size, dtype=bool)
            quality_saturated = np.zeros(size, dtype=bool)
        else:
            quality = quality_by_node[node]
            if not np.array_equal(direction.timestamps, quality.timestamps):
                raise ValueError("Direction and quality timestamps must align.")
            shadow = np.asarray(quality.shadow_quality, dtype=float)
            delayed = np.asarray(quality.feedback_quality, dtype=float)
            quality_valid = np.asarray(quality.valid, dtype=bool)
            quality_saturated = np.asarray(
                quality.boundary_saturated, dtype=bool,
            )
        phase = np.asarray(direction.anchored_phase, dtype=float)
        rt_residual = np.asarray(rt.anchored_residual_rt, dtype=float)
        feature_matrix = np.column_stack([
            np.sin(phase), np.cos(phase), rt.anchored_rt[:, 0],
            rt.anchored_rt[:, 1], direction.bump_concentration,
            direction.anchored_residual,
            rt_residual[:, 0], rt_residual[:, 1],
            np.maximum(direction.anchor_age, rt.anchor_age), shadow, delayed,
        ])
        valid = np.asarray(direction.valid, dtype=bool) & np.asarray(
            rt.valid, dtype=bool,
        ) & quality_valid
        saturated = np.asarray(rt.saturated_at_boundary, dtype=bool) | (
            quality_saturated
        )
        result[node] = NavigationBrainStateHistory(
            node_id=node, timestamps=direction.timestamps.copy(),
            decoded_phase=phase.copy(), decoded_rt=rt.anchored_rt.copy(),
            feature_matrix=feature_matrix, feature_names=FEATURE_NAMES,
            valid=valid, boundary_saturated=saturated,
            anchor_rejected=rt.anchor_rejected.copy(),
            reference_rebased=rt.reference_rebased.copy(),
            shadow_quality=shadow.copy(),
            delayed_feedback_quality=delayed.copy(),
        )
    return result
