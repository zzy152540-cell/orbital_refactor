from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from orbital_core.dynamics import propagate_absolute_orbit

from .navigation_brain_state import (
    NavigationBrainStateHistory,
    build_navigation_brain_states,
)
from .navigation_place_cells import (
    NavigationPlaceCellConfig,
    NavigationPlaceCellHistory,
    build_navigation_place_cell_histories,
)
from .orbital_direction_runner import (
    OrbitalDirectionHistory,
    run_orbital_direction_states,
)
from .orbital_phase_adapter import OrbitalPlaneFrame
from .orbital_rt_grid_runner import (
    OrbitalRTGridHistory,
    run_orbital_rt_grid_states,
)

Array = np.ndarray


@dataclass(frozen=True)
class MultiTargetBrainSnapshot:
    """Display-ready CANN state for one persistent cooperative target."""

    target_id: str
    timestamp: float
    decoded_phase_rad: float
    decoded_rt_m: Array
    direction_ring: Array
    place_heatmap_3x3: Array
    valid: bool


@dataclass(frozen=True)
class MultiTargetBrainSidecar:
    """Target-keyed brain-inspired histories kept outside the estimator."""

    timestamps: Array
    direction_by_target: Mapping[str, OrbitalDirectionHistory]
    rt_by_target: Mapping[str, OrbitalRTGridHistory]
    navigation_by_target: Mapping[str, NavigationBrainStateHistory]
    place_cells_by_target: Mapping[str, NavigationPlaceCellHistory]

    def snapshot(self, target_id: str, index: int) -> MultiTargetBrainSnapshot:
        target = str(target_id)
        if target not in self.navigation_by_target:
            raise KeyError(f"Unknown cooperative target: {target!r}.")
        direction = self.direction_by_target[target]
        navigation = self.navigation_by_target[target]
        place = self.place_cells_by_target[target]
        if not -self.timestamps.size <= index < self.timestamps.size:
            raise IndexError("Brain-sidecar snapshot index is out of range.")
        direction_ring = np.asarray(direction.neural_activity[index], dtype=float)
        phase_count = place.activity.shape[1] // 9
        heatmap = np.asarray(place.activity[index], dtype=float).reshape(
            phase_count, 3, 3,
        ).sum(axis=0)
        return MultiTargetBrainSnapshot(
            target_id=target,
            timestamp=float(self.timestamps[index]),
            decoded_phase_rad=float(navigation.decoded_phase[index]),
            decoded_rt_m=np.array(navigation.decoded_rt[index], copy=True),
            direction_ring=np.array(direction_ring, copy=True),
            place_heatmap_3x3=np.array(heatmap, copy=True),
            valid=bool(navigation.valid[index] and place.valid[index]),
        )


def build_multitarget_brain_sidecar(
    *,
    timestamps: Array,
    state_history_by_target: Mapping[str, Array],
    anchor_confidence_by_target: Mapping[str, Array] | None = None,
) -> MultiTargetBrainSidecar:
    """Build target-centric direction and place-cell states from posteriors.

    Global cooperative target IDs, rather than observer-local track names, are
    the keys.  Therefore observer membership changes do not reset or exchange
    the brain-inspired representation.  The propagated first posterior is the
    RT reference; no truth state is consumed.
    """

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    states = {
        str(target_id): np.asarray(history, dtype=float)
        for target_id, history in state_history_by_target.items()
    }
    if not states:
        raise ValueError("At least one cooperative target history is required.")
    if (
        times.size == 0
        or np.any(~np.isfinite(times))
        or (times.size > 1 and np.any(np.diff(times) <= 0.0))
    ):
        raise ValueError("timestamps must be finite, nonempty, and increasing.")
    for history in states.values():
        if history.shape != (times.size, 6) or np.any(~np.isfinite(history)):
            raise ValueError("Target histories must have finite shape (N, 6).")

    masks = {
        target_id: np.ones(times.size, dtype=bool) for target_id in states
    }
    confidence = _confidence_histories(
        states, times.size, anchor_confidence_by_target,
    )
    direction = run_orbital_direction_states(
        timestamps=times,
        posterior_state_history_by_node=states,
        frame_by_node={
            target_id: OrbitalPlaneFrame.from_ascending_node_of_state_eci(
                history[0]
            )
            for target_id, history in states.items()
        },
        anchor_mask_by_node=masks,
        anchor_confidence_by_node=confidence,
        retain_activity=True,
    )
    relative_times = times - times[0]
    references = {
        target_id: propagate_absolute_orbit(history[0], relative_times)
        for target_id, history in states.items()
    }
    rt = run_orbital_rt_grid_states(
        timestamps=times,
        posterior_state_history_by_node=states,
        reference_state_history_by_node=references,
        anchor_mask_by_node=masks,
        anchor_confidence_by_node=confidence,
    )
    navigation = build_navigation_brain_states(
        direction_by_node=direction,
        rt_by_node=rt,
    )
    place_cells = build_navigation_place_cell_histories(
        navigation_by_node=navigation,
        config=NavigationPlaceCellConfig.display_3x3(extent_m=2_000.0),
    )
    return MultiTargetBrainSidecar(
        timestamps=times.copy(),
        direction_by_target=direction,
        rt_by_target=rt,
        navigation_by_target=navigation,
        place_cells_by_target=place_cells,
    )


def _confidence_histories(states, size, supplied):
    if supplied is None:
        return {
            target_id: np.ones(size, dtype=float) for target_id in states
        }
    unknown = set(supplied) - set(states)
    if unknown:
        raise ValueError(
            f"Anchor confidence contains unknown targets: {sorted(unknown)}"
        )
    result = {}
    for target_id in states:
        values = np.asarray(
            supplied.get(target_id, np.ones(size)), dtype=float,
        )
        if (
            values.shape != (size,)
            or np.any(~np.isfinite(values))
            or np.any((values < 0.0) | (values > 1.0))
        ):
            raise ValueError(
                "Per-target anchor confidence must have shape (N,) and lie "
                "in [0, 1]."
            )
        result[target_id] = values
    return result
