from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from orbital_core.dynamics import rk4_step_absolute

from .multitarget_sidecar import MultiTargetBrainSnapshot
from .navigation_place_cells import (
    NavigationPlaceCellConfig,
    NavigationPlaceCellEncoder,
)
from .orbital_direction_state import OrbitalDirectionState
from .orbital_phase_adapter import OrbitalPlaneFrame
from .orbital_rt_grid_state import OrbitalRTGridState

Array = np.ndarray


@dataclass
class _TargetBrainRuntime:
    direction: OrbitalDirectionState
    rt: OrbitalRTGridState
    reference_state: Array
    last_posterior_state: Array
    timestamp: float


class OnlineMultiTargetBrainSidecar:
    """Persistent target-keyed CANN states for an online estimator."""

    def __init__(self, *, display_extent_m: float = 2_000.0) -> None:
        self.place_config = NavigationPlaceCellConfig.display_3x3(
            extent_m=display_extent_m,
        )
        self.place_encoder = NavigationPlaceCellEncoder(self.place_config)
        self._runtime_by_target: dict[str, _TargetBrainRuntime] = {}

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._runtime_by_target))

    def update(
        self,
        *,
        timestamp: float,
        state_by_target: Mapping[str, Array],
        covariance_by_target: Mapping[str, Array] | None = None,
    ) -> dict[str, MultiTargetBrainSnapshot]:
        current = {
            str(target_id): np.asarray(state, dtype=float).reshape(6)
            for target_id, state in state_by_target.items()
        }
        covariance = dict(covariance_by_target or {})
        snapshots = {}
        for target_id, state in sorted(current.items()):
            confidence = _anchor_confidence(covariance.get(target_id))
            runtime = self._runtime_by_target.get(target_id)
            if runtime is None:
                snapshots[target_id] = self._initialize(
                    target_id, float(timestamp), state,
                )
            else:
                snapshots[target_id] = self._advance(
                    runtime, float(timestamp), state, confidence,
                )
        return snapshots

    def remove(self, target_id: str) -> bool:
        return self._runtime_by_target.pop(str(target_id), None) is not None

    def reset(self) -> None:
        self._runtime_by_target.clear()

    def _initialize(self, target_id, timestamp, state):
        direction = OrbitalDirectionState(
            node_id=target_id,
            frame=OrbitalPlaneFrame.from_ascending_node_of_state_eci(state),
        )
        direction_snapshot = direction.initialize_from_state(
            timestamp=timestamp, state_eci=state,
        )
        rt = OrbitalRTGridState(node_id=target_id)
        rt_snapshot = rt.initialize(
            timestamp=timestamp,
            state_eci=state,
            reference_state_eci=state,
        )
        self._runtime_by_target[target_id] = _TargetBrainRuntime(
            direction=direction,
            rt=rt,
            reference_state=state.copy(),
            last_posterior_state=state.copy(),
            timestamp=timestamp,
        )
        return self._snapshot(
            target_id, timestamp, direction_snapshot, rt_snapshot,
        )

    def _advance(self, runtime, timestamp, state, confidence):
        delta_time = timestamp - runtime.timestamp
        if not np.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("Online brain timestamps must be strictly increasing.")
        predicted_state = rk4_step_absolute(
            runtime.last_posterior_state, delta_time,
        )
        reference_state = rk4_step_absolute(
            runtime.reference_state, delta_time,
        )
        runtime.direction.predict_from_state(
            timestamp=timestamp,
            predicted_state_eci=predicted_state,
        )
        direction_snapshot = runtime.direction.anchor_from_state(
            timestamp=timestamp,
            posterior_state_eci=state,
            confidence=confidence,
            trusted=True,
        )
        runtime.rt.predict(
            timestamp=timestamp,
            predicted_state_eci=predicted_state,
            reference_state_eci=reference_state,
        )
        rt_snapshot = runtime.rt.anchor(
            timestamp=timestamp,
            posterior_state_eci=state,
            reference_state_eci=reference_state,
            confidence=confidence,
            trusted=True,
        )
        runtime.reference_state = reference_state
        runtime.last_posterior_state = state.copy()
        runtime.timestamp = timestamp
        return self._snapshot(
            runtime.direction.node_id, timestamp,
            direction_snapshot, rt_snapshot,
        )

    def _snapshot(self, target_id, timestamp, direction, rt):
        place = self.place_encoder.encode(
            phase=direction.decoded_phase,
            radial_position=rt.decoded_rt[0],
            along_track_position=rt.decoded_rt[1],
        )
        heatmap = place.activity.reshape(
            self.place_config.phase_cell_count, 3, 3,
        ).sum(axis=0)
        return MultiTargetBrainSnapshot(
            target_id=target_id,
            timestamp=timestamp,
            decoded_phase_rad=direction.decoded_phase,
            decoded_rt_m=rt.decoded_rt.copy(),
            direction_ring=direction.neural_activity.copy(),
            place_heatmap_3x3=heatmap,
            valid=bool(direction.valid and rt.valid and place.valid),
        )


def _anchor_confidence(covariance) -> float:
    if covariance is None:
        return 1.0
    value = np.asarray(covariance, dtype=float)
    if value.shape != (6, 6) or np.any(~np.isfinite(value)):
        raise ValueError("Target covariance must have finite shape (6, 6).")
    position_variance = max(float(np.trace(value[:3, :3]) / 3.0), 0.0)
    position_sigma = np.sqrt(position_variance)
    return float(np.clip(1.0 / (1.0 + position_sigma / 1_000.0), 0.05, 1.0))
