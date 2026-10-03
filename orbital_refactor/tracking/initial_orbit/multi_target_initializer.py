from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tracking.data_contracts import TargetInitialState

from .data_contracts import InitialOrbitEstimate, IODObservation, IODStatus
from .initializer import initialize_target, target_initial_state_from_iod
from .observation_buffer import IODBufferConfig, TargetIODObservationBuffer


@dataclass(frozen=True)
class MultiTargetIODUpdate:
    estimates_by_target: Mapping[str, InitialOrbitEstimate]
    initialized_states_by_target: Mapping[str, TargetInitialState]
    waiting_target_ids: tuple[str, ...]


class MultiTargetIODManager:
    """Accumulate independent short arcs and initialize known target IDs."""

    def __init__(
        self,
        *,
        track_id_by_target: Mapping[str, str] | None = None,
        buffer_config: IODBufferConfig | None = None,
    ):
        self.track_id_by_target = dict(track_id_by_target or {})
        self.buffer_config = buffer_config or IODBufferConfig()
        self._buffers: dict[str, TargetIODObservationBuffer] = {}
        self._initialized: dict[str, TargetInitialState] = {}
        self._last_estimate: dict[str, InitialOrbitEstimate] = {}

    @property
    def initialized_states(self) -> Mapping[str, TargetInitialState]:
        return dict(self._initialized)

    def ingest(self, observations) -> MultiTargetIODUpdate:
        touched = set()
        for observation in observations:
            if not isinstance(observation, IODObservation):
                raise TypeError("MultiTargetIODManager requires IODObservation values.")
            target_id = observation.target_id
            if target_id in self._initialized:
                continue
            buffer = self._buffers.setdefault(
                target_id,
                TargetIODObservationBuffer(target_id, self.buffer_config),
            )
            buffer.add(observation)
            touched.add(target_id)
        estimates = {}
        initialized = {}
        for target_id in sorted(touched):
            buffer = self._buffers[target_id]
            if not buffer.is_ready():
                continue
            estimate = initialize_target(
                buffer.observations,
                target_id=target_id,
                track_id=self.track_id_by_target.get(target_id),
            )
            estimates[target_id] = estimate
            self._last_estimate[target_id] = estimate
            if estimate.status is IODStatus.SUCCESS:
                state = target_initial_state_from_iod(estimate)
                self._initialized[target_id] = state
                initialized[target_id] = state
                buffer.clear()
        waiting = tuple(sorted(
            target_id for target_id in self._buffers
            if target_id not in self._initialized
        ))
        return MultiTargetIODUpdate(
            estimates_by_target=estimates,
            initialized_states_by_target=initialized,
            waiting_target_ids=waiting,
        )

    def retire(self, target_id: str) -> None:
        key = str(target_id)
        self._buffers.pop(key, None)
        self._initialized.pop(key, None)
        self._last_estimate.pop(key, None)
