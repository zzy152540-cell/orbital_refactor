from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, replace

import numpy as np

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
        initialization_cache: MutableMapping | None = None,
    ):
        self.track_id_by_target = dict(track_id_by_target or {})
        self.buffer_config = buffer_config or IODBufferConfig()
        self.initialization_cache = initialization_cache
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
        ready = []
        for target_id in sorted(touched):
            buffer = self._buffers[target_id]
            if not buffer.is_ready():
                continue
            observations = tuple(buffer.observations)
            cache_key = _initialization_cache_key(
                target_id, self.track_id_by_target.get(target_id), observations,
            )
            estimate = (
                None
                if self.initialization_cache is None
                else self.initialization_cache.get(cache_key)
            )
            if estimate is not None:
                estimate = replace(
                    estimate,
                    target_id=target_id,
                    track_id=(
                        self.track_id_by_target.get(target_id)
                        or f"track-{target_id}"
                    ),
                )
            ready.append((target_id, observations, cache_key, estimate))

        missing = [item for item in ready if item[3] is None]
        computed_by_target = {
            item[0]: initialize_target(
                item[1],
                target_id=item[0],
                track_id=self.track_id_by_target.get(item[0]),
            )
            for item in missing
        }

        for target_id, _observations, cache_key, cached in ready:
            estimate = (
                cached if cached is not None else computed_by_target[target_id]
            )
            if self.initialization_cache is not None and cached is None:
                self.initialization_cache[cache_key] = estimate
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


def _initialization_cache_key(_target_id, _track_id, observations):
    """Fingerprint the physical short arc, independent of local identity.

    Autonomous nodes are allowed to assign different candidate labels to the
    same detections.  Those labels do not enter the orbit solution and must not
    prevent paired experiment arms from reusing an identical deterministic
    solve.  The cached estimate is rebound to the requesting local identity on
    lookup.
    """

    return (
        tuple(
            (
                float(item.timestamp),
                str(item.observer_id),
                str(item.modality),
                np.asarray(item.observer_state_eci, dtype=float).shape,
                np.asarray(item.observer_state_eci, dtype=float).tobytes(),
                np.asarray(item.measurement, dtype=float).shape,
                np.asarray(item.measurement, dtype=float).tobytes(),
                np.asarray(item.covariance, dtype=float).shape,
                np.asarray(item.covariance, dtype=float).tobytes(),
                bool(item.valid_flag),
            )
            for item in observations
        ),
    )
