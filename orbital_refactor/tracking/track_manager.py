from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)

from .data_contracts import (
    GlobalTargetEstimate,
    MultiTargetOutput,
    TargetInitialState,
    TargetTrack,
    TargetTrackKey,
)
from .track_lifecycle import TrackLifecycle


class TrackManager:
    """Maintain independent known-target tracks across aligned epochs."""

    def __init__(
        self,
        *,
        scene_id: str,
        initial_states: Mapping[str, TargetInitialState],
        process_noise_acceleration: float = 1e-4,
        max_coast_epochs: int = 3,
    ) -> None:
        self.scene_id = str(scene_id).strip()
        if not self.scene_id:
            raise ValueError("scene_id must be a non-empty identifier.")
        if not initial_states:
            raise ValueError("initial_states must contain at least one known target.")
        if process_noise_acceleration < 0.0:
            raise ValueError("process_noise_acceleration must be non-negative.")
        if int(max_coast_epochs) < 0:
            raise ValueError("max_coast_epochs must be non-negative.")
        values = dict(initial_states)
        if any(key != value.target_id for key, value in values.items()):
            raise ValueError("initial_states keys must match TargetInitialState.target_id.")
        timestamps = {value.timestamp for value in values.values()}
        if len(timestamps) != 1:
            raise ValueError("All initial target states must share one timestamp.")
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.max_coast_epochs = int(max_coast_epochs)
        self._timestamp = next(iter(timestamps))
        self._has_stepped = False
        self._missed_epochs = {target_id: 0 for target_id in values}
        self._tracks = {
            target_id: TargetTrack(
                key=TargetTrackKey(self.scene_id, initial.track_id),
                target_id=target_id,
                lifecycle=TrackLifecycle.INITIALIZING,
                estimate=initial,
            )
            for target_id, initial in values.items()
        }

    @property
    def timestamp(self) -> float:
        return float(self._timestamp)

    @property
    def tracks(self) -> Mapping[str, TargetTrack]:
        return MappingProxyType(dict(self._tracks))

    def step(self, output: MultiTargetOutput) -> Mapping[str, TargetTrack]:
        if output.scene_id != self.scene_id:
            raise ValueError("MultiTargetOutput scene_id does not match TrackManager.")
        timestamp = float(output.timestamp)
        if timestamp < self._timestamp or (self._has_stepped and timestamp <= self._timestamp):
            raise ValueError("TrackManager epochs must be strictly increasing after the first step.")
        unknown = set(output.estimates_by_target) - set(self._tracks)
        if unknown:
            raise ValueError(f"Output contains unregistered targets: {sorted(unknown)}")

        for target_id, track in tuple(self._tracks.items()):
            if track.lifecycle == TrackLifecycle.TERMINATED:
                continue
            estimate = output.estimates_by_target.get(target_id)
            if estimate is not None:
                if estimate.track_id != track.key.track_id:
                    raise ValueError(
                        f"Target {target_id!r} changed track_id from "
                        f"{track.key.track_id!r} to {estimate.track_id!r}."
                    )
                self._missed_epochs[target_id] = 0
                self._tracks[target_id] = replace(
                    track,
                    lifecycle=TrackLifecycle.TRACKING,
                    estimate=estimate,
                )
                continue

            predicted = _propagate_estimate(
                track.estimate,
                timestamp,
                process_noise_acceleration=self.process_noise_acceleration,
            )
            missed = self._missed_epochs[target_id] + 1
            self._missed_epochs[target_id] = missed
            lifecycle = (
                TrackLifecycle.LOST
                if missed > self.max_coast_epochs
                else TrackLifecycle.COASTING
            )
            self._tracks[target_id] = replace(
                track,
                lifecycle=lifecycle,
                estimate=predicted,
            )

        self._timestamp = timestamp
        self._has_stepped = True
        return self.tracks

    def terminate(self, target_id: str) -> TargetTrack:
        key = str(target_id)
        if key not in self._tracks:
            raise KeyError(key)
        track = replace(self._tracks[key], lifecycle=TrackLifecycle.TERMINATED)
        self._tracks[key] = track
        return track

    def register(self, initial_state: TargetInitialState) -> TargetTrack:
        """Register a newly initialized target while the manager is running."""

        target_id = str(initial_state.target_id)
        if target_id in self._tracks:
            raise ValueError(f"Target {target_id!r} is already registered.")
        if initial_state.timestamp > self._timestamp:
            raise ValueError("A new target cannot begin after the manager timestamp.")
        estimate = (
            initial_state
            if np.isclose(initial_state.timestamp, self._timestamp)
            else _propagate_estimate(
                initial_state,
                self._timestamp,
                process_noise_acceleration=self.process_noise_acceleration,
            )
        )
        track = TargetTrack(
            key=TargetTrackKey(self.scene_id, initial_state.track_id),
            target_id=target_id,
            lifecycle=TrackLifecycle.INITIALIZING,
            estimate=estimate,
        )
        self._tracks[target_id] = track
        self._missed_epochs[target_id] = 0
        return track


def _propagate_estimate(
    estimate: GlobalTargetEstimate | TargetInitialState,
    timestamp: float,
    *,
    process_noise_acceleration: float,
) -> GlobalTargetEstimate:
    delta_time = float(timestamp - estimate.timestamp)
    if delta_time < 0.0:
        raise ValueError("Cannot propagate a target estimate backward in time.")
    if delta_time == 0.0:
        state = np.asarray(estimate.state_eci, dtype=float).copy()
        covariance = np.asarray(estimate.covariance_eci, dtype=float).copy()
    else:
        propagate = lambda value: rk4_step_absolute(value, delta_time)
        state = propagate(estimate.state_eci)
        transition = numerical_jacobian_discrete(propagate, estimate.state_eci)
        covariance = (
            transition @ estimate.covariance_eci @ transition.T
            + make_process_noise(delta_time, process_noise_acceleration)
        )
        covariance = 0.5 * (covariance + covariance.T)
    return GlobalTargetEstimate(
        target_id=estimate.target_id,
        track_id=estimate.track_id,
        timestamp=timestamp,
        state_eci=state,
        covariance_eci=covariance,
        contributing_observer_ids=(),
        node_weights={},
        valid_flag=True,
    )
