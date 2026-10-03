from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from interfaces.data_objects import InitialState, ModuleInput
from orbital_core.dynamics import make_process_noise
from tracking.cooperative_pipeline import run_cooperative_target_fusion
from tracking.data_contracts import (
    MultiTargetOutput,
    TargetInitialState,
    TargetTrack,
)
from tracking.local_target_filter import run_local_target_filter
from tracking.track_manager import TrackManager

from .autonomous_sequence import _tracking_observation
from .data_contracts import AssociationResult, UnlabeledIODObservation
from .pipeline import AssociationPipelineUpdate, MultiTargetAssociationPipeline


@dataclass(frozen=True)
class OnlineTrackingUpdate:
    timestamp: float
    association: AssociationResult
    initialized_states_by_target: Mapping[str, TargetInitialState]
    output: MultiTargetOutput | None
    tracks_by_target: Mapping[str, TargetTrack]


class OnlineMultiTargetTracker:
    """Persistent one-epoch-at-a-time target association and estimation API."""

    def __init__(
        self,
        *,
        scene_id: str,
        association_pipeline: MultiTargetAssociationPipeline | None = None,
        process_noise_acceleration: float = 1e-4,
        max_coast_epochs: int = 3,
    ):
        self.scene_id = str(scene_id)
        self.association_pipeline = association_pipeline or MultiTargetAssociationPipeline()
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.max_coast_epochs = int(max_coast_epochs)
        self._manager: TrackManager | None = None
        self._local_priors: dict[tuple[str, str], TargetInitialState] = {}
        self._observer_states: dict[tuple[str, float], np.ndarray] = {}
        self._observer_quaternions: dict[tuple[str, float], np.ndarray] = {}
        self._last_timestamp: float | None = None

    @property
    def tracks(self):
        return {} if self._manager is None else self._manager.tracks

    def step(
        self,
        *,
        timestamp: float,
        observations,
        observer_states_eci: Mapping[str, np.ndarray],
        q_eci2pri_by_observer: Mapping[str, np.ndarray],
    ) -> OnlineTrackingUpdate:
        epoch = float(timestamp)
        if self._last_timestamp is not None and epoch <= self._last_timestamp:
            raise ValueError("Online tracking timestamps must be strictly increasing.")
        if set(observer_states_eci) != set(q_eci2pri_by_observer):
            raise ValueError("Observer states and attitudes must use identical IDs.")
        values = tuple(observations)
        if any(not isinstance(item, UnlabeledIODObservation) for item in values):
            raise TypeError("Online tracker requires unlabeled IOD observations.")
        if any(not np.isclose(item.timestamp, epoch) for item in values):
            raise ValueError("Every observation must match the online step timestamp.")
        for observer_id in observer_states_eci:
            key = (str(observer_id), epoch)
            self._observer_states[key] = np.asarray(
                observer_states_eci[observer_id], dtype=float,
            ).reshape(6).copy()
            self._observer_quaternions[key] = np.asarray(
                q_eci2pri_by_observer[observer_id], dtype=float,
            ).reshape(4).copy()

        estimates = {
            target_id: track.estimate
            for target_id, track in self.tracks.items()
        }
        routed = self.association_pipeline.step(values, estimates)
        births = dict(routed.autonomous_iod.initialized_states_by_target)
        self._register_births(births)
        reports = self._run_local_updates(epoch, routed)
        output = None
        if self._manager is not None and epoch > self._manager.timestamp:
            output = run_cooperative_target_fusion(
                scene_id=self.scene_id,
                timestamp=epoch,
                reports=reports,
                expected_target_ids=self._manager.tracks,
            )
            self._manager.step(output)
        self._last_timestamp = epoch
        return OnlineTrackingUpdate(
            timestamp=epoch,
            association=routed.association,
            initialized_states_by_target=births,
            output=output,
            tracks_by_target=dict(self.tracks),
        )

    def _register_births(self, births):
        if not births:
            return
        if self._manager is None:
            self._manager = TrackManager(
                scene_id=self.scene_id,
                initial_states=births,
                process_noise_acceleration=self.process_noise_acceleration,
                max_coast_epochs=self.max_coast_epochs,
            )
            return
        for state in births.values():
            self._manager.register(state)

    def _run_local_updates(self, epoch, routed: AssociationPipelineUpdate):
        grouped = defaultdict(list)
        for target_id, items in routed.observations_by_target.items():
            for item in items:
                grouped[(item.observer_id, target_id)].append(item)
        reports = []
        for (observer_id, target_id), items in sorted(grouped.items()):
            prior = self._local_priors.get((observer_id, target_id))
            if prior is None:
                prior = self._manager.tracks[target_id].estimate
                prior = TargetInitialState(
                    target_id=target_id,
                    track_id=prior.track_id,
                    timestamp=prior.timestamp,
                    state_eci=prior.state_eci,
                    covariance_eci=prior.covariance_eci,
                )
            if epoch <= prior.timestamp:
                continue
            previous_key = (observer_id, float(prior.timestamp))
            current_key = (observer_id, epoch)
            if previous_key not in self._observer_states:
                raise ValueError(f"Missing retained observer state for {previous_key!r}.")
            observations = [
                _tracking_observation(
                    item, target_id, self._observer_quaternions[current_key],
                )
                for item in items
            ]
            dt = epoch - prior.timestamp
            module_input = ModuleInput(
                initial_state=InitialState(
                    target_id=target_id,
                    timestamp=prior.timestamp,
                    state_estimate=prior.state_eci - self._observer_states[previous_key],
                    covariance=prior.covariance_eci.copy(),
                ),
                sensor_measurements=observations,
                config={
                    "runtime": {
                        "timestamps": np.array([prior.timestamp, epoch]),
                        "chief_state_history_eci": np.vstack([
                            self._observer_states[previous_key],
                            self._observer_states[current_key],
                        ]),
                        "q_eci2pri_history": np.vstack([
                            self._observer_quaternions[previous_key],
                            self._observer_quaternions[current_key],
                        ]),
                        "node_id": observer_id,
                    },
                    "filter": {
                        "architecture": "federated_ci",
                        "process_noise": make_process_noise(
                            dt, self.process_noise_acceleration,
                        ),
                        "reset_feedback": True,
                        "ci_objective": "trace",
                        "ci_grid_points": 31,
                    },
                    "modalities": {},
                },
            )
            report = run_local_target_filter(
                module_input,
                track_id=prior.track_id,
            )
            reports.append(report)
            self._local_priors[(observer_id, target_id)] = TargetInitialState(
                target_id=target_id,
                track_id=prior.track_id,
                timestamp=epoch,
                state_eci=report.state_eci,
                covariance_eci=report.covariance_eci,
            )
        return reports

