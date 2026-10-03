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
from tracking.maneuver_detection import ManeuverAssessment, ManeuverDetector
from tracking.duplicate_tracks import DuplicateTrackResolver
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
    maneuver_by_target: Mapping[str, ManeuverAssessment]
    retired_target_ids: tuple[str, ...]
    merged_target_aliases: Mapping[str, str]


class OnlineMultiTargetTracker:
    """Persistent one-epoch-at-a-time target association and estimation API."""

    def __init__(
        self,
        *,
        scene_id: str,
        association_pipeline: MultiTargetAssociationPipeline | None = None,
        process_noise_acceleration: float = 1e-4,
        max_coast_epochs: int = 3,
        max_lost_epochs: int = 10,
        maneuver_detector: ManeuverDetector | None = None,
        duplicate_track_resolver: DuplicateTrackResolver | None = None,
    ):
        self.scene_id = str(scene_id)
        self.association_pipeline = association_pipeline or MultiTargetAssociationPipeline()
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.max_coast_epochs = int(max_coast_epochs)
        self.max_lost_epochs = int(max_lost_epochs)
        self.maneuver_detector = maneuver_detector or ManeuverDetector()
        self.duplicate_track_resolver = (
            duplicate_track_resolver or DuplicateTrackResolver()
        )
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
        raw_births = dict(routed.autonomous_iod.initialized_states_by_target)
        resolution = self.duplicate_track_resolver.resolve(raw_births, self.tracks)
        births = dict(resolution.accepted_births)
        for duplicate_id in resolution.aliases:
            self.association_pipeline.autonomous_iod_manager.retire_target(
                duplicate_id,
            )
        self._register_births(births)
        reports = self._run_local_updates(epoch, routed)
        output = None
        maneuver_assessments = {}
        retired_target_ids = ()
        if self._manager is not None and epoch > self._manager.timestamp:
            prior_estimates = {
                target_id: track.estimate
                for target_id, track in self._manager.tracks.items()
            }
            output = run_cooperative_target_fusion(
                scene_id=self.scene_id,
                timestamp=epoch,
                reports=reports,
                expected_target_ids=self._manager.tracks,
            )
            maneuver_assessments = {
                target_id: self.maneuver_detector.assess(
                    prior_estimates[target_id], estimate,
                )
                for target_id, estimate in output.estimates_by_target.items()
            }
            self._manager.step(
                output,
                maneuver_suspected_target_ids=(
                    target_id
                    for target_id, assessment in maneuver_assessments.items()
                    if assessment.suspected
                ),
            )
            reported_tracks = dict(self._manager.tracks)
            retired_target_ids = tuple(sorted(
                target_id for target_id, track in reported_tracks.items()
                if track.lifecycle.value == "TERMINATED"
            ))
            for target_id in retired_target_ids:
                self.association_pipeline.autonomous_iod_manager.retire_target(
                    target_id,
                )
                self._local_priors = {
                    key: value for key, value in self._local_priors.items()
                    if key[1] != target_id
                }
            self._manager.remove_terminated()
        else:
            reported_tracks = dict(self.tracks)
        self._last_timestamp = epoch
        return OnlineTrackingUpdate(
            timestamp=epoch,
            association=routed.association,
            initialized_states_by_target=births,
            output=output,
            tracks_by_target=reported_tracks,
            maneuver_by_target=dict(maneuver_assessments),
            retired_target_ids=retired_target_ids,
            merged_target_aliases=dict(resolution.aliases),
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
                max_lost_epochs=self.max_lost_epochs,
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
            process_noise_scale = self.maneuver_detector.process_noise_scale(target_id)
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
                            dt, self.process_noise_acceleration * process_noise_scale,
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

