from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import time

import numpy as np

from brain_inspired.measurement_preprocessor import (
    MeasurementPreprocessor,
    validate_preprocessed_stream,
)
from interfaces.data_objects import InitialState, ModuleInput, Observation
from orbital_core.coordinates import rotate_pri_to_eci
from orbital_core.dynamics import make_process_noise
from orbital_core.quality import quality_score_from_covariance
from tracking.cooperative_pipeline import run_cooperative_target_fusion
from tracking.data_contracts import (
    MultiTargetOutput,
    TargetInitialState,
    TargetNodeReport,
    TargetTrack,
)
from tracking.local_target_filter import run_local_target_filter
from tracking.maneuver_detection import ManeuverAssessment, ManeuverDetector
from tracking.multi_observer_bearing_filter import (
    batch_multi_observer_bearing_update,
)
from tracking.inverse_depth_bearing_filter import (
    inverse_depth_multi_observer_bearing_update,
)
from tracking.multi_observer_multimodal_filter import (
    unified_multi_observer_multimodal_update,
)
from tracking.multi_epoch_bearing_filter import MultiEpochBearingWindow
from tracking.multi_epoch_multimodal_filter import FixedLagMultimodalWindow
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
    local_reports: tuple[TargetNodeReport, ...]
    suppressed_birth_ids: tuple[str, ...] = ()
    joint_bearing_update_count: int = 0
    joint_bearing_measurement_count: int = 0
    unified_multimodal_update_count: int = 0
    unified_multimodal_observation_count: int = 0
    cann_preprocessed_observation_count: int = 0
    cann_preprocessing_seconds: float = 0.0
    cann_action_counts: Mapping[str, int] = field(default_factory=dict)
    iod_diagnostics_by_target: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ConsensusFeedbackApplication:
    timestamp: float
    applied_observer_target_keys: tuple[tuple[str, str], ...]
    skipped_observer_target_reasons: Mapping[tuple[str, str], str]


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
        initial_states: Mapping[str, TargetInitialState] | None = None,
        maximum_track_count: int | None = None,
        preserve_angular_radial_nullspace: bool = False,
        enable_multi_observer_bearing_update: bool = False,
        enable_unified_multimodal_update: bool = False,
        enable_local_unified_multimodal_update: bool = False,
        enable_inverse_depth_los_update: bool = False,
        unified_multimodal_window_epochs: int = 1,
        multi_epoch_bearing_window_epochs: int = 1,
        measurement_preprocessor_factory: Callable[
            [str, str], MeasurementPreprocessor
        ] | None = None,
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
        self.measurement_preprocessor_factory = measurement_preprocessor_factory
        if maximum_track_count is not None and int(maximum_track_count) < 1:
            raise ValueError("maximum_track_count must be positive when supplied.")
        self.maximum_track_count = (
            None if maximum_track_count is None else int(maximum_track_count)
        )
        self.preserve_angular_radial_nullspace = bool(
            preserve_angular_radial_nullspace
        )
        self.enable_multi_observer_bearing_update = bool(
            enable_multi_observer_bearing_update
        )
        self.enable_unified_multimodal_update = bool(
            enable_unified_multimodal_update
        )
        self.enable_local_unified_multimodal_update = bool(
            enable_local_unified_multimodal_update
        )
        self.enable_inverse_depth_los_update = bool(
            enable_inverse_depth_los_update
        )
        if int(unified_multimodal_window_epochs) < 1:
            raise ValueError(
                "unified_multimodal_window_epochs must be positive."
            )
        self.unified_multimodal_window_epochs = int(
            unified_multimodal_window_epochs
        )
        if int(multi_epoch_bearing_window_epochs) < 1:
            raise ValueError(
                "multi_epoch_bearing_window_epochs must be positive."
            )
        self.multi_epoch_bearing_window_epochs = int(
            multi_epoch_bearing_window_epochs
        )
        self._multi_epoch_bearing_windows: dict[
            str, MultiEpochBearingWindow
        ] = {}
        self._multimodal_windows: dict[str, FixedLagMultimodalWindow] = {}
        self._measurement_preprocessor_by_stream: dict[
            tuple[str, str], MeasurementPreprocessor
        ] = {}
        self._last_preprocessed_observation_count = 0
        self._last_preprocessing_seconds = 0.0
        self._last_cann_action_counts: Counter[str] = Counter()
        self._last_joint_bearing_update_count = 0
        self._last_joint_bearing_measurement_count = 0
        self._last_unified_multimodal_update_count = 0
        self._last_unified_multimodal_observation_count = 0
        configured_initial_states = dict(initial_states or {})
        if (
            self.maximum_track_count is not None
            and len(configured_initial_states) > self.maximum_track_count
        ):
            raise ValueError(
                "initial_states exceeds the configured maximum_track_count."
            )
        self._manager: TrackManager | None = (
            TrackManager(
                scene_id=self.scene_id,
                initial_states=configured_initial_states,
                process_noise_acceleration=self.process_noise_acceleration,
                max_coast_epochs=self.max_coast_epochs,
                max_lost_epochs=self.max_lost_epochs,
            )
            if configured_initial_states else None
        )
        self._local_priors: dict[tuple[str, str], TargetInitialState] = {}
        self._local_prior_information_ids: dict[
            tuple[str, str], tuple[str, ...]
        ] = {}
        self._observer_states: dict[tuple[str, float], np.ndarray] = {}
        self._observer_quaternions: dict[tuple[str, float], np.ndarray] = {}
        self._last_timestamp: float | None = None

    @property
    def tracks(self):
        return {} if self._manager is None else self._manager.tracks

    def apply_consensus_feedback(self, consensus_step) -> ConsensusFeedbackApplication:
        """Use per-node Consensus-CI posteriors as the next local-filter priors.

        The feedback is accepted only for the most recently processed epoch,
        for observers present in that epoch, and for the currently active
        target/track identity.  This keeps delayed or stale network output from
        silently reviving a retired track.
        """

        if self._last_timestamp is None:
            raise ValueError("Tracking must process an epoch before feedback is applied.")
        timestamp = float(consensus_step.timestamp)
        if not np.isclose(timestamp, self._last_timestamp):
            raise ValueError("Consensus feedback must match the latest tracking epoch.")

        applied = []
        skipped = {}
        for observer_id, estimates in consensus_step.estimates_by_node.items():
            for target_id, estimate in estimates.items():
                key = (str(observer_id), str(target_id))
                reason = None
                if (key[0], timestamp) not in self._observer_states:
                    reason = "observer_state_unavailable"
                elif target_id not in self.tracks:
                    reason = "target_not_active"
                elif self.tracks[target_id].estimate.track_id != estimate.track_id:
                    reason = "track_identity_mismatch"
                elif not estimate.valid_flag:
                    reason = "invalid_estimate"
                if reason is not None:
                    skipped[key] = reason
                    continue
                self._local_priors[key] = TargetInitialState(
                    target_id=estimate.target_id,
                    track_id=estimate.track_id,
                    timestamp=timestamp,
                    state_eci=estimate.state_eci,
                    covariance_eci=estimate.covariance_eci,
                )
                self._local_prior_information_ids[key] = estimate.information_ids
                applied.append(key)
        return ConsensusFeedbackApplication(
            timestamp=timestamp,
            applied_observer_target_keys=tuple(sorted(applied)),
            skipped_observer_target_reasons=dict(sorted(skipped.items())),
        )

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
        suppressed_birth_ids = ()
        if self.maximum_track_count is not None:
            capacity = max(self.maximum_track_count - len(self.tracks), 0)
            accepted_ids = tuple(sorted(births))[:capacity]
            suppressed_birth_ids = tuple(sorted(set(births) - set(accepted_ids)))
            births = {target_id: births[target_id] for target_id in accepted_ids}
        for duplicate_id in resolution.aliases:
            self.association_pipeline.autonomous_iod_manager.retire_target(
                duplicate_id,
            )
        for suppressed_id in suppressed_birth_ids:
            self.association_pipeline.autonomous_iod_manager.retire_target(
                suppressed_id,
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
                self._local_prior_information_ids = {
                    key: value
                    for key, value in self._local_prior_information_ids.items()
                    if key[1] != target_id
                }
                self._measurement_preprocessor_by_stream = {
                    key: value
                    for key, value in self._measurement_preprocessor_by_stream.items()
                    if key[1] != target_id
                }
                self._multi_epoch_bearing_windows.pop(target_id, None)
                self._multimodal_windows.pop(target_id, None)
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
            local_reports=tuple(reports),
            suppressed_birth_ids=suppressed_birth_ids,
            joint_bearing_update_count=self._last_joint_bearing_update_count,
            joint_bearing_measurement_count=(
                self._last_joint_bearing_measurement_count
            ),
            unified_multimodal_update_count=(
                self._last_unified_multimodal_update_count
            ),
            unified_multimodal_observation_count=(
                self._last_unified_multimodal_observation_count
            ),
            cann_preprocessed_observation_count=(
                self._last_preprocessed_observation_count
            ),
            cann_preprocessing_seconds=self._last_preprocessing_seconds,
            cann_action_counts=dict(sorted(self._last_cann_action_counts.items())),
            iod_diagnostics_by_target={
                target_id: dict(estimate.diagnostics)
                for target_id, estimate in (
                    routed.autonomous_iod.estimates_by_target.items()
                )
            },
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
        self._last_preprocessed_observation_count = 0
        self._last_preprocessing_seconds = 0.0
        self._last_cann_action_counts.clear()
        self._last_joint_bearing_update_count = 0
        self._last_joint_bearing_measurement_count = 0
        self._last_unified_multimodal_update_count = 0
        self._last_unified_multimodal_observation_count = 0
        grouped = defaultdict(list)
        for target_id, items in routed.observations_by_target.items():
            for item in items:
                grouped[(item.observer_id, target_id)].append(item)
        reports, unified_targets = self._unified_multimodal_reports(
            epoch, routed.observations_by_target,
        )
        joint_reports, jointly_updated_targets = self._joint_bearing_reports(
            epoch,
            {
                target_id: values
                for target_id, values in routed.observations_by_target.items()
                if target_id not in unified_targets
            },
        )
        reports.extend(joint_reports)
        jointly_updated_targets.update(unified_targets)
        for (observer_id, target_id), items in sorted(grouped.items()):
            if target_id in jointly_updated_targets:
                continue
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
            if self.enable_local_unified_multimodal_update:
                values = tuple(
                    item for item in items
                    if item.valid_flag and item.modality in {"RADAR", "LOS"}
                )
                values = self._preprocess_unified_observations(
                    target_id=target_id,
                    observations=values,
                    timestamp=epoch,
                )
                if not values:
                    continue
                process_noise_scale = (
                    self.maneuver_detector.process_noise_scale(target_id)
                )
                update = unified_multi_observer_multimodal_update(
                    prior,
                    values,
                    timestamp=epoch,
                    process_noise_acceleration=(
                        self.process_noise_acceleration * process_noise_scale
                    ),
                )
                prior_information_ids = self._local_prior_information_ids.get(
                    (observer_id, target_id), ()
                )
                information_ids = tuple(dict.fromkeys((
                    *prior_information_ids,
                    *update.measurement_ids,
                )))
                total = max(update.observation_count, 1)
                modality_weights = {
                    name.lower(): count / total
                    for name, count in update.modality_counts.items()
                }
                quality = quality_score_from_covariance(update.covariance_eci)
                report = TargetNodeReport(
                    observer_id=observer_id,
                    target_id=target_id,
                    track_id=prior.track_id,
                    timestamp=epoch,
                    state_eci=update.state_eci,
                    covariance_eci=update.covariance_eci,
                    quality_score=float(np.clip(
                        quality / (1.0 + quality), 0.0, 1.0,
                    )),
                    valid_flag=True,
                    modality_weights=modality_weights,
                    used_measurement_ids=information_ids,
                    information_capabilities=update.information_capabilities,
                )
                reports.append(report)
                self._local_priors[(observer_id, target_id)] = (
                    TargetInitialState(
                        target_id=target_id,
                        track_id=prior.track_id,
                        timestamp=epoch,
                        state_eci=update.state_eci,
                        covariance_eci=update.covariance_eci,
                    )
                )
                self._local_prior_information_ids[(observer_id, target_id)] = (
                    information_ids
                )
                continue
            observations = [
                _tracking_observation(
                    item, target_id, self._observer_quaternions[current_key],
                )
                for item in items
            ]
            observations = self._preprocess_measurements(
                observer_id=observer_id,
                target_id=target_id,
                observations=observations,
                timestamp=epoch,
            )
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
                        "preserve_angular_radial_nullspace": (
                            self.preserve_angular_radial_nullspace
                        ),
                        "cross_modality_anchor": "rad",
                        "cross_modality_gate_threshold": 25.0,
                    },
                    "modalities": {},
                },
            )
            report = run_local_target_filter(
                module_input,
                track_id=prior.track_id,
            )
            prior_information_ids = self._local_prior_information_ids.get(
                (observer_id, target_id), ()
            )
            report = replace(
                report,
                used_measurement_ids=tuple(dict.fromkeys((
                    *prior_information_ids,
                    *report.used_measurement_ids,
                ))),
            )
            reports.append(report)
            self._local_priors[(observer_id, target_id)] = TargetInitialState(
                target_id=target_id,
                track_id=prior.track_id,
                timestamp=epoch,
                state_eci=report.state_eci,
                covariance_eci=report.covariance_eci,
            )
            self._local_prior_information_ids[(observer_id, target_id)] = (
                report.used_measurement_ids
            )
        return reports

    def _unified_multimodal_reports(self, epoch, observations_by_target):
        """Apply one predicted prior to all current radar and LOS blocks.

        This is a centralized measurement-level validation baseline.  It is
        deliberately opt-in and emits one target report rather than pretending
        that the stacked update is a distributed node-local posterior.
        """

        if not self.enable_unified_multimodal_update or self._manager is None:
            return [], set()
        reports = []
        updated_targets = set()
        for target_id, target_observations in sorted(
            observations_by_target.items()
        ):
            if target_id not in self._manager.tracks:
                continue
            values = tuple(
                item for item in target_observations
                if item.valid_flag and item.modality in {"RADAR", "LOS"}
            )
            if not values:
                continue
            values = self._preprocess_unified_observations(
                target_id=target_id,
                observations=values,
                timestamp=epoch,
            )
            if not values:
                continue
            prior = self._manager.tracks[target_id].estimate
            if epoch <= prior.timestamp:
                continue
            process_noise_scale = self.maneuver_detector.process_noise_scale(
                target_id
            )
            process_noise = self.process_noise_acceleration * process_noise_scale
            use_inverse_depth = (
                self.enable_inverse_depth_los_update
                and not any(item.modality == "RADAR" for item in values)
                and len({item.observer_id for item in values}) >= 2
            )
            if use_inverse_depth:
                self._multimodal_windows.pop(target_id, None)
                update = inverse_depth_multi_observer_bearing_update(
                    prior,
                    values,
                    timestamp=epoch,
                    process_noise_acceleration=process_noise,
                )
            elif self.unified_multimodal_window_epochs > 1:
                window = self._multimodal_windows.get(target_id)
                if window is None:
                    window = FixedLagMultimodalWindow(
                        prior,
                        maximum_epochs=self.unified_multimodal_window_epochs,
                        process_noise_acceleration=process_noise,
                    )
                    self._multimodal_windows[target_id] = window
                window.append(values, timestamp=epoch)
                update = window.estimate()
            else:
                update = unified_multi_observer_multimodal_update(
                    prior,
                    values,
                    timestamp=epoch,
                    process_noise_acceleration=process_noise,
                )
            information_ids = tuple(dict.fromkeys((
                *getattr(prior, "information_ids", ()),
                *update.measurement_ids,
            )))
            observation_count = getattr(
                update,
                "observation_count",
                getattr(update, "measurement_count", len(values)),
            )
            total = max(observation_count, 1)
            modality_weights = {
                name.lower(): count / total
                for name, count in update.modality_counts.items()
            }
            quality = quality_score_from_covariance(update.covariance_eci)
            reports.append(TargetNodeReport(
                observer_id=update.observer_ids[0],
                target_id=target_id,
                track_id=prior.track_id,
                timestamp=epoch,
                state_eci=update.state_eci,
                covariance_eci=update.covariance_eci,
                quality_score=float(np.clip(
                    quality / (1.0 + quality), 0.0, 1.0,
                )),
                valid_flag=True,
                modality_weights=modality_weights,
                used_measurement_ids=information_ids,
                information_capabilities=getattr(
                    update, "information_capabilities", (),
                ),
            ))
            updated_targets.add(target_id)
            shared_prior = TargetInitialState(
                target_id=target_id,
                track_id=prior.track_id,
                timestamp=epoch,
                state_eci=update.state_eci,
                covariance_eci=update.covariance_eci,
            )
            for observer_id in update.observer_ids:
                key = (observer_id, target_id)
                self._local_priors[key] = shared_prior
                self._local_prior_information_ids[key] = information_ids
            self._multi_epoch_bearing_windows.pop(target_id, None)
            self._last_unified_multimodal_update_count += 1
            self._last_unified_multimodal_observation_count += (
                len(values)
            )
        return reports, updated_targets

    def _joint_bearing_reports(self, epoch, observations_by_target):
        if not self.enable_multi_observer_bearing_update or self._manager is None:
            return [], set()
        reports = []
        updated_targets = set()
        for target_id, target_observations in sorted(
            observations_by_target.items()
        ):
            values = tuple(item for item in target_observations if item.valid_flag)
            if any(item.modality == "RADAR" for item in values):
                self._multi_epoch_bearing_windows.pop(target_id, None)
                continue
            if (
                self.enable_local_unified_multimodal_update
                and any(
                    item.metadata.get("rangeProxyM") is not None
                    or item.metadata.get("rangeRateProxyMps") is not None
                    for item in values
                )
            ):
                # A LOS carrier with an independent radial-backup block is not
                # a pure bearing-only epoch.  Leave it for the node-local
                # unified update so the radial rows are neither discarded here
                # nor double-counted in a second update.
                self._multi_epoch_bearing_windows.pop(target_id, None)
                continue
            lines = tuple(item for item in values if item.modality == "LOS")
            observer_ids = tuple(sorted({item.observer_id for item in lines}))
            if len(observer_ids) < 2 or target_id not in self._manager.tracks:
                continue
            lines = self._preprocess_joint_bearing_lines(
                target_id=target_id,
                observations=lines,
                timestamp=epoch,
            )
            observer_ids = tuple(sorted({item.observer_id for item in lines}))
            if len(observer_ids) < 2:
                continue
            prior = self._manager.tracks[target_id].estimate
            single_epoch_update = batch_multi_observer_bearing_update(
                prior,
                lines,
                timestamp=epoch,
                process_noise_acceleration=self.process_noise_acceleration,
            )
            if self.multi_epoch_bearing_window_epochs > 1:
                window = self._multi_epoch_bearing_windows.get(target_id)
                if window is None:
                    window = MultiEpochBearingWindow(
                        prior,
                        maximum_epochs=self.multi_epoch_bearing_window_epochs,
                        process_noise_acceleration=(
                            self.process_noise_acceleration
                        ),
                    )
                    self._multi_epoch_bearing_windows[target_id] = window
                window.append(
                    lines, timestamp=epoch, restart_prior=prior,
                )
                candidate = window.estimate()
                update = _consistent_multi_epoch_candidate(
                    single_epoch_update, candidate,
                )
            else:
                update = single_epoch_update
            information_ids = tuple(dict.fromkeys((
                *getattr(prior, "information_ids", ()),
                *update.measurement_ids,
            )))
            total = max(update.measurement_count, 1)
            modality_weights = {
                name.lower(): count / total
                for name, count in update.modality_counts.items()
            }
            report = TargetNodeReport(
                observer_id=observer_ids[0],
                target_id=target_id,
                track_id=prior.track_id,
                timestamp=epoch,
                state_eci=update.state_eci,
                covariance_eci=update.covariance_eci,
                quality_score=float(np.clip(
                    quality_score_from_covariance(update.covariance_eci)
                    / (1.0 + quality_score_from_covariance(update.covariance_eci)),
                    0.0,
                    1.0,
                )),
                valid_flag=True,
                modality_weights=modality_weights,
                used_measurement_ids=information_ids,
            )
            reports.append(report)
            updated_targets.add(target_id)
            joint_prior = TargetInitialState(
                target_id=target_id,
                track_id=prior.track_id,
                timestamp=epoch,
                state_eci=update.state_eci,
                covariance_eci=update.covariance_eci,
            )
            for observer_id in observer_ids:
                key = (observer_id, target_id)
                self._local_priors[key] = joint_prior
                self._local_prior_information_ids[key] = information_ids
            self._last_joint_bearing_update_count += 1
            self._last_joint_bearing_measurement_count += update.measurement_count
        return reports, updated_targets

    def _preprocess_joint_bearing_lines(
        self, *, target_id, observations, timestamp,
    ):
        """Run per-stream measurement preprocessing before the joint LOS EKF.

        The CANN boundary operates on the same SPRI optical/infrared
        ``Observation`` contract as the local filters.  The joint estimator,
        however, consumes unit LOS vectors in ECI.  This adapter performs that
        round trip while retaining measurement identity and covariance.
        """

        return tuple(
            item for item in self._preprocess_unified_observations(
                target_id=target_id,
                observations=observations,
                timestamp=timestamp,
            )
            if item.modality == "LOS"
        )

    def _preprocess_unified_observations(
        self, *, target_id, observations, timestamp,
    ):
        """Round-trip radar and angular observations through optional CANN."""

        if self.measurement_preprocessor_factory is None:
            return tuple(observations)
        grouped = defaultdict(list)
        for item in observations:
            grouped[item.observer_id].append(item)
        result = []
        for observer_id, items in sorted(grouped.items()):
            quaternion_key = (observer_id, float(timestamp))
            quaternion = self._observer_quaternions[quaternion_key]
            source = [
                _tracking_observation(item, target_id, quaternion)
                for item in items
            ]
            processed = self._preprocess_measurements(
                observer_id=observer_id,
                target_id=target_id,
                observations=source,
                timestamp=timestamp,
            )
            for item, observation in zip(items, processed, strict=True):
                if not observation.valid_flag:
                    continue
                if item.modality == "RADAR":
                    result.append(replace(
                        item,
                        measurement=np.asarray(
                            observation.measurement, dtype=float,
                        ).reshape(2),
                        covariance=np.asarray(
                            observation.covariance, dtype=float,
                        ).reshape(2, 2),
                        valid_flag=True,
                        metadata={
                            **item.metadata,
                            **observation.metadata,
                            "cann_preprocessed": True,
                        },
                    ))
                    continue
                line_spri, line_jacobian_spri = (
                    _angular_observation_to_spri_line_and_jacobian(
                        observation,
                    )
                )
                line_eci = rotate_pri_to_eci(line_spri, quaternion)
                line_jacobian_eci = np.column_stack([
                    rotate_pri_to_eci(
                        line_jacobian_spri[:, axis], quaternion,
                    )
                    for axis in range(2)
                ])
                line_covariance_eci = (
                    line_jacobian_eci
                    @ np.asarray(observation.covariance, dtype=float)
                    @ line_jacobian_eci.T
                )
                metadata = {
                    **item.metadata,
                    **observation.metadata,
                    "cann_preprocessed": True,
                    "lineOfSightCovarianceEci": line_covariance_eci,
                }
                result.append(replace(
                    item,
                    measurement=line_eci,
                    covariance=np.asarray(observation.covariance, dtype=float),
                    valid_flag=True,
                    metadata=metadata,
                ))
        return tuple(result)

    def _preprocess_measurements(
        self, *, observer_id, target_id, observations, timestamp,
    ):
        if self.measurement_preprocessor_factory is None:
            return observations
        key = (str(observer_id), str(target_id))
        processor = self._measurement_preprocessor_by_stream.get(key)
        if processor is None:
            processor = self.measurement_preprocessor_factory(*key)
            if processor is None or not callable(getattr(processor, "process", None)):
                raise TypeError(
                    "measurement_preprocessor_factory must return a processor."
                )
            self._measurement_preprocessor_by_stream[key] = processor
        started = time.perf_counter()
        processed = processor.process(
            list(observations), np.asarray([timestamp], dtype=float),
        )
        self._last_preprocessing_seconds += time.perf_counter() - started
        result = validate_preprocessed_stream(
            observations, processed, observer_id=observer_id,
        )
        self._last_preprocessed_observation_count += len(result)
        for observation in result:
            modality = str(observation.modality).upper()
            diagnostics = observation.metadata.get("cann_diagnostics", {})
            detailed_pending = any(
                bool(value) and str(name).endswith("_recovery_pending")
                for name, value in diagnostics.items()
            )
            for name, value in diagnostics.items():
                action = str(name)
                if not bool(value):
                    continue
                if action == "recovery_pending" and detailed_pending:
                    continue
                if (
                    action.endswith("_substituted")
                    or action.endswith("_reanchored")
                    or action.endswith("_recovery_pending")
                    or action == "recovery_pending"
                ):
                    self._last_cann_action_counts[f"{modality}.{action}"] += 1
        return result


def _angular_observation_to_spri_line(observation: Observation) -> np.ndarray:
    line, _ = _angular_observation_to_spri_line_and_jacobian(observation)
    return line


def _angular_observation_to_spri_line_and_jacobian(
    observation: Observation,
) -> tuple[np.ndarray, np.ndarray]:
    measurement = np.asarray(observation.measurement, dtype=float).reshape(2)
    modality = str(observation.modality).upper()
    if modality == "OPTICAL":
        raw_line = np.array(
            [measurement[0], measurement[1], 1.0], dtype=float,
        )
        raw_jacobian = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
    elif modality == "INFRARED":
        azimuth, elevation = measurement
        cosine_elevation = np.cos(elevation)
        sine_elevation = np.sin(elevation)
        cosine_azimuth = np.cos(azimuth)
        sine_azimuth = np.sin(azimuth)
        line = np.array([
            cosine_elevation * cosine_azimuth,
            cosine_elevation * sine_azimuth,
            sine_elevation,
        ])
        jacobian = np.array([
            [-cosine_elevation * sine_azimuth,
             -sine_elevation * cosine_azimuth],
            [cosine_elevation * cosine_azimuth,
             -sine_elevation * sine_azimuth],
            [0.0, cosine_elevation],
        ])
        return line, jacobian
    else:
        raise ValueError(
            "Joint bearing preprocessing accepts optical or infrared data only."
        )
    norm = float(np.linalg.norm(raw_line))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("Preprocessed angular observation produced an invalid LOS.")
    line = raw_line / norm
    normalization_jacobian = (
        np.eye(3) - np.outer(line, line)
    ) / norm
    return line, normalization_jacobian @ raw_jacobian


def _consistent_multi_epoch_candidate(
    reference,
    candidate,
    *,
    gate_squared_mahalanobis: float = 25.0,
    maximum_position_shift_m: float = 50.0,
    maximum_velocity_shift_mps: float = 0.5,
):
    """Use a window solution only when it agrees with the stable EKF path."""

    state_difference = np.asarray(candidate.state_eci - reference.state_eci)
    if np.linalg.norm(state_difference[:3]) > maximum_position_shift_m:
        return reference
    if np.linalg.norm(state_difference[3:]) > maximum_velocity_shift_mps:
        return reference
    covariance = (
        np.asarray(candidate.covariance_eci, dtype=float)
        + np.asarray(reference.covariance_eci, dtype=float)
    )
    covariance = 0.5 * (covariance + covariance.T)
    distance = float(
        state_difference @ np.linalg.pinv(covariance) @ state_difference
    )
    if not np.isfinite(distance) or distance > gate_squared_mahalanobis:
        return reference
    return candidate

