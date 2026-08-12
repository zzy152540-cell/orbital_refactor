from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping

import numpy as np
from orbital_core.measurement_integrity import MeasurementIntegrityPolicy

from cooperative.exact_transport_protocol import apply_exact_transport_state_message
from cooperative.multi_neighbor_schmidt import (
    MultiNeighborSchmidtState,
    MultiNeighborSchmidtUpdateResult,
    multi_neighbor_schmidt_batch_update,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_absolute_position_update,
    multi_neighbor_schmidt_update,
)
from cooperative.schmidt_refresh import refresh_consider_neighbor
from cooperative.neighbor_measurement_quality import (
    NeighborLinkQuality,
    NeighborMeasurementQualityPolicy,
)
from interfaces.data_objects import (
    AbsolutePositionObservation,
    CovarianceTransportEvent,
    ObservationMessage,
    StateMessage,
)

Array = np.ndarray


@dataclass(frozen=True)
class RemoteTransportEvent:
    neighbor_id: str
    event: CovarianceTransportEvent


@dataclass(frozen=True)
class CoordinatorMessageResult:
    accepted: bool
    reason: str
    replayed_event_count: int = 0


@dataclass(frozen=True)
class ResynchronizationBaseline:
    neighbor_id: str
    lineage_id: str
    timestamp: float
    state_estimate: np.ndarray
    covariance: np.ndarray


@dataclass
class ReplayPerformanceStats:
    replay_count: int = 0
    batch_count: int = 0
    total_replay_seconds: float = 0.0
    maximum_replay_seconds: float = 0.0
    total_replay_span: float = 0.0
    maximum_replay_span: float = 0.0
    replayed_remote_events: int = 0
    replayed_observations: int = 0
    maximum_batch_size: int = 0
    fallback_count: int = 0
    maximum_remote_event_count: int = 0
    maximum_observation_count: int = 0
    maximum_checkpoint_count: int = 0
    maximum_posterior_state_count: int = 0
    maximum_pinned_checkpoint_count: int = 0
    maximum_resync_required_count: int = 0
    maximum_retained_journal_count: int = 0


class MultiNeighborReplayCoordinator:
    """Own one coherent replay timeline for a node and all its neighbors."""

    def __init__(
        self, initial_state: MultiNeighborSchmidtState, *,
        process_noise_acceleration: float = 1e-4,
        history_window: float | None = None,
        max_pinned_age: float | None = None,
        max_retained_events: int | None = None,
        nis_gate_threshold_by_modality: Mapping[str, float] | None = None,
        nis_inflation_threshold_by_modality: Mapping[str, float] | None = None,
        maximum_measurement_covariance_scale_by_modality: Mapping[str, float] | None = None,
        integrity_policy_by_modality: Mapping[
            str, MeasurementIntegrityPolicy
        ] | None = None,
        relative_observation_order: tuple[str, ...] | None = None,
        relative_observation_order_start_time: float | None = None,
        batch_relative_observations: bool = False,
        batch_relative_observations_start_time: float | None = None,
        neighbor_linearization_state_by_node_and_time: Mapping[
            tuple[str, float], Array
        ] | None = None,
        neighbor_uncertainty_inflation_by_modality: Mapping[
            str, float
        ] | None = None,
        neighbor_measurement_quality_policy: (
            NeighborMeasurementQualityPolicy | None
        ) = None,
        neighbor_link_quality_by_node_and_time: Mapping[
            tuple[str, str, float], NeighborLinkQuality
        ] | None = None,
    ) -> None:
        if history_window is not None and history_window < 0.0:
            raise ValueError("history_window cannot be negative.")
        if max_pinned_age is not None and max_pinned_age <= 0.0:
            raise ValueError("max_pinned_age must be positive.")
        if max_retained_events is not None and max_retained_events < 1:
            raise ValueError("max_retained_events must be at least one.")
        self.state = initial_state
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.history_window = history_window
        self.max_pinned_age = max_pinned_age
        self.max_retained_events = max_retained_events
        self.nis_gate_threshold_by_modality = {
            str(modality): float(threshold)
            for modality, threshold in (nis_gate_threshold_by_modality or {}).items()
        }
        self.nis_inflation_threshold_by_modality = {
            str(modality): float(threshold)
            for modality, threshold in (nis_inflation_threshold_by_modality or {}).items()
        }
        self.maximum_measurement_covariance_scale_by_modality = {
            str(modality): float(scale)
            for modality, scale in (
                maximum_measurement_covariance_scale_by_modality or {}
            ).items()
        }
        self.integrity_policy_by_modality = dict(
            integrity_policy_by_modality or {}
        )
        self.relative_observation_rank = {
            modality: index
            for index, modality in enumerate(relative_observation_order or ())
        }
        self.relative_observation_order_start_time = (
            None if relative_observation_order_start_time is None
            else float(relative_observation_order_start_time)
        )
        self.batch_relative_observations = bool(batch_relative_observations)
        self.batch_relative_observations_start_time = (
            None if batch_relative_observations_start_time is None
            else float(batch_relative_observations_start_time)
        )
        self.neighbor_linearization_state_by_node_and_time = {
            (str(node), float(timestamp)): np.asarray(
                state, dtype=float
            ).reshape(6).copy()
            for (node, timestamp), state in (
                neighbor_linearization_state_by_node_and_time or {}
            ).items()
        }
        self.neighbor_uncertainty_inflation_by_modality = {
            str(modality): float(value)
            for modality, value in (
                neighbor_uncertainty_inflation_by_modality or {}
            ).items()
        }
        self.neighbor_measurement_quality_policy = (
            neighbor_measurement_quality_policy
        )
        self.neighbor_link_quality_by_node_and_time = dict(
            neighbor_link_quality_by_node_and_time or {}
        )
        if any(
            not np.isfinite(value) or value < 0.0
            for value in self.neighbor_uncertainty_inflation_by_modality.values()
        ):
            raise ValueError(
                "Neighbor uncertainty inflation must be finite and nonnegative."
            )
        if not all(
            isinstance(value, MeasurementIntegrityPolicy)
            for value in self.integrity_policy_by_modality.values()
        ):
            raise TypeError("Integrity policies must be MeasurementIntegrityPolicy values.")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in self.nis_gate_threshold_by_modality.values()
        ):
            raise ValueError("NIS gate thresholds must be finite and positive.")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in self.nis_inflation_threshold_by_modality.values()
        ):
            raise ValueError("NIS inflation thresholds must be finite and positive.")
        if any(
            not np.isfinite(value) or value < 1.0
            for value in self.maximum_measurement_covariance_scale_by_modality.values()
        ):
            raise ValueError("Maximum covariance scales must be finite and at least one.")
        self._checkpoints = {float(initial_state.timestamp): initial_state}
        self._posterior_states = {float(initial_state.timestamp): initial_state}
        self._observations: dict[str, ObservationMessage] = {}
        self._absolute_observations: dict[str, AbsolutePositionObservation] = {}
        self.integrity_by_information_id = {}
        self.last_relative_update: MultiNeighborSchmidtUpdateResult | None = None
        self._remote_events: dict[tuple[str, str], RemoteTransportEvent] = {}
        self._pinned_checkpoints: dict[tuple[str, str | None], tuple[float, MultiNeighborSchmidtState]] = {}
        self._resync_required: dict[tuple[str, str | None], str] = {}
        self.performance = ReplayPerformanceStats()
        self._record_resource_peaks()

    def advance(self, timestamp: float) -> MultiNeighborSchmidtState:
        timestamp = float(timestamp)
        if timestamp <= float(self.state.timestamp):
            raise ValueError("advance timestamp must be later than current state.")
        self.state = multi_neighbor_schmidt_predict(
            self.state, timestamp,
            process_noise_acceleration=self.process_noise_acceleration,
        )
        self._checkpoints[timestamp] = self.state
        self._posterior_states[timestamp] = self.state
        self._prune(timestamp)
        self._enforce_resource_limits(timestamp)
        self._record_resource_peaks()
        return self.state

    def apply_observation(
        self, observation: ObservationMessage, *,
        neighbor_linearization_state: Array | None = None,
    ) -> float:
        if not np.isclose(float(observation.timestamp), float(self.state.timestamp)):
            raise ValueError("Observation must be applied at the coordinator's current time.")
        self._observations[observation.information_id] = observation
        if neighbor_linearization_state is None:
            neighbor_linearization_state = (
                self.neighbor_linearization_state_by_node_and_time.get((
                    _counterpart(self.state, observation),
                    float(observation.timestamp),
                ))
            )
        update = multi_neighbor_schmidt_update(
            self.state, observation,
            neighbor_linearization_state=neighbor_linearization_state,
            neighbor_uncertainty_inflation=(
                self._neighbor_inflation(observation)
            ),
            nis_gate_threshold=self.nis_gate_threshold_by_modality.get(
                observation.modality
            ),
            nis_inflation_threshold=self.nis_inflation_threshold_by_modality.get(
                observation.modality
            ),
            maximum_measurement_covariance_scale=(
                self.maximum_measurement_covariance_scale_by_modality.get(
                    observation.modality, 1.0
                )
            ),
            integrity_policy=self.integrity_policy_by_modality.get(
                observation.modality
            ),
        )
        self.state = update.state
        self.last_relative_update = update
        self.integrity_by_information_id[observation.information_id] = (
            update.integrity
        )
        self._posterior_states[float(observation.timestamp)] = self.state
        self._enforce_resource_limits(float(self.state.timestamp))
        self._record_resource_peaks()
        return update.nis

    def apply_observation_batch(
        self, observations: tuple[ObservationMessage, ...], *,
        neighbor_linearization_state: Array | None = None,
    ) -> MultiNeighborSchmidtUpdateResult:
        if not observations:
            raise ValueError("Observation batch cannot be empty.")
        if any(
            not np.isclose(float(item.timestamp), float(self.state.timestamp))
            for item in observations
        ):
            raise ValueError("Observation batch must be applied at current time.")
        for observation in observations:
            self._observations[observation.information_id] = observation
        if neighbor_linearization_state is None:
            neighbor_linearization_state = (
                self.neighbor_linearization_state_by_node_and_time.get((
                    _counterpart(self.state, observations[0]),
                    float(observations[0].timestamp),
                ))
            )
        update = multi_neighbor_schmidt_batch_update(
            self.state, observations,
            neighbor_linearization_state=neighbor_linearization_state,
            neighbor_uncertainty_inflation_by_modality=(
                {
                    observation.modality: self._neighbor_inflation(observation)
                    for observation in observations
                }
            ),
        )
        self.state = update.state
        self.last_relative_update = update
        for observation in observations:
            self.integrity_by_information_id[observation.information_id] = (
                update.integrity
            )
        self._posterior_states[float(observations[0].timestamp)] = self.state
        self._enforce_resource_limits(float(self.state.timestamp))
        self._record_resource_peaks()
        return update

    def apply_delayed_observation(
        self, observation: ObservationMessage,
    ) -> float | None:
        """Insert one arrived observation at source time and replay once."""

        information_id = observation.information_id
        existing = self._observations.get(information_id)
        if existing is not None:
            if not _same_observation(existing, observation):
                raise ValueError("Conflicting copies share a physical observation ID.")
            return None
        timestamp = float(observation.timestamp)
        current_timestamp = float(self.state.timestamp)
        if timestamp > current_timestamp:
            raise ValueError("Observation timestamp cannot be in the future.")
        if np.isclose(timestamp, current_timestamp):
            return self.apply_observation(observation)
        if timestamp not in self._checkpoints:
            raise ValueError("Observation timestamp is unavailable in replay history.")
        self._observations[information_id] = observation
        nis_by_id = self._replay_from(timestamp)
        self._enforce_resource_limits(current_timestamp)
        self._record_resource_peaks()
        return nis_by_id[information_id]

    def apply_absolute_observation(
        self, observation: AbsolutePositionObservation,
    ) -> float:
        if not np.isclose(float(observation.timestamp), float(self.state.timestamp)):
            raise ValueError(
                "Absolute observation must be applied at the coordinator's current time."
            )
        self._absolute_observations[observation.information_id] = observation
        update = multi_neighbor_schmidt_absolute_position_update(
            self.state, observation,
            nis_gate_threshold=self.nis_gate_threshold_by_modality.get(
                "ABSOLUTE_POSITION"
            ),
            nis_inflation_threshold=self.nis_inflation_threshold_by_modality.get(
                "ABSOLUTE_POSITION"
            ),
            maximum_measurement_covariance_scale=(
                self.maximum_measurement_covariance_scale_by_modality.get(
                    "ABSOLUTE_POSITION", 1.0
                )
            ),
            integrity_policy=self.integrity_policy_by_modality.get(
                "ABSOLUTE_POSITION"
            ),
        )
        self.state = update.state
        self.integrity_by_information_id[observation.information_id] = (
            update.integrity
        )
        self._posterior_states[float(observation.timestamp)] = self.state
        self._enforce_resource_limits(float(self.state.timestamp))
        self._record_resource_peaks()
        return update.nis

    def apply_delayed_absolute_observation(
        self, observation: AbsolutePositionObservation,
    ) -> float | None:
        information_id = observation.information_id
        if information_id in self._absolute_observations:
            return None
        timestamp = float(observation.timestamp)
        current_timestamp = float(self.state.timestamp)
        if timestamp > current_timestamp:
            raise ValueError("Absolute observation timestamp cannot be in the future.")
        if np.isclose(timestamp, current_timestamp):
            return self.apply_absolute_observation(observation)
        if timestamp not in self._checkpoints:
            raise ValueError("Absolute observation timestamp is unavailable in replay history.")
        self._absolute_observations[information_id] = observation
        nis_by_id = self._replay_from(timestamp)
        self._enforce_resource_limits(current_timestamp)
        self._record_resource_peaks()
        return nis_by_id[information_id]

    def apply_state_message(
        self, message: StateMessage, *, expected_lineage_id: str | None = None,
    ) -> CoordinatorMessageResult:
        return self._apply_state_messages(
            ((message, expected_lineage_id),), allow_fallback=False,
        )[0]

    def apply_state_messages(
        self, messages: tuple[tuple[StateMessage, str | None], ...],
    ) -> tuple[CoordinatorMessageResult, ...]:
        return self._apply_state_messages(messages, allow_fallback=len(messages) > 1)

    def _apply_state_messages(
        self, messages: tuple[tuple[StateMessage, str | None], ...], *,
        allow_fallback: bool,
    ) -> tuple[CoordinatorMessageResult, ...]:
        if not messages:
            return ()
        self.performance.batch_count += 1
        self.performance.maximum_batch_size = max(
            self.performance.maximum_batch_size, len(messages)
        )
        results: list[CoordinatorMessageResult | None] = [None] * len(messages)
        staged = []
        all_new_keys = []
        for index, (message, expected_lineage_id) in enumerate(messages):
            validation = self._validated_checkpoint(message, expected_lineage_id)
            if isinstance(validation, CoordinatorMessageResult):
                results[index] = validation
                continue
            checkpoint, reference_timestamp, link_key = validation
            neighbor_id = str(message.source_node_id); new_keys = []
            failure = None
            for event in message.transport_events:
                event_ids = tuple(str(value) for value in event.information_ids)
                event_key = _transport_event_key(event)
                if event_key is None:
                    failure = CoordinatorMessageResult(False, "missing_event_information_id"); break
                key = (neighbor_id, event_key)
                existing = self._remote_events.get(key)
                if existing is not None:
                    if not _same_event(existing.event, event):
                        failure = CoordinatorMessageResult(False, "conflicting_event_information_id")
                    continue
                self._remote_events[key] = RemoteTransportEvent(neighbor_id, event)
                new_keys.append(key); all_new_keys.append(key)
            if failure is not None:
                for key in new_keys: self._remote_events.pop(key, None)
                results[index] = failure; continue
            staged.append((
                index, message, expected_lineage_id, checkpoint,
                reference_timestamp, link_key, new_keys,
            ))
        if staged:
            earliest = min(staged, key=lambda item: item[4])
            self._replay_from(earliest[4], starting_state=earliest[3])
            mismatched = False
            for _, message, _, _, _, _, _ in staged:
                endpoint = self._posterior_states.get(float(message.timestamp))
                neighbor_id = str(message.source_node_id)
                if not (endpoint is not None and np.allclose(
                    endpoint.neighbor_state_by_id[neighbor_id], message.state_estimate,
                    rtol=1e-9, atol=1e-7,
                ) and np.allclose(
                    endpoint.neighbor_covariance(neighbor_id), message.covariance,
                    rtol=1e-8, atol=1e-10,
                )):
                    mismatched = True; break
            if mismatched:
                for key in all_new_keys: self._remote_events.pop(key, None)
                self._replay_from(earliest[4], starting_state=earliest[3])
                if allow_fallback:
                    self.performance.fallback_count += 1
                    for index, message, expected_lineage_id, *_ in staged:
                        results[index] = self.apply_state_message(
                            message, expected_lineage_id=expected_lineage_id,
                        )
                else:
                    for index, *_ in staged:
                        results[index] = CoordinatorMessageResult(
                            False, "event_bundle_endpoint_mismatch"
                        )
            else:
                for index, message, _, _, _, link_key, new_keys in staged:
                    self._pinned_checkpoints[link_key] = (
                        float(message.timestamp), self._posterior_states[float(message.timestamp)]
                    )
                    self._resync_required.pop(link_key, None)
                    results[index] = CoordinatorMessageResult(True, "accepted", len(new_keys))
                self._enforce_resource_limits(float(self.state.timestamp))
        self._record_resource_peaks()
        if any(result is None for result in results):
            raise RuntimeError("Every batched state message must produce a result.")
        return tuple(results)  # type: ignore[return-value]

    def _validated_checkpoint(self, message, expected_lineage_id):
        if not message.transport_events:
            return CoordinatorMessageResult(False, "missing_transport_events")
        if message.reference_timestamp is None:
            return CoordinatorMessageResult(False, "missing_provenance")
        reference_timestamp = float(message.reference_timestamp)
        link_key = (str(message.source_node_id), message.lineage_id)
        if link_key in self._resync_required:
            return CoordinatorMessageResult(False, "resync_required")
        pinned = self._pinned_checkpoints.get(link_key)
        candidates = [self._checkpoints.get(reference_timestamp), self._posterior_states.get(reference_timestamp)]
        if pinned is not None and np.isclose(pinned[0], reference_timestamp): candidates.append(pinned[1])
        last_reason = "history_unavailable"
        for checkpoint in candidates:
            if checkpoint is None: continue
            validation = apply_exact_transport_state_message(
                checkpoint, message, expected_lineage_id=expected_lineage_id,
            )
            if validation.accepted:
                return checkpoint, reference_timestamp, link_key
            last_reason = validation.reason
        return CoordinatorMessageResult(False, last_reason)

    @property
    def checkpoint_timestamps(self) -> tuple[float, ...]:
        return tuple(sorted(self._checkpoints))

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    @property
    def pinned_checkpoint_count(self) -> int:
        return len(self._pinned_checkpoints)

    @property
    def oldest_pinned_timestamp(self) -> float | None:
        if not self._pinned_checkpoints:
            return None
        return min(value[0] for value in self._pinned_checkpoints.values())

    @property
    def resynchronization_requirements(self) -> dict[tuple[str, str | None], str]:
        return dict(self._resync_required)

    def establish_resynchronized_link(
        self, *, neighbor_id: str, lineage_id: str,
    ) -> ResynchronizationBaseline:
        neighbor_id = str(neighbor_id); lineage_id = str(lineage_id)
        if neighbor_id not in self.state.neighbor_ids:
            raise KeyError(f"Unknown consider neighbor: {neighbor_id}")
        for key in tuple(self._pinned_checkpoints):
            if key[0] == neighbor_id:
                self._pinned_checkpoints.pop(key, None)
        for key in tuple(self._resync_required):
            if key[0] == neighbor_id:
                self._resync_required.pop(key, None)
        self._remote_events = {
            key: value for key, value in self._remote_events.items()
            if value.neighbor_id != neighbor_id
        }
        link_key = (neighbor_id, lineage_id)
        self._pinned_checkpoints[link_key] = (float(self.state.timestamp), self.state)
        return ResynchronizationBaseline(
            neighbor_id=neighbor_id, lineage_id=lineage_id,
            timestamp=float(self.state.timestamp),
            state_estimate=self.state.neighbor_state_by_id[neighbor_id].copy(),
            covariance=self.state.neighbor_covariance(neighbor_id),
        )

    def _replay_from(
        self, reference_timestamp: float,
        starting_state: MultiNeighborSchmidtState | None = None,
    ) -> dict[str, float]:
        started = perf_counter()
        current_timestamp = float(self.state.timestamp)
        current = self._checkpoints[reference_timestamp] if starting_state is None else starting_state
        event_times = {
            float(item.event.timestamp) for item in self._remote_events.values()
            if reference_timestamp <= float(item.event.timestamp) <= current_timestamp
        }
        observation_times = {
            float(item.timestamp) for item in self._observations.values()
            if reference_timestamp <= float(item.timestamp) <= current_timestamp
        }
        absolute_observation_times = {
            float(item.timestamp) for item in self._absolute_observations.values()
            if reference_timestamp <= float(item.timestamp) <= current_timestamp
        }
        times = sorted(
            event_times | observation_times | absolute_observation_times
            | {current_timestamp}
        )
        remote_count = 0; observation_count = 0
        nis_by_id: dict[str, float] = {}
        for timestamp in times:
            if timestamp > float(current.timestamp):
                current = multi_neighbor_schmidt_predict(
                    current, timestamp,
                    process_noise_acceleration=self.process_noise_acceleration,
                )
            if starting_state is None or timestamp > reference_timestamp:
                self._checkpoints[timestamp] = current
            remote = sorted(
                (item for item in self._remote_events.values()
                 if np.isclose(float(item.event.timestamp), timestamp)),
                key=lambda item: (
                    item.neighbor_id,
                    _transport_event_key(item.event) or "",
                ),
            )
            for item in remote:
                event_ids = _transport_event_tracking_ids(item.event)
                used = set(event_ids) & set(current.transport_information_ids)
                if used == set(event_ids):
                    continue
                if used:
                    raise ValueError("Transport event partially overlaps replay state.")
                current = refresh_consider_neighbor(
                    current, neighbor_id=item.neighbor_id,
                    neighbor_state=item.event.state_estimate, mode="exact_transport",
                    error_transition=item.event.error_transition,
                    independent_process_noise=item.event.independent_process_noise,
                )
                current = replace(
                    current,
                    transport_information_ids=(*current.transport_information_ids, *event_ids),
                )
                remote_count += 1
            observations = sorted(
                (item for item in self._observations.values()
                 if np.isclose(float(item.timestamp), timestamp)),
                key=lambda item: (
                    self.relative_observation_rank.get(
                        item.modality, len(self.relative_observation_rank)
                    ) if self.relative_observation_order_start_time is None
                    or float(item.timestamp)
                    > self.relative_observation_order_start_time else 0,
                    item.information_id,
                ),
            )
            absolute_observations = sorted(
                (item for item in self._absolute_observations.values()
                 if np.isclose(float(item.timestamp), timestamp)),
                key=lambda item: item.information_id,
            )
            for observation in absolute_observations:
                if observation.information_id not in current.information_ids:
                    update = multi_neighbor_schmidt_absolute_position_update(
                        current, observation,
                        nis_gate_threshold=self.nis_gate_threshold_by_modality.get(
                            "ABSOLUTE_POSITION"
                        ),
                        nis_inflation_threshold=self.nis_inflation_threshold_by_modality.get(
                            "ABSOLUTE_POSITION"
                        ),
                        maximum_measurement_covariance_scale=(
                            self.maximum_measurement_covariance_scale_by_modality.get(
                                "ABSOLUTE_POSITION", 1.0
                            )
                        ),
                        integrity_policy=self.integrity_policy_by_modality.get(
                            "ABSOLUTE_POSITION"
                        ),
                    )
                    current = update.state
                    nis_by_id[observation.information_id] = update.nis
                    self.integrity_by_information_id[
                        observation.information_id
                    ] = update.integrity
                    observation_count += 1
            pending_observations = [
                item for item in observations
                if item.information_id not in current.information_ids
            ]
            batches = _relative_observation_batches(
                pending_observations,
                enabled=self.batch_relative_observations,
                start_time=self.batch_relative_observations_start_time,
            )
            for batch in batches:
                if len(batch) > 1:
                    neighbor_id = _counterpart(current, batch[0])
                    update = multi_neighbor_schmidt_batch_update(
                        current, batch,
                        neighbor_linearization_state=(
                            self.neighbor_linearization_state_by_node_and_time.get(
                                (neighbor_id, timestamp)
                            )
                        ),
                        neighbor_uncertainty_inflation_by_modality=(
                            {
                                item.modality: self._neighbor_inflation(item)
                                for item in batch
                            }
                        ),
                    )
                    current = update.state
                    for observation in batch:
                        nis_by_id[observation.information_id] = update.nis
                        self.integrity_by_information_id[
                            observation.information_id
                        ] = update.integrity
                        observation_count += 1
                    continue
                observation = batch[0]
                if observation.information_id not in current.information_ids:
                    update = multi_neighbor_schmidt_update(
                        current, observation,
                        neighbor_linearization_state=(
                            self.neighbor_linearization_state_by_node_and_time.get(
                                (_counterpart(current, observation), timestamp)
                            )
                        ),
                        neighbor_uncertainty_inflation=(
                            self._neighbor_inflation(observation)
                        ),
                        nis_gate_threshold=self.nis_gate_threshold_by_modality.get(
                            observation.modality
                        ),
                        nis_inflation_threshold=self.nis_inflation_threshold_by_modality.get(
                            observation.modality
                        ),
                        maximum_measurement_covariance_scale=(
                            self.maximum_measurement_covariance_scale_by_modality.get(
                                observation.modality, 1.0
                            )
                        ),
                        integrity_policy=self.integrity_policy_by_modality.get(
                            observation.modality
                        ),
                    )
                    current = update.state
                    nis_by_id[observation.information_id] = update.nis
                    self.integrity_by_information_id[
                        observation.information_id
                    ] = update.integrity
                    observation_count += 1
            self._posterior_states[timestamp] = current
        self.state = current
        elapsed = perf_counter() - started
        span = current_timestamp - reference_timestamp
        self.performance.replay_count += 1
        self.performance.total_replay_seconds += elapsed
        self.performance.maximum_replay_seconds = max(self.performance.maximum_replay_seconds, elapsed)
        self.performance.total_replay_span += span
        self.performance.maximum_replay_span = max(self.performance.maximum_replay_span, span)
        self.performance.replayed_remote_events += remote_count
        self.performance.replayed_observations += observation_count
        return nis_by_id

    def _prune(self, current_timestamp: float) -> None:
        if self.history_window is None:
            return
        oldest = current_timestamp - self.history_window
        pinned_oldest = self.oldest_pinned_timestamp
        journal_oldest = oldest if pinned_oldest is None else min(oldest, pinned_oldest)
        self._checkpoints = {
            timestamp: state for timestamp, state in self._checkpoints.items()
            if timestamp >= oldest
        }
        self._observations = {
            key: value for key, value in self._observations.items()
            if float(value.timestamp) >= journal_oldest
        }
        self._absolute_observations = {
            key: value for key, value in self._absolute_observations.items()
            if float(value.timestamp) >= journal_oldest
        }
        self._posterior_states = {
            timestamp: state for timestamp, state in self._posterior_states.items()
            if timestamp >= oldest
        }
        self._remote_events = {
            key: value for key, value in self._remote_events.items()
            if float(value.event.timestamp) >= journal_oldest
        }

    def _enforce_resource_limits(self, current_timestamp: float) -> None:
        for link_key, (timestamp, _) in tuple(self._pinned_checkpoints.items()):
            reason = None
            if self.max_pinned_age is not None and current_timestamp - timestamp > self.max_pinned_age:
                reason = "max_pinned_age_exceeded"
            if reason is None and self.max_retained_events is not None:
                retained_observations = sum(
                    float(item.timestamp) >= timestamp for item in self._observations.values()
                )
                retained_observations += sum(
                    float(item.timestamp) >= timestamp
                    for item in self._absolute_observations.values()
                )
                retained_remote = sum(
                    item.neighbor_id == link_key[0] and float(item.event.timestamp) >= timestamp
                    for item in self._remote_events.values()
                )
                if retained_observations + retained_remote > self.max_retained_events:
                    reason = "max_retained_events_exceeded"
            if reason is not None:
                self._pinned_checkpoints.pop(link_key, None)
                self._resync_required[link_key] = reason

    def _neighbor_inflation(self, observation) -> float:
        fixed = self.neighbor_uncertainty_inflation_by_modality.get(
            observation.modality, 0.0
        )
        if self.neighbor_measurement_quality_policy is None:
            return fixed
        neighbor_id = _counterpart(self.state, observation)
        quality = self.neighbor_link_quality_by_node_and_time.get(
            (
                self.state.active_node_id,
                neighbor_id,
                float(observation.timestamp),
            ),
            NeighborLinkQuality(),
        )
        dynamic = self.neighbor_measurement_quality_policy.inflation(
            modality=observation.modality,
            age=quality.age,
            consecutive_losses=quality.consecutive_losses,
            resynchronization_required=quality.resynchronization_required,
        )
        return max(fixed, dynamic)

    def _record_resource_peaks(self) -> None:
        performance = self.performance
        performance.maximum_remote_event_count = max(
            performance.maximum_remote_event_count, len(self._remote_events)
        )
        performance.maximum_observation_count = max(
            performance.maximum_observation_count,
            len(self._observations) + len(self._absolute_observations),
        )
        performance.maximum_checkpoint_count = max(
            performance.maximum_checkpoint_count, len(self._checkpoints)
        )
        performance.maximum_posterior_state_count = max(
            performance.maximum_posterior_state_count, len(self._posterior_states)
        )
        performance.maximum_pinned_checkpoint_count = max(
            performance.maximum_pinned_checkpoint_count,
            len(self._pinned_checkpoints),
        )
        performance.maximum_resync_required_count = max(
            performance.maximum_resync_required_count, len(self._resync_required)
        )
        performance.maximum_retained_journal_count = max(
            performance.maximum_retained_journal_count,
            len(self._remote_events) + len(self._observations)
            + len(self._absolute_observations),
        )


def _relative_observation_batches(observations, *, enabled, start_time):
    if not enabled:
        return tuple((item,) for item in observations)
    groups = []
    current_key = None
    current = []
    for observation in observations:
        use_batch = (
            start_time is None or float(observation.timestamp) > float(start_time)
        )
        key = (
            float(observation.timestamp),
            str(observation.observer_id),
            str(observation.target_id),
        ) if use_batch else (observation.information_id,)
        if current and key != current_key:
            groups.append(tuple(current))
            current = []
        current_key = key
        current.append(observation)
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _counterpart(state, observation):
    return (
        str(observation.target_id)
        if str(observation.observer_id) == state.active_node_id
        else str(observation.observer_id)
    )


def _same_event(left: CovarianceTransportEvent, right: CovarianceTransportEvent) -> bool:
    return (
        np.isclose(float(left.timestamp), float(right.timestamp))
        and left.event_id == right.event_id
        and tuple(left.information_ids) == tuple(right.information_ids)
        and np.allclose(left.state_estimate, right.state_estimate)
        and np.allclose(left.error_transition, right.error_transition)
        and np.allclose(left.independent_process_noise, right.independent_process_noise)
        and _optional_array_equal(
            left.source_error_transition, right.source_error_transition
        )
        and _optional_array_equal(
            left.source_process_noise, right.source_process_noise
        )
    )


def _transport_event_key(event: CovarianceTransportEvent) -> str | None:
    if event.event_id is not None and str(event.event_id):
        return f"event:{event.event_id}"
    information_ids = tuple(str(value) for value in event.information_ids)
    return f"information:{'|'.join(information_ids)}" if information_ids else None


def _transport_event_tracking_ids(event: CovarianceTransportEvent) -> tuple[str, ...]:
    if event.event_id is not None and str(event.event_id):
        return (f"transport-event:{event.event_id}",)
    return tuple(str(value) for value in event.information_ids)


def _optional_array_equal(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.allclose(left, right))


def _same_observation(left: ObservationMessage, right: ObservationMessage) -> bool:
    return (
        np.isclose(float(left.timestamp), float(right.timestamp))
        and str(left.observer_id) == str(right.observer_id)
        and str(left.target_id) == str(right.target_id)
        and str(left.modality) == str(right.modality)
        and str(left.frame) == str(right.frame)
        and np.allclose(left.measurement, right.measurement)
        and np.allclose(left.covariance, right.covariance)
    )
