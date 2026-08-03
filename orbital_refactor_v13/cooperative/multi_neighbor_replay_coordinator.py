from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from cooperative.exact_transport_protocol import apply_exact_transport_state_message
from cooperative.multi_neighbor_schmidt import (
    MultiNeighborSchmidtState,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_update,
)
from cooperative.schmidt_refresh import refresh_consider_neighbor
from interfaces.data_objects import CovarianceTransportEvent, ObservationMessage, StateMessage


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
        self._checkpoints = {float(initial_state.timestamp): initial_state}
        self._posterior_states = {float(initial_state.timestamp): initial_state}
        self._observations: dict[str, ObservationMessage] = {}
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

    def apply_observation(self, observation: ObservationMessage) -> float:
        if not np.isclose(float(observation.timestamp), float(self.state.timestamp)):
            raise ValueError("Observation must be applied at the coordinator's current time.")
        self._observations[observation.information_id] = observation
        update = multi_neighbor_schmidt_update(self.state, observation)
        self.state = update.state
        self._posterior_states[float(observation.timestamp)] = self.state
        self._enforce_resource_limits(float(self.state.timestamp))
        self._record_resource_peaks()
        return update.nis

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
                if not event_ids:
                    failure = CoordinatorMessageResult(False, "missing_event_information_id"); break
                key = (neighbor_id, "|".join(event_ids))
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
    ) -> None:
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
        times = sorted(event_times | observation_times | {current_timestamp})
        remote_count = 0; observation_count = 0
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
                    tuple(str(value) for value in item.event.information_ids),
                ),
            )
            for item in remote:
                event_ids = tuple(str(value) for value in item.event.information_ids)
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
                key=lambda item: item.information_id,
            )
            for observation in observations:
                if observation.information_id not in current.information_ids:
                    current = multi_neighbor_schmidt_update(current, observation).state
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
                retained_remote = sum(
                    item.neighbor_id == link_key[0] and float(item.event.timestamp) >= timestamp
                    for item in self._remote_events.values()
                )
                if retained_observations + retained_remote > self.max_retained_events:
                    reason = "max_retained_events_exceeded"
            if reason is not None:
                self._pinned_checkpoints.pop(link_key, None)
                self._resync_required[link_key] = reason

    def _record_resource_peaks(self) -> None:
        performance = self.performance
        performance.maximum_remote_event_count = max(
            performance.maximum_remote_event_count, len(self._remote_events)
        )
        performance.maximum_observation_count = max(
            performance.maximum_observation_count, len(self._observations)
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
            len(self._remote_events) + len(self._observations),
        )


def _same_event(left: CovarianceTransportEvent, right: CovarianceTransportEvent) -> bool:
    return (
        np.isclose(float(left.timestamp), float(right.timestamp))
        and tuple(left.information_ids) == tuple(right.information_ids)
        and np.allclose(left.state_estimate, right.state_estimate)
        and np.allclose(left.error_transition, right.error_transition)
        and np.allclose(left.independent_process_noise, right.independent_process_noise)
    )
