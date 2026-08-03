from __future__ import annotations

from dataclasses import dataclass, replace

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


class MultiNeighborReplayCoordinator:
    """Own one coherent replay timeline for a node and all its neighbors."""

    def __init__(
        self, initial_state: MultiNeighborSchmidtState, *,
        process_noise_acceleration: float = 1e-4,
        history_window: float | None = None,
    ) -> None:
        if history_window is not None and history_window < 0.0:
            raise ValueError("history_window cannot be negative.")
        self.state = initial_state
        self.process_noise_acceleration = float(process_noise_acceleration)
        self.history_window = history_window
        self._checkpoints = {float(initial_state.timestamp): initial_state}
        self._posterior_states = {float(initial_state.timestamp): initial_state}
        self._observations: dict[str, ObservationMessage] = {}
        self._remote_events: dict[tuple[str, str], RemoteTransportEvent] = {}

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
        return self.state

    def apply_observation(self, observation: ObservationMessage) -> float:
        if not np.isclose(float(observation.timestamp), float(self.state.timestamp)):
            raise ValueError("Observation must be applied at the coordinator's current time.")
        self._observations[observation.information_id] = observation
        update = multi_neighbor_schmidt_update(self.state, observation)
        self.state = update.state
        self._posterior_states[float(observation.timestamp)] = self.state
        return update.nis

    def apply_state_message(
        self, message: StateMessage, *, expected_lineage_id: str | None = None,
    ) -> CoordinatorMessageResult:
        if not message.transport_events:
            return CoordinatorMessageResult(False, "missing_transport_events")
        reference_timestamp = message.reference_timestamp
        if reference_timestamp is None:
            return CoordinatorMessageResult(False, "missing_provenance")
        checkpoint = self._checkpoints.get(float(reference_timestamp))
        if checkpoint is None:
            return CoordinatorMessageResult(False, "history_unavailable")
        validation = apply_exact_transport_state_message(
            checkpoint, message, expected_lineage_id=expected_lineage_id,
        )
        if not validation.accepted:
            posterior_checkpoint = self._posterior_states.get(float(reference_timestamp))
            if posterior_checkpoint is not None:
                posterior_validation = apply_exact_transport_state_message(
                    posterior_checkpoint, message,
                    expected_lineage_id=expected_lineage_id,
                )
                if posterior_validation.accepted:
                    checkpoint = posterior_checkpoint
                    validation = posterior_validation
        if not validation.accepted:
            return CoordinatorMessageResult(False, validation.reason)
        neighbor_id = str(message.source_node_id)
        new_count = 0
        new_keys = []
        for event in message.transport_events:
            event_ids = tuple(str(value) for value in event.information_ids)
            if not event_ids:
                return CoordinatorMessageResult(False, "missing_event_information_id")
            key = (neighbor_id, "|".join(event_ids))
            existing = self._remote_events.get(key)
            if existing is not None:
                if not _same_event(existing.event, event):
                    return CoordinatorMessageResult(False, "conflicting_event_information_id")
                continue
            self._remote_events[key] = RemoteTransportEvent(neighbor_id, event)
            new_keys.append(key)
            new_count += 1
        self._replay_from(float(reference_timestamp), starting_state=checkpoint)
        endpoint = self._posterior_states.get(float(message.timestamp))
        endpoint_matches = endpoint is not None and np.allclose(
            endpoint.neighbor_state_by_id[neighbor_id], message.state_estimate,
            rtol=1e-9, atol=1e-7,
        ) and np.allclose(
            endpoint.neighbor_covariance(neighbor_id), message.covariance,
            rtol=1e-8, atol=1e-10,
        )
        if not endpoint_matches:
            for key in new_keys:
                self._remote_events.pop(key, None)
            self._replay_from(float(reference_timestamp), starting_state=checkpoint)
            return CoordinatorMessageResult(False, "event_bundle_endpoint_mismatch")
        return CoordinatorMessageResult(True, "accepted", new_count)

    @property
    def checkpoint_timestamps(self) -> tuple[float, ...]:
        return tuple(sorted(self._checkpoints))

    def _replay_from(
        self, reference_timestamp: float,
        starting_state: MultiNeighborSchmidtState | None = None,
    ) -> None:
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
            observations = sorted(
                (item for item in self._observations.values()
                 if np.isclose(float(item.timestamp), timestamp)),
                key=lambda item: item.information_id,
            )
            for observation in observations:
                if observation.information_id not in current.information_ids:
                    current = multi_neighbor_schmidt_update(current, observation).state
            self._posterior_states[timestamp] = current
        self.state = current

    def _prune(self, current_timestamp: float) -> None:
        if self.history_window is None:
            return
        oldest = current_timestamp - self.history_window
        self._checkpoints = {
            timestamp: state for timestamp, state in self._checkpoints.items()
            if timestamp >= oldest
        }
        self._observations = {
            key: value for key, value in self._observations.items()
            if float(value.timestamp) >= oldest
        }
        self._posterior_states = {
            timestamp: state for timestamp, state in self._posterior_states.items()
            if timestamp >= oldest
        }
        self._remote_events = {
            key: value for key, value in self._remote_events.items()
            if float(value.event.timestamp) >= oldest
        }


def _same_event(left: CovarianceTransportEvent, right: CovarianceTransportEvent) -> bool:
    return (
        np.isclose(float(left.timestamp), float(right.timestamp))
        and tuple(left.information_ids) == tuple(right.information_ids)
        and np.allclose(left.state_estimate, right.state_estimate)
        and np.allclose(left.error_transition, right.error_transition)
        and np.allclose(left.independent_process_noise, right.independent_process_noise)
    )
