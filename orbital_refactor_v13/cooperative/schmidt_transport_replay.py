from __future__ import annotations

from typing import Iterable

from cooperative.multi_neighbor_schmidt import MultiNeighborSchmidtState, multi_neighbor_schmidt_predict, multi_neighbor_schmidt_update
from cooperative.schmidt_refresh import refresh_consider_neighbor
from interfaces.data_objects import CovarianceTransportEvent, ObservationMessage


def replay_transport_event_bundle(
    state: MultiNeighborSchmidtState, *, neighbor_id: str,
    events: Iterable[CovarianceTransportEvent],
    observations: Iterable[ObservationMessage], current_timestamp: float,
    process_noise_acceleration: float = 1e-4,
) -> MultiNeighborSchmidtState:
    """Merge remote update events and local observations in timestamp order."""
    remote_by_time = {}
    for event in events:
        remote_by_time.setdefault(float(event.timestamp), []).append(event)
    observation_by_time = {}
    for observation in observations:
        timestamp = float(observation.timestamp)
        if float(state.timestamp) <= timestamp <= float(current_timestamp):
            observation_by_time.setdefault(timestamp, []).append(observation)
    times = sorted(set(remote_by_time) | set(observation_by_time) | {float(current_timestamp)})
    current = state
    for timestamp in times:
        if timestamp > float(current.timestamp):
            current = multi_neighbor_schmidt_predict(
                current, timestamp,
                process_noise_acceleration=process_noise_acceleration,
            )
        for event in remote_by_time.get(timestamp, []):
            current = refresh_consider_neighbor(
                current, neighbor_id=neighbor_id,
                neighbor_state=event.state_estimate, mode="exact_transport",
                error_transition=event.error_transition,
                independent_process_noise=event.independent_process_noise,
            )
        for observation in sorted(observation_by_time.get(timestamp, []), key=lambda item: item.information_id):
            if observation.information_id not in current.information_ids:
                current = multi_neighbor_schmidt_update(current, observation).state
    return current
