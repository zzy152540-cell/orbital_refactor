from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cooperative.multi_neighbor_schmidt import (
    MultiNeighborSchmidtState,
    multi_neighbor_schmidt_predict,
    multi_neighbor_schmidt_update,
)
from interfaces.data_objects import ObservationMessage


@dataclass(frozen=True)
class SchmidtReplayResult:
    state: MultiNeighborSchmidtState
    replayed_information_ids: tuple[str, ...]
    nis_by_information_id: dict[str, float]


def replay_schmidt_events(
    state: MultiNeighborSchmidtState, *, current_timestamp: float,
    observations: Iterable[ObservationMessage],
    process_noise_acceleration: float = 1e-4,
) -> SchmidtReplayResult:
    """Replay timestamped observations after a historical state correction."""
    end = float(current_timestamp)
    if end < float(state.timestamp):
        raise ValueError("current_timestamp precedes the replay checkpoint.")
    grouped: dict[float, list[ObservationMessage]] = {}
    for observation in observations:
        timestamp = float(observation.timestamp)
        if timestamp < float(state.timestamp) or timestamp > end:
            continue
        grouped.setdefault(timestamp, []).append(observation)
    event_times = sorted(set(grouped) | ({end} if end > float(state.timestamp) else set()))
    replayed = []; nis = {}; current = state
    for timestamp in event_times:
        if timestamp > float(current.timestamp):
            current = multi_neighbor_schmidt_predict(
                current, timestamp,
                process_noise_acceleration=process_noise_acceleration,
            )
        for observation in sorted(grouped.get(timestamp, []), key=lambda item: item.information_id):
            if observation.information_id in current.information_ids:
                continue
            update = multi_neighbor_schmidt_update(current, observation)
            current = update.state
            replayed.append(observation.information_id)
            nis[observation.information_id] = update.nis
    return SchmidtReplayResult(current, tuple(replayed), nis)
