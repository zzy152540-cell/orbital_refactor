from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from cooperative.topology import NetworkTopology
from interfaces.data_objects import (
    AbsolutePositionObservation,
    ObservationMessage,
    StateMessage,
)

Array = np.ndarray


def route_relative_observations(
    observations: Iterable[ObservationMessage],
    *,
    times: Array,
    topology: NetworkTopology,
    observation_usage: str,
    allow_delayed: bool = False,
    modality_update_order: tuple[str, ...] | None = None,
    modality_update_order_start_time: float | None = None,
) -> dict[float, dict[str, list[ObservationMessage]]]:
    """Validate and route relative observations by arrival epoch and owner."""

    result = {float(timestamp): {} for timestamp in times}
    seen_message_ids: set[str] = set()
    for observation in observations:
        source_timestamp = float(observation.timestamp)
        if source_timestamp not in result:
            raise ValueError("Observation timestamp is not in timestamps.")
        observer = str(observation.observer_id)
        target = str(observation.target_id)
        if target not in topology.neighbors(observer):
            raise ValueError("Observation endpoints must share a topology edge.")
        if observation.message_id in seen_message_ids:
            raise ValueError(
                "Observation message_id values must be globally unique."
            )
        seen_message_ids.add(observation.message_id)
        owners = (
            (observer, target)
            if observation_usage == "both_endpoints"
            and observation.metadata.get("shared_delivery", True)
            else (observer,)
        )
        for owner in owners:
            route_timestamp = source_timestamp
            if owner != observer and observation.arrival_timestamp is not None:
                route_timestamp = float(observation.arrival_timestamp)
            if route_timestamp > float(times[-1]):
                continue
            if route_timestamp not in result:
                raise ValueError(
                    "Observation arrival timestamp is not in timestamps."
                )
            if not allow_delayed and route_timestamp > source_timestamp:
                raise ValueError(
                    "Delayed shared observations require exact event replay."
                )
            result[route_timestamp].setdefault(owner, []).append(observation)
    modality_rank = {
        modality: index
        for index, modality in enumerate(modality_update_order or ())
    }
    for per_owner in result.values():
        for owner, messages in per_owner.items():
            messages.sort(key=lambda item: (
                modality_rank.get(item.modality, len(modality_rank))
                if modality_update_order_start_time is None
                or float(item.timestamp) > modality_update_order_start_time
                else 0,
                item.information_id,
            ))
            unique = {}
            for message in messages:
                unique.setdefault(message.information_id, message)
            per_owner[owner] = list(unique.values())
    return result


def route_absolute_observations(
    observations: Iterable[AbsolutePositionObservation],
    *,
    times: Array,
    node_ids: set[str],
    allow_delayed: bool,
) -> dict[float, dict[str, list[AbsolutePositionObservation]]]:
    """Validate and route absolute observations by arrival epoch and node."""

    result = {float(timestamp): {} for timestamp in times}
    seen_ids = set()
    for observation in observations:
        if not observation.valid_flag:
            continue
        node_id = str(observation.satellite_id)
        if node_id not in node_ids:
            raise ValueError(
                "Absolute observation satellite is not a topology node."
            )
        source_timestamp = float(observation.timestamp)
        if source_timestamp not in result:
            raise ValueError(
                "Absolute observation timestamp is not in timestamps."
            )
        if observation.information_id in seen_ids:
            raise ValueError("Absolute observation IDs must be globally unique.")
        seen_ids.add(observation.information_id)
        route_timestamp = (
            source_timestamp
            if observation.arrival_timestamp is None
            else float(observation.arrival_timestamp)
        )
        if route_timestamp > float(times[-1]):
            continue
        if route_timestamp not in result:
            raise ValueError(
                "Absolute observation arrival timestamp is not in timestamps."
            )
        if not allow_delayed and route_timestamp > source_timestamp:
            raise ValueError(
                "Delayed absolute observations require exact event replay."
            )
        result[route_timestamp].setdefault(node_id, []).append(observation)
    for per_node in result.values():
        for values in per_node.values():
            values.sort(key=lambda item: item.information_id)
    return result


def prepare_state_messages(
    messages_by_receiver: Mapping[str, Iterable[StateMessage]],
    node_ids: tuple[str, ...],
    topology: NetworkTopology,
) -> dict[str, list[StateMessage]]:
    """Validate and deterministically order transported state messages."""

    if set(messages_by_receiver) - set(node_ids):
        raise ValueError(
            "State-message receiver IDs must belong to the topology."
        )
    result = {node_id: [] for node_id in node_ids}
    for receiver_id, messages in messages_by_receiver.items():
        for message in messages:
            if str(message.source_node_id) not in topology.neighbors(receiver_id):
                raise ValueError(
                    "State-message source must be a receiver neighbor."
                )
            result[receiver_id].append(message)
        result[receiver_id].sort(key=lambda item: (
            float(
                item.timestamp
                if item.arrival_timestamp is None
                else item.arrival_timestamp
            ),
            float(item.timestamp),
            str(item.source_node_id),
        ))
    return result
