"""Read-only adapter from a completed network-filter run to display frames."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np

from cooperative.topology import NetworkTopology
from interfaces.data_objects import AbsolutePositionObservation, ObservationMessage
from visualization.data_contract import (
    VisualEdge,
    VisualNavigationState,
    VisualNodeState,
    VisualObservation,
    VisualizationFrame,
)


def network_history_visualization_frames(
    *, history, truth_history_by_node: Mapping[str, np.ndarray],
    topology: NetworkTopology,
    observation_messages: Iterable[ObservationMessage] = (),
    absolute_position_observations: Iterable[AbsolutePositionObservation] = (),
    raw_sensor_data_by_information_id: Mapping[
        str, tuple[np.ndarray, str]
    ] | None = None,
    navigation_by_epoch=None, cann_by_epoch=None,
    edges_by_epoch=None, events_by_epoch=None,
    scenario_id: str, run_id: str,
) -> tuple[VisualizationFrame, ...]:
    """Project existing histories into immutable frames without estimator feedback."""

    timestamps = np.asarray(history.timestamps, dtype=float)
    node_ids = tuple(history.node_ids)
    _validate_histories(history, truth_history_by_node, node_ids, timestamps.size)
    observations_by_epoch = _group_by_timestamp(observation_messages, timestamps)
    absolute_by_epoch = _group_by_timestamp(
        absolute_position_observations, timestamps,
    )
    edges = _topology_edges(topology)

    frames = []
    for index, timestamp in enumerate(timestamps):
        nodes = tuple(
            VisualNodeState(
                node_id=node_id,
                truth_state=truth_history_by_node[node_id][index],
                estimate_state=history.active_state_history_by_node[node_id][index],
                covariance_diagonal=np.diag(
                    history.active_covariance_history_by_node[node_id][index]
                ),
                covariance=history.active_covariance_history_by_node[node_id][index],
            )
            for node_id in node_ids
        )
        observations = tuple(
            _visual_observation(
                message, history=history, epoch_index=index,
                raw_payload=(
                    None if raw_sensor_data_by_information_id is None
                    else raw_sensor_data_by_information_id.get(
                        message.information_id
                    )
                ),
            )
            for message in observations_by_epoch[index]
        )
        available_absolute_nodes = {
            item.satellite_id for item in absolute_by_epoch[index]
            if item.valid_flag
        }
        navigation = tuple(
            VisualNavigationState(
                node_id=node_id,
                absolute_navigation_available=node_id in available_absolute_nodes,
                last_absolute_navigation_timestamp=_last_absolute_timestamp(
                    node_id, index=index, grouped=absolute_by_epoch,
                ),
                status=(
                    "AVAILABLE" if node_id in available_absolute_nodes
                    else "UNAVAILABLE"
                ),
            )
            for node_id in node_ids
        ) if navigation_by_epoch is None else tuple(navigation_by_epoch[index])
        errors = np.asarray([
            nodes[node_index].estimate_state[:3]
            - nodes[node_index].truth_state[:3]
            for node_index in range(len(nodes))
        ])
        frames.append(VisualizationFrame(
            scenario_id=scenario_id, run_id=run_id,
            timestamp=float(timestamp), epoch_index=index,
            nodes=nodes,
            edges=(edges if edges_by_epoch is None else tuple(edges_by_epoch[index])),
            observations=observations,
            navigation=navigation,
            cann=() if cann_by_epoch is None else tuple(cann_by_epoch[index]),
            events=() if events_by_epoch is None else tuple(events_by_epoch[index]),
            metadata={
                "fleet_position_rmse_m": float(np.sqrt(np.mean(
                    np.sum(np.square(errors), axis=1)
                ))),
                "truth_usage": "DISPLAY_ONLY",
            },
        ))
    return tuple(frames)


def _validate_histories(history, truth, node_ids, epoch_count):
    if set(node_ids) != set(truth):
        raise ValueError("Truth and estimator histories must cover identical nodes.")
    if set(topology_node for topology_node in node_ids) != set(node_ids):
        raise ValueError("History contains duplicate nodes.")
    for node_id in node_ids:
        estimate = np.asarray(history.active_state_history_by_node[node_id])
        covariance = np.asarray(history.active_covariance_history_by_node[node_id])
        node_truth = np.asarray(truth[node_id])
        if estimate.shape[0] != epoch_count or node_truth.shape != estimate.shape:
            raise ValueError("Truth and estimate histories must align by epoch.")
        if covariance.shape != (epoch_count, estimate.shape[1], estimate.shape[1]):
            raise ValueError("Covariance history has an incompatible shape.")


def _group_by_timestamp(items, timestamps):
    grouped = defaultdict(list)
    for item in items:
        timestamp = float(item.timestamp)
        matches = np.flatnonzero(np.isclose(timestamps, timestamp, atol=1e-9, rtol=0.0))
        if matches.size == 1:
            grouped[int(matches[0])].append(item)
    return tuple(tuple(grouped[index]) for index in range(timestamps.size))


def _topology_edges(topology):
    pairs = {
        tuple(sorted((node_id, neighbor)))
        for node_id in topology.node_ids
        for neighbor in topology.neighbors(node_id)
    }
    return tuple(
        VisualEdge(left, right, "CONFIGURED_TOPOLOGY")
        for left, right in sorted(pairs)
    )


def _visual_observation(message, *, history, epoch_index, raw_payload=None):
    information_id = message.information_id
    nis = history.nis_history_by_node.get(message.observer_id, [{}])[epoch_index].get(
        information_id
    )
    integrity = history.integrity_history_by_node.get(
        message.observer_id, [{}]
    )[epoch_index].get(information_id)
    status = getattr(integrity, "status", "NOT_PROCESSED")
    covariance = np.asarray(message.covariance)
    return VisualObservation(
        observer_id=message.observer_id, target_id=message.target_id,
        modality=message.modality, measurement=message.measurement,
        covariance_diagonal=np.diag(covariance),
        visible=bool(message.metadata.get("geometrically_visible", message.valid_flag)),
        valid=bool(message.valid_flag), processing_status=str(status),
        frame=message.frame, nis=None if nis is None else float(nis),
        raw_sensor_data=None if raw_payload is None else raw_payload[0],
        raw_data_kind=None if raw_payload is None else raw_payload[1],
        metadata={
            "information_id": information_id,
            "source_timestamp": message.source_timestamp,
            "arrival_timestamp": message.arrival_timestamp,
        },
    )


def _last_absolute_timestamp(node_id, *, index, grouped):
    latest = None
    for epoch in range(index + 1):
        for item in grouped[epoch]:
            if item.satellite_id == node_id and item.valid_flag:
                latest = float(item.timestamp)
    return latest
