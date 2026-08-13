from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.topology_policy import (
    GraphObservation,
    validate_deployment_graph_observation,
)


MODALITIES = ("RANGE", "RANGE_RATE", "RADAR", "AZ_EL", "INFRARED", "OPTICAL")
NIS_DOF = {"RANGE": 1.0, "RANGE_RATE": 1.0, "RADAR": 2.0,
           "AZ_EL": 2.0, "INFRARED": 2.0, "OPTICAL": 2.0}
NODE_METRICS = (
    "absolute_navigation_available",
    "history_checkpoint_count", "pinned_checkpoint_count",
    "retained_journal_count", "pending_delivery_count",
    "resync_required_neighbor_count", "suspended_neighbor_count",
    "replay_count", "fallback_count",
)
GRAPH_METRICS = (
    "topology_version", "candidate_edge_count", "active_edge_count",
    "pending_delivery_count", "cumulative_accepted_message_count",
    "cumulative_rejected_message_count", "cumulative_transmitted_message_count",
    "cumulative_dropped_message_count", "cumulative_stale_topology_message_count",
    "cumulative_protocol_rejected_message_count",
    "decisions_since_topology_switch", "topology_cooldown_remaining",
    "minimum_topology_dwell_decisions",
)
NODE_FEATURE_NAMES = (
    *(f"centered_position_{axis}" for axis in "xyz"),
    *(f"centered_velocity_{axis}" for axis in "xyz"),
    *(f"log_covariance_diagonal_{index}" for index in range(6)),
    "covariance_available",
    *(f"log1p_{name}" for name in NODE_METRICS),
    *(f"available_{name}" for name in NODE_METRICS),
)
EDGE_FEATURE_NAMES = (
    "log1p_distance", "geometrically_visible", "communication_available",
    "log1p_delay", "packet_loss_rate", "observation_age_available",
    "log1p_observation_age", "previously_active", "estimation_dependency",
    *(f"modality_{name}" for name in MODALITIES),
    *(f"normalized_nis_{name}" for name in MODALITIES),
    *(f"nis_available_{name}" for name in MODALITIES),
    *(f"log1p_nis_sample_count_{name}" for name in MODALITIES),
    *(f"log1p_anomaly_count_{name}" for name in MODALITIES),
)
GLOBAL_FEATURE_NAMES = (
    "log1p_timestamp",
    *(f"log1p_{name}" for name in GRAPH_METRICS),
    *(f"available_{name}" for name in GRAPH_METRICS),
)


@dataclass(frozen=True)
class V15PolicyTensor:
    schema_version: str
    node_ids: tuple[str, ...]
    node_feature_names: tuple[str, ...]
    node_features: np.ndarray
    candidate_edges: tuple[tuple[str, str], ...]
    edge_index: np.ndarray
    edge_feature_names: tuple[str, ...]
    edge_features: np.ndarray
    global_feature_names: tuple[str, ...]
    global_features: np.ndarray


def tensorize_v15_policy_observation(
    observation: GraphObservation,
    *, position_scale: float = 1.0e7,
    velocity_scale: float = 1.0e4,
    distance_scale: float = 1.0e7,
    delay_scale: float = 10.0,
    age_scale: float = 10.0,
) -> V15PolicyTensor:
    """Convert a deployment-safe observation into a normalized immutable tensor."""

    validate_deployment_graph_observation(observation)
    scales = (position_scale, velocity_scale, distance_scale, delay_scale, age_scale)
    if any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("V15 policy tensor scales must be finite and positive.")
    nodes = tuple(sorted(observation.nodes, key=lambda value: value.node_id))
    node_ids = tuple(node.node_id for node in nodes)
    node_index = {node: index for index, node in enumerate(node_ids)}
    states = np.asarray([node.state for node in nodes], dtype=float)
    if states.shape != (len(nodes), 6) or not np.all(np.isfinite(states)):
        raise ValueError("V15 policy tensors require finite six-state nodes.")
    centered = states - np.mean(states, axis=0, keepdims=True)
    node_rows = []
    for index, node in enumerate(nodes):
        covariance_available = node.covariance_diagonal is not None
        covariance = np.asarray(
            node.covariance_diagonal if covariance_available else np.zeros(6),
            dtype=float,
        )
        if covariance.shape != (6,) or np.any(covariance < 0.0):
            raise ValueError("Node covariance diagonal must have six nonnegative values.")
        metrics = dict(node.estimator_metrics)
        node_rows.append((
            *(centered[index, :3] / position_scale),
            *(centered[index, 3:] / velocity_scale),
            *np.log1p(covariance), float(covariance_available),
            *(_log_metrics(metrics, NODE_METRICS)),
            *(float(name in metrics) for name in NODE_METRICS),
        ))
    edges = tuple(sorted(observation.candidate_edges, key=lambda value: value.nodes))
    previous = set(observation.previous_active_edges)
    dependencies = set(observation.estimation_dependency_edges)
    edge_rows = []
    for edge in edges:
        nis = {str(name).upper(): value for name, value in edge.nis_by_modality}
        counts = {str(name).upper(): value
                  for name, value in edge.nis_sample_count_by_modality}
        anomalies = {str(name).upper(): value
                     for name, value in edge.consecutive_anomaly_count_by_modality}
        modalities = {str(name).upper() for name in edge.measurement_modalities}
        edge_rows.append((
            np.log1p(edge.distance / distance_scale),
            float(edge.geometrically_visible), float(edge.communication_available),
            np.log1p(edge.delay / delay_scale), edge.packet_loss_rate,
            float(edge.observation_age is not None),
            np.log1p((edge.observation_age or 0.0) / age_scale),
            float(edge.nodes in previous), float(edge.nodes in dependencies),
            *(float(name in modalities) for name in MODALITIES),
            *(float(nis.get(name, 0.0)) / NIS_DOF[name] for name in MODALITIES),
            *(float(name in nis) for name in MODALITIES),
            *(np.log1p(float(counts.get(name, 0))) for name in MODALITIES),
            *(np.log1p(float(anomalies.get(name, 0))) for name in MODALITIES),
        ))
    graph = dict(observation.graph_metrics)
    tensor = V15PolicyTensor(
        schema_version="v15.0-policy-normalized",
        node_ids=node_ids,
        node_feature_names=NODE_FEATURE_NAMES,
        node_features=_readonly(node_rows),
        candidate_edges=tuple(edge.nodes for edge in edges),
        edge_index=_readonly([
            (node_index[left], node_index[right])
            for left, right in (edge.nodes for edge in edges)
        ], dtype=np.int64, transpose=True, empty_shape=(2, 0)),
        edge_feature_names=EDGE_FEATURE_NAMES,
        edge_features=_readonly(edge_rows, empty_shape=(0, len(EDGE_FEATURE_NAMES))),
        global_feature_names=GLOBAL_FEATURE_NAMES,
        global_features=_readonly((
            np.log1p(max(0.0, float(observation.timestamp))),
            *(_log_metrics(graph, GRAPH_METRICS)),
            *(float(name in graph) for name in GRAPH_METRICS),
        )),
    )
    return tensor


def _log_metrics(values, names):
    result = []
    for name in names:
        value = float(values.get(name, 0.0))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Policy metric {name} must be finite and nonnegative.")
        result.append(np.log1p(value))
    return tuple(result)


def _readonly(values, *, dtype=float, transpose=False, empty_shape=None):
    array = np.asarray(values, dtype=dtype)
    if array.size == 0 and empty_shape is not None:
        array = np.empty(empty_shape, dtype=dtype)
    elif transpose:
        array = array.T
    array.setflags(write=False)
    return array
