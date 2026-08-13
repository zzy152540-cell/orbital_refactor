from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cooperative.link_lifecycle import LinkLifecycleState
from cooperative.topology_policy import (
    GraphMeasurementFeature,
    GraphObservation,
    GraphObservationProvenance,
    UndirectedEdge,
    build_graph_observation,
    normalized_undirected_edge,
    validate_deployment_graph_observation,
)


def build_online_graph_observation(
    orchestrator,
    *,
    candidate_edges: tuple[UndirectedEdge, ...] | None = None,
    measurement_modalities_by_edge: Mapping[UndirectedEdge, tuple[str, ...]] | None = None,
    geometrically_visible_by_edge: Mapping[UndirectedEdge, bool] | None = None,
    nis_by_modality_by_edge=None,
    nis_sample_count_by_modality_by_edge=None,
    consecutive_anomaly_count_by_modality_by_edge=None,
    observation_age_by_edge=None,
    measurements: tuple[GraphMeasurementFeature, ...] = (),
    additional_graph_metrics: Mapping[str, float] | None = None,
    additional_node_metrics_by_node: Mapping[
        str, Mapping[str, float]
    ] | None = None,
) -> GraphObservation:
    """Build a truth-free V15 policy snapshot from live orchestrator state."""

    if not orchestrator.sessions:
        raise ValueError("Online graph observation requires orchestrator sessions.")
    states = {
        node: np.asarray(session.state.active_state, dtype=float)
        for node, session in orchestrator.sessions.items()
    }
    timestamps = {float(session.state.timestamp)
                  for session in orchestrator.sessions.values()}
    if len(timestamps) != 1:
        raise ValueError("Orchestrator sessions must share one decision timestamp.")
    covariance = {
        node: session.state.active_covariance
        for node, session in orchestrator.sessions.items()
    }
    edges = _candidate_edges(orchestrator, candidate_edges)
    distance = {
        edge: float(np.linalg.norm(states[edge[0]][:3] - states[edge[1]][:3]))
        for edge in edges
    }
    active = _active_edges(orchestrator.active_neighbors_by_node)
    step_totals = _step_totals(orchestrator.step_history)
    additions = additional_node_metrics_by_node or {}
    unknown = set(additions) - set(states)
    if unknown:
        raise ValueError("Additional node metrics reference unknown nodes.")
    estimator_metrics = {
        node: {**_node_metrics(orchestrator, node), **additions.get(node, {})}
        for node in states
    }
    graph_metrics = {
        "topology_version": float(orchestrator.topology_version),
        "candidate_edge_count": float(len(edges)),
        "active_edge_count": float(len(active)),
        "pending_delivery_count": float(sum(
            len(values) for values in orchestrator.pending_deliveries.values()
        )),
        **step_totals,
        **(additional_graph_metrics or {}),
    }
    observation = build_graph_observation(
        timestamp=timestamps.pop(),
        state_by_node=states,
        covariance_by_node=covariance,
        estimator_metrics_by_node=estimator_metrics,
        candidate_distance_by_edge=distance,
        previous_active_edges=active,
        estimation_dependency_edges=active,
        measurement_modalities_by_edge=measurement_modalities_by_edge,
        geometrically_visible_by_edge=geometrically_visible_by_edge,
        communication_available_by_edge={
            edge: _edge_is_available(orchestrator, edge) for edge in edges
        },
        delay_by_edge={
            edge: _edge_delay(orchestrator, edge) for edge in edges
        },
        packet_loss_rate_by_edge={
            edge: _edge_loss(orchestrator, edge) for edge in edges
        },
        nis_by_modality_by_edge=nis_by_modality_by_edge,
        nis_sample_count_by_modality_by_edge=(
            nis_sample_count_by_modality_by_edge
        ),
        consecutive_anomaly_count_by_modality_by_edge=(
            consecutive_anomaly_count_by_modality_by_edge
        ),
        observation_age_by_edge=observation_age_by_edge,
        graph_metrics=graph_metrics,
        measurements=measurements,
        provenance=GraphObservationProvenance(
            schema_version="v15.0-online-orchestrator",
            state_source="estimator",
            geometry_source="estimator",
            online_decision_safe=True,
        ),
    )
    validate_deployment_graph_observation(observation)
    return observation


def _candidate_edges(orchestrator, requested):
    available = {
        normalized_undirected_edge(node, neighbor)
        for node in orchestrator.topology.node_ids
        for neighbor in orchestrator.topology.neighbors(node)
    }
    if requested is None:
        return tuple(sorted(available))
    edges = tuple(sorted(normalized_undirected_edge(*edge) for edge in requested))
    if len(set(edges)) != len(edges) or set(edges) - available:
        raise ValueError("Candidate edges must be unique orchestrator topology edges.")
    return edges


def _active_edges(active_neighbors_by_node):
    return tuple(sorted({
        normalized_undirected_edge(node, neighbor)
        for node, neighbors in active_neighbors_by_node.items()
        for neighbor in neighbors
    }))


def _edge_is_available(orchestrator, edge):
    left, right = edge
    return bool(
        orchestrator.sessions[left].link_by_neighbor[right].state
        != LinkLifecycleState.RESYNC_REQUIRED
        and orchestrator.sessions[right].link_by_neighbor[left].state
        != LinkLifecycleState.RESYNC_REQUIRED
    )


def _edge_delay(orchestrator, edge):
    values = []
    for receiver, source in (edge, edge[::-1]):
        channel = orchestrator.channels[(receiver, source)]
        values.append(float(channel.delay_by_source[source]))
    return max(values)


def _edge_loss(orchestrator, edge):
    values = []
    for receiver, source in (edge, edge[::-1]):
        channel = orchestrator.channels[(receiver, source)]
        values.append(float(channel.packet_loss_rate[source]))
    return max(values)


def _node_metrics(orchestrator, node):
    session = orchestrator.sessions[node]
    performance = session.coordinator.performance
    lifecycles = tuple(session.link_by_neighbor.values())
    return {
        "history_checkpoint_count": float(session.coordinator.checkpoint_count),
        "pinned_checkpoint_count": float(
            session.coordinator.pinned_checkpoint_count
        ),
        "retained_journal_count": float(
            session.coordinator.retained_journal_count
        ),
        "pending_delivery_count": float(len(orchestrator.pending_deliveries[node])),
        "resync_required_neighbor_count": float(sum(
            lifecycle.state == LinkLifecycleState.RESYNC_REQUIRED
            for lifecycle in lifecycles
        )),
        "suspended_neighbor_count": float(sum(
            lifecycle.state == LinkLifecycleState.SUSPENDED
            for lifecycle in lifecycles
        )),
        "replay_count": float(performance.replay_count),
        "fallback_count": float(performance.fallback_count),
    }


def _step_totals(steps):
    names = (
        "accepted_message_count", "rejected_message_count",
        "transmitted_message_count", "dropped_message_count",
        "stale_topology_message_count", "protocol_rejected_message_count",
    )
    return {
        f"cumulative_{name}": float(sum(getattr(step, name) for step in steps))
        for name in names
    }
