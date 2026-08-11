from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cooperative.topology_policy import (
    GraphObservation,
    TopologyAction,
    build_graph_observation,
    normalized_undirected_edge,
)


@dataclass(frozen=True)
class GraphOutcome:
    """Filter and communication results produced after executing an action."""

    timestamp: float
    node_metrics: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    edge_nis_by_modality: tuple[
        tuple[tuple[str, str], tuple[tuple[str, float], ...]], ...
    ]
    graph_metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GraphTransition:
    """Strict causal sample: pre-state, action, outcome, next pre-state."""

    pre_observation: GraphObservation
    action: TopologyAction
    outcome: GraphOutcome
    next_pre_observation: GraphObservation | None = None

    def __post_init__(self) -> None:
        if (
            self.next_pre_observation is not None
            and self.next_pre_observation.timestamp
            <= self.pre_observation.timestamp
        ):
            raise ValueError("GraphTransition timestamps must increase.")
        if self.outcome.timestamp != self.pre_observation.timestamp:
            raise ValueError("GraphTransition outcome must match action epoch.")
        candidate_edges = {edge.nodes for edge in self.pre_observation.candidate_edges}
        if set(self.action.active_edges) - candidate_edges:
            raise ValueError(
                "GraphTransition action must use decision-epoch candidate edges."
            )


@dataclass(frozen=True)
class WalkerGraphDataset:
    """Chronological causal transitions recorded from one online Walker run."""

    feature_version: str
    transitions: tuple[GraphTransition, ...]

    @property
    def observations(self) -> tuple[GraphObservation, ...]:
        """Compatibility view of the observation trajectory."""

        if not self.transitions:
            return ()
        return (
            self.transitions[0].pre_observation,
            *(
                transition.next_pre_observation
                for transition in self.transitions
                if transition.next_pre_observation is not None
            ),
        )


def build_walker_graph_outcome(
    *, timestamp, step_result, relative_observations,
    prior_covariance_by_node, action, previous_active_edges,
) -> GraphOutcome:
    """Record only information produced by executing an epoch action."""
    observation_by_id = {
        observation.information_id: observation
        for observation in relative_observations
    }
    nis_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    node_nis: dict[str, list[float]] = {
        node: [] for node in step_result.result_by_node
    }
    for node, result in step_result.result_by_node.items():
        for information_id, value in result.nis_by_information_id.items():
            node_nis[node].append(float(value))
            observation = observation_by_id.get(information_id)
            if observation is None:
                continue
            edge = normalized_undirected_edge(
                observation.observer_id, observation.target_id
            )
            nis_values.setdefault(edge, {}).setdefault(
                observation.modality, []
            ).append(float(value))
    node_metrics = {
        node: tuple(sorted({
            "nis_count": float(len(values)),
            "prior_covariance_trace": float(np.trace(
                prior_covariance_by_node[node]
            )),
            "posterior_covariance_trace": float(np.trace(
                step_result.result_by_node[node].state.active_covariance
            )),
            **({"mean_nis": float(np.mean(values))} if values else {}),
        }.items()))
        for node, values in node_nis.items()
    }
    return GraphOutcome(
        timestamp=float(timestamp),
        node_metrics=tuple(sorted(node_metrics.items())),
        edge_nis_by_modality=tuple(sorted(
            (edge, tuple(sorted({
                modality: float(np.mean(values))
                for modality, values in by_modality.items()
            }.items())))
            for edge, by_modality in nis_values.items()
        )),
        graph_metrics=tuple(sorted({
            "active_edge_count": len(action.active_edges),
            "added_edge_count": len(
                set(action.active_edges) - set(previous_active_edges)
            ),
            "removed_edge_count": len(
                set(previous_active_edges) - set(action.active_edges)
            ),
            "accepted_message_count": step_result.accepted_message_count,
            "rejected_message_count": step_result.rejected_message_count,
            "transmitted_message_count": step_result.transmitted_message_count,
            "dropped_message_count": step_result.dropped_message_count,
            "resynchronization_count": len(step_result.resynchronized_links),
            "stale_topology_message_count": (
                step_result.stale_topology_message_count
            ),
            "protocol_rejected_message_count": (
                step_result.protocol_rejected_message_count
            ),
        }.items())),
    )


def build_pre_walker_graph_observation(
    *, timestamp, plan_observation, state_by_node, covariance_by_node,
    relative_observations, packet_loss_rate, communication_delay,
    previous_active_edges=None, estimation_dependency_edges=(),
) -> GraphObservation:
    """Build policy input using only information available before the action."""

    modalities_by_edge: dict[tuple[str, str], set[str]] = {}
    for observation in relative_observations:
        edge = normalized_undirected_edge(
            observation.observer_id, observation.target_id
        )
        modalities_by_edge.setdefault(edge, set()).add(observation.modality)
    candidate_distances = {
        edge.nodes: edge.distance for edge in plan_observation.candidate_edges
    }
    return build_graph_observation(
        timestamp=timestamp,
        state_by_node=state_by_node,
        covariance_by_node=covariance_by_node,
        candidate_distance_by_edge=candidate_distances,
        previous_active_edges=(
            plan_observation.previous_active_edges
            if previous_active_edges is None else previous_active_edges
        ),
        estimation_dependency_edges=estimation_dependency_edges,
        measurement_modalities_by_edge={
            edge: tuple(values) for edge, values in modalities_by_edge.items()
        },
        delay_by_edge={edge: communication_delay for edge in candidate_distances},
        packet_loss_rate_by_edge={
            edge: packet_loss_rate for edge in candidate_distances
        },
    )
