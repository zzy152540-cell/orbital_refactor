from __future__ import annotations

from cooperative.topology_policy import GraphObservation, UndirectedEdge


def select_top_k_addition_edges(
    observation: GraphObservation, *, top_k_per_node: int | None,
) -> tuple[UndirectedEdge, ...]:
    """Select a sparse, truth-free set of feasible non-active addition edges."""

    if top_k_per_node is None:
        return tuple(sorted(
            edge.nodes for edge in observation.candidate_edges
            if edge.nodes not in set(observation.previous_active_edges)
        ))
    if top_k_per_node < 0:
        raise ValueError("Top-K candidate count cannot be negative.")
    active = set(observation.previous_active_edges)
    feasible = tuple(
        edge for edge in observation.candidate_edges
        if edge.nodes not in active
        and edge.geometrically_visible
        and edge.communication_available
    )
    selected = set()
    for node in sorted(value.node_id for value in observation.nodes):
        incident = sorted(
            (edge for edge in feasible if node in edge.nodes),
            key=lambda edge: (
                edge.distance, edge.packet_loss_rate, edge.delay, edge.nodes,
            ),
        )
        selected.update(edge.nodes for edge in incident[:top_k_per_node])
    return tuple(sorted(selected))
