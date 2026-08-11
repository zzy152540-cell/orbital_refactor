import numpy as np

from cooperative.topology_policy import (
    GraphEdgeFeature, GraphNodeFeature, GraphObservation, TopologyAction,
)
from experiments.action_graph_features import action_graph_metrics, action_pair_metrics


def _observation():
    nodes = tuple(
        GraphNodeFeature(node, state, (1.0,) * 6)
        for node, state in (
            ("a", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ("b", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
            ("c", (0.0, 1.0, 0.0, -1.0, 0.0, 0.0)),
        )
    )
    edges = tuple(
        GraphEdgeFeature(edge, 1.0, measurement_modalities=("RADAR",))
        for edge in (("a", "b"), ("a", "c"), ("b", "c"))
    )
    return GraphObservation(
        0.0, nodes, edges, previous_active_edges=(("a", "b"), ("b", "c"))
    )


def test_action_graph_metrics_distinguish_tree_and_complete_graph():
    observation = _observation()
    tree = action_graph_metrics(
        observation, TopologyAction("tree", (("a", "b"), ("b", "c")))
    )
    complete = action_graph_metrics(
        observation,
        TopologyAction("complete", (("a", "b"), ("a", "c"), ("b", "c"))),
    )

    assert tree.negative_bridge_count == -2.0
    assert complete.negative_bridge_count == 0.0
    assert complete.algebraic_connectivity > tree.algebraic_connectivity
    assert complete.normalized_information_rank >= tree.normalized_information_rank
    assert np.isfinite(complete.information_log_pseudodeterminant)


def test_action_graph_metrics_do_not_fabricate_optical_attitude():
    observation = _observation()
    optical_edges = tuple(
        GraphEdgeFeature(
            edge.nodes, edge.distance,
            measurement_modalities=("RADAR", "OPTICAL"),
        )
        for edge in observation.candidate_edges
    )
    with_optical = GraphObservation(
        observation.timestamp, observation.nodes, optical_edges,
        previous_active_edges=observation.previous_active_edges,
    )

    metrics = action_graph_metrics(
        with_optical,
        TopologyAction("tree", observation.previous_active_edges),
    )

    assert metrics.normalized_information_rank > 0.0


def test_swap_pair_metrics_separate_added_removed_and_retained_information():
    observation = _observation()
    swap = TopologyAction("swap", (("a", "b"), ("a", "c")))

    metrics = action_pair_metrics(observation, swap)

    assert metrics.added_information_rank > 0.0
    assert metrics.removed_information_rank > 0.0
    assert 0.0 <= metrics.added_removed_subspace_complementarity <= 1.0
    assert 0.0 <= metrics.added_retained_nullspace_fill <= 1.0
    assert 0.0 <= metrics.removed_retained_unique_information <= 1.0
