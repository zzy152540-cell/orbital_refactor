from dataclasses import replace

import numpy as np
import pytest

from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from cooperative.online_graph_observation import build_online_graph_observation
from cooperative.topology import fully_connected_topology
from cooperative.topology_policy import GraphObservation
from cooperative.v15_policy_tensor import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    tensorize_v15_policy_observation,
)


def _observation():
    nodes = ("a", "b", "c")
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node={
            "a": np.array([7e6, 0, 0, 0, 7e3, 0]),
            "b": np.array([7e6 + 3e3, 4e3, 0, 0, 7e3, 0]),
            "c": np.array([7e6, 0, 12e3, 0, 7e3, 0]),
        },
        initial_covariance_by_node={node: np.eye(6) for node in nodes},
        topology=fully_connected_topology(nodes),
    )
    return build_online_graph_observation(
        orchestrator,
        measurement_modalities_by_edge={("a", "b"): ("RANGE",)},
        nis_by_modality_by_edge={("a", "b"): {"RANGE": 2.0}},
        nis_sample_count_by_modality_by_edge={("a", "b"): {"RANGE": 3}},
    )


def test_v15_tensor_is_normalized_masked_and_read_only():
    tensor = tensorize_v15_policy_observation(_observation())
    assert tensor.schema_version == "v15.0-policy-normalized"
    assert tensor.node_features.shape == (3, len(NODE_FEATURE_NAMES))
    assert tensor.edge_features.shape == (3, len(EDGE_FEATURE_NAMES))
    assert np.all(np.isfinite(tensor.node_features))
    assert np.all(np.isfinite(tensor.edge_features))
    edge = tensor.candidate_edges.index(("a", "b"))
    assert tensor.edge_features[edge, EDGE_FEATURE_NAMES.index("normalized_nis_RANGE")] == 2.0
    assert tensor.edge_features[edge, EDGE_FEATURE_NAMES.index("nis_available_RANGE")] == 1.0
    missing = tensor.candidate_edges.index(("a", "c"))
    assert tensor.edge_features[missing, EDGE_FEATURE_NAMES.index("nis_available_RANGE")] == 0.0
    with pytest.raises(ValueError):
        tensor.node_features[0, 0] = 1.0


def test_v15_tensor_canonicalizes_node_order_and_is_translation_invariant():
    observation = _observation()
    base = tensorize_v15_policy_observation(observation)
    offset = np.array([1e6, -2e6, 3e6, 100.0, -200.0, 300.0])
    shifted_nodes = tuple(
        replace(node, state=tuple(np.asarray(node.state) + offset))
        for node in reversed(observation.nodes)
    )
    shifted = tensorize_v15_policy_observation(replace(
        observation, nodes=shifted_nodes,
    ))
    assert shifted.node_ids == base.node_ids
    np.testing.assert_allclose(shifted.node_features, base.node_features)
    np.testing.assert_array_equal(shifted.edge_index, base.edge_index)


def test_v15_tensor_rejects_unspecified_or_truth_unsafe_observation():
    observation = _observation()
    unsafe = GraphObservation(
        timestamp=observation.timestamp,
        nodes=observation.nodes,
        candidate_edges=observation.candidate_edges,
    )
    with pytest.raises(ValueError, match="not marked"):
        tensorize_v15_policy_observation(unsafe)
