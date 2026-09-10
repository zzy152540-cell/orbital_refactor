from types import SimpleNamespace

import numpy as np

from cooperative.topology import NetworkTopology
from experiments.v15_dynamic_visualization import phased_visualization_overlays
from interfaces.data_objects import StateMessage


def test_phased_overlays_distinguish_configured_active_and_flow_edges():
    topology = NetworkTopology({"a": ("b",), "b": ("a",)})
    message = StateMessage(
        source_node_id="a", target_node_id="b", timestamp=0.0,
        state_estimate=np.zeros(6), covariance=np.eye(6), quality_score=1.0,
        arrival_timestamp=0.0,
    )
    case = {
        "timestamps": np.array([0.0, 2.0, 4.0, 6.0]),
        "topology": topology, "state_messages": {"a": [], "b": [message]},
        "active_neighbors_by_timestamp": {
            0.0: {"a": ("b",), "b": ("a",)},
            2.0: {"a": ("b",), "b": ("a",)},
            4.0: {"a": (), "b": ()},
            6.0: {"a": ("b",), "b": ("a",)},
        },
    }
    edges, events = phased_visualization_overlays(
        case=case, dropout_nodes=("a",), inactive_edge=("a", "b"),
        dropout_window=(0.0, 2.0), topology_window=(4.0, 4.0),
    )
    assert {edge.edge_type for edge in edges[0]} == {
        "CONFIGURED_TOPOLOGY", "ACTIVE_TOPOLOGY", "ACTUAL_INFORMATION_FLOW"
    }
    assert [edge.status for edge in edges[2]
            if edge.edge_type == "CONFIGURED_TOPOLOGY"] == ["INACTIVE"]
    assert not any(edge.edge_type == "ACTIVE_TOPOLOGY" for edge in edges[2])
    assert sum(len(value) for value in events) == 4
