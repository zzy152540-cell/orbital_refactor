import numpy as np

from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from cooperative.online_graph_observation import build_online_graph_observation
from cooperative.topology import fully_connected_topology
from cooperative.topology_policy import validate_deployment_graph_observation


def _orchestrator():
    nodes = ("a", "b", "c")
    return NetworkSchmidtOrchestrator(
        initial_state_by_node={
            "a": np.array([0, 0, 0, 0, 0, 0], dtype=float),
            "b": np.array([3, 4, 0, 0, 0, 0], dtype=float),
            "c": np.array([0, 0, 12, 0, 0, 0], dtype=float),
        },
        initial_covariance_by_node={node: np.eye(6) for node in nodes},
        topology=fully_connected_topology(nodes),
        packet_loss_rate_by_link={("a", "b"): 0.2, ("b", "a"): 0.1},
        communication_delay_by_link={("a", "b"): 2.0, ("b", "a"): 1.0},
    )


def test_online_adapter_builds_deployment_safe_estimator_observation():
    observation = build_online_graph_observation(
        _orchestrator(),
        measurement_modalities_by_edge={("a", "b"): ("RANGE",)},
    )

    validate_deployment_graph_observation(observation)
    assert observation.provenance.schema_version == "v15.0-online-orchestrator"
    assert observation.provenance.state_source == "estimator"
    assert observation.provenance.geometry_source == "estimator"
    assert len(observation.candidate_edges) == 3
    edge = next(value for value in observation.candidate_edges
                if value.nodes == ("a", "b"))
    assert edge.distance == 5.0
    assert edge.delay == 2.0
    assert edge.packet_loss_rate == 0.2
    assert edge.measurement_modalities == ("RANGE",)
    assert set(observation.previous_active_edges) == {
        ("a", "b"), ("a", "c"), ("b", "c")
    }
    assert "history_checkpoint_count" in dict(
        observation.nodes[0].estimator_metrics
    )


def test_online_adapter_rejects_unknown_candidate_edge():
    try:
        build_online_graph_observation(
            _orchestrator(), candidate_edges=(("a", "unknown"),)
        )
    except ValueError as error:
        assert "topology edges" in str(error)
    else:
        raise AssertionError("Unknown candidate edge was accepted.")
