import numpy as np

from cooperative.consensus_ci import run_consensus_ci_step
from cooperative.communication_channel import CommunicationChannel
from cooperative.satellite_node import SatelliteNode
from cooperative.topology import NetworkTopology, chain_topology
from orbital_core.measurements import measure_relative_range


def _node(node_id, offset, covariance_scale):
    return SatelliteNode(
        node_id=node_id,
        state=np.array([offset, 0.0, 0.0, 0.0, 7.5, 0.0]),
        covariance=np.eye(6) * covariance_scale,
    )


def test_chain_topology_builds_symmetric_neighbors():
    topology = chain_topology(["sat_01", "sat_02", "sat_03"])
    assert topology.neighbors("sat_01") == ("sat_02",)
    assert topology.neighbors("sat_02") == ("sat_01", "sat_03")
    assert topology.neighbors("sat_03") == ("sat_02",)


def test_topology_rejects_asymmetric_edges():
    try:
        NetworkTopology({"sat_01": ["sat_02"], "sat_02": []})
    except ValueError as exc:
        assert "symmetric" in str(exc)
    else:
        raise AssertionError("Expected asymmetric topology to be rejected.")


def test_consensus_ci_keeps_per_node_estimates_without_global_state():
    nodes = {
        "sat_01": _node("sat_01", 0.0, 9.0),
        "sat_02": _node("sat_02", 10.0, 4.0),
        "sat_03": _node("sat_03", 20.0, 16.0),
    }
    result = run_consensus_ci_step(
        nodes=nodes,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        timestamp=10.0,
        grid_points=11,
    )

    assert set(result.estimates) == {"sat_01", "sat_02", "sat_03"}
    assert not hasattr(result, "global_state")
    assert result.received_reports_by_node["sat_01"] == ["sat_02"]
    assert result.received_reports_by_node["sat_02"] == ["sat_01", "sat_03"]
    assert result.received_reports_by_node["sat_03"] == ["sat_02"]
    for node_id, estimate in result.estimates.items():
        assert estimate.node_id == node_id
        assert estimate.state.shape == (6,)
        assert estimate.covariance.shape == (6, 6)
        assert abs(sum(result.node_weight_by_node[node_id].values()) - 1.0) < 1e-12


def test_consensus_ci_respects_packet_loss():
    nodes = {
        "sat_01": _node("sat_01", 0.0, 9.0),
        "sat_02": _node("sat_02", 10.0, 4.0),
    }
    result = run_consensus_ci_step(
        nodes=nodes,
        topology=chain_topology(["sat_01", "sat_02"]),
        timestamp=0.0,
        communication_channel=CommunicationChannel(
            packet_loss_rate={"sat_01": 1.0, "sat_02": 0.0},
            random_seed=1,
        ),
    )

    assert result.received_reports_by_node["sat_01"] == ["sat_02"]
    assert result.received_reports_by_node["sat_02"] == []
    assert result.node_weight_by_node["sat_02"] == {"sat_02": 1.0}


def test_measure_relative_range_uses_absolute_positions():
    state_i = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    state_j = np.array([4.0, 6.0, 3.0, 0.0, 0.0, 0.0])
    assert measure_relative_range(state_i, state_j) == 5.0
    assert measure_relative_range(state_i, state_j, noise=0.5) == 5.5
