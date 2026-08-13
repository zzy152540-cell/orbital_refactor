import pytest

from cooperative.topology_policy import (
    GraphEdgeFeature, GraphMeasurementFeature, GraphNodeFeature, GraphObservation,
    GraphObservationProvenance,
    LowChurnConnectedTreePolicy,
    build_graph_observation, validate_deployment_graph_observation,
)
from experiments.v14_walker_dynamic_topology import (
    _select_connected_edges, build_v14_walker_dynamic_topology_plan,
)


def test_walker_plan_exposes_policy_ready_graph_observations():
    plan = build_v14_walker_dynamic_topology_plan(duration=4.0, dt=2.0)
    assert set(plan.graph_observation_by_timestamp) == set(plan.topology_by_timestamp)
    first = plan.graph_observation_by_timestamp[0.0]
    assert len(first.nodes) == 20
    assert all(len(node.state) == 6 for node in first.nodes)
    assert first.previous_active_edges == ()
    assert all(edge.geometrically_visible and edge.communication_available
               for edge in first.candidate_edges)


def test_policy_adapter_exactly_matches_legacy_selector_entry_point():
    nodes = ("a", "b", "c", "d")
    ranges = {("a", "b"): 1.0, ("b", "c"): 2.0, ("c", "d"): 3.0,
              ("a", "d"): 4.0, ("a", "c"): 5.0}
    previous = {("a", "d"), ("b", "c"), ("c", "d")}
    observation = GraphObservation(
        2.0, tuple(GraphNodeFeature(node, ()) for node in nodes),
        tuple(GraphEdgeFeature(edge, distance) for edge, distance in ranges.items()),
        tuple(sorted(previous)),
    )
    decision = LowChurnConnectedTreePolicy(3).select(observation)
    legacy = _select_connected_edges(nodes, ranges, previous_edges=previous,
                                     maximum_degree=3)
    assert set(decision.active_edges) == legacy
    assert decision.policy_name == "LowChurnConnectedTreePolicy"


def test_policy_rejects_disconnected_candidate_graph():
    observation = GraphObservation(
        0.0, tuple(GraphNodeFeature(node, ()) for node in ("a", "b", "c")),
        (GraphEdgeFeature(("a", "b"), 1.0),),
    )
    with pytest.raises(ValueError, match="disconnected"):
        LowChurnConnectedTreePolicy().select(observation)


def test_walker_plan_accepts_an_injected_topology_policy():
    class RecordingPolicy:
        def __init__(self):
            self.observations = []
            self.delegate = LowChurnConnectedTreePolicy(maximum_degree=3)

        def select(self, observation):
            self.observations.append(observation)
            return self.delegate.select(observation)

    policy = RecordingPolicy()
    plan = build_v14_walker_dynamic_topology_plan(
        duration=4.0, dt=2.0, topology_policy=policy,
    )

    assert tuple(item.timestamp for item in policy.observations) == (0.0, 2.0, 4.0)
    assert tuple(policy.observations) == tuple(
        plan.graph_observation_by_timestamp[timestamp]
        for timestamp in (0.0, 2.0, 4.0)
    )


def test_graph_observation_factory_accepts_filter_and_communication_features():
    observation = build_graph_observation(
        timestamp=2.0,
        state_by_node={"a": [1.0, 2.0], "b": [3.0, 4.0]},
        covariance_by_node={
            "a": [[4.0, 0.0], [0.0, 9.0]],
            "b": [[1.0, 0.0], [0.0, 1.0]],
        },
        estimator_metrics_by_node={"a": {"nees": 2.5}},
        candidate_distance_by_edge={("b", "a"): 1200.0},
        measurement_modalities_by_edge={("a", "b"): ("RADAR", "OPTICAL")},
        communication_available_by_edge={("a", "b"): False},
        delay_by_edge={("a", "b"): 2.0},
        packet_loss_rate_by_edge={("a", "b"): 0.1},
        nis_by_modality_by_edge={("a", "b"): {"RADAR": 1.5}},
        measurements=(GraphMeasurementFeature(
            "a", "b", "RADAR", "ECI",
            ((4.0, 0.0), (0.0, 0.01)),
        ),),
        nis_sample_count_by_modality_by_edge={
            ("a", "b"): {"RADAR": 3}
        },
        consecutive_anomaly_count_by_modality_by_edge={
            ("a", "b"): {"RADAR": 1}
        },
    )

    assert observation.nodes[0].covariance_diagonal == (4.0, 9.0)
    assert observation.nodes[0].estimator_metrics == (("nees", 2.5),)
    edge = observation.candidate_edges[0]
    assert edge.nodes == ("a", "b")
    assert edge.measurement_modalities == ("OPTICAL", "RADAR")
    assert not edge.communication_available
    assert edge.delay == 2.0
    assert edge.packet_loss_rate == 0.1
    assert edge.nis_by_modality == (("RADAR", 1.5),)
    assert edge.nis_sample_count_by_modality == (("RADAR", 3),)
    assert edge.consecutive_anomaly_count_by_modality == (("RADAR", 1),)
    assert observation.measurements[0].covariance[0][0] == 4.0


def test_deployment_validation_requires_explicit_safe_provenance():
    observation = build_graph_observation(
        timestamp=0.0,
        state_by_node={"a": [0.0] * 6, "b": [1.0] * 6},
        candidate_distance_by_edge={("a", "b"): 1.0},
    )
    with pytest.raises(ValueError, match="not marked"):
        validate_deployment_graph_observation(observation)

    safe = build_graph_observation(
        timestamp=0.0,
        state_by_node={"a": [0.0] * 6, "b": [1.0] * 6},
        candidate_distance_by_edge={("a", "b"): 1.0},
        provenance=GraphObservationProvenance(
            schema_version="v15.0-online",
            state_source="estimator",
            geometry_source="measurement",
            online_decision_safe=True,
        ),
    )
    validate_deployment_graph_observation(safe)


def test_online_safe_provenance_rejects_truth_geometry():
    with pytest.raises(ValueError, match="estimated or measured geometry"):
        GraphObservationProvenance(
            schema_version="v15.0-online",
            state_source="estimator",
            geometry_source="truth",
            online_decision_safe=True,
        )
