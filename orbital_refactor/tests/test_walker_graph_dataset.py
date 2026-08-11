import numpy as np

from cooperative.topology_policy import (
    LowChurnConnectedTreePolicy,
    TopologyAction,
)
from experiments.v14_walker_dynamic_topology import (
    build_v14_walker_dynamic_topology_plan,
    run_v14_walker_online_dynamic_filter_smoke,
)


def test_online_walker_records_filter_and_link_graph_features():
    result = run_v14_walker_online_dynamic_filter_smoke(
        duration=4.0, dt=2.0, packet_loss_rate=0.1,
        communication_delay=2.0,
    )

    dataset = result.graph_dataset
    assert dataset.feature_version == "v14.2-causal"
    assert tuple(item.timestamp for item in dataset.observations) == (0.0, 2.0, 4.0)
    assert len(dataset.transitions) == 3
    assert tuple(
        (
            item.pre_observation.timestamp,
            (
                item.next_pre_observation.timestamp
                if item.next_pre_observation is not None else None
            ),
        )
        for item in dataset.transitions
    ) == ((0.0, 2.0), (2.0, 4.0), (4.0, None))
    for transition in dataset.transitions:
        assert transition.outcome.timestamp == transition.pre_observation.timestamp
        assert set(transition.action.active_edges) <= {
            edge.nodes for edge in transition.pre_observation.candidate_edges
        }
    for observation in dataset.observations:
        assert len(observation.nodes) == 20
        assert all(node.covariance_diagonal is not None for node in observation.nodes)
        assert all(len(node.covariance_diagonal) == 6 for node in observation.nodes)
        assert all(np.all(np.asarray(node.covariance_diagonal) >= 0.0)
                   for node in observation.nodes)
        assert all(edge.delay == 2.0 for edge in observation.candidate_edges)
        assert all(edge.packet_loss_rate == 0.1
                   for edge in observation.candidate_edges)
        assert all(edge.communication_available
                   for edge in observation.candidate_edges)
        assert observation.graph_metrics == ()
        assert observation.estimation_dependency_edges
        assert all(node.estimator_metrics == () for node in observation.nodes)
        assert all(edge.nis_by_modality == ()
                   for edge in observation.candidate_edges)
    for transition in dataset.transitions:
        metrics = dict(transition.outcome.graph_metrics)
        assert metrics["active_edge_count"] == 19.0
        assert metrics["transmitted_message_count"] >= 0.0
        assert metrics["dropped_message_count"] >= 0.0
        for _, node_metrics in transition.outcome.node_metrics:
            values = dict(node_metrics)
            assert values["prior_covariance_trace"] > 0.0
            assert values["posterior_covariance_trace"] > 0.0


def test_online_walker_graph_dataset_is_reproducible():
    left = run_v14_walker_online_dynamic_filter_smoke(duration=4.0, dt=2.0)
    right = run_v14_walker_online_dynamic_filter_smoke(duration=4.0, dt=2.0)

    assert left.graph_dataset == right.graph_dataset


def test_action_outcomes_do_not_leak_into_decision_time_observation():
    reliable = run_v14_walker_online_dynamic_filter_smoke(
        duration=4.0, dt=2.0, packet_loss_rate=0.0,
    ).graph_dataset
    lossy = run_v14_walker_online_dynamic_filter_smoke(
        duration=4.0, dt=2.0, packet_loss_rate=1.0,
    ).graph_dataset

    reliable_first = reliable.transitions[0]
    lossy_first = lossy.transitions[0]
    assert reliable_first.pre_observation.nodes == lossy_first.pre_observation.nodes
    assert reliable_first.action == lossy_first.action
    assert dict(reliable_first.outcome.graph_metrics)["dropped_message_count"] == 0.0
    assert dict(lossy_first.outcome.graph_metrics)["dropped_message_count"] > 0.0


def test_online_policy_is_called_with_each_causal_pre_observation():
    class RecordingPolicy:
        def __init__(self):
            self.observations = []
            self.delegate = LowChurnConnectedTreePolicy(maximum_degree=3)

        def select(self, observation):
            self.observations.append(observation)
            return self.delegate.select(observation)

    policy = RecordingPolicy()
    result = run_v14_walker_online_dynamic_filter_smoke(
        duration=4.0, dt=2.0, topology_policy=policy,
    )

    assert tuple(policy.observations) == result.graph_dataset.observations
    assert len(policy.observations) == 3
    assert all(observation.graph_metrics == ()
               for observation in policy.observations)


def test_online_default_policy_matches_the_existing_walker_plan():
    plan = build_v14_walker_dynamic_topology_plan(duration=4.0, dt=2.0)
    result = run_v14_walker_online_dynamic_filter_smoke(
        duration=4.0, dt=2.0,
    )

    assert tuple(
        transition.action.active_edges
        for transition in result.graph_dataset.transitions
    ) == tuple(
        record.active_undirected_edges for record in plan.epoch_records
    )
    assert result.topology_change_count == plan.topology_change_count


def test_online_policy_can_activate_a_candidate_outside_the_legacy_plan():
    class OneExtraEdgePolicy:
        def __init__(self):
            self.delegate = LowChurnConnectedTreePolicy(maximum_degree=3)

        def select(self, observation):
            baseline = self.delegate.select(observation)
            selected = set(baseline.active_edges)
            extra = next(
                edge.nodes for edge in observation.candidate_edges
                if edge.nodes not in selected
            )
            return TopologyAction(
                "one_extra_edge", tuple(sorted(selected | {extra}))
            )

    result = run_v14_walker_online_dynamic_filter_smoke(
        duration=2.0, dt=2.0, topology_policy=OneExtraEdgePolicy(),
    )

    assert all(
        transition.action.policy_name == "one_extra_edge"
        for transition in result.graph_dataset.transitions
    )
    assert all(
        dict(transition.outcome.graph_metrics)["active_edge_count"] == 20.0
        for transition in result.graph_dataset.transitions
    )
