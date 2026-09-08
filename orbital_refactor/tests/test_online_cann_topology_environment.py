import numpy as np

from brain_inspired.navigation_graph_features import (
    NAVIGATION_GRAPH_NODE_METRIC_NAMES,
)
from experiments.topology_control_environment import (
    CompactFleetScenarioDistribution,
    TopologyControlEnvironment,
)


def test_short_online_cann_environment_closes_interface_without_truth_features():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=4, decision_interval_epochs=1,
        relative_modalities=("RANGE",), cann_policy_features=True,
        cann_anchor_interval_epochs=2,
    )
    state = environment.reset(seed=3)
    assert state.policy_tensor.schema_version == "v15.1-cann-policy-normalized"
    assert state.observation.provenance.online_decision_safe
    assert state.observation.provenance.state_source == "estimator"
    assert state.observation.provenance.geometry_source == "estimator"
    for node in state.observation.nodes:
        metrics = dict(node.estimator_metrics)
        assert set(NAVIGATION_GRAPH_NODE_METRIC_NAMES) <= set(metrics)
    first = state.policy_tensor.node_features.copy()
    while True:
        step = environment.step(0)
        assert not step.action_resolution.used_fallback
        assert step.state.policy_tensor.schema_version == (
            "v15.1-cann-policy-normalized"
        )
        if step.terminated:
            final = step.state.policy_tensor.node_features
            break
    assert not np.array_equal(first, final)


def test_cann_disabled_environment_remains_v150_and_reproducible():
    options = dict(
        node_count=3, episode_epochs=2, relative_modalities=("RANGE",),
    )
    implicit = TopologyControlEnvironment(**options).reset(seed=5)
    explicit = TopologyControlEnvironment(
        **options, cann_policy_features=False,
    ).reset(seed=5)
    assert implicit.policy_tensor.schema_version == "v15.0-policy-normalized"
    np.testing.assert_array_equal(
        implicit.policy_tensor.node_features,
        explicit.policy_tensor.node_features,
    )
    np.testing.assert_array_equal(
        implicit.policy_tensor.edge_features,
        explicit.policy_tensor.edge_features,
    )


def test_online_cann_stops_anchoring_during_navigation_dropout():
    environment = TopologyControlEnvironment(
        node_count=3, episode_epochs=6, relative_modalities=("RANGE",),
        cann_policy_features=True, randomize_stage1_conditions=True,
        compact_scenario_distribution=CompactFleetScenarioDistribution(
            packet_loss_range=(0.0, 0.0),
            communication_delay_range=(0.0, 0.0),
            navigation_dropout_node_count=1,
        ),
    )
    state = environment.reset(seed=8)
    dropout_node, windows = next(iter(
        environment._episode_conditions["navigation_dropout_by_node"].items()
    ))
    start, end = windows[0]
    ages_during_dropout = []
    while True:
        timestamp = state.observation.timestamp
        node = next(item for item in state.observation.nodes
                    if item.node_id == dropout_node)
        metrics = dict(node.estimator_metrics)
        if start <= timestamp <= end:
            assert metrics["absolute_navigation_available"] == 0.0
            ages_during_dropout.append(metrics["cann_anchor_age_s"])
        step = environment.step(0)
        state = step.state
        if step.terminated:
            break
    assert ages_during_dropout
    assert max(ages_during_dropout) > 0.0
