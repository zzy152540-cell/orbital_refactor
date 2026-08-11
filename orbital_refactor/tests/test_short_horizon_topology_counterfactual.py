import numpy as np

from experiments.short_horizon_topology_counterfactual import (
    build_counterfactual_actions,
    run_short_horizon_topology_counterfactual,
)


def test_three_node_actions_cover_keep_add_swap_and_remove():
    actions = build_counterfactual_actions(
        node_ids=("a", "b", "c"),
        candidate_edges=(("a", "b"), ("a", "c"), ("b", "c")),
        baseline_edges=(("a", "b"), ("b", "c")),
    )

    assert tuple(action.kind for action in actions) == (
        "keep", "add", "swap", "swap", "remove", "remove",
    )
    assert all(
        len(action.topology.active_edges) == 2
        for action in actions if action.kind in {"keep", "swap"}
    )
    assert all(
        len(action.topology.active_edges) == 3
        for action in actions if action.kind == "add"
    )


def test_short_horizon_rollouts_share_prefix_and_measure_only_future_window():
    result = run_short_horizon_topology_counterfactual(
        node_count=3, seed=0, decision_epoch=1, horizon_epochs=2,
        relative_modalities=("RANGE",),
    )

    assert result.decision_observation.timestamp == 2.0
    assert len(result.rollouts) == 6
    reference = result.rollouts[0]
    assert all(
        rollout.decision_state_by_node == reference.decision_state_by_node
        and rollout.decision_covariance_diagonal_by_node
        == reference.decision_covariance_diagonal_by_node
        for rollout in result.rollouts
    )
    assert all(np.isfinite(rollout.metrics.position_rmse)
               for rollout in result.rollouts)
    keep = next(
        rollout for rollout in result.rollouts
        if rollout.action.kind == "keep"
    )
    added = next(
        rollout for rollout in result.rollouts
        if rollout.action.kind == "add"
    )
    assert added.metrics.transmitted_message_count > (
        keep.metrics.transmitted_message_count
    )
    edge_by_nodes = {
        edge.nodes: edge
        for edge in result.decision_observation.candidate_edges
    }
    assert edge_by_nodes[("sat_01", "sat_03")].observation_age == 2.0
    assert edge_by_nodes[("sat_01", "sat_03")].nis_by_modality == ()
    assert (
        edge_by_nodes[("sat_01", "sat_03")].nis_sample_count_by_modality
        == ()
    )
    assert edge_by_nodes[("sat_01", "sat_02")].observation_age == 0.0
    assert edge_by_nodes[("sat_01", "sat_02")].nis_by_modality
    assert edge_by_nodes[("sat_01", "sat_02")].nis_sample_count_by_modality
    assert result.decision_observation.measurements
    optical = [
        measurement for measurement in result.decision_observation.measurements
        if measurement.modality == "OPTICAL"
    ]
    assert not optical


def test_physical_counterfactual_exports_body_optical_measurement_semantics():
    result = run_short_horizon_topology_counterfactual(
        node_count=3, seed=0, decision_epoch=1, horizon_epochs=1,
        relative_modalities=("RADAR", "INFRARED", "OPTICAL"),
    )

    optical = [
        measurement for measurement in result.decision_observation.measurements
        if measurement.modality == "OPTICAL"
    ]
    assert optical
    assert all(measurement.frame == "BODY" for measurement in optical)
    assert all(measurement.quaternion_i2b_wxyz is not None
               for measurement in optical)
    assert all(len(measurement.covariance) == 2 for measurement in optical)


def test_short_horizon_counterfactual_is_reproducible():
    arguments = dict(
        node_count=3, seed=2, decision_epoch=1, horizon_epochs=1,
        relative_modalities=("RANGE",),
    )
    left = run_short_horizon_topology_counterfactual(**arguments)
    right = run_short_horizon_topology_counterfactual(**arguments)

    assert left == right


def test_future_seed_changes_only_post_decision_noise():
    arguments = dict(
        node_count=3, seed=2, decision_epoch=2, horizon_epochs=2,
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    left = run_short_horizon_topology_counterfactual(
        **arguments, future_seed=100,
    )
    right = run_short_horizon_topology_counterfactual(
        **arguments, future_seed=101,
    )
    repeated = run_short_horizon_topology_counterfactual(
        **arguments, future_seed=100,
    )

    assert left.decision_observation == right.decision_observation
    assert left.rollouts != right.rollouts
    assert left == repeated


def test_future_update_order_changes_only_post_decision_fusion_order():
    arguments = dict(
        node_count=3, seed=2, future_seed=100,
        decision_epoch=2, horizon_epochs=1,
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    forward = run_short_horizon_topology_counterfactual(
        **arguments,
        future_relative_update_order=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    reverse = run_short_horizon_topology_counterfactual(
        **arguments,
        future_relative_update_order=("AZ_EL", "RANGE_RATE", "RANGE"),
    )

    assert forward.decision_observation == reverse.decision_observation
    assert forward.rollouts != reverse.rollouts
    with np.testing.assert_raises_regex(
        ValueError, "future_relative_update_order"
    ):
        run_short_horizon_topology_counterfactual(
            **arguments,
            future_relative_update_order=("RANGE", "AZ_EL"),
        )


def test_decision_time_edge_failure_is_exported_and_shared_by_all_actions():
    failed = ("sat_01", "sat_02")
    result = run_short_horizon_topology_counterfactual(
        node_count=3, seed=0, future_seed=100,
        decision_epoch=2, horizon_epochs=2,
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
        inactive_edges_after_decision=(failed,),
    )
    feature = next(
        edge for edge in result.decision_observation.candidate_edges
        if edge.nodes == failed
    )
    reference = result.rollouts[0]

    assert feature.communication_available is False
    assert all(
        rollout.decision_state_by_node == reference.decision_state_by_node
        and rollout.decision_covariance_diagonal_by_node
        == reference.decision_covariance_diagonal_by_node
        for rollout in result.rollouts
    )


def test_absolute_navigation_dropout_is_exported_as_node_status():
    result = run_short_horizon_topology_counterfactual(
        node_count=3, seed=0, future_seed=100,
        decision_epoch=2, horizon_epochs=2,
        absolute_navigation_dropout_nodes_after_decision=("sat_01",),
    )
    metrics = {
        node.node_id: dict(node.estimator_metrics)
        for node in result.decision_observation.nodes
    }

    assert metrics["sat_01"]["absolute_navigation_available"] == 0.0
    assert metrics["sat_02"]["absolute_navigation_available"] == 1.0


def test_disturbance_can_precede_the_topology_decision():
    result = run_short_horizon_topology_counterfactual(
        node_count=3, seed=0, future_seed=100,
        decision_epoch=4, horizon_epochs=1,
        inactive_edges_after_decision=(("sat_01", "sat_02"),),
        absolute_navigation_dropout_nodes_after_decision=("sat_01",),
        disturbance_start_epoch=1,
    )

    assert result.decision_observation.timestamp == 8.0
    with np.testing.assert_raises_regex(ValueError, "disturbance_start_epoch"):
        run_short_horizon_topology_counterfactual(
            decision_epoch=2, horizon_epochs=1,
            disturbance_start_epoch=3,
        )
