import numpy as np
import pytest
from types import SimpleNamespace

from cooperative.topology_policy import (
    GraphEdgeFeature,
    GraphNodeFeature,
    GraphObservation,
    TopologyAction,
)
from experiments.edge_marginal_information import (
    TopologyRolloutMetrics,
    covariance_summary,
    evaluate_candidate_edge_marginals,
    topology_rollout_metrics_from_history,
)


def _metrics(edge_count):
    return TopologyRolloutMetrics(
        mean_covariance_trace=10.0 - edge_count,
        mean_covariance_logdet=5.0 - 0.5 * edge_count,
        position_rmse=4.0 - 0.25 * edge_count,
        mean_nees=6.0 + 0.1 * edge_count,
        nees_95_coverage=0.8 + 0.01 * edge_count,
        worst_node_position_rmse=5.0 - 0.2 * edge_count,
        transmitted_message_count=2 * edge_count,
        topology_change_count=edge_count,
    )


def test_edge_marginal_evaluator_scores_additions_and_deletions():
    observation = GraphObservation(
        0.0,
        tuple(GraphNodeFeature(node, ()) for node in ("a", "b", "c")),
        tuple(GraphEdgeFeature(edge, 1.0) for edge in (
            ("a", "b"), ("a", "c"), ("b", "c"),
        )),
    )
    baseline = TopologyAction("baseline", (("a", "b"), ("b", "c")))
    calls = []

    def evaluate(action):
        calls.append(action.active_edges)
        return _metrics(len(action.active_edges))

    values = evaluate_candidate_edge_marginals(
        observation=observation,
        baseline_action=baseline,
        evaluate_action=evaluate,
    )

    assert tuple(value.edge for value in values) == (
        ("a", "b"), ("a", "c"), ("b", "c"),
    )
    assert all(value.covariance_trace_reduction == 1.0 for value in values)
    assert all(value.covariance_logdet_reduction == 0.5 for value in values)
    assert all(value.position_rmse_reduction == 0.25 for value in values)
    assert all(value.transmitted_message_cost == 2 for value in values)
    assert len(calls) == 4  # repeated baseline actions are evaluated once


def test_edge_marginal_evaluator_rejects_non_candidate_actions():
    observation = GraphObservation(
        0.0,
        tuple(GraphNodeFeature(node, ()) for node in ("a", "b", "c")),
        (GraphEdgeFeature(("a", "b"), 1.0),),
    )
    with pytest.raises(ValueError, match="non-candidate"):
        evaluate_candidate_edge_marginals(
            observation=observation,
            baseline_action=TopologyAction("bad", (("a", "c"),)),
            evaluate_action=lambda action: _metrics(len(action.active_edges)),
        )


def test_covariance_summary_uses_trace_and_stable_logdet():
    trace, logdet = covariance_summary({
        "a": np.diag([2.0, 3.0]),
        "b": np.diag([4.0, 5.0]),
    })

    assert trace == 7.0
    assert logdet == pytest.approx(0.5 * (np.log(6.0) + np.log(20.0)))

    with pytest.raises(ValueError, match="positive definite"):
        covariance_summary({"a": np.diag([1.0, 0.0])})


def test_network_history_adapter_exports_accuracy_and_consistency_metrics():
    truth = {"a": np.zeros((2, 6)), "b": np.zeros((2, 6))}
    states = {
        "a": np.array([[1.0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0]]),
        "b": np.array([[0.0, 1, 0, 0, 0, 0], [0.0, 2, 0, 0, 0, 0]]),
    }
    covariances = {node: np.repeat(np.eye(6)[None, :, :], 2, axis=0)
                   for node in truth}
    history = SimpleNamespace(
        node_ids=("a", "b"),
        active_state_history_by_node=states,
        active_covariance_history_by_node=covariances,
    )

    metrics = topology_rollout_metrics_from_history(
        history=history, truth_by_node=truth,
        transmitted_message_count=7, replay_count=2,
    )

    assert metrics.mean_covariance_trace == 6.0
    assert metrics.mean_covariance_logdet == 0.0
    assert metrics.position_rmse == pytest.approx(np.sqrt(10.0 / 4.0))
    assert metrics.worst_node_position_rmse == pytest.approx(np.sqrt(5.0 / 2.0))
    assert metrics.mean_nees == 2.5
    assert metrics.transmitted_message_count == 7
    assert metrics.replay_count == 2


def test_network_history_adapter_can_limit_metrics_to_a_future_window():
    truth = {"a": np.zeros((3, 6))}
    states = {
        "a": np.array([
            [10.0, 0, 0, 0, 0, 0],
            [2.0, 0, 0, 0, 0, 0],
            [4.0, 0, 0, 0, 0, 0],
        ])
    }
    covariances = {"a": np.repeat(np.eye(6)[None, :, :], 3, axis=0)}
    history = SimpleNamespace(
        node_ids=("a",),
        active_state_history_by_node=states,
        active_covariance_history_by_node=covariances,
    )

    metrics = topology_rollout_metrics_from_history(
        history=history,
        truth_by_node=truth,
        start_index=1,
        stop_index=3,
    )

    assert metrics.position_rmse == pytest.approx(np.sqrt(10.0))
    with pytest.raises(ValueError, match="nonempty valid"):
        topology_rollout_metrics_from_history(
            history=history,
            truth_by_node=truth,
            start_index=2,
            stop_index=2,
        )
