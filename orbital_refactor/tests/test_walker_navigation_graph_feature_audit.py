from experiments.walker_navigation_graph_feature_audit import (
    run_walker_navigation_graph_feature_audit,
)


def test_short_navigation_graph_feature_audit_is_valid():
    result = run_walker_navigation_graph_feature_audit(
        duration=4.0, dt=2.0, seed=1, anchor_interval_samples=1,
    )
    assert result.summary["node_count"] == len(result.histories) > 0
    assert result.summary["metric_count"] == 12
    assert result.summary["varying_metric_count"] > 0
    assert result.summary["effective_rank"] > 0.0
    assert result.summary["valid_fraction"] == 1.0
