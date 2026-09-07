from experiments.walker_navigation_brain_state_audit import (
    run_walker_navigation_brain_state_audit,
)


def test_short_walker_navigation_state_audit_is_valid():
    result = run_walker_navigation_brain_state_audit(duration=10.0, dt=2.0)
    summary = result.summary
    assert summary["node_count"] == 20
    assert summary["epoch_count"] == 6
    assert summary["feature_count"] == 11
    assert 0 < summary["varying_feature_count"] <= 11
    assert summary["effective_rank"] > 0.0
    assert summary["valid_fraction"] == 1.0
    assert summary["boundary_saturation_fraction"] == 0.0
