import numpy as np

from examples.run_distributed_consensus_ci import evaluate_fleet_rmse, run_demo


def test_distributed_consensus_demo_runs_and_returns_metrics():
    case, history = run_demo()
    metrics = evaluate_fleet_rmse(
        truth_state_history_by_node=case.truth_state_history_by_node,
        estimated_state_history_by_node=history.state_history_by_node,
    )

    assert set(history.node_ids) == {"sat_01", "sat_02", "sat_03"}
    assert history.communication_stats.attempted_report_count > 0
    assert history.communication_stats.pending_report_count == 0
    assert np.isfinite(metrics.fleet_position_rmse)
    assert np.isfinite(metrics.fleet_velocity_rmse)
