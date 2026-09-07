from experiments.walker_navigation_place_cell_audit import (
    run_walker_navigation_place_cell_audit,
)


def test_short_walker_place_cell_audit_preserves_read_only_boundary():
    result = run_walker_navigation_place_cell_audit(
        duration=4.0, dt=2.0, seed=3, anchor_interval_samples=1,
    )
    summary = result.summary
    assert summary["cell_count"] == 108
    assert summary["valid_fraction"] == 1.0
    assert summary["phase_reconstruction_rmse_rad"] < 0.01
    assert summary["activity_effective_rank"] > 0.0
    assert "not an independent observation" in summary["interpretation"]
    for node, navigation in result.navigation.histories.items():
        assert result.histories[node].timestamps.shape == navigation.timestamps.shape
