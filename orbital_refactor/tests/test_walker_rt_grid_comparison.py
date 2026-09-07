from experiments.walker_rt_grid_comparison import (
    generate_walker_rt_grid_figure,
    run_walker_rt_grid_comparison,
)


def test_short_walker_rt_grid_comparison_is_valid(tmp_path):
    result = run_walker_rt_grid_comparison(
        duration=4.0, dt=2.0, anchor_interval_samples=1,
        anchor_outage_window=(2.0, 2.0), rate_bias_rt_mps=(0.1, 0.2),
    )
    assert set(result.metrics) == {
        "free_running", "periodic_anchor", "interrupted_anchor",
        "adaptive_anchor",
        "adaptive_interrupted_anchor",
    }
    assert all(metric["valid_fraction"] == 1.0
               for metric in result.metrics.values())
    assert all(metric["boundary_saturation_fraction"] == 0.0
               for metric in result.metrics.values())
    assert all(metric["posterior_tracking_rt_norm_rmse_m"] >= 0.0
               for metric in result.metrics.values())
    node = result.representative_node
    assert result.interrupted_anchor[node].cue_applied.tolist() == [
        False, False, True,
    ]
    assert generate_walker_rt_grid_figure(
        result, tmp_path / "rt.png",
    ).exists()
