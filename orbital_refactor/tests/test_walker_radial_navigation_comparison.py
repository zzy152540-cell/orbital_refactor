from experiments.walker_radial_navigation_comparison import (
    generate_walker_radial_comparison_figure,
    run_walker_radial_navigation_comparison,
)


def test_short_walker_radial_comparison_is_finite_and_causal(tmp_path):
    result = run_walker_radial_navigation_comparison(
        duration=4.0, dt=2.0, anchor_interval_samples=1,
        anchor_outage_window=(2.0, 2.0), radial_rate_bias_mps=0.1,
    )
    assert set(result.metrics) == {
        "free_running", "periodic_anchor", "interrupted_anchor",
    }
    assert all(metric["valid_fraction"] == 1.0
               for metric in result.metrics.values())
    node = result.representative_node
    assert result.interrupted_anchor[node].cue_applied.tolist() == [
        False, False, True,
    ]
    path = generate_walker_radial_comparison_figure(
        result, tmp_path / "radial.png",
    )
    assert path.exists()
