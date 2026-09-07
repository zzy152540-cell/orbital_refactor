from dataclasses import replace

from brain_inspired.orbital_direction_state import OrbitalDirectionConfig
from brain_inspired.ring_cann import RingCANNConfig
from experiments.walker_direction_navigation_comparison import (
    generate_walker_direction_comparison_figure,
    run_walker_direction_navigation_comparison,
)


def test_short_walker_direction_comparison_is_causal_and_finite(tmp_path):
    config = OrbitalDirectionConfig(
        ring=replace(
            RingCANNConfig(), num_neurons=36, internal_dt=0.002,
            initialization_duration=0.02,
        ),
    )
    result = run_walker_direction_navigation_comparison(
        duration=4.0, dt=2.0, seed=0, anchor_interval_samples=1,
        anchor_outage_window=(2.0, 2.0), direction_config=config,
        phase_rate_bias_deg_per_hour=1.0,
    )
    assert set(result.metrics) == {
        "free_running", "periodic_anchor", "interrupted_anchor",
    }
    assert all(
        metrics["valid_fraction"] == 1.0
        for metrics in result.metrics.values()
    )
    node = result.representative_node
    assert result.periodic_anchor[node].cue_applied.tolist() == [
        False, True, True,
    ]
    assert result.interrupted_anchor[node].cue_applied.tolist() == [
        False, False, True,
    ]
    assert result.phase_rate_bias_deg_per_hour == 1.0
    path = generate_walker_direction_comparison_figure(
        result, tmp_path / "direction.png",
    )
    assert path.exists()
