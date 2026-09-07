from experiments.walker_rt_grid_stress_matrix import (
    _shift_reference_radially,
    run_walker_rt_grid_stress_matrix,
)
import numpy as np


def test_short_rt_grid_stress_matrix_covers_required_cases():
    result = run_walker_rt_grid_stress_matrix(
        duration=10.0, dt=2.0, anchor_interval_samples=1,
        outlier_interval_samples=2,
    )
    assert set(result.metrics) == {
        "constant_bias", "step_bias", "ramp_bias", "anchor_outliers",
        "extreme_unanchored",
        "large_reference_offset",
    }
    assert all(set(policies) == {"fixed", "adaptive", "rolling"}
               for policies in result.metrics.values())
    assert result.outlier_count_per_node > 0
    assert all(
        values["valid_fraction"] == 1.0
        for policies in result.metrics.values()
        for values in policies.values()
    )


def test_radial_reference_shift_preserves_orbital_scale():
    reference = np.array([[7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]])
    shifted = _shift_reference_radially(reference, -15_000.0)
    assert np.linalg.norm(shifted[0, :3]) == 6_985_000.0
    assert np.linalg.norm(reference[0, :3]) == 7.0e6
