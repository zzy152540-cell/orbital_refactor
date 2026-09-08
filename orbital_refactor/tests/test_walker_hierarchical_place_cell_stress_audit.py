from experiments.walker_hierarchical_place_cell_stress_audit import (
    run_walker_hierarchical_place_cell_stress_audit,
)


def test_short_walker_hierarchical_place_audit_covers_rt_stress_cases():
    result = run_walker_hierarchical_place_cell_stress_audit(
        duration=10.0, dt=2.0, anchor_interval_samples=1,
    )
    assert set(result.metrics) == {
        "constant_bias", "step_bias", "ramp_bias", "anchor_outliers",
        "extreme_unanchored", "large_reference_offset",
    }
    assert all(set(policies) == {"fixed", "adaptive", "rolling"}
               for policies in result.metrics.values())
    assert all(values["representation_rmse_m"] >= 0.0
               for policies in result.metrics.values()
               for values in policies.values())
    assert all(-1.0 <= values["entropy_error_correlation"] <= 1.0
               for policies in result.metrics.values()
               for values in policies.values())
