from experiments.walker_rt_multiscale_comparison import (
    run_walker_rt_multiscale_comparison,
)


def test_short_walker_multiscale_comparison_is_complete():
    result = run_walker_rt_multiscale_comparison(
        duration=4.0, dt=2.0, reference_shift_m=15_000.0,
    )
    assert set(result.metrics) == {
        "nominal_reference", "large_reference_offset",
    }
    assert all(set(methods) == {
        "single_10km", "single_50km", "rolling_10km", "multiscale",
    } for methods in result.metrics.values())
    assert all(
        values["valid_fraction"] == 1.0
        for methods in result.metrics.values()
        for values in methods.values()
    )
    assert result.metrics["large_reference_offset"]["multiscale"][
        "boundary_saturation_fraction"
    ] == 0.0
