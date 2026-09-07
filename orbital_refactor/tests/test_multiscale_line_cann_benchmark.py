from experiments.multiscale_line_cann_benchmark import (
    run_multiscale_line_cann_benchmark,
)


def test_multiscale_line_benchmark_crosses_cells_without_saturation():
    result = run_multiscale_line_cann_benchmark(
        duration=20.0, dt=1.0, amplitude=2_000.0, period=20.0,
    )
    assert result.metrics["coarse_cell_change_count"] > 0
    assert result.metrics["boundary_saturation_fraction"] == 0.0
    assert result.metrics["multiscale_rmse"] < 1_000.0
    assert result.metrics["fine_mean_bump_width"] < result.metrics[
        "coarse_mean_bump_width"
    ]
