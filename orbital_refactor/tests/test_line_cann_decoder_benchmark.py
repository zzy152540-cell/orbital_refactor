from experiments.line_cann_decoder_benchmark import (
    run_line_cann_decoder_benchmark,
)


def test_peak_fit_removes_static_boundary_bias():
    result = run_line_cann_decoder_benchmark(center_count=21)
    assert result["summary"]["peak_fit"]["edge_rmse"] < 1e-10
    assert (
        result["summary"]["hybrid"]["maximum_absolute_error"]
        < result["summary"]["centroid"]["maximum_absolute_error"]
    )
    assert (
        result["summary"]["peak_fit"]["maximum_absolute_error"]
        < result["summary"]["centroid"]["maximum_absolute_error"]
    )
