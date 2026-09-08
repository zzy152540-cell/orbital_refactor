from experiments.navigation_place_cell_stress_matrix import (
    run_navigation_place_cell_stress_matrix,
)


def test_short_place_cell_stress_matrix_exposes_scale_tradeoff():
    result = run_navigation_place_cell_stress_matrix(duration=20.0, dt=2.0)
    assert result.configuration_cell_counts == {
        "compact": 108, "wide": 972, "dual_scale": 1080,
        "hierarchical": 1272,
    }
    nominal = result.metrics["nominal"]
    large = result.metrics["large_smooth_offset"]
    assert nominal["compact"]["boundary_saturation_fraction"] == 0.0
    assert large["compact"]["boundary_saturation_fraction"] > 0.0
    assert large["wide"]["boundary_saturation_fraction"] == 0.0
    assert (
        large["dual_scale"]["rt_reconstruction_rmse_m"]
        < large["compact"]["rt_reconstruction_rmse_m"]
    )
    assert (
        large["hierarchical"]["rt_reconstruction_rmse_m"]
        < large["wide"]["rt_reconstruction_rmse_m"]
    )
    assert large["hierarchical"]["scale_transition_count"] >= 0
    excursion = result.metrics["coarse_support_excursion"]["hierarchical"]
    assert excursion["fine_scale_active_fraction"] < 1.0
    assert excursion["scale_transition_count"] > 0
