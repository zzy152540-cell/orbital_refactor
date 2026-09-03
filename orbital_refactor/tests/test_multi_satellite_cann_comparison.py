import numpy as np

from experiments.multi_satellite_cann_comparison import (
    run_multi_satellite_cann_comparison,
)


def test_short_three_satellite_cann_comparison_runs_paired_paths():
    result = run_multi_satellite_cann_comparison(
        duration=8.0, seed=2,
        fault_by_node={"sat_01": {"radar": (4.0,)}},
    )
    summary = result["summary"]
    assert set(summary["baseline_local_position_rmse_m"]) == {
        "sat_01", "sat_02", "sat_03",
    }
    assert np.isfinite(summary["baseline_position_rmse_m"])
    assert np.isfinite(summary["cann_position_rmse_m"])
    diagnostics = summary["fault_diagnostics"]["sat_01"]
    assert set(diagnostics) == {"baseline", "cann"}
    assert 0.0 <= diagnostics["baseline"]["mean_ci_weight"] <= 1.0
    assert np.isfinite(diagnostics["cann"]["local_position_rmse_m"])
