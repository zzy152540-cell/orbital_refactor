from pathlib import Path

import numpy as np

from examples.run_parameter_sweep import run_sweep


def test_parameter_sweep_exports_three_algorithm_rows():
    output = Path("results/_test_parameter_sweep.csv")
    summary = Path("results/_test_parameter_sweep_summary.csv")
    try:
        rows = run_sweep(
            seed_count=1,
            angle_sigmas_deg=[0.1],
            process_noises=[1e-8],
            modes=["range_rate"],
            output_path=output,
            ci_grid_points=5,
        )
        assert len(rows) == 3
        assert {row["algorithm"] for row in rows} == {
            "centralized",
            "local_6d",
            "fleet_ci",
        }
        assert output.exists()
        assert summary.exists()
        assert all(np.isfinite(float(row["position_rmse"])) for row in rows)
        assert all(np.isfinite(float(row["mean_nees"])) for row in rows)
    finally:
        if output.exists():
            output.unlink()
        if summary.exists():
            summary.unlink()
