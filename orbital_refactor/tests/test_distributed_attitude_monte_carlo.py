import csv

import numpy as np

from examples.run_v13_4_distributed_monte_carlo import (
    METHODS,
    export_results,
    run_monte_carlo,
)


def test_distributed_attitude_monte_carlo_groups_communication_scenarios(tmp_path):
    raw, summary = run_monte_carlo(
        [501],
        packet_loss_rates=[0.0, 0.25],
        delays=[0.0, 2.0],
        duration=4.0,
        ci_grid_points=5,
    )

    scenario_count = 4
    assert len(raw) == scenario_count * len(METHODS)
    assert len(summary) == scenario_count * (len(METHODS) + 1)
    assert not any(row["failed"] for row in raw)
    paired = [
        row for row in summary if row["method"] == "paired_covariance_effect"
    ]
    assert len(paired) == scenario_count
    assert all(row["successful_runs"] == 1 for row in paired)
    assert all(
        0.0 <= row["angle_nis_consistency_win_rate"] <= 1.0
        for row in paired
    )
    lossy = [row for row in raw if row["packet_loss_rate"] == 0.25]
    assert all(np.isfinite(row["realized_packet_loss_rate"]) for row in lossy)
    assert all(np.isfinite(row["pre_ci_position_rmse"]) for row in raw)
    assert all(np.isfinite(row["ci_position_gain"]) for row in raw)

    local_only = [
        row
        for row in raw
        if row["method"] == "mekf_with_covariance_local_only"
    ]
    by_delay = {}
    for row in local_only:
        by_delay.setdefault(row["delay"], []).append(row)
    for rows in by_delay.values():
        assert rows[0]["position_rmse"] == rows[1]["position_rmse"]
        assert rows[0]["velocity_rmse"] == rows[1]["velocity_rmse"]

    raw_path, summary_path = export_results(
        raw,
        summary,
        tmp_path / "distributed_attitude_mc.csv",
    )
    with raw_path.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(raw)
    with summary_path.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(summary)
