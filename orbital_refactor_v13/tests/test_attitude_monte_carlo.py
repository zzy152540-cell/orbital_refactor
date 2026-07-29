import csv

import numpy as np

from examples.run_v13_4_attitude_monte_carlo import (
    METHODS,
    export_results,
    run_monte_carlo,
)


def test_attitude_monte_carlo_aggregates_and_exports(tmp_path):
    raw, summary = run_monte_carlo(
        [101, 102, 103],
        duration=20.0,
    )

    assert len(raw) == 3 * len(METHODS)
    assert len(summary) == len(METHODS) + 1
    assert not any(row["failed"] for row in raw)
    paired = next(
        row for row in summary if row["method"] == "paired_covariance_effect"
    )
    assert paired["successful_runs"] == 3
    assert paired["failure_rate"] == 0.0
    assert 0.0 <= paired["angle_nis_consistency_win_rate"] <= 1.0
    for row in summary[:-1]:
        assert row["runs"] == 3
        assert np.isfinite(row["mean_angle_nis_mean"])

    raw_path, summary_path = export_results(
        raw,
        summary,
        tmp_path / "attitude_mc.csv",
    )
    with raw_path.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(raw)
    with summary_path.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(summary)
