import csv

import numpy as np

from examples.run_v13_4_attitude_consistency_monte_carlo import (
    export_results,
    run_monte_carlo,
)


def test_attitude_consistency_monte_carlo_exports_nees_and_sensor_nis(tmp_path):
    rows, summary, failures = run_monte_carlo(
        [701, 702, 703],
        duration=20.0,
        satellite_count=2,
    )

    assert len(rows) == 3 * 2 * 11
    assert failures == []
    assert summary["successful_seeds"] == 3
    assert summary["failed_seeds"] == 0
    assert summary["failure_rate"] == 0.0
    for key in (
        "attitude_nees_sample_mean",
        "gyro_nis_sample_mean",
        "star_tracker_nis_sample_mean",
        "attitude_nees_ratio_to_dimension",
        "gyro_nis_ratio_to_dimension",
        "star_tracker_nis_ratio_to_dimension",
    ):
        assert np.isfinite(summary[key])

    raw_path, summary_path, failures_path = export_results(
        rows,
        summary,
        failures,
        tmp_path / "attitude_consistency.csv",
    )
    with raw_path.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(rows)
    with summary_path.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 1
    with failures_path.open(encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == []
