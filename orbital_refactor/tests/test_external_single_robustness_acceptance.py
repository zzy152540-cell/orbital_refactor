import numpy as np

from experiments.external_single_robustness_acceptance import (
    run_external_single_robustness_acceptance,
)


def test_short_p08_pairs_each_full_modality_outage_with_reference():
    report = run_external_single_robustness_acceptance(
        seeds=(0,), duration=4.0, dt=2.0,
    )

    assert {group.missing_modality for group in report.groups} == {
        "opt", "ir", "rad",
    }
    assert len(report.runs) == 3
    assert not report.formal_sample_size_met
    assert not report.passed
    for run in report.runs:
        assert run.missing_modality_valid_count == 0
        assert run.remaining_modality_valid_count > 0
        assert run.finite
        assert np.isfinite(run.position_rmse_increase_percent)
        assert (
            run.dropout_optical_valid_count
            + run.dropout_infrared_valid_count
            + run.dropout_radar_valid_count
            == run.remaining_modality_valid_count
        )
        assert np.isclose(
            run.dropout_mean_ci_weight_optical
            + run.dropout_mean_ci_weight_infrared
            + run.dropout_mean_ci_weight_radar,
            1.0,
        )
        assert np.all(np.isfinite([
            run.reference_local_position_rmse_optical_m,
            run.reference_local_position_rmse_infrared_m,
            run.reference_local_position_rmse_radar_m,
            run.dropout_local_position_rmse_optical_m,
            run.dropout_local_position_rmse_infrared_m,
            run.dropout_local_position_rmse_radar_m,
        ]))
