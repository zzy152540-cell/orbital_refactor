import numpy as np

from experiments.external_single_accuracy_acceptance import (
    run_external_single_accuracy_acceptance,
)


def test_short_p07_pair_uses_matching_measurements_and_finite_metrics():
    report = run_external_single_accuracy_acceptance(
        seeds=(0,), duration=4.0, dt=2.0,
    )

    assert len(report.runs) == 1
    run = report.runs[0]
    assert run.measurement_counts_identical
    assert run.finite
    assert run.best_single_modality in {"opt", "ir", "rad"}
    assert run.best_single_modality_position_rmse_m == min(
        run.optical_only_position_rmse_m,
        run.infrared_only_position_rmse_m,
        run.radar_only_position_rmse_m,
    )
    assert np.isfinite(run.ekf_mean_nis_optical)
    assert np.isfinite(run.federated_mean_nis_optical)
    assert np.isfinite(run.early_position_improvement_percent)
    assert np.isfinite(run.stable_position_improvement_percent)
    assert np.isfinite(run.federated_mean_ci_weight_radar)
    assert np.isfinite(run.federated_local_position_rmse_optical_m)
    assert np.isfinite(run.federated_local_mean_covariance_trace_infrared)
    assert np.isclose(
        run.federated_mean_ci_weight_optical
        + run.federated_mean_ci_weight_infrared
        + run.federated_mean_ci_weight_radar,
        1.0,
    )
    assert np.isfinite(report.mean_position_improvement_percent)
    assert report.position_improvement_95_half_width_percent == 0.0
    assert not report.formal_sample_size_met
    assert not report.passed
