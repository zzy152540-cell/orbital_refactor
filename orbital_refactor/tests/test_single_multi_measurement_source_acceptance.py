from experiments.single_multi_measurement_source_acceptance import (
    run_single_multi_measurement_source_acceptance,
)


def test_measurement_source_acceptance_runs_paired_seed():
    summary, rows = run_single_multi_measurement_source_acceptance(
        seeds=(0,), duration=2.0, dt=2.0,
        calibration_seed=7, calibration_samples=2,
    )
    assert summary.seeds == (0,)
    assert summary.run_count == 1
    assert summary.all_single_modalities_observable
    assert summary.all_position_rmse_finite
    assert len(rows) == 4
