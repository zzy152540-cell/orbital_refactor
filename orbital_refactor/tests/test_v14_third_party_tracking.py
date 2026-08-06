from experiments.v14_third_party_tracking import (
    run_v14_third_party_tracking_experiment,
)


def test_third_party_tracking_experiment_updates_only_remote_target_safely():
    result = run_v14_third_party_tracking_experiment(
        seeds=1, duration=8.0, dt=2.0,
    )

    assert result.propagate_only.mode == "propagate_only"
    assert result.propagate_only.mean_nis_by_modality == {}
    assert set(result.independent_approximation.mean_nis_by_modality) == {
        "RANGE", "RANGE_RATE", "AZ_EL",
    }
    assert set(result.schmidt_consistent.mean_nis_by_modality) == {
        "RANGE", "RANGE_RATE", "AZ_EL",
    }
    assert result.schmidt_consistent.mean_position_rmse > 0.0
    assert result.independent_approximation.psd_failure_count == 0
    assert result.schmidt_consistent.psd_failure_count == 0
