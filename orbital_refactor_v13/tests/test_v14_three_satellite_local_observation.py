from experiments.v14_three_satellite_local_observation import (
    run_v14_three_satellite_local_observation_experiment,
)


def test_three_satellite_local_observation_experiment_uses_all_local_edges():
    result = run_v14_three_satellite_local_observation_experiment(
        seeds=1, duration=8.0, dt=2.0,
    )

    assert len(result.summary_by_case_and_mode) == 4
    assert 0.0 < result.visibility_summary.overall.visibility_rate < 1.0
    exact = result.summary_by_case_and_mode[
        ("visibility_limited", "exact_transport_event_replay")
    ]
    assert set(exact.mean_nis_by_modality) == {"RANGE", "RANGE_RATE", "AZ_EL"}
    assert set(exact.mean_position_rmse_by_node) == {"sat_a", "sat_b", "sat_c"}
    assert set(exact.mean_nees_by_node) == {"sat_a", "sat_b", "sat_c"}
    assert set(exact.observation_count_by_directed_edge) == {
        ("sat_a", "sat_b"), ("sat_b", "sat_a"),
    }
    assert exact.message_acceptance_rate == 1.0
    assert exact.psd_failure_count == 0
