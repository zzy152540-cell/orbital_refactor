from experiments.v14_three_satellite_local_observation import (
    run_v14_three_satellite_body_scheduling_experiment,
)


def test_three_satellite_body_scheduler_limits_optical_target_per_observer():
    result = run_v14_three_satellite_body_scheduling_experiment(
        seeds=1, duration=8.0, dt=2.0,
    )

    assert result.eci_upper_bound.psd_failure_count == 0
    assert result.body_scheduled.psd_failure_count == 0
    assert set(result.body_scheduled.mean_nis_by_modality) == {
        "RANGE", "RANGE_RATE", "AZ_EL",
    }
    selected = result.scheduling.selected_count_by_directed_edge
    assert selected[("sat_a", "sat_b")] == 5
    assert selected[("sat_a", "sat_c")] == 0
    assert result.scheduling.maximum_unobserved_visible_epochs_by_directed_edge[
        ("sat_a", "sat_b")
    ] == 0
