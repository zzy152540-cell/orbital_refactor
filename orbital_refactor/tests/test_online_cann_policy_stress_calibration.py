from experiments.online_cann_policy_stress_calibration import (
    run_online_cann_policy_stress_calibration,
)


def test_stress_calibration_activates_events_without_changing_filter():
    results = run_online_cann_policy_stress_calibration(
        seed=5, node_count=3, episode_epochs=4, dt=0.2,
    )
    assert all(item.base_filter_identical_to_nominal for item in results)
    assert any(
        count > 0
        for item in results[1:]
        for _, count in item.event_count_by_feature
    )
