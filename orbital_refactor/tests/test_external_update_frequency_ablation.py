from experiments.external_update_frequency_ablation import (
    run_external_update_frequency_ablation,
)


def test_short_p06_ablation_reports_nested_algorithm_profiles():
    report = run_external_update_frequency_ablation(
        seeds=(0,), duration=0.2, dt=0.2,
    )
    assert report.walker_definition == (20, 10, 1)
    assert len(report.records) == 3
    assert set(report.mean_latency_by_profile_ms) == {
        "production_exact_replay_three_modal",
        "propagate_only_three_modal",
        "propagate_only_absolute_only",
    }
    assert all(item.epoch_count == 2 for item in report.records)
    assert all(item.mean_epoch_latency_ms > 0.0 for item in report.records)
