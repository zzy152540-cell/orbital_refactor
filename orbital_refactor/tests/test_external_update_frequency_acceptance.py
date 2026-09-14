from experiments.external_update_frequency_acceptance import (
    run_external_update_frequency_acceptance,
)


def test_short_p06_prescan_reports_three_scopes_without_gui():
    report = run_external_update_frequency_acceptance(
        seeds=(0,), duration=0.4, dt=0.2,
    )
    assert report.walker_definition == (20, 10, 1)
    assert report.formal_profile == "no_cann"
    assert report.formal_latency_distribution_available is True
    assert [group.profile for group in report.groups] == [
        "no_cann", "representative_cann", "all_node_cann",
    ]
    assert len(report.records) == 3
    assert all(record.total_seconds > 0 for record in report.records)
    assert all(len(record.epoch_latency_ms) == 3 for record in report.records)
    assert report.groups[0].latency_granularity == "measured_per_epoch"
    assert all(group.mean_update_frequency_hz > 0 for group in report.groups)
