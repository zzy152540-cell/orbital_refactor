from experiments.external_cooperative_accuracy_acceptance import (
    run_external_cooperative_accuracy_acceptance,
)


def test_short_p09_run_is_paired_and_distinguishes_cooperative_arm():
    report = run_external_cooperative_accuracy_acceptance(
        seeds=(0,), duration=4.0, dt=2.0, formal_minimum_run_count=2,
    )

    assert report.walker_definition == (20, 10, 1)
    assert report.run_count == 1
    assert len(report.node_records) == 20
    assert not report.formal_sample_size_met
    assert not report.formal_duration_met
    assert not report.passed
    record = report.records[0]
    assert record.independent_position_rmse_m > 0.0
    assert record.cooperative_position_rmse_m > 0.0
    assert record.independent_run_seconds >= 0.0
    assert record.cooperative_run_seconds >= 0.0
