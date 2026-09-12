from experiments.external_cooperative_information_gain_audit import (
    run_external_cooperative_information_gain_audit,
)


def test_short_information_gain_audit_records_both_neighbors():
    report = run_external_cooperative_information_gain_audit(
        seeds=(0,), duration=4.0, dt=2.0,
    )

    assert report.sample_count > 0
    assert {group.neighbor_id for group in report.groups} == {
        "sat_p02_s01", "sat_p04_s01",
    }
    assert all(
        record.predicted_position_covariance_reduction >= 0.0
        for record in report.records
    )
