from experiments.infrared_filter_outage_shadow import (
    run_infrared_filter_outage_shadow,
)


def test_outage_shadow_is_paired_and_keeps_ci_out_of_network_filter_claim():
    report = run_infrared_filter_outage_shadow(
        seeds=(0,), duration=2.0, dt=2.0, calibration_samples=12,
    )
    assert report.paired_raw_images
    assert not report.ci_weight_available
    assert len(report.runs) == 3
    assert len(report.groups) == 3
    assert all(group.run_count == 1 for group in report.groups)
