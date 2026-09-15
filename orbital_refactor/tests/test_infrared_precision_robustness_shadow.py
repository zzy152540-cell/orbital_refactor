from experiments.infrared_precision_robustness_shadow import (
    run_infrared_precision_robustness_shadow,
)


def test_short_precision_shadow_is_paired_and_keeps_baseline_default():
    report = run_infrared_precision_robustness_shadow(
        seeds=(0,), duration=4.0, dt=2.0,
        infrared_angle_sigmas_deg=(0.05, 0.00625),
    )
    assert report.baseline_unchanged
    assert report.baseline_sigma_deg == 0.05
    assert len(report.runs) == 2
    assert len(report.groups) == 2
    assert all(group.run_count == 1 for group in report.groups)
