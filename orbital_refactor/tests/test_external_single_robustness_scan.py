from experiments.external_single_robustness_scan import (
    run_external_single_robustness_scan,
)


def test_short_p08_scan_covers_policy_and_modality_axes():
    report = run_external_single_robustness_scan(
        seeds=(0,), duration=4.0, dt=2.0, outage_durations=(2.0,),
        ci_objectives=("trace",), reset_feedback_modes=(True, False),
    )

    assert len(report.records) == 6
    assert len(report.groups) == 6
    assert {record.missing_modality for record in report.records} == {
        "opt", "ir", "rad",
    }
    assert {record.reset_feedback for record in report.records} == {True, False}
