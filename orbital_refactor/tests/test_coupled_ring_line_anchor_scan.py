from experiments.coupled_ring_line_anchor_scan import scan_rolling_anchor_parameters


def test_rolling_anchor_scan_reports_each_parameter_case_pair():
    result = scan_rolling_anchor_parameters(
        seeds=(0,), baselines=(60.0,), gains=(0.1,),
        duration=700.0, dt=10.0,
    )
    assert len(result["rows"]) == 2
    assert result["best"]["minimum_bias_baseline_s"] == 60.0
    assert result["best"]["line_cue_gain"] == 0.1
