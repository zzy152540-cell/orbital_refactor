from experiments.coupled_ring_line_evaluation import evaluate_coupled_ring_line


def test_short_coupled_evaluation_preserves_seed_partition():
    result = evaluate_coupled_ring_line(
        seeds=(0,), duration=20.0, dt=2.0, outage_window=(8.0, 12.0),
    )
    assert result["evaluation_seeds"] == [0]
    assert len(result["rows"]) == 1
    assert result["summary"]["mean_outage_rmse_deg"] >= 0.0


def test_coupled_evaluation_accepts_rolling_configuration():
    result = evaluate_coupled_ring_line(
        seeds=(0,), duration=40.0, dt=10.0,
        outage_window=(20.0, 30.0), bias_anchor_mode="rolling_cue",
        minimum_bias_baseline=10.0, line_cue_gain=0.2,
    )
    assert result["bias_anchor_mode"] == "rolling_cue"
    assert result["minimum_bias_baseline"] == 10.0
    assert result["line_cue_gain"] == 0.2
