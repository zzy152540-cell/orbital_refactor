from experiments.coupled_ring_line_robustness import (
    evaluate_coupled_ring_line_robustness,
)


def test_coupled_ring_line_robustness_exercises_all_three_cases():
    result = evaluate_coupled_ring_line_robustness(
        seeds=range(1), duration=700.0, dt=10.0,
    )
    summaries = result["summary"]
    assert set(summaries) == {
        "initial_phase_offset", "time_varying_rate_bias",
        "insufficient_pre_outage_baseline",
    }
    short = [
        row for row in result["rows"]
        if row["case"] == "insufficient_pre_outage_baseline"
    ][0]
    assert short["pre_outage_bias_observation_count"] == 0
    assert all(row["maximum_abs_error_deg"] < 180.0 for row in result["rows"])
