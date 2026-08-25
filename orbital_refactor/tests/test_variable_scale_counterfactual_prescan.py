from experiments.variable_scale_counterfactual_prescan import (
    run_variable_scale_counterfactual_prescan,
)


def test_bounded_prescan_reports_later_nonkeep_opportunities():
    summary = run_variable_scale_counterfactual_prescan(
        condition_seeds=(320,), decision_indices=(0, 1),
        horizon_decisions=1, maximum_actions_per_kind=1,
    )
    assert len(summary["records"]) == 2
    assert set(summary["summary_by_node_count"]) == {"5"}
    assert set(summary["summary_by_decision_index"]) == {"0", "1"}
    assert summary["overall"]["keep_reward_rms"] > 0.0
    assert summary["overall"]["all_nonkeep_gain_rms"] >= 0.0
    for record in summary["records"]:
        assert record["evaluated_action_count"] >= 2
        assert record["best_nonkeep"]["kind"] in {"add", "swap", "remove"}
        assert record["legal_action_kind_counts"]["keep"] == 1
        assert "keep_cumulative_reward" in record
