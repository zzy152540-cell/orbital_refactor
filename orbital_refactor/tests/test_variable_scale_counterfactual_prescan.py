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
    for record in summary["records"]:
        assert record["evaluated_action_count"] >= 2
        assert record["best_nonkeep"]["kind"] in {"add", "swap", "remove"}
        assert record["legal_action_kind_counts"]["keep"] == 1
