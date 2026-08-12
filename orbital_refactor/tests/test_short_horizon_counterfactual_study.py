import pytest

from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)


def test_short_horizon_study_records_actions_relative_to_keep():
    study = run_short_horizon_counterfactual_study(
        node_counts=(3,),
        seeds=(0,),
        decision_epochs=(1,),
        horizon_epochs=(2,),
    )

    assert len(study.records) == 6
    by_kind = {
        summary.action_kind: summary
        for summary in study.summaries_by_action_kind
    }
    assert set(by_kind) == {"keep", "add", "swap", "remove"}
    keep = next(
        record for record in study.records if record.action_kind == "keep"
    )
    assert keep.position_rmse_reduction == 0.0
    assert keep.transmitted_message_cost == 0
    assert keep.communication_risk_score_gain == 0.0
    assert keep.resynchronization_cost == 0
    assert by_kind["add"].mean_transmitted_message_cost > 0.0
    assert by_kind["swap"].mean_transmitted_message_cost == 0.0
    assert by_kind["remove"].mean_transmitted_message_cost < 0.0
    assert study.swap_oracle_summary.group_count == 1
    assert len(study.swap_predictor_summaries) == 11
    assert all(summary.group_count == 1
               for summary in study.swap_predictor_summaries)
    assert all(summary.best_swap_hit_rate in {0.0, 1.0}
               for summary in study.swap_predictor_summaries)
    assert len(study.swap_nis_retention_gate_summaries) == 5
    assert all(
        summary.group_count == 1
        for summary in study.swap_nis_retention_gate_summaries
    )
    assert all(
        0.0 <= summary.swap_execution_rate <= 1.0
        for summary in study.swap_nis_retention_gate_summaries
    )
    assert len(study.swap_abstention_summaries) == 11
    assert all(
        summary.group_count == 1
        for summary in study.swap_abstention_summaries
    )


def test_short_horizon_study_matrix_is_reproducible():
    arguments = dict(
        node_counts=(3,),
        seeds=(0, 1),
        decision_epochs=(1, 2),
        horizon_epochs=(1,),
    )
    left = run_short_horizon_counterfactual_study(**arguments)
    right = run_short_horizon_counterfactual_study(**arguments)

    assert left == right
    assert left.swap_oracle_summary.group_count == 4


@pytest.mark.parametrize(
    "arguments, message",
    (
        ({"seeds": (0, 0)}, "seeds"),
        ({"decision_epochs": (-1,)}, "decision_epochs"),
        ({"horizon_epochs": (0,)}, "horizon_epochs"),
        ({"node_counts": (4,)}, "node_counts"),
    ),
)
def test_short_horizon_study_rejects_invalid_matrix(arguments, message):
    with pytest.raises(ValueError, match=message):
        run_short_horizon_counterfactual_study(**arguments)
