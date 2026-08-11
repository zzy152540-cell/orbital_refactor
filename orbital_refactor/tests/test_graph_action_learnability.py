from dataclasses import replace

import pytest

from experiments.graph_action_learnability import (
    ActionCostWeights,
    analyze_graph_action_learnability,
    consistency_safe_action_kind_oracles,
    stratify_graph_action_learnability,
)
from experiments.short_horizon_counterfactual_study import (
    run_short_horizon_counterfactual_study,
)


def _controlled_study():
    study = run_short_horizon_counterfactual_study(
        seeds=(0,), decision_epochs=(1,), horizon_epochs=(1,),
    )
    keep = next(record for record in study.records
                if record.action_kind == "keep")
    unsafe = replace(
        keep, action_kind="add", position_rmse_reduction=0.20,
        nees_calibration_improvement=-0.10,
        nees_coverage_calibration_improvement=0.02,
        transmitted_message_cost=1, topology_change_cost=1,
    )
    safe_expensive = replace(
        keep, action_kind="swap", position_rmse_reduction=0.10,
        worst_node_position_rmse_reduction=0.08,
        nees_calibration_improvement=0.02,
        nees_coverage_calibration_improvement=0.01,
        transmitted_message_cost=10, topology_change_cost=1,
    )
    safe_efficient = replace(
        keep, action_kind="remove", position_rmse_reduction=0.08,
        worst_node_position_rmse_reduction=0.06,
        nees_calibration_improvement=0.01,
        nees_coverage_calibration_improvement=0.01,
        transmitted_message_cost=0, topology_change_cost=1,
    )
    return replace(
        study, records=(keep, unsafe, safe_expensive, safe_efficient)
    )


def test_learnability_report_separates_accuracy_safety_and_cost_oracles():
    report = analyze_graph_action_learnability(
        _controlled_study(),
        cost_weights=ActionCostWeights(
            transmitted_message=0.01, topology_change=0.001,
        ),
    )

    assert report.group_count == 1
    assert report.action_count == 4
    assert report.safe_positive_action_availability_rate == 1.0
    assert report.summary("keep").mean_position_rmse_reduction == 0.0
    assert (
        report.summary("unconstrained_oracle").action_kind_counts
        == (("add", 1),)
    )
    assert (
        report.summary("consistency_safe_oracle").action_kind_counts
        == (("swap", 1),)
    )
    assert (
        report.summary("cost_aware_safe_oracle").action_kind_counts
        == (("remove", 1),)
    )
    assert (
        report.summary("consistency_safe_oracle")
        .nees_calibration_violation_rate == 0.0
    )


def test_learnability_report_validates_weights_and_real_study():
    with pytest.raises(ValueError, match="weights"):
        ActionCostWeights(transmitted_message=-1.0)

    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1), decision_epochs=(1,), horizon_epochs=(1,),
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    report = analyze_graph_action_learnability(study)

    assert report.group_count == 2
    assert report.action_count == len(study.records)
    assert report.summary("unconstrained_oracle").worst_position_rmse_reduction >= 0.0
    assert report.summary("consistency_safe_oracle").worst_position_rmse_reduction >= 0.0


def test_learnability_strata_and_action_kind_oracles_preserve_abstention():
    study = run_short_horizon_counterfactual_study(
        seeds=(0, 1), decision_epochs=(1, 3), horizon_epochs=(1, 2),
    )
    by_horizon = stratify_graph_action_learnability(
        study, dimension="horizon_epochs"
    )
    kinds = consistency_safe_action_kind_oracles(study)

    assert tuple(value.value for value in by_horizon) == (1, 2)
    assert sum(value.report.group_count for value in by_horizon) == 8
    assert {value.policy for value in kinds} == {
        "consistency_safe_add_oracle",
        "consistency_safe_remove_oracle",
        "consistency_safe_swap_oracle",
    }
    assert all(value.worst_position_rmse_reduction >= 0.0 for value in kinds)
    with pytest.raises(ValueError, match="Unsupported"):
        stratify_graph_action_learnability(study, dimension="seed")
