from dataclasses import replace

import pytest

from experiments.monte_carlo_action_opportunity import (
    RobustOpportunityCriteria,
    analyze_monte_carlo_action_opportunities,
)
from experiments.monte_carlo_graph_action_dataset import (
    build_monte_carlo_graph_action_dataset,
)


def test_robust_opportunity_requires_mean_probability_and_lower_tail():
    dataset = build_monte_carlo_graph_action_dataset(
        scenario_id="controlled",
        prefix_seeds=(0,),
        future_noise_seeds=(100, 101),
        decision_epochs=(1,),
        horizon_epochs=(1,),
    )
    group = dataset.groups[0]
    changed = []
    for action in group.actions:
        updates = {}
        if action.action_kind == "add":
            updates = dict(
                mean_position_rmse_reduction=0.10,
                safe_positive_gain_probability=0.80,
                tenth_percentile_position_rmse_reduction=0.01,
                lower_tail_mean_position_rmse_reduction=0.01,
                mean_position_rmse_reduction_confidence_interval=(0.01, 0.2),
                consistency_non_degrading_probability=0.80,
            )
        elif action.action_kind == "swap":
            updates = dict(
                mean_position_rmse_reduction=0.20,
                safe_positive_gain_probability=0.50,
                tenth_percentile_position_rmse_reduction=0.10,
            )
        elif action.action_kind == "remove":
            updates = dict(
                mean_position_rmse_reduction=0.05,
                safe_positive_gain_probability=0.80,
                tenth_percentile_position_rmse_reduction=-0.01,
            )
        changed.append(replace(action, **updates))
    controlled = replace(
        dataset,
        groups=(replace(group, actions=tuple(changed)),),
    )
    report = analyze_monte_carlo_action_opportunities(controlled)

    assert report.overall.expected_positive_opportunity_rate == 1.0
    assert report.overall.robust_opportunity_rate == 1.0
    assert report.overall.selected_action_kind_counts == (("add", 1),)
    assert report.overall.mean_selected_expected_gain == 0.10


def test_opportunity_criteria_validate_probability():
    with pytest.raises(ValueError, match="probability"):
        RobustOpportunityCriteria(minimum_safe_positive_probability=1.1)
    with pytest.raises(ValueError, match="Consistency probability"):
        RobustOpportunityCriteria(
            minimum_consistency_non_degrading_probability=1.1
        )


def test_robust_opportunity_rejects_negative_confidence_lower_bound():
    dataset = build_monte_carlo_graph_action_dataset(
        scenario_id="controlled", prefix_seeds=(0,),
        future_noise_seeds=(100, 101), decision_epochs=(1,),
        horizon_epochs=(1,),
    )
    group = dataset.groups[0]
    changed = tuple(
        replace(
            action,
            mean_position_rmse_reduction=0.1,
            safe_positive_gain_probability=1.0,
            tenth_percentile_position_rmse_reduction=0.01,
            lower_tail_mean_position_rmse_reduction=0.01,
            mean_position_rmse_reduction_confidence_interval=(-0.01, 0.2),
            consistency_non_degrading_probability=1.0,
        ) if action.action_kind == "add" else action
        for action in group.actions
    )
    report = analyze_monte_carlo_action_opportunities(
        replace(dataset, groups=(replace(group, actions=changed),))
    )

    assert report.overall.robust_opportunity_rate == 0.0
    assert report.overall.selected_action_kind_counts == (("keep", 1),)
