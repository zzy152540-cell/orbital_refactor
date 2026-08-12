from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from experiments.monte_carlo_graph_action_dataset import (
    MonteCarloActionTarget,
    MonteCarloGraphActionDataset,
    MonteCarloGraphActionGroup,
)


@dataclass(frozen=True)
class RobustOpportunityCriteria:
    minimum_expected_gain: float = 0.0
    minimum_safe_positive_probability: float = 0.75
    minimum_tenth_percentile_gain: float = 0.0
    minimum_mean_confidence_lower_bound: float = 0.0
    minimum_lower_tail_mean_gain: float = 0.0
    minimum_consistency_non_degrading_probability: float = 0.75
    allowed_action_kinds: tuple[str, ...] = ("add", "swap")

    def __post_init__(self):
        if not 0.0 <= self.minimum_safe_positive_probability <= 1.0:
            raise ValueError("Safe-positive probability must be in [0, 1].")
        if not 0.0 <= self.minimum_consistency_non_degrading_probability <= 1.0:
            raise ValueError("Consistency probability must be in [0, 1].")
        allowed = tuple(dict.fromkeys(self.allowed_action_kinds))
        if not allowed or set(allowed) - {"add", "swap", "remove"}:
            raise ValueError("Allowed actions must use add, swap, or remove.")


@dataclass(frozen=True)
class MonteCarloOpportunitySummary:
    scope: str
    group_count: int
    expected_positive_opportunity_rate: float
    robust_opportunity_rate: float
    mean_selected_expected_gain: float
    median_selected_expected_gain: float
    worst_selected_tenth_percentile_gain: float
    mean_selected_safe_positive_probability: float
    selected_action_kind_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class MonteCarloOpportunityReport:
    criteria: RobustOpportunityCriteria
    overall: MonteCarloOpportunitySummary
    by_scenario: tuple[MonteCarloOpportunitySummary, ...]


def analyze_monte_carlo_action_opportunities(
    dataset: MonteCarloGraphActionDataset,
    *,
    criteria: RobustOpportunityCriteria = RobustOpportunityCriteria(),
) -> MonteCarloOpportunityReport:
    if not dataset.groups:
        raise ValueError("Opportunity analysis requires Monte Carlo groups.")
    return MonteCarloOpportunityReport(
        criteria=criteria,
        overall=_summarize("all", dataset.groups, criteria),
        by_scenario=tuple(
            _summarize(
                scenario_id,
                tuple(group for group in dataset.groups
                      if group.scenario_id == scenario_id),
                criteria,
            )
            for scenario_id in dataset.scenario_ids
        ),
    )


def _summarize(scope, groups, criteria):
    selected = []
    expected_positive = []
    robust_available = []
    for group in groups:
        keep = _unique_keep(group)
        alternatives = [
            action for action in group.actions
            if action.action_kind in criteria.allowed_action_kinds
        ]
        expected_positive.append(any(
            action.mean_position_rmse_reduction
            > criteria.minimum_expected_gain
            for action in alternatives
        ))
        robust = [
            action for action in alternatives
            if _is_robust(action, criteria)
        ]
        robust_available.append(bool(robust))
        selected.append(
            max(
                robust,
                key=lambda action: (
                    action.lower_tail_mean_position_rmse_reduction,
                    action.mean_position_rmse_reduction,
                ),
            )
            if robust else keep
        )
    gains = np.asarray([
        action.mean_position_rmse_reduction for action in selected
    ])
    kinds = Counter(action.action_kind for action in selected)
    return MonteCarloOpportunitySummary(
        scope=scope,
        group_count=len(groups),
        expected_positive_opportunity_rate=float(np.mean(expected_positive)),
        robust_opportunity_rate=float(np.mean(robust_available)),
        mean_selected_expected_gain=float(np.mean(gains)),
        median_selected_expected_gain=float(np.median(gains)),
        worst_selected_tenth_percentile_gain=float(min(
            action.tenth_percentile_position_rmse_reduction
            for action in selected
        )),
        mean_selected_safe_positive_probability=float(np.mean([
            action.safe_positive_gain_probability for action in selected
        ])),
        selected_action_kind_counts=tuple(sorted(kinds.items())),
    )


def _is_robust(action: MonteCarloActionTarget, criteria):
    return (
        action.mean_position_rmse_reduction
        > criteria.minimum_expected_gain
        and action.safe_positive_gain_probability
        >= criteria.minimum_safe_positive_probability
        and action.tenth_percentile_position_rmse_reduction
        >= criteria.minimum_tenth_percentile_gain
        and action.mean_position_rmse_reduction_confidence_interval[0]
        >= criteria.minimum_mean_confidence_lower_bound
        and action.lower_tail_mean_position_rmse_reduction
        >= criteria.minimum_lower_tail_mean_gain
        and action.consistency_non_degrading_probability
        >= criteria.minimum_consistency_non_degrading_probability
    )


def _unique_keep(group: MonteCarloGraphActionGroup):
    keep = [
        action for action in group.actions if action.action_kind == "keep"
    ]
    if len(keep) != 1:
        raise ValueError("Every Monte Carlo group requires exactly one keep.")
    return keep[0]
