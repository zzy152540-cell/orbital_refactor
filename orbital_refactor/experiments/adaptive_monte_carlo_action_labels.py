from __future__ import annotations

from dataclasses import dataclass, replace

from experiments.monte_carlo_action_opportunity import RobustOpportunityCriteria
from experiments.monte_carlo_graph_action_dataset import (
    aggregate_monte_carlo_action_rollouts,
    MonteCarloGraphActionGroup,
    MonteCarloGraphActionDataset,
)
from experiments.short_horizon_topology_counterfactual import (
    run_short_horizon_topology_counterfactual,
)


@dataclass(frozen=True)
class AdaptiveSamplingReport:
    dataset: MonteCarloGraphActionDataset
    initial_future_noise_seeds: tuple[int, ...]
    extended_future_noise_seeds: tuple[int, ...]
    extended_action_count: int
    total_action_count: int


def build_adaptive_monte_carlo_graph_action_dataset(
    *, scenario_id: str, prefix_seeds, initial_future_noise_seeds,
    extended_future_noise_seeds,
    criteria: RobustOpportunityCriteria = RobustOpportunityCriteria(),
    probability_margin: float = 0.2, gain_margin: float = 0.02,
    **experiment_options,
) -> AdaptiveSamplingReport:
    """Screen all actions, then relabel only borderline actions at full depth."""
    initial = _unique_seeds(initial_future_noise_seeds, "initial")
    extended = _unique_seeds(extended_future_noise_seeds, "extended")
    if not set(initial).issubset(extended):
        raise ValueError("Extended future seeds must include all initial seeds.")
    if len(extended) <= len(initial):
        raise ValueError("Extended sampling must add future branches.")
    if probability_margin < 0.0 or gain_margin < 0.0:
        raise ValueError("Adaptive margins cannot be negative.")
    prefixes = _unique_seeds(prefix_seeds, "prefix")
    decision_epochs = tuple(experiment_options.pop(
        "decision_epochs", (1, 3, 5)
    ))
    horizon_options = tuple(experiment_options.pop(
        "horizon_epochs", (1, 3)
    ))
    relative_modalities = tuple(experiment_options.get(
        "relative_modalities", ("RANGE", "RANGE_RATE", "AZ_EL")
    ))
    final_groups = []
    extended_count = 0
    total_action_count = 0
    for prefix_seed in prefixes:
      for decision_epoch in decision_epochs:
       for horizon in horizon_options:
        initial_results = _run_results(
            prefix_seed, decision_epoch, horizon, initial, None,
            experiment_options,
        )
        initial_actions = aggregate_monte_carlo_action_rollouts(
            initial_results, initial,
            experiment_options.get("severe_relative_loss_threshold", 0.05),
        )
        group = MonteCarloGraphActionGroup(
            scenario_id=scenario_id,
            prefix_seed=prefix_seed,
            decision_epoch=decision_epoch,
            horizon_epochs=horizon,
            decision_observation=initial_results[0].decision_observation,
            actions=initial_actions,
        )
        total_action_count += len(group.actions)
        keep = next(action for action in group.actions
                    if action.action_kind == "keep")
        borderline = tuple(
            action for action in group.actions
            if _is_borderline(action, criteria, probability_margin, gain_margin)
        )
        if not borderline:
            final_groups.append(group)
            continue
        selected_edges = (keep.active_edges,) + tuple(
            action.active_edges for action in borderline
        )
        additional = tuple(seed for seed in extended if seed not in initial)
        selected_initial_results = tuple(
            replace(
                result,
                rollouts=tuple(
                    rollout for rollout in result.rollouts
                    if rollout.action.topology.active_edges in selected_edges
                ),
            )
            for result in initial_results
        )
        additional_results = _run_results(
            group.prefix_seed, group.decision_epoch, group.horizon_epochs,
            additional, selected_edges, experiment_options,
        )
        refined_actions = aggregate_monte_carlo_action_rollouts(
            selected_initial_results + additional_results,
            extended,
            experiment_options.get("severe_relative_loss_threshold", 0.05),
        )
        refined_by_edges = {action.active_edges: action
                            for action in refined_actions}
        final_groups.append(replace(
            group, actions=tuple(
                refined_by_edges.get(action.active_edges, action)
                for action in group.actions
            ),
        ))
        extended_count += len(borderline)
    return AdaptiveSamplingReport(
        dataset=MonteCarloGraphActionDataset(
            scenario_ids=(scenario_id,),
            prefix_seeds=prefixes,
            future_noise_seeds=extended,
            relative_modalities=relative_modalities,
            groups=tuple(final_groups),
        ),
        initial_future_noise_seeds=initial,
        extended_future_noise_seeds=extended,
        extended_action_count=extended_count,
        total_action_count=total_action_count,
    )


def _is_borderline(action, criteria, probability_margin, gain_margin):
    if action.action_kind not in criteria.allowed_action_kinds:
        return False
    return (
        action.safe_positive_gain_probability
        >= criteria.minimum_safe_positive_probability - probability_margin
        and action.consistency_non_degrading_probability
        >= criteria.minimum_consistency_non_degrading_probability
        - probability_margin
        and action.mean_position_rmse_reduction
        >= criteria.minimum_expected_gain - gain_margin
        and action.tenth_percentile_position_rmse_reduction
        >= criteria.minimum_tenth_percentile_gain - gain_margin
        and action.lower_tail_mean_position_rmse_reduction
        >= criteria.minimum_lower_tail_mean_gain - gain_margin
        and action.mean_position_rmse_reduction_confidence_interval[0]
        >= criteria.minimum_mean_confidence_lower_bound - gain_margin
    )


def _unique_seeds(values, label):
    seeds = tuple(int(value) for value in values)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"{label} future seeds must be nonempty and unique.")
    return seeds


def _run_results(
    prefix_seed, decision_epoch, horizon_epochs, future_seeds,
    action_active_edges, options,
):
    run_options = dict(options)
    run_options.pop("severe_relative_loss_threshold", None)
    return tuple(
        run_short_horizon_topology_counterfactual(
            seed=prefix_seed,
            future_seed=future_seed,
            decision_epoch=decision_epoch,
            horizon_epochs=horizon_epochs,
            action_active_edges=action_active_edges,
            **run_options,
        )
        for future_seed in future_seeds
    )
