from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cooperative.topology_policy import GraphObservation, UndirectedEdge
from experiments.short_horizon_topology_counterfactual import (
    run_short_horizon_topology_counterfactual,
)


@dataclass(frozen=True)
class MonteCarloActionTarget:
    action_kind: str
    active_edges: tuple[UndirectedEdge, ...]
    added_edges: tuple[UndirectedEdge, ...]
    removed_edges: tuple[UndirectedEdge, ...]
    future_noise_seeds: tuple[int, ...]
    mean_position_rmse_reduction: float
    position_rmse_reduction_standard_deviation: float
    mean_position_rmse_reduction_confidence_interval: tuple[float, float]
    tenth_percentile_position_rmse_reduction: float
    lower_tail_mean_position_rmse_reduction: float
    mean_relative_position_rmse_reduction: float
    relative_position_rmse_reduction_standard_deviation: float
    mean_relative_position_rmse_reduction_confidence_interval: tuple[
        float, float
    ]
    tenth_percentile_relative_position_rmse_reduction: float
    lower_tail_mean_relative_position_rmse_reduction: float
    positive_gain_probability: float
    consistency_non_degrading_probability: float
    safe_positive_gain_probability: float
    safe_positive_gain_probability_confidence_interval: tuple[float, float]
    severe_relative_loss_threshold: float
    severe_relative_loss_probability: float
    severe_relative_loss_probability_confidence_interval: tuple[float, float]
    mean_worst_node_rmse_reduction: float
    mean_position_rmse_reduction_by_node: tuple[tuple[str, float], ...]
    mean_relative_position_rmse_reduction_by_node: tuple[
        tuple[str, float], ...
    ]
    mean_covariance_trace_reduction: float
    mean_covariance_trace_reduction_by_node: tuple[tuple[str, float], ...]
    mean_nees_by_node: tuple[tuple[str, float], ...]
    mean_nees_calibration_improvement_by_node: tuple[
        tuple[str, float], ...
    ]
    mean_nis_by_node_and_modality: tuple[
        tuple[str, str, float, float, float, float], ...
    ]
    mean_transmitted_message_cost: float
    mean_replay_cost: float
    mean_topology_change_cost: float


@dataclass(frozen=True)
class MonteCarloGraphActionGroup:
    scenario_id: str
    prefix_seed: int
    decision_epoch: int
    horizon_epochs: int
    decision_observation: GraphObservation
    actions: tuple[MonteCarloActionTarget, ...]


@dataclass(frozen=True)
class MonteCarloGraphActionDataset:
    scenario_ids: tuple[str, ...]
    prefix_seeds: tuple[int, ...]
    future_noise_seeds: tuple[int, ...]
    relative_modalities: tuple[str, ...]
    groups: tuple[MonteCarloGraphActionGroup, ...]


@dataclass(frozen=True)
class MonteCarloScenarioSplit:
    training: MonteCarloGraphActionDataset
    validation: MonteCarloGraphActionDataset
    test: MonteCarloGraphActionDataset


def build_monte_carlo_graph_action_dataset(
    *,
    scenario_id: str,
    prefix_seeds: Iterable[int],
    future_noise_seeds: Iterable[int],
    node_count: int = 3,
    decision_epochs: tuple[int, ...] = (1, 3, 5),
    horizon_epochs: tuple[int, ...] = (1, 3),
    dt: float = 2.0,
    relative_modalities: tuple[str, ...] = (
        "RANGE", "RANGE_RATE", "AZ_EL",
    ),
    future_relative_update_order: tuple[str, ...] | None = None,
    truth_initial_state_by_node: dict[str, np.ndarray] | None = None,
    severe_relative_loss_threshold: float = 0.05,
    inactive_edges_after_decision: tuple[UndirectedEdge, ...] = (),
    absolute_navigation_dropout_nodes_after_decision: tuple[str, ...] = (),
    disturbance_start_epoch: int | None = None,
) -> MonteCarloGraphActionDataset:
    """Build conditional action targets from fixed-prefix future rollouts."""

    if not scenario_id:
        raise ValueError("scenario_id cannot be empty.")
    if severe_relative_loss_threshold < 0.0:
        raise ValueError("severe_relative_loss_threshold cannot be negative.")
    prefixes = _unique_nonempty(prefix_seeds, "prefix_seeds")
    futures = _unique_nonempty(future_noise_seeds, "future_noise_seeds")
    groups = []
    for prefix_seed in prefixes:
        for decision_epoch in decision_epochs:
            for horizon in horizon_epochs:
                results = tuple(
                    run_short_horizon_topology_counterfactual(
                        node_count=node_count,
                        seed=prefix_seed,
                        future_seed=future_seed,
                        decision_epoch=decision_epoch,
                        horizon_epochs=horizon,
                        dt=dt,
                        relative_modalities=relative_modalities,
                        future_relative_update_order=(
                            future_relative_update_order
                        ),
                        truth_initial_state_by_node=truth_initial_state_by_node,
                        inactive_edges_after_decision=(
                            inactive_edges_after_decision
                        ),
                        absolute_navigation_dropout_nodes_after_decision=(
                            absolute_navigation_dropout_nodes_after_decision
                        ),
                        disturbance_start_epoch=disturbance_start_epoch,
                    )
                    for future_seed in futures
                )
                observation = results[0].decision_observation
                if any(
                    result.decision_observation != observation
                    for result in results[1:]
                ):
                    raise RuntimeError(
                        "Future-noise rollouts changed the decision prefix."
                    )
                groups.append(MonteCarloGraphActionGroup(
                    scenario_id=scenario_id,
                    prefix_seed=prefix_seed,
                    decision_epoch=decision_epoch,
                    horizon_epochs=horizon,
                    decision_observation=observation,
                    actions=_aggregate_actions(
                        results, futures, severe_relative_loss_threshold
                    ),
                ))
    return MonteCarloGraphActionDataset(
        scenario_ids=(scenario_id,),
        prefix_seeds=prefixes,
        future_noise_seeds=futures,
        relative_modalities=tuple(relative_modalities),
        groups=tuple(groups),
    )


def combine_monte_carlo_graph_action_datasets(
    *datasets: MonteCarloGraphActionDataset,
) -> MonteCarloGraphActionDataset:
    if not datasets:
        raise ValueError("At least one Monte Carlo dataset is required.")
    scenario_ids = tuple(
        scenario_id
        for dataset in datasets for scenario_id in dataset.scenario_ids
    )
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Combined Monte Carlo scenario IDs must be unique.")
    if len({dataset.relative_modalities for dataset in datasets}) != 1:
        raise ValueError("Combined datasets must share measurement modalities.")
    return MonteCarloGraphActionDataset(
        scenario_ids=scenario_ids,
        prefix_seeds=tuple(sorted(set(
            seed for dataset in datasets for seed in dataset.prefix_seeds
        ))),
        future_noise_seeds=tuple(sorted(set(
            seed for dataset in datasets for seed in dataset.future_noise_seeds
        ))),
        relative_modalities=datasets[0].relative_modalities,
        groups=tuple(
            group for dataset in datasets for group in dataset.groups
        ),
    )


def split_monte_carlo_dataset_by_scenario(
    dataset: MonteCarloGraphActionDataset,
    *,
    training_scenarios: tuple[str, ...],
    validation_scenarios: tuple[str, ...],
    test_scenarios: tuple[str, ...],
) -> MonteCarloScenarioSplit:
    partitions = tuple(map(set, (
        training_scenarios, validation_scenarios, test_scenarios,
    )))
    if any(not values for values in partitions) or any(
        partitions[left] & partitions[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError(
            "Training, validation, and test scenarios must be disjoint."
        )
    if set().union(*partitions) - set(dataset.scenario_ids):
        raise ValueError("Requested scenario IDs are absent from the dataset.")
    return MonteCarloScenarioSplit(*(
        _scenario_subset(dataset, scenario_ids)
        for scenario_ids in partitions
    ))


def _scenario_subset(dataset, scenario_ids):
    groups = tuple(
        group for group in dataset.groups
        if group.scenario_id in scenario_ids
    )
    return MonteCarloGraphActionDataset(
        scenario_ids=tuple(sorted(scenario_ids)),
        prefix_seeds=tuple(sorted({
            group.prefix_seed for group in groups
        })),
        future_noise_seeds=dataset.future_noise_seeds,
        relative_modalities=dataset.relative_modalities,
        groups=groups,
    )


def _aggregate_actions(results, future_seeds, severe_relative_loss_threshold):
    reference_actions = tuple(
        rollout.action for rollout in results[0].rollouts
    )
    if any(
        tuple(rollout.action for rollout in result.rollouts)
        != reference_actions
        for result in results[1:]
    ):
        raise RuntimeError("Future rollouts must share one action matrix.")
    values = []
    for action_index, action in enumerate(reference_actions):
        gains, relative_gains, worst_gains, consistency = [], [], [], []
        node_gains = {}
        node_relative_gains = {}
        covariance_reductions = []
        node_covariance_reductions = {}
        node_nees = {}
        node_nees_improvements = {}
        nis_totals = {}
        message_costs, replay_costs, topology_costs = [], [], []
        for result in results:
            keep = result.rollouts[0].metrics
            current = result.rollouts[action_index].metrics
            gains.append(keep.position_rmse - current.position_rmse)
            relative_gains.append(
                (keep.position_rmse - current.position_rmse)
                / max(keep.position_rmse, 1e-12)
            )
            worst_gains.append(
                keep.worst_node_position_rmse
                - current.worst_node_position_rmse
            )
            keep_by_node = dict(keep.position_rmse_by_node)
            current_by_node = dict(current.position_rmse_by_node)
            for node in keep_by_node:
                gain = keep_by_node[node] - current_by_node[node]
                node_gains.setdefault(node, []).append(gain)
                node_relative_gains.setdefault(node, []).append(
                    gain / max(keep_by_node[node], 1e-12)
                )
            covariance_reductions.append(
                keep.mean_covariance_trace - current.mean_covariance_trace
            )
            keep_covariance_by_node = dict(
                keep.mean_covariance_trace_by_node
            )
            current_covariance_by_node = dict(
                current.mean_covariance_trace_by_node
            )
            for node in keep_covariance_by_node:
                node_covariance_reductions.setdefault(node, []).append(
                    keep_covariance_by_node[node]
                    - current_covariance_by_node[node]
                )
            keep_nees_by_node = dict(keep.mean_nees_by_node)
            current_nees_by_node = dict(current.mean_nees_by_node)
            for node in keep_nees_by_node:
                node_nees.setdefault(node, []).append(
                    current_nees_by_node[node]
                )
                node_nees_improvements.setdefault(node, []).append(
                    abs(keep_nees_by_node[node] - 6.0)
                    - abs(current_nees_by_node[node] - 6.0)
                )
            for summary in current.nis_by_node_and_modality:
                key = (summary.node_id, summary.modality)
                totals = nis_totals.setdefault(
                    key, [0.0, 0.0, 0.0, 0]
                )
                totals[0] += summary.mean_nis * summary.sample_count
                totals[1] += summary.coverage_95 * summary.sample_count
                totals[2] += (
                    summary.upper_violation_rate * summary.sample_count
                )
                totals[3] += summary.sample_count
            consistency.append(
                abs(current.mean_nees - 6.0) <= abs(keep.mean_nees - 6.0)
                and abs(current.nees_95_coverage - 0.95)
                <= abs(keep.nees_95_coverage - 0.95)
            )
            message_costs.append(
                current.transmitted_message_count
                - keep.transmitted_message_count
            )
            replay_costs.append(current.replay_count - keep.replay_count)
            topology_costs.append(
                current.topology_change_count - keep.topology_change_count
            )
        gains = np.asarray(gains, dtype=float)
        relative_gains = np.asarray(relative_gains, dtype=float)
        consistency = np.asarray(consistency, dtype=bool)
        tail_threshold = float(np.quantile(gains, 0.1))
        tail = gains[gains <= tail_threshold]
        relative_tail_threshold = float(np.quantile(relative_gains, 0.1))
        relative_tail = relative_gains[
            relative_gains <= relative_tail_threshold
        ]
        safe_positive = (gains > 0.0) & consistency
        severe_loss = relative_gains < -severe_relative_loss_threshold
        values.append(MonteCarloActionTarget(
            action_kind=action.kind,
            active_edges=action.topology.active_edges,
            added_edges=action.added_edges,
            removed_edges=action.removed_edges,
            future_noise_seeds=future_seeds,
            mean_position_rmse_reduction=float(np.mean(gains)),
            position_rmse_reduction_standard_deviation=float(np.std(gains)),
            mean_position_rmse_reduction_confidence_interval=(
                _mean_confidence_interval(gains)
            ),
            tenth_percentile_position_rmse_reduction=tail_threshold,
            lower_tail_mean_position_rmse_reduction=float(np.mean(tail)),
            mean_relative_position_rmse_reduction=float(
                np.mean(relative_gains)
            ),
            relative_position_rmse_reduction_standard_deviation=float(
                np.std(relative_gains)
            ),
            mean_relative_position_rmse_reduction_confidence_interval=(
                _mean_confidence_interval(relative_gains)
            ),
            tenth_percentile_relative_position_rmse_reduction=(
                relative_tail_threshold
            ),
            lower_tail_mean_relative_position_rmse_reduction=float(
                np.mean(relative_tail)
            ),
            positive_gain_probability=float(np.mean(gains > 0.0)),
            consistency_non_degrading_probability=float(np.mean(consistency)),
            safe_positive_gain_probability=float(np.mean(safe_positive)),
            safe_positive_gain_probability_confidence_interval=(
                _wilson_interval(int(np.sum(safe_positive)), len(gains))
            ),
            severe_relative_loss_threshold=severe_relative_loss_threshold,
            severe_relative_loss_probability=float(np.mean(severe_loss)),
            severe_relative_loss_probability_confidence_interval=(
                _wilson_interval(int(np.sum(severe_loss)), len(gains))
            ),
            mean_worst_node_rmse_reduction=float(np.mean(worst_gains)),
            mean_position_rmse_reduction_by_node=tuple(
                (node, float(np.mean(values)))
                for node, values in sorted(node_gains.items())
            ),
            mean_relative_position_rmse_reduction_by_node=tuple(
                (node, float(np.mean(values)))
                for node, values in sorted(node_relative_gains.items())
            ),
            mean_covariance_trace_reduction=float(
                np.mean(covariance_reductions)
            ),
            mean_covariance_trace_reduction_by_node=tuple(
                (node, float(np.mean(values)))
                for node, values in sorted(
                    node_covariance_reductions.items()
                )
            ),
            mean_nees_by_node=tuple(
                (node, float(np.mean(values)))
                for node, values in sorted(node_nees.items())
            ),
            mean_nees_calibration_improvement_by_node=tuple(
                (node, float(np.mean(values)))
                for node, values in sorted(node_nees_improvements.items())
            ),
            mean_nis_by_node_and_modality=tuple(
                (
                    node, modality,
                    float(totals[0] / totals[3]),
                    float(totals[1] / totals[3]),
                    float(totals[2] / totals[3]),
                    float(totals[3] / len(results)),
                )
                for (node, modality), totals in sorted(nis_totals.items())
            ),
            mean_transmitted_message_cost=float(np.mean(message_costs)),
            mean_replay_cost=float(np.mean(replay_costs)),
            mean_topology_change_cost=float(np.mean(topology_costs)),
        ))
    return tuple(values)


def _mean_confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean - half_width, mean + half_width


def _wilson_interval(successes, count):
    z = 1.96
    probability = successes / count
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    half_width = (
        z / denominator
        * np.sqrt(
            probability * (1.0 - probability) / count
            + z * z / (4.0 * count * count)
        )
    )
    return float(center - half_width), float(center + half_width)


def _unique_nonempty(values, name):
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique.")
    return result
