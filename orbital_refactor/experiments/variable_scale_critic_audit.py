from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from experiments.training.variable_scale_topology_ppo import (
    ACTION_KINDS,
    apply_variable_scale_penalties,
)
from experiments.training.topology_ppo import (
    collect_topology_rollout,
    prepare_topology_rollout,
)
from experiments.training.topology_ppo_stage1 import (
    Stage1PenaltyWeights,
    build_stage1_environment,
)


def audit_variable_scale_critic(
    model,
    curriculum,
    *,
    condition_seeds: tuple[int, ...],
    noise_seed: int = 0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    penalty_weights: Stage1PenaltyWeights = Stage1PenaltyWeights(),
    counterfactual_keep_reward: bool = False,
    return_scale_by_node_count: tuple[tuple[int, float], ...] = (),
) -> dict[str, object]:
    """Compare one frozen Critic with MC returns and GAE by fleet scale."""

    records = []
    for condition_seed in condition_seeds:
        configuration = curriculum.configuration_for_condition(condition_seed)
        environment = build_stage1_environment(configuration)
        rollout = collect_topology_rollout(
            environment, model,
            seed=noise_seed,
            condition_seed=condition_seed,
            deterministic=True,
            counterfactual_keep_reward=counterfactual_keep_reward,
        )
        penalized = apply_variable_scale_penalties(
            rollout,
            node_count=configuration.node_count,
            decision_interval_epochs=configuration.decision_interval_epochs,
            weights=penalty_weights,
            return_scale=dict(return_scale_by_node_count).get(
                configuration.node_count, 1.0
            ),
        )
        prepared = prepare_topology_rollout(
            penalized,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=False,
        )
        monte_carlo_returns = _discounted_returns(
            penalized.rewards.detach().cpu().numpy(), gamma,
            final_value=penalized.final_value,
        )
        for decision_index, transition in enumerate(penalized.transitions):
            kind_index = int(
                transition.group.action_kind_index[
                    transition.action_index
                ].item()
            )
            records.append({
                "condition_seed": int(condition_seed),
                "node_count": int(configuration.node_count),
                "decision_index": int(decision_index),
                "action_kind": ACTION_KINDS[kind_index],
                "penalized_reward": float(transition.reward),
                "predicted_value": float(transition.value),
                "monte_carlo_return": float(
                    monte_carlo_returns[decision_index]
                ),
                "value_error": float(
                    transition.value - monte_carlo_returns[decision_index]
                ),
                "gae_advantage": float(prepared.advantages[decision_index]),
            })
    return {
        "condition_seeds": list(condition_seeds),
        "noise_seed": int(noise_seed),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "counterfactual_keep_reward": bool(counterfactual_keep_reward),
        "return_scale_by_node_count": [
            list(item) for item in return_scale_by_node_count
        ],
        "records": records,
        "summary_by_node_count": _group_summaries(records, "node_count"),
        "summary_by_action_kind": _group_summaries(records, "action_kind"),
        "overall": _value_summary(records),
    }


def _discounted_returns(rewards, gamma, *, final_value=0.0):
    returns = np.empty_like(rewards, dtype=float)
    accumulator = float(final_value)
    for index in range(len(rewards) - 1, -1, -1):
        accumulator = float(rewards[index]) + gamma * accumulator
        returns[index] = accumulator
    return returns


def _group_summaries(records, key):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    return {
        str(value): _value_summary(items)
        for value, items in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def _value_summary(records):
    targets = np.asarray([
        item["monte_carlo_return"] for item in records
    ], dtype=float)
    values = np.asarray([
        item["predicted_value"] for item in records
    ], dtype=float)
    errors = values - targets
    advantages = np.asarray([
        item["gae_advantage"] for item in records
    ], dtype=float)
    target_variance = float(np.var(targets))
    explained_variance = (
        0.0
        if target_variance <= 1.0e-15
        else 1.0 - float(np.var(errors)) / target_variance
    )
    correlation = (
        0.0
        if len(targets) < 2
        or np.std(targets) <= 1.0e-15
        or np.std(values) <= 1.0e-15
        else float(np.corrcoef(targets, values)[0, 1])
    )
    return {
        "transition_count": len(records),
        "action_kind_counts": dict(sorted(Counter(
            item["action_kind"] for item in records
        ).items())),
        "target_mean": float(np.mean(targets)),
        "target_standard_deviation": float(np.std(targets)),
        "predicted_value_mean": float(np.mean(values)),
        "predicted_value_standard_deviation": float(np.std(values)),
        "mean_value_error": float(np.mean(errors)),
        "value_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "explained_variance": explained_variance,
        "target_value_correlation": correlation,
        "gae_advantage_mean": float(np.mean(advantages)),
        "gae_advantage_standard_deviation": float(np.std(advantages)),
        "positive_gae_fraction": float(np.mean(advantages > 0.0)),
    }
