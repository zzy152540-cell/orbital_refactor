from __future__ import annotations

import numpy as np

from experiments.training.topology_ppo import collect_topology_rollout
from experiments.training.topology_ppo_stage1 import (
    Stage1PenaltyWeights,
    build_stage1_environment,
)
from experiments.variable_scale_critic_audit import _discounted_returns
from experiments.training.variable_scale_topology_ppo import (
    apply_variable_scale_penalties,
)


def audit_variable_scale_value_features(
    model,
    curriculum,
    *,
    training_condition_seeds: tuple[int, ...],
    test_condition_seeds: tuple[int, ...],
    counterfactual_keep_reward: bool,
    return_scale_by_node_count: tuple[tuple[int, float], ...] = (),
    noise_seed: int = 0,
    gamma: float = 0.99,
    ridge_penalties: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
    penalty_weights: Stage1PenaltyWeights = Stage1PenaltyWeights(),
) -> dict[str, object]:
    """Test whether current pooled Critic inputs linearly predict held-out returns."""

    if set(training_condition_seeds) & set(test_condition_seeds):
        raise ValueError("Value-feature training and test conditions must be disjoint.")
    if not training_condition_seeds or not test_condition_seeds:
        raise ValueError("Value-feature audit requires training and test conditions.")
    if not ridge_penalties or any(value < 0.0 for value in ridge_penalties):
        raise ValueError("Ridge penalties must be nonempty and nonnegative.")
    scales = dict(return_scale_by_node_count)
    training = _collect_records(
        model, curriculum, training_condition_seeds,
        counterfactual_keep_reward=counterfactual_keep_reward,
        return_scales=scales, noise_seed=noise_seed, gamma=gamma,
        penalty_weights=penalty_weights,
    )
    test = _collect_records(
        model, curriculum, test_condition_seeds,
        counterfactual_keep_reward=counterfactual_keep_reward,
        return_scales=scales, noise_seed=noise_seed, gamma=gamma,
        penalty_weights=penalty_weights,
    )
    by_scale = {}
    for node_count in (5, 10, 20):
        train_rows = [item for item in training if item["node_count"] == node_count]
        test_rows = [item for item in test if item["node_count"] == node_count]
        by_scale[str(node_count)] = {
            "training_transition_count": len(train_rows),
            "test_transition_count": len(test_rows),
            "constant_baseline": _constant_baseline(train_rows, test_rows),
            "timestamp_ridge": _ridge_sweep(
                train_rows, test_rows, (0,), ridge_penalties,
            ),
            "global_state_ridge": _ridge_sweep(
                train_rows, test_rows,
                tuple(range(training[0]["global_feature_count"])),
                ridge_penalties,
            ),
            "full_state_ridge": _ridge_sweep(
                train_rows, test_rows, None, ridge_penalties,
            ),
        }
    return {
        "training_condition_seeds": list(training_condition_seeds),
        "test_condition_seeds": list(test_condition_seeds),
        "noise_seed": int(noise_seed),
        "gamma": float(gamma),
        "ridge_penalties": list(ridge_penalties),
        "counterfactual_keep_reward": bool(counterfactual_keep_reward),
        "feature_count": len(training[0]["features"]),
        "summary_by_node_count": by_scale,
    }


def _collect_records(
    model, curriculum, condition_seeds, *, counterfactual_keep_reward,
    return_scales, noise_seed, gamma, penalty_weights,
):
    records = []
    for condition_seed in condition_seeds:
        configuration = curriculum.configuration_for_condition(condition_seed)
        environment = build_stage1_environment(configuration)
        rollout = collect_topology_rollout(
            environment, model, seed=noise_seed,
            condition_seed=condition_seed, deterministic=True,
            counterfactual_keep_reward=counterfactual_keep_reward,
        )
        penalized = apply_variable_scale_penalties(
            rollout, node_count=configuration.node_count,
            decision_interval_epochs=configuration.decision_interval_epochs,
            weights=penalty_weights,
            return_scale=return_scales.get(configuration.node_count, 1.0),
        )
        returns = _discounted_returns(
            penalized.rewards.detach().cpu().numpy(), gamma,
            final_value=penalized.final_value,
        )
        for decision_index, (transition, target) in enumerate(zip(
            penalized.transitions, returns,
        )):
            records.append({
                "condition_seed": int(condition_seed),
                "node_count": int(configuration.node_count),
                "decision_index": int(decision_index),
                "target": float(target),
                "features": _critic_features(transition.group),
                "global_feature_count": int(
                    transition.group.action_features.shape[1] - 7
                ),
            })
    return records


def _critic_features(group):
    arrays = (
        group.node_features.detach().cpu().numpy(),
        group.candidate_edge_features.detach().cpu().numpy(),
    )
    node, edge = arrays
    global_count = group.action_features.shape[1] - 7
    global_features = (
        group.action_features[0, -global_count:].detach().cpu().numpy()
        if global_count else np.empty(0)
    )
    return np.concatenate((
        global_features,
        node.mean(axis=0), node.max(axis=0),
        edge.mean(axis=0), edge.max(axis=0),
    )).astype(float).tolist()


def _constant_baseline(training, test):
    train_targets = np.asarray([item["target"] for item in training])
    test_targets = np.asarray([item["target"] for item in test])
    prediction = np.full_like(test_targets, np.mean(train_targets))
    return _regression_summary(test_targets, prediction)


def _ridge_summary(training, test, feature_indices, penalty):
    train_x = np.asarray([item["features"] for item in training])
    test_x = np.asarray([item["features"] for item in test])
    if feature_indices is not None:
        train_x = train_x[:, feature_indices]
        test_x = test_x[:, feature_indices]
    train_y = np.asarray([item["target"] for item in training])
    test_y = np.asarray([item["target"] for item in test])
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    train_z, test_z = (train_x - mean) / scale, (test_x - mean) / scale
    centered_y = train_y - train_y.mean()
    gram = train_z.T @ train_z + penalty * np.eye(train_z.shape[1])
    weights = np.linalg.solve(gram, train_z.T @ centered_y)
    prediction = train_y.mean() + test_z @ weights
    return _regression_summary(test_y, prediction)


def _ridge_sweep(training, test, feature_indices, penalties):
    return {
        str(float(penalty)): _ridge_summary(
            training, test, feature_indices, penalty,
        )
        for penalty in penalties
    }


def _regression_summary(target, prediction):
    error = prediction - target
    variance = float(np.var(target))
    correlation = (
        0.0 if np.std(target) <= 1.0e-15 or np.std(prediction) <= 1.0e-15
        else float(np.corrcoef(target, prediction)[0, 1])
    )
    return {
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "explained_variance": (
            0.0 if variance <= 1.0e-15
            else 1.0 - float(np.var(error)) / variance
        ),
        "correlation": correlation,
    }
