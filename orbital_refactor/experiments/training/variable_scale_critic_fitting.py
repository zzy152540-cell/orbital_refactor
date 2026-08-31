from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from experiments.training.topology_ppo import collect_topology_rollout
from experiments.training.topology_ppo_stage1 import (
    Stage1PenaltyWeights,
    build_stage1_environment,
)
from experiments.training.variable_scale_topology_ppo import (
    apply_variable_scale_penalties,
)
from experiments.variable_scale_critic_audit import _discounted_returns


@dataclass(frozen=True)
class CriticFittingConfiguration:
    learning_rate: float = 3.0e-4
    epochs: int = 200
    minibatch_size: int = 32
    weight_decay: float = 1.0e-3
    gamma: float = 0.99
    gradient_clip_norm: float = 1.0
    seed: int = 0


def fit_frozen_actor_critic(
    model,
    curriculum,
    *,
    training_condition_seeds: tuple[int, ...],
    validation_condition_seeds: tuple[int, ...],
    configuration: CriticFittingConfiguration = CriticFittingConfiguration(),
    counterfactual_keep_reward: bool = False,
    return_scale_by_node_count: tuple[tuple[int, float], ...] = (),
    penalty_weights: Stage1PenaltyWeights = Stage1PenaltyWeights(),
) -> dict[str, object]:
    """Fit only the Critic to fixed-Actor, finite-horizon MC returns.

    A zero terminal bootstrap keeps targets independent of the Critic being
    tested. Condition sets must be disjoint so validation remains meaningful.
    """

    _validate_inputs(training_condition_seeds, validation_condition_seeds,
                     configuration)
    scales = dict(return_scale_by_node_count)
    training = _collect_examples(
        model, curriculum, training_condition_seeds,
        counterfactual_keep_reward=counterfactual_keep_reward,
        return_scales=scales, gamma=configuration.gamma,
        penalty_weights=penalty_weights,
    )
    validation = _collect_examples(
        model, curriculum, validation_condition_seeds,
        counterfactual_keep_reward=counterfactual_keep_reward,
        return_scales=scales, gamma=configuration.gamma,
        penalty_weights=penalty_weights,
    )

    actor_before = {
        name: value.detach().clone() for name, value in model.actor.state_dict().items()
    }
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    critic_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith((
            "critic.", "critic_phase_projection.", "critic_scale_calibration.",
        )):
            parameter.requires_grad_(True)
            critic_parameters.append(parameter)
    if not critic_parameters:
        raise ValueError("Model exposes no trainable Critic parameters.")

    before = _evaluation_summary(model, training, validation)
    best_validation_explained_variance = float("-inf")
    best_epoch = 0
    best_critic_state = None
    optimizer = torch.optim.AdamW(
        critic_parameters, lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )
    generator = torch.Generator().manual_seed(configuration.seed)
    history = []
    for epoch in range(configuration.epochs):
        model.train()
        order = torch.randperm(len(training), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), configuration.minibatch_size):
            batch = [training[index] for index in order[
                start:start + configuration.minibatch_size
            ]]
            prediction = torch.stack([model(item[0]).value for item in batch])
            target = prediction.new_tensor([item[1] for item in batch])
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                critic_parameters, configuration.gradient_clip_norm,
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in {0, configuration.epochs - 1} or (epoch + 1) % 20 == 0:
            metrics = _evaluation_summary(model, training, validation)
            validation_ev = metrics["validation"]["overall"][
                "explained_variance"
            ]
            if validation_ev > best_validation_explained_variance:
                best_validation_explained_variance = validation_ev
                best_epoch = epoch + 1
                best_critic_state = {
                    name: parameter.detach().clone()
                    for name, parameter in model.state_dict().items()
                    if name.startswith((
                        "critic.", "critic_phase_projection.",
                        "critic_scale_calibration.",
                    ))
                }
            history.append({
                "epoch": epoch + 1,
                "minibatch_loss_mean": float(np.mean(losses)),
                "training": metrics["training"],
                "validation": metrics["validation"],
            })
    if best_critic_state is None:  # pragma: no cover - epochs are validated
        raise RuntimeError("Critic fitting produced no validation checkpoint.")
    model.load_state_dict(best_critic_state, strict=False)
    after = _evaluation_summary(model, training, validation)

    actor_unchanged = all(torch.equal(value, actor_before[name]) for name, value in (
        model.actor.state_dict().items()
    ))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])
    model.eval()
    return {
        "target_definition": "finite_horizon_discounted_mc_zero_terminal_bootstrap",
        "training_condition_seeds": list(training_condition_seeds),
        "validation_condition_seeds": list(validation_condition_seeds),
        "training_transition_count": len(training),
        "validation_transition_count": len(validation),
        "actor_unchanged": actor_unchanged,
        "selected_checkpoint_epoch": best_epoch,
        "selected_validation_explained_variance": (
            best_validation_explained_variance
        ),
        "before": before,
        "after": after,
        "history": history,
    }


def _collect_examples(
    model, curriculum, condition_seeds, *, counterfactual_keep_reward,
    return_scales, gamma, penalty_weights,
):
    examples = []
    model.eval()
    with torch.no_grad():
        for condition_seed in condition_seeds:
            configuration = curriculum.configuration_for_condition(condition_seed)
            rollout = collect_topology_rollout(
                build_stage1_environment(configuration), model, seed=0,
                condition_seed=condition_seed, deterministic=True,
                counterfactual_keep_reward=counterfactual_keep_reward,
            )
            penalized = apply_variable_scale_penalties(
                rollout, node_count=configuration.node_count,
                decision_interval_epochs=configuration.decision_interval_epochs,
                weights=penalty_weights,
                return_scale=return_scales.get(configuration.node_count, 1.0),
            )
            targets = _discounted_returns(
                penalized.rewards.detach().cpu().numpy(), gamma, final_value=0.0,
            )
            examples.extend(
                (transition.group, float(target), int(configuration.node_count))
                for transition, target in zip(penalized.transitions, targets)
            )
    return examples


def _evaluation_summary(model, training, validation):
    return {
        "training": _summaries(model, training),
        "validation": _summaries(model, validation),
    }


def _summaries(model, examples):
    grouped = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for group, target, node_count in examples:
            grouped[node_count].append((target, float(model(group).value)))
    return {
        "overall": _regression_summary([
            item for rows in grouped.values() for item in rows
        ]),
        "by_node_count": {
            str(node_count): _regression_summary(rows)
            for node_count, rows in sorted(grouped.items())
        },
    }


def _regression_summary(rows):
    target = np.asarray([row[0] for row in rows], dtype=float)
    prediction = np.asarray([row[1] for row in rows], dtype=float)
    error = prediction - target
    variance = float(np.var(target))
    correlation = 0.0
    if len(rows) > 1 and np.std(target) > 1.0e-15 and np.std(prediction) > 1.0e-15:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        "count": len(rows),
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "explained_variance": (
            0.0 if variance <= 1.0e-15
            else 1.0 - float(np.var(error)) / variance
        ),
        "correlation": correlation,
    }


def _validate_inputs(training, validation, configuration):
    if not training or not validation:
        raise ValueError("Critic fitting requires training and validation conditions.")
    if set(training) & set(validation):
        raise ValueError("Critic training and validation conditions must be disjoint.")
    if min(
        configuration.learning_rate, configuration.epochs,
        configuration.minibatch_size, configuration.gamma,
        configuration.gradient_clip_norm,
    ) <= 0.0:
        raise ValueError("Critic fitting controls must be positive.")
    if configuration.gamma > 1.0 or configuration.weight_decay < 0.0:
        raise ValueError("Critic gamma/weight decay is invalid.")
