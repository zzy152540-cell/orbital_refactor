from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import csv

import numpy as np
import torch

from experiments.training.graph_action_gnn import torch_snapshot_action_group
from experiments.topology_control_baselines import (
    AlwaysKeepPolicy,
    run_topology_control_baseline_episode,
)
from experiments.training.topology_ppo import (
    TopologyActorCritic,
    collect_topology_rollout,
    prepare_topology_rollout,
)
from experiments.training.topology_ppo_stage1 import (
    Stage1PenaltyWeights,
    build_stage1_environment,
)
from experiments.topology_snapshot_counterfactual import build_online_snapshot_action_tensor
from experiments.variable_scale_counterfactual_prescan import _bounded_action_ids
from experiments.variable_scale_topology_curriculum import VariableScaleTopologyCurriculum
from experiments.training.variable_scale_topology_ppo import (
    apply_variable_scale_penalties,
)


ACTION_KINDS = ("keep", "add", "swap", "remove")


def load_variable_scale_model(
    checkpoint_path: str | Path,
    *, branch: str = "warm_start",
    condition_seed: int = 1500,
) -> tuple[TopologyActorCritic, dict[str, object]]:
    """Load one Actor-Critic branch from a variable-scale PPO result."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    configuration = checkpoint["configuration"]
    state_key = {
        "warm_start": "warm_model_state_dict",
        "random_init": "random_model_state_dict",
    }.get(branch, f"{branch}_model_state_dict")
    if state_key not in checkpoint:
        if branch == "warm_start" and "model_state_dict" in checkpoint:
            state_key = "model_state_dict"
        else:
            raise ValueError(f"Checkpoint has no model branch {branch!r}.")
    curriculum = VariableScaleTopologyCurriculum()
    environment = build_stage1_environment(
        curriculum.configuration_for_condition(condition_seed)
    )
    state = environment.reset(seed=0, condition_seed=condition_seed)
    snapshot, _ = build_online_snapshot_action_tensor(state)
    group = torch_snapshot_action_group(snapshot)
    model = TopologyActorCritic(
        node_feature_count=group.node_features.shape[1],
        candidate_edge_feature_count=group.candidate_edge_features.shape[1],
        measurement_feature_count=group.measurement_features.shape[1],
        action_feature_count=group.action_features.shape[1],
        global_feature_count=len(state.policy_tensor.global_feature_names),
        hidden_size=32,
        message_passing_steps=2,
        explicit_action_pairing=bool(configuration.get("explicit_action_pairing", True)),
        critic_timestamp_horizon=configuration.get("critic_timestamp_horizon"),
        critic_scale_calibration_node_counts=tuple(
            configuration.get("critic_scale_calibration_node_counts", ())
        ),
    )
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return model, configuration


def run_variable_scale_policy_diagnostics(
    model: TopologyActorCritic,
    *,
    condition_seeds: tuple[int, ...],
    noise_seeds: tuple[int, ...] = (0,),
    decision_indices: tuple[int, ...] = (0, 3, 6, 9),
    horizon_decisions: int = 2,
    maximum_actions_per_kind: int = 2,
    trajectories: tuple[str, ...] = ("keep", "policy"),
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    penalty_weights: Stage1PenaltyWeights = Stage1PenaltyWeights(),
    curriculum: VariableScaleTopologyCurriculum | None = None,
) -> dict[str, object]:
    """Jointly audit baseline scale, reward terms, policy mass, and advantages."""

    if not condition_seeds or not noise_seeds:
        raise ValueError("Diagnostic conditions and noise seeds must be nonempty.")
    curriculum = curriculum or VariableScaleTopologyCurriculum()
    requested = set(int(value) for value in decision_indices)
    baselines, states, selected_advantages = [], [], []
    for condition_seed in condition_seeds:
        configuration = curriculum.configuration_for_condition(condition_seed)
        for noise_seed in noise_seeds:
            keep = run_topology_control_baseline_episode(
                build_stage1_environment(configuration), AlwaysKeepPolicy(),
                seed=noise_seed, condition_seed=condition_seed,
            )
            baselines.append({
                "condition_seed": int(condition_seed),
                "noise_seed": int(noise_seed),
                "node_count": int(configuration.node_count),
                "keep_final_position_rmse": float(keep.final_position_rmse),
                "keep_final_worst_node_position_rmse": float(
                    keep.final_worst_node_position_rmse
                ),
                "keep_cumulative_absolute_reward": float(keep.cumulative_reward),
            })
            for trajectory in trajectories:
                if trajectory not in {"keep", "policy"}:
                    raise ValueError("Diagnostic trajectory must be keep or policy.")
                states.extend(_audit_trajectory(
                    model, configuration, condition_seed=condition_seed,
                    noise_seed=noise_seed, requested_decisions=requested,
                    horizon_decisions=horizon_decisions,
                    maximum_actions_per_kind=maximum_actions_per_kind,
                    penalty_weights=penalty_weights,
                    trajectory=trajectory,
                ))
            selected_advantages.extend(_selected_policy_advantages(
                model, configuration, condition_seed=condition_seed,
                noise_seed=noise_seed, gamma=gamma, gae_lambda=gae_lambda,
                penalty_weights=penalty_weights,
            ))
    return {
        "role": "variable_scale_policy_imbalance_diagnostic",
        "condition_seeds": list(condition_seeds),
        "noise_seeds": list(noise_seeds),
        "decision_indices": sorted(requested),
        "horizon_decisions": int(horizon_decisions),
        "maximum_actions_per_kind": int(maximum_actions_per_kind),
        "trajectories": list(trajectories),
        "penalty_weights": asdict(penalty_weights),
        "baseline_records": baselines,
        "state_records": states,
        "selected_policy_advantages": selected_advantages,
        "baseline_by_node_count": _baseline_summary(baselines),
        "action_by_node_count": _action_summary(states),
        "action_by_node_count_and_trajectory": _action_summary(
            states, split_trajectory=True
        ),
        "availability_by_node_count_and_trajectory": _availability_summary(states),
        "selected_advantage_by_node_count": _advantage_summary(selected_advantages),
    }


def _audit_trajectory(
    model, configuration, *, condition_seed, noise_seed, requested_decisions,
    horizon_decisions, maximum_actions_per_kind, penalty_weights, trajectory,
):
    environment = build_stage1_environment(configuration)
    state = environment.reset(seed=noise_seed, condition_seed=condition_seed)
    records = []
    decision_index = 0
    while True:
        if decision_index in requested_decisions:
            snapshot, action_ids = build_online_snapshot_action_tensor(state)
            group = torch_snapshot_action_group(snapshot)
            with torch.no_grad():
                output = model(group)
                actor = model.actor(group)
            distribution = output.distribution
            probabilities = distribution.action_log_probabilities.exp().cpu().numpy()
            type_probabilities = distribution.type_probabilities.cpu().numpy()
            action_index_by_id = {
                int(action_id): index for index, action_id in enumerate(action_ids)
            }
            selected_ids = _bounded_action_ids(
                state.action_space, maximum_actions_per_kind
            )
            outcomes = []
            for action_id in selected_ids:
                trial = deepcopy(environment)
                cumulative_reward = 0.0
                costs = np.zeros(6, dtype=float)
                for horizon_index in range(horizon_decisions):
                    step = trial.step(action_id if horizon_index == 0 else 0)
                    cumulative_reward += float(step.reward)
                    current = step.constraint_costs
                    costs += np.asarray((
                        current.transmitted_messages, current.dropped_messages,
                        current.replay_count, current.resynchronization_count,
                        current.topology_switch, current.action_fallback,
                    ), dtype=float)
                    if step.terminated or step.truncated:
                        break
                action = state.action_space.actions[action_id]
                action_index = action_index_by_id[action_id]
                outcomes.append({
                    "action_id": int(action_id),
                    "action_kind": action.kind,
                    "action_probability": float(probabilities[action_index]),
                    "cumulative_absolute_reward": float(cumulative_reward),
                    "transmitted_messages": float(costs[0]),
                    "resynchronizations": float(costs[3]),
                    "topology_switches": float(costs[4]),
                })
            keep_reward = next(
                item["cumulative_absolute_reward"] for item in outcomes
                if item["action_kind"] == "keep"
            )
            for item in outcomes:
                task_gain = item["cumulative_absolute_reward"] - keep_reward
                communication_penalty = (
                    penalty_weights.communication * item["transmitted_messages"]
                    / (configuration.node_count * configuration.decision_interval_epochs)
                )
                resynchronization_penalty = (
                    penalty_weights.resynchronization * item["resynchronizations"]
                    / configuration.node_count
                )
                switch_penalty = (
                    penalty_weights.topology_switch * item["topology_switches"]
                )
                item.update({
                    "counterfactual_task_gain": float(task_gain),
                    "communication_penalty": float(communication_penalty),
                    "resynchronization_penalty": float(resynchronization_penalty),
                    "topology_switch_penalty": float(switch_penalty),
                    "training_objective": float(
                        task_gain - communication_penalty
                        - resynchronization_penalty - switch_penalty
                    ),
                })
            keep_objective = next(
                item["training_objective"] for item in outcomes
                if item["action_kind"] == "keep"
            )
            for item in outcomes:
                item["objective_gain_over_keep"] = float(
                    item["training_objective"] - keep_objective
                )
            legal_counts = Counter(
                action.kind for action, legal in zip(
                    state.action_space.actions, state.action_space.legal_mask
                ) if legal
            )
            type_order = np.argsort(type_probabilities)[::-1]
            type_margin = (
                float(type_probabilities[type_order[0]] - type_probabilities[type_order[1]])
                if len(type_order) > 1 else float("inf")
            )
            selected_kind_index = int(type_order[0])
            selected_kind_mask = (
                group.action_kind_index.cpu().numpy() == selected_kind_index
            )
            selected_kind_utilities = np.sort(
                actor.utility.detach().cpu().numpy()[selected_kind_mask]
            )[::-1]
            conditional_margin = (
                float(selected_kind_utilities[0] - selected_kind_utilities[1])
                if len(selected_kind_utilities) > 1 else float("inf")
            )
            objective_order = sorted(
                outcomes, key=lambda item: item["objective_gain_over_keep"],
                reverse=True,
            )
            oracle_kind = objective_order[0]["action_kind"]
            oracle_margin = (
                float(objective_order[0]["objective_gain_over_keep"]
                      - objective_order[1]["objective_gain_over_keep"])
                if len(objective_order) > 1 else float("inf")
            )
            records.append({
                "condition_seed": int(condition_seed),
                "noise_seed": int(noise_seed),
                "node_count": int(configuration.node_count),
                "decision_index": int(decision_index),
                "trajectory": trajectory,
                "critic_value": float(output.value.item()),
                "type_probabilities": {
                    kind: float(type_probabilities[index])
                    for index, kind in enumerate(ACTION_KINDS)
                },
                "type_probability_margin": type_margin,
                "selected_type_utility_margin": conditional_margin,
                "audited_best_action_kind": oracle_kind,
                "audited_best_objective_margin": oracle_margin,
                "audited_best_kind_matches": (
                    ACTION_KINDS[selected_kind_index] == oracle_kind
                ),
                "deterministic_action_kind": ACTION_KINDS[int(
                    group.action_kind_index[int(distribution.mode().item())].item()
                )],
                "legal_action_kind_counts": dict(sorted(legal_counts.items())),
                "outcomes": outcomes,
            })
        if trajectory == "keep":
            trajectory_action_id = 0
        else:
            snapshot, action_ids = build_online_snapshot_action_tensor(state)
            with torch.no_grad():
                selected_index = int(model(
                    torch_snapshot_action_group(snapshot)
                ).distribution.mode().item())
            trajectory_action_id = int(action_ids[selected_index])
        step = environment.step(trajectory_action_id)
        state = step.state
        decision_index += 1
        if step.terminated or step.truncated:
            break
    return records


def _selected_policy_advantages(
    model, configuration, *, condition_seed, noise_seed, gamma, gae_lambda,
    penalty_weights,
):
    rollout = collect_topology_rollout(
        build_stage1_environment(configuration), model, seed=noise_seed,
        condition_seed=condition_seed, deterministic=True,
        counterfactual_keep_reward=True,
    )
    penalized = apply_variable_scale_penalties(
        rollout, node_count=configuration.node_count,
        decision_interval_epochs=configuration.decision_interval_epochs,
        weights=penalty_weights,
    )
    prepared = prepare_topology_rollout(
        penalized, gamma=gamma, gae_lambda=gae_lambda,
        normalize_advantages=False,
    )
    return [{
        "condition_seed": int(condition_seed),
        "noise_seed": int(noise_seed),
        "node_count": int(configuration.node_count),
        "decision_index": int(index),
        "action_kind": ACTION_KINDS[int(
            transition.group.action_kind_index[transition.action_index].item()
        )],
        "reward": float(penalized.transitions[index].reward),
        "advantage": float(prepared.advantages[index].item()),
        "critic_value": float(transition.value),
    } for index, transition in enumerate(rollout.transitions)]


def _baseline_summary(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["node_count"]].append(record)
    return {str(node_count): {
        "episode_count": len(values),
        "mean_keep_final_position_rmse": float(np.mean([
            item["keep_final_position_rmse"] for item in values
        ])),
        "std_keep_final_position_rmse": float(np.std([
            item["keep_final_position_rmse"] for item in values
        ])),
        "mean_keep_final_worst_node_position_rmse": float(np.mean([
            item["keep_final_worst_node_position_rmse"] for item in values
        ])),
    } for node_count, values in sorted(grouped.items())}


def _action_summary(records, *, split_trajectory=False):
    grouped = defaultdict(list)
    for record in records:
        for outcome in record["outcomes"]:
            key = (
                record["node_count"],
                record["trajectory"] if split_trajectory else None,
                outcome["action_kind"],
            )
            grouped[key].append(
                (record, outcome)
            )
    result = {}
    for (node_count, trajectory, kind), values in sorted(grouped.items()):
        outcomes = [item[1] for item in values]
        node_result = result.setdefault(str(node_count), {})
        target = node_result.setdefault(trajectory, {}) if split_trajectory else node_result
        target[kind] = {
            "evaluated_count": len(outcomes),
            "mean_action_probability": float(np.mean([
                item["action_probability"] for item in outcomes
            ])),
            "mean_counterfactual_task_gain": float(np.mean([
                item["counterfactual_task_gain"] for item in outcomes
            ])),
            "positive_task_gain_fraction": float(np.mean([
                item["counterfactual_task_gain"] > 0.0 for item in outcomes
            ])),
            "mean_communication_penalty": float(np.mean([
                item["communication_penalty"] for item in outcomes
            ])),
            "mean_resynchronization_penalty": float(np.mean([
                item["resynchronization_penalty"] for item in outcomes
            ])),
            "mean_topology_switch_penalty": float(np.mean([
                item["topology_switch_penalty"] for item in outcomes
            ])),
            "mean_training_objective": float(np.mean([
                item["training_objective"] for item in outcomes
            ])),
            "positive_training_objective_fraction": float(np.mean([
                item["training_objective"] > 0.0 for item in outcomes
            ])),
            "mean_objective_gain_over_keep": float(np.mean([
                item["objective_gain_over_keep"] for item in outcomes
            ])),
            "positive_objective_gain_over_keep_fraction": float(np.mean([
                item["objective_gain_over_keep"] > 0.0 for item in outcomes
            ])),
        }
    return result


def write_policy_diagnostic_csv(summary: dict[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "condition_seed", "noise_seed", "node_count", "trajectory",
        "decision_index", "action_id", "action_kind", "action_probability",
        "counterfactual_task_gain", "communication_penalty",
        "resynchronization_penalty", "topology_switch_penalty",
        "training_objective", "objective_gain_over_keep",
    )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in summary["state_records"]:
            common = {name: record[name] for name in fields[:5]}
            for outcome in record["outcomes"]:
                writer.writerow({
                    **common,
                    **{name: outcome[name] for name in fields[5:]},
                })
    return destination


def generate_policy_diagnostic_figure(
    summary: dict[str, object], path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    node_counts = ["5", "10", "20"]
    colors = {"keep": "#777777", "add": "#2b6cb0", "swap": "#d69e2e", "remove": "#c53030"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    baselines = summary["baseline_by_node_count"]
    axes[0].bar(node_counts, [
        baselines.get(node, {}).get("mean_keep_final_position_rmse", np.nan)
        for node in node_counts
    ], color="#4c78a8")
    axes[0].set(title="Always-keep baseline", xlabel="Node count", ylabel="Final position RMSE (m)")
    availability = summary["availability_by_node_count_and_trajectory"]
    x = np.arange(len(node_counts), dtype=float)
    width = 0.19
    for index, kind in enumerate(ACTION_KINDS):
        axes[1].bar(x + (index - 1.5) * width, [
            availability.get(node, {}).get("policy", {}).get(
                "mean_type_probability_by_kind", {}
            ).get(kind, 0.0) for node in node_counts
        ], width=width, label=kind, color=colors[kind])
    axes[1].set(title="Actor type probability", xlabel="Node count", ylabel="Probability", xticks=x, xticklabels=node_counts)
    actions = summary["action_by_node_count_and_trajectory"]
    for index, kind in enumerate(ACTION_KINDS):
        axes[2].bar(x + (index - 1.5) * width, [
            actions.get(node, {}).get("policy", {}).get(kind, {}).get(
                "mean_objective_gain_over_keep", np.nan
            ) for node in node_counts
        ], width=width, label=kind, color=colors[kind])
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set(title="Counterfactual objective advantage", xlabel="Node count", ylabel="Gain over keep", xticks=x, xticklabels=node_counts)
    axes[1].legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _availability_summary(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["node_count"], record["trajectory"])].append(record)
    result = {}
    for (node_count, trajectory), values in sorted(grouped.items()):
        target = result.setdefault(str(node_count), {})
        target[trajectory] = {
            "audited_state_count": len(values),
            "deterministic_action_kind_counts": dict(sorted(Counter(
                item["deterministic_action_kind"] for item in values
            ).items())),
            "legal_state_fraction_by_kind": {
                kind: float(np.mean([
                    item["legal_action_kind_counts"].get(kind, 0) > 0
                    for item in values
                ]))
                for kind in ACTION_KINDS
            },
            "mean_type_probability_by_kind": {
                kind: float(np.mean([
                    item["type_probabilities"][kind] for item in values
                ]))
                for kind in ACTION_KINDS
            },
        }
    return result


def _advantage_summary(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["node_count"], record["action_kind"])].append(record)
    result = {}
    for (node_count, kind), values in sorted(grouped.items()):
        result.setdefault(str(node_count), {})[kind] = {
            "selected_count": len(values),
            "mean_advantage": float(np.mean([item["advantage"] for item in values])),
            "positive_advantage_fraction": float(np.mean([
                item["advantage"] > 0.0 for item in values
            ])),
            "mean_penalized_reward": float(np.mean([
                item["reward"] for item in values
            ])),
        }
    return result
