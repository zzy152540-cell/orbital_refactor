from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from experiments.topology_control_baselines import (
    AlwaysKeepPolicy,
    run_topology_control_baseline_episode,
)
from experiments.topology_ppo import collect_topology_rollout
from experiments.topology_ppo_stage1 import (
    build_stage1_environment,
    five_node_stratified_physical_ppo_configuration,
    train_stage1_ppo,
)


def _summarize_training(result):
    diagnostics = result.diagnostics
    type_probabilities = np.asarray([
        item.initial_type_probabilities for item in diagnostics
    ])
    return {
        "episode_count": len(diagnostics),
        "mean_task_return": float(np.mean([
            item.task_return for item in diagnostics
        ])),
        "mean_penalized_return": float(np.mean([
            item.penalized_return for item in diagnostics
        ])),
        "mean_final_position_rmse": float(np.mean([
            item.final_position_rmse for item in diagnostics
        ])),
        "mean_positive_advantage_fraction": float(np.mean([
            item.update.positive_advantage_fraction for item in diagnostics
        ])),
        "mean_advantage": float(np.mean([
            item.update.advantage_mean for item in diagnostics
        ])),
        "mean_advantage_standard_deviation": float(np.mean([
            item.update.advantage_standard_deviation for item in diagnostics
        ])),
        "mean_explained_variance_before_update": float(np.mean([
            item.update.explained_variance_before_update for item in diagnostics
        ])),
        "mean_initial_type_probabilities": tuple(
            float(value) for value in type_probabilities.mean(axis=0)
        ),
        "topology_switches": float(sum(
            item.topology_switches for item in diagnostics
        )),
        "by_action_kind": _summarize_action_kinds(diagnostics),
        "batch_diagnostics": tuple(
            asdict(item) for item in result.batch_diagnostics
        ),
    }


def _summarize_action_kinds(diagnostics):
    grouped = {}
    for episode in diagnostics:
        for item in episode.action_kind_diagnostics:
            values = grouped.setdefault(item.action_kind, [])
            values.append(item)
    summary = {}
    for action_kind, values in sorted(grouped.items()):
        count = sum(item.transition_count for item in values)
        weighted = lambda field: float(sum(
            getattr(item, field) * item.transition_count for item in values
        ) / count)
        summary[action_kind] = {
            "transition_count": count,
            "actor_transition_count": sum(
                item.actor_transition_count for item in values
            ),
            "mean_advantage": weighted("mean_advantage"),
            "positive_advantage_fraction": weighted(
                "positive_advantage_fraction"
            ),
            "mean_task_reward": weighted("mean_task_reward"),
            "mean_penalized_reward": weighted("mean_penalized_reward"),
            "mean_transmitted_messages": weighted(
                "mean_transmitted_messages"
            ),
            "mean_resynchronization_count": weighted(
                "mean_resynchronization_count"
            ),
            "mean_topology_switch": weighted("mean_topology_switch"),
        }
    return summary


def _evaluate_model(configuration, model, *, condition_seeds, noise_seeds):
    keep_rmse, model_rmse = [], []
    actions = Counter()
    switches = resynchronizations = fallbacks = 0.0
    for condition_seed in condition_seeds:
        for noise_seed in noise_seeds:
            keep = run_topology_control_baseline_episode(
                build_stage1_environment(configuration), AlwaysKeepPolicy(),
                seed=noise_seed, condition_seed=condition_seed,
            )
            environment = build_stage1_environment(configuration)
            rollout = collect_topology_rollout(
                environment, model, seed=noise_seed,
                condition_seed=condition_seed, deterministic=True,
            )
            keep_rmse.append(keep.final_position_rmse)
            model_rmse.append(float(environment._metrics()[0]))
            actions.update(
                ("keep", "add", "swap", "remove")[int(
                    transition.group.action_kind_index[
                        transition.action_index
                    ].item()
                )]
                for transition in rollout.transitions
            )
            totals = rollout.cost_matrix.sum(dim=0)
            switches += float(totals[4])
            resynchronizations += float(totals[3])
            fallbacks += float(totals[5])
    improvements = np.asarray(keep_rmse) - np.asarray(model_rmse)
    return {
        "episode_count": len(model_rmse),
        "mean_keep_rmse": float(np.mean(keep_rmse)),
        "mean_model_rmse": float(np.mean(model_rmse)),
        "mean_rmse_improvement": float(np.mean(improvements)),
        "improved_episode_count": int(np.count_nonzero(improvements > 0.0)),
        "worst_rmse_improvement": float(np.min(improvements)),
        "action_kind_counts": dict(sorted(actions.items())),
        "topology_switches": switches,
        "resynchronizations": resynchronizations,
        "fallbacks": fallbacks,
    }


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run the frozen stratified-physical PPO warm-start pilot."
    )
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument(
        "--preserve-warm-start-type-head", action="store_true",
        help="Keep the supervised action-type head instead of resetting it.",
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true",
        help="Collect training diagnostics without repeating closed-loop evaluation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    configuration = five_node_stratified_physical_ppo_configuration(
        policy_seed=arguments.policy_seed,
    )
    random_result = train_stage1_ppo(configuration)
    warm_result = train_stage1_ppo(
        configuration,
        warm_start_checkpoint=str(arguments.warm_start),
        reset_warm_start_type_head=(
            not arguments.preserve_warm_start_type_head
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "configuration": asdict(configuration),
        "warm_start_checkpoint": str(arguments.warm_start),
        "preserve_warm_start_type_head": bool(
            arguments.preserve_warm_start_type_head
        ),
        "random_model_state_dict": random_result.model.state_dict(),
        "warm_model_state_dict": warm_result.model.state_dict(),
        "summary": None,
    }
    torch.save(checkpoint, arguments.output)
    condition_seeds = tuple(range(232, 240))
    noise_seeds = tuple(range(4))
    summary = {
        "policy_seed": arguments.policy_seed,
        "preserve_warm_start_type_head": bool(
            arguments.preserve_warm_start_type_head
        ),
        "condition_seeds": condition_seeds,
        "noise_seeds": noise_seeds,
        "random_training": _summarize_training(random_result),
        "warm_training": _summarize_training(warm_result),
        "random_init": None,
        "warm_start": None,
    }
    if not arguments.skip_evaluation:
        summary["random_init"] = _evaluate_model(
            configuration, random_result.model,
            condition_seeds=condition_seeds, noise_seeds=noise_seeds,
        )
        summary["warm_start"] = _evaluate_model(
            configuration, warm_result.model,
            condition_seeds=condition_seeds, noise_seeds=noise_seeds,
        )
    checkpoint["summary"] = summary
    torch.save(checkpoint, arguments.output)
    summary_path = arguments.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
