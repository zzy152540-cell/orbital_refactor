from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from experiments.run_v15_variable_scale_ppo_pilot import _evaluate
from experiments.variable_scale_topology_ppo import (
    VariableScalePPOConfiguration,
    train_variable_scale_topology_ppo,
)


def _evaluation_summary(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["node_count"])].append(record)
    return {
        str(node_count): {
            "episode_count": len(values),
            "mean_rmse_improvement": float(np.mean([
                item["model_rmse_improvement"] for item in values
            ])),
            "worst_rmse_improvement": float(min(
                item["model_rmse_improvement"] for item in values
            )),
            "improved_episode_count": int(sum(
                item["model_rmse_improvement"] > 0.0 for item in values
            )),
            "action_kind_counts": dict(sorted(sum((
                Counter(item["action_kind_counts"]) for item in values
            ), Counter()).items())),
            "fallbacks": float(sum(item["fallbacks"] for item in values)),
        }
        for node_count, values in sorted(grouped.items())
    }


def _training_summary(result):
    grouped = defaultdict(list)
    for diagnostic in result.diagnostics:
        grouped[diagnostic.node_count].append(diagnostic)
    return {
        str(node_count): {
            "episode_count": len(values),
            "mean_task_return": float(np.mean([
                item.task_return for item in values
            ])),
            "mean_absolute_task_return": float(np.mean([
                item.absolute_task_return for item in values
            ])),
            "mean_penalized_return": float(np.mean([
                item.penalized_return for item in values
            ])),
            "mean_unnormalized_penalized_return": float(np.mean([
                item.unnormalized_penalized_return for item in values
            ])),
            "mean_topology_switches": float(np.mean([
                item.topology_switches for item in values
            ])),
            "action_kind_counts": dict(sorted(sum((
                Counter(dict(item.action_kind_counts)) for item in values
            ), Counter()).items())),
        }
        for node_count, values in sorted(grouped.items())
    }


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Compare warm and random PPO over one full scale cycle."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--training-condition-offset", type=int, default=440)
    parser.add_argument(
        "--evaluation-conditions", type=int, nargs="+",
        default=(460, 461, 462, 463, 464, 466),
    )
    arguments = parser.parse_args(argv)
    configuration = VariableScalePPOConfiguration(
        training_episodes=20,
        rollout_batch_episodes=20,
        training_condition_seed_offset=arguments.training_condition_offset,
        training_condition_seed_count=20,
        environment_seed_count=4,
        policy_seed=arguments.policy_seed,
        update_epochs=4,
        minibatch_size=32,
    )
    random_result = train_variable_scale_topology_ppo(configuration)
    warm_result = train_variable_scale_topology_ppo(
        configuration,
        warm_start_checkpoint=str(arguments.warm_start),
        reset_warm_start_type_head=False,
    )
    evaluation_conditions = tuple(arguments.evaluation_conditions)
    random_evaluation = _evaluate(
        random_result.model, configuration.curriculum,
        evaluation_conditions, noise_seed=0,
    )
    warm_evaluation = _evaluate(
        warm_result.model, configuration.curriculum,
        evaluation_conditions, noise_seed=0,
    )
    summary = {
        "configuration": asdict(configuration),
        "evaluation_conditions": list(evaluation_conditions),
        "random_init": {
            "training": _training_summary(random_result),
            "batch_diagnostics": [
                asdict(item) for item in random_result.batch_diagnostics
            ],
            "evaluation": random_evaluation,
            "evaluation_by_node_count": _evaluation_summary(random_evaluation),
        },
        "warm_start": {
            "training": _training_summary(warm_result),
            "batch_diagnostics": [
                asdict(item) for item in warm_result.batch_diagnostics
            ],
            "evaluation": warm_evaluation,
            "evaluation_by_node_count": _evaluation_summary(warm_evaluation),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = arguments.output.with_suffix(".pt")
    torch.save({
        "role": "variable_scale_topology_ppo_cycle_comparison",
        "configuration": asdict(configuration),
        "random_model_state_dict": random_result.model.state_dict(),
        "warm_model_state_dict": warm_result.model.state_dict(),
        "evaluation_conditions": evaluation_conditions,
    }, checkpoint)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
