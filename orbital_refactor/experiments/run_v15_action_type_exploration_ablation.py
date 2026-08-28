from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from experiments.run_v15_variable_scale_ppo_cycle import (
    _evaluation_summary,
    _training_summary,
)
from experiments.run_v15_variable_scale_ppo_pilot import _evaluate
from experiments.variable_scale_topology_ppo import (
    VariableScalePPOConfiguration,
    train_variable_scale_topology_ppo,
)


def _branch(result, evaluation):
    return {
        "training": _training_summary(result),
        "batch_diagnostics": [asdict(item) for item in result.batch_diagnostics],
        "episode_diagnostics": [asdict(item) for item in result.diagnostics],
        "evaluation": evaluation,
        "evaluation_by_node_count": _evaluation_summary(evaluation),
    }


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Pair a standard warm PPO run with type-floor exploration."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--training-episodes", type=int, default=20)
    parser.add_argument("--rollout-batch-episodes", type=int, default=10)
    parser.add_argument("--training-condition-offset", type=int, default=1520)
    parser.add_argument("--type-probability-floor", type=float, default=0.05)
    parser.add_argument(
        "--evaluation-conditions", type=int, nargs="+",
        default=(1580, 1581, 1582, 1583, 1584, 1586),
    )
    arguments = parser.parse_args(argv)
    baseline = VariableScalePPOConfiguration(
        training_episodes=arguments.training_episodes,
        rollout_batch_episodes=arguments.rollout_batch_episodes,
        training_condition_seed_offset=arguments.training_condition_offset,
        training_condition_seed_count=arguments.training_episodes,
        environment_seed_count=4,
        policy_seed=arguments.policy_seed,
        learning_rate=1.0e-4,
        update_epochs=4,
        minibatch_size=32,
        target_kl=0.02,
        explicit_action_pairing=True,
        counterfactual_keep_reward=True,
        critic_scale_calibration_node_counts=(5, 10, 20),
        critic_weight_decay=1.0e-3,
    )
    explored = replace(
        baseline,
        action_type_probability_floor=arguments.type_probability_floor,
    )
    baseline_result = train_variable_scale_topology_ppo(
        baseline, warm_start_checkpoint=str(arguments.warm_start),
        reset_warm_start_type_head=False,
    )
    explored_result = train_variable_scale_topology_ppo(
        explored, warm_start_checkpoint=str(arguments.warm_start),
        reset_warm_start_type_head=False,
    )
    evaluation_conditions = tuple(arguments.evaluation_conditions)
    baseline_evaluation = _evaluate(
        baseline_result.model, baseline.curriculum,
        evaluation_conditions, noise_seed=0,
    )
    explored_evaluation = _evaluate(
        explored_result.model, explored.curriculum,
        evaluation_conditions, noise_seed=0,
    )
    summary = {
        "role": "variable_scale_action_type_exploration_ablation",
        "baseline_configuration": asdict(baseline),
        "explored_configuration": asdict(explored),
        "evaluation_conditions": list(evaluation_conditions),
        "baseline": _branch(baseline_result, baseline_evaluation),
        "type_floor": _branch(explored_result, explored_evaluation),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "role": summary["role"],
        "baseline_configuration": asdict(baseline),
        "explored_configuration": asdict(explored),
        "baseline_model_state_dict": baseline_result.model.state_dict(),
        "explored_model_state_dict": explored_result.model.state_dict(),
        "evaluation_conditions": evaluation_conditions,
    }, arguments.output.with_suffix(".pt"))
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(arguments.output)
    return arguments.output


if __name__ == "__main__":
    main()
