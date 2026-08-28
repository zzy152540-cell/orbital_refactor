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
        description="Compare absolute and keep-difference resource penalties."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--training-episodes", type=int, default=20)
    parser.add_argument("--rollout-batch-episodes", type=int, default=10)
    parser.add_argument("--training-condition-offset", type=int, default=1600)
    parser.add_argument(
        "--evaluation-conditions", type=int, nargs="+",
        default=(1660, 1661, 1662, 1663, 1664, 1666),
    )
    arguments = parser.parse_args(argv)
    absolute = VariableScalePPOConfiguration(
        training_episodes=arguments.training_episodes,
        rollout_batch_episodes=arguments.rollout_batch_episodes,
        training_condition_seed_offset=arguments.training_condition_offset,
        training_condition_seed_count=arguments.training_episodes,
        environment_seed_count=4,
        policy_seed=arguments.policy_seed,
        learning_rate=1.0e-4, update_epochs=4, minibatch_size=32,
        target_kl=0.02, explicit_action_pairing=True,
        counterfactual_keep_reward=True,
        critic_scale_calibration_node_counts=(5, 10, 20),
        critic_weight_decay=1.0e-3,
    )
    difference = replace(
        absolute, difference_resource_penalties_from_keep=True,
    )
    absolute_result = train_variable_scale_topology_ppo(
        absolute, warm_start_checkpoint=str(arguments.warm_start),
        reset_warm_start_type_head=False,
    )
    difference_result = train_variable_scale_topology_ppo(
        difference, warm_start_checkpoint=str(arguments.warm_start),
        reset_warm_start_type_head=False,
    )
    conditions = tuple(arguments.evaluation_conditions)
    absolute_evaluation = _evaluate(
        absolute_result.model, absolute.curriculum, conditions, noise_seed=0,
    )
    difference_evaluation = _evaluate(
        difference_result.model, difference.curriculum, conditions, noise_seed=0,
    )
    summary = {
        "role": "variable_scale_resource_reward_form_ablation",
        "absolute_configuration": asdict(absolute),
        "difference_configuration": asdict(difference),
        "evaluation_conditions": list(conditions),
        "absolute": _branch(absolute_result, absolute_evaluation),
        "difference_from_keep": _branch(difference_result, difference_evaluation),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "role": summary["role"],
        "absolute_configuration": asdict(absolute),
        "difference_configuration": asdict(difference),
        "absolute_model_state_dict": absolute_result.model.state_dict(),
        "difference_model_state_dict": difference_result.model.state_dict(),
        "evaluation_conditions": conditions,
    }, arguments.output.with_suffix(".pt"))
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(arguments.output)
    return arguments.output


if __name__ == "__main__":
    main()
