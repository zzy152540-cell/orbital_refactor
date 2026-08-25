from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from experiments.run_v15_variable_scale_ppo_cycle import (
    _evaluation_summary,
    _training_summary,
)
from experiments.run_v15_variable_scale_ppo_pilot import _evaluate
from experiments.variable_scale_topology_ppo import (
    AUDITED_COUNTERFACTUAL_RETURN_SCALES,
    VariableScalePPOConfiguration,
    train_variable_scale_topology_ppo,
)


def _branch_summary(result, evaluation):
    return {
        "training": _training_summary(result),
        "batch_diagnostics": [
            asdict(item) for item in result.batch_diagnostics
        ],
        "episode_diagnostics": [
            asdict(item) for item in result.diagnostics
        ],
        "evaluation": evaluation,
        "evaluation_by_node_count": _evaluation_summary(evaluation),
    }


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run an aligned multi-batch mixed-scale PPO comparison."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--training-episodes", type=int, default=60)
    parser.add_argument("--rollout-batch-episodes", type=int, default=10)
    parser.add_argument("--critic-timestamp-horizon", type=float)
    parser.add_argument("--counterfactual-keep-reward", action="store_true")
    parser.add_argument(
        "--normalize-counterfactual-return-by-scale", action="store_true",
    )
    parser.add_argument("--critic-scale-calibration", action="store_true")
    parser.add_argument("--training-condition-offset", type=int, default=500)
    parser.add_argument(
        "--evaluation-conditions", type=int, nargs="+",
        default=(580, 581, 582, 583, 584, 586),
    )
    arguments = parser.parse_args(argv)
    configuration = VariableScalePPOConfiguration(
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
        critic_timestamp_horizon=arguments.critic_timestamp_horizon,
        counterfactual_keep_reward=arguments.counterfactual_keep_reward,
        return_scale_by_node_count=(
            AUDITED_COUNTERFACTUAL_RETURN_SCALES
            if arguments.normalize_counterfactual_return_by_scale else ()
        ),
        critic_scale_calibration_node_counts=(
            (5, 10, 20) if arguments.critic_scale_calibration else ()
        ),
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
        "comparison_role": "aligned_actor_multibatch_initialization_ablation",
        "configuration": asdict(configuration),
        "evaluation_conditions": list(evaluation_conditions),
        "random_init": _branch_summary(random_result, random_evaluation),
        "warm_start": _branch_summary(warm_result, warm_evaluation),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "role": summary["comparison_role"],
        "configuration": asdict(configuration),
        "random_model_state_dict": random_result.model.state_dict(),
        "warm_model_state_dict": warm_result.model.state_dict(),
        "evaluation_conditions": evaluation_conditions,
    }, arguments.output.with_suffix(".pt"))
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
