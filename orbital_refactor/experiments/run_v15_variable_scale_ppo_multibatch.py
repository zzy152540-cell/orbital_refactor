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
from experiments.training.variable_scale_topology_ppo import (
    AUDITED_COUNTERFACTUAL_RETURN_SCALES,
    VariableScalePPOConfiguration,
    train_variable_scale_topology_ppo,
)


def _branch_summary(result, evaluation, randomized_evaluation=None):
    summary = {
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
    if randomized_evaluation is not None:
        summary["randomized_walker_evaluation"] = randomized_evaluation
        summary["randomized_walker_evaluation_by_node_count"] = (
            _evaluation_summary(randomized_evaluation)
        )
    return summary


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run an aligned multi-batch mixed-scale PPO comparison."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument(
        "--initializations", choices=("both", "random", "warm"), default="both",
        help="Train both initialization branches or only the selected branch.",
    )
    parser.add_argument("--training-episodes", type=int, default=60)
    parser.add_argument("--rollout-batch-episodes", type=int, default=10)
    parser.add_argument("--critic-timestamp-horizon", type=float)
    parser.add_argument("--counterfactual-keep-reward", action="store_true")
    parser.add_argument("--difference-resource-penalties-from-keep", action="store_true")
    parser.add_argument(
        "--normalize-counterfactual-return-by-scale", action="store_true",
    )
    parser.add_argument("--critic-scale-calibration", action="store_true")
    parser.add_argument("--critic-weight-decay", type=float, default=0.0)
    parser.add_argument("--action-type-probability-floor", type=float, default=0.0)
    parser.add_argument("--walker-randomization-start-episode", type=int)
    parser.add_argument("--walker-randomization-full-episode", type=int)
    parser.add_argument(
        "--walker-randomization-max-probability", type=float, default=1.0,
    )
    parser.add_argument("--stratify-walker-randomization-by-batch", action="store_true")
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--resume-random", type=Path)
    parser.add_argument("--resume-warm", type=Path)
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
        difference_resource_penalties_from_keep=(
            arguments.difference_resource_penalties_from_keep
        ),
        return_scale_by_node_count=(
            AUDITED_COUNTERFACTUAL_RETURN_SCALES
            if arguments.normalize_counterfactual_return_by_scale else ()
        ),
        critic_scale_calibration_node_counts=(
            (5, 10, 20) if arguments.critic_scale_calibration else ()
        ),
        critic_weight_decay=arguments.critic_weight_decay,
        action_type_probability_floor=arguments.action_type_probability_floor,
        walker_randomization_start_episode=(
            arguments.walker_randomization_start_episode
        ),
        walker_randomization_full_episode=(
            arguments.walker_randomization_full_episode
        ),
        walker_randomization_max_probability=(
            arguments.walker_randomization_max_probability
        ),
        stratify_walker_randomization_by_batch=(
            arguments.stratify_walker_randomization_by_batch
        ),
    )
    checkpoint_directory = arguments.checkpoint_directory
    random_checkpoint = (
        checkpoint_directory / "random_training.pt"
        if checkpoint_directory is not None else None
    )
    warm_checkpoint = (
        checkpoint_directory / "warm_training.pt"
        if checkpoint_directory is not None else None
    )
    random_result = None
    warm_result = None
    if arguments.initializations in ("both", "random"):
        random_result = train_variable_scale_topology_ppo(
            configuration,
            training_checkpoint=random_checkpoint,
            resume_training_checkpoint=arguments.resume_random,
        )
    if arguments.initializations in ("both", "warm"):
        warm_result = train_variable_scale_topology_ppo(
            configuration,
            warm_start_checkpoint=str(arguments.warm_start),
            reset_warm_start_type_head=False,
            training_checkpoint=warm_checkpoint,
            resume_training_checkpoint=arguments.resume_warm,
        )
    evaluation_conditions = tuple(arguments.evaluation_conditions)
    randomized_curriculum = replace(
        configuration.curriculum, randomize_walker_initialization=True,
    )
    summary = {
        "comparison_role": "aligned_actor_multibatch_initialization_ablation",
        "configuration": asdict(configuration),
        "evaluation_conditions": list(evaluation_conditions),
    }
    saved = {
        "role": summary["comparison_role"],
        "configuration": asdict(configuration),
        "evaluation_conditions": evaluation_conditions,
    }
    for name, result in (("random_init", random_result), ("warm_start", warm_result)):
        if result is None:
            continue
        fixed_evaluation = _evaluate(
            result.model, configuration.curriculum,
            evaluation_conditions, noise_seed=0,
        )
        randomized_evaluation = _evaluate(
            result.model, randomized_curriculum,
            evaluation_conditions, noise_seed=0,
        )
        summary[name] = _branch_summary(
            result, fixed_evaluation, randomized_evaluation,
        )
        checkpoint_key = (
            "random_model_state_dict"
            if name == "random_init" else "warm_model_state_dict"
        )
        saved[checkpoint_key] = result.model.state_dict()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, arguments.output.with_suffix(".pt"))
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
