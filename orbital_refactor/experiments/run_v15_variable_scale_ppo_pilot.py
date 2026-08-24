from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import torch

from experiments.topology_control_baselines import (
    AlwaysKeepPolicy,
    run_topology_control_baseline_episode,
)
from experiments.topology_ppo import collect_topology_rollout
from experiments.topology_ppo_stage1 import build_stage1_environment
from experiments.variable_scale_topology_ppo import (
    ACTION_KINDS,
    VariableScalePPOConfiguration,
    train_variable_scale_topology_ppo,
)


def _evaluate(model, curriculum, condition_seeds, noise_seed):
    records = []
    for condition_seed in condition_seeds:
        configuration = curriculum.configuration_for_condition(condition_seed)
        keep = run_topology_control_baseline_episode(
            build_stage1_environment(configuration), AlwaysKeepPolicy(),
            seed=noise_seed, condition_seed=condition_seed,
        )
        environment = build_stage1_environment(configuration)
        rollout = collect_topology_rollout(
            environment, model, seed=noise_seed,
            condition_seed=condition_seed, deterministic=True,
        )
        model_rmse = float(environment._metrics()[0])
        kinds = Counter(
            ACTION_KINDS[int(
                transition.group.action_kind_index[
                    transition.action_index
                ].item()
            )]
            for transition in rollout.transitions
        )
        records.append({
            "condition_seed": int(condition_seed),
            "node_count": configuration.node_count,
            "keep_final_position_rmse": keep.final_position_rmse,
            "model_final_position_rmse": model_rmse,
            "model_rmse_improvement": keep.final_position_rmse - model_rmse,
            "action_kind_counts": dict(sorted(kinds.items())),
            "topology_switches": float(sum(
                transition.costs[4] for transition in rollout.transitions
            )),
            "fallbacks": float(sum(
                transition.costs[5] for transition in rollout.transitions
            )),
        })
    return records


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run the first shared 5/10/20-node V15 PPO pilot."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-episodes", type=int, default=6)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument(
        "--evaluation-conditions", type=int, nargs="+",
        default=(420, 421, 422),
    )
    arguments = parser.parse_args(argv)
    configuration = VariableScalePPOConfiguration(
        training_episodes=arguments.training_episodes,
        rollout_batch_episodes=arguments.training_episodes,
        training_condition_seed_offset=400,
        training_condition_seed_count=arguments.training_episodes,
        environment_seed_count=min(4, arguments.training_episodes),
        policy_seed=arguments.policy_seed,
        update_epochs=2,
        minibatch_size=32,
    )
    result = train_variable_scale_topology_ppo(
        configuration,
        warm_start_checkpoint=str(arguments.warm_start),
        reset_warm_start_type_head=False,
    )
    evaluation = _evaluate(
        result.model, configuration.curriculum,
        tuple(arguments.evaluation_conditions), noise_seed=0,
    )
    summary = {
        "configuration": asdict(configuration),
        "training_diagnostics": [
            asdict(item) for item in result.diagnostics
        ],
        "batch_diagnostics": [
            asdict(item) for item in result.batch_diagnostics
        ],
        "evaluation": evaluation,
        "mean_evaluation_rmse_improvement": float(sum(
            item["model_rmse_improvement"] for item in evaluation
        ) / len(evaluation)),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = arguments.output.with_suffix(".pt")
    torch.save({
        "role": "variable_scale_topology_ppo_pilot",
        "model_state_dict": result.model.state_dict(),
        "configuration": asdict(configuration),
        "evaluation": evaluation,
    }, checkpoint)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
