from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.training.topology_ppo import build_warm_started_actor_critic
from experiments.variable_scale_critic_audit import (
    audit_variable_scale_critic,
)
from experiments.variable_scale_topology_curriculum import (
    VariableScaleTopologyCurriculum,
)


def _load_model(
    warm_start, state_dict, *, critic_timestamp_horizon=None,
    critic_scale_calibration_node_counts=(),
):
    model = build_warm_started_actor_critic(
        warm_start, reset_type_head=False,
        critic_timestamp_horizon=critic_timestamp_horizon,
        critic_scale_calibration_node_counts=(
            critic_scale_calibration_node_counts
        ),
    )
    model.load_state_dict(state_dict)
    return model


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Audit frozen mixed-scale PPO Critics by scale and action type."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition-seeds", type=int, nargs="+",
        default=(600, 601, 602, 603, 604, 606),
    )
    arguments = parser.parse_args(argv)
    checkpoint = torch.load(
        arguments.ppo_checkpoint, map_location="cpu", weights_only=False,
    )
    curriculum = VariableScaleTopologyCurriculum()
    critic_timestamp_horizon = checkpoint["configuration"].get(
        "critic_timestamp_horizon"
    )
    counterfactual_keep_reward = bool(
        checkpoint["configuration"].get("counterfactual_keep_reward", False)
    )
    return_scale_by_node_count = tuple(
        (int(node), float(scale))
        for node, scale in checkpoint["configuration"].get(
            "return_scale_by_node_count", ()
        )
    )
    critic_scale_calibration_node_counts = tuple(
        int(value) for value in checkpoint["configuration"].get(
            "critic_scale_calibration_node_counts", ()
        )
    )
    conditions = tuple(arguments.condition_seeds)
    summary = {
        "audit_role": "frozen_multibatch_critic_mc_return_audit",
        "ppo_checkpoint": str(arguments.ppo_checkpoint),
        "random_init": audit_variable_scale_critic(
            _load_model(
                arguments.warm_start,
                checkpoint["random_model_state_dict"],
                critic_timestamp_horizon=critic_timestamp_horizon,
                critic_scale_calibration_node_counts=(
                    critic_scale_calibration_node_counts
                ),
            ),
            curriculum,
            condition_seeds=conditions,
            counterfactual_keep_reward=counterfactual_keep_reward,
            return_scale_by_node_count=return_scale_by_node_count,
        ),
        "warm_start": audit_variable_scale_critic(
            _load_model(
                arguments.warm_start,
                checkpoint["warm_model_state_dict"],
                critic_timestamp_horizon=critic_timestamp_horizon,
                critic_scale_calibration_node_counts=(
                    critic_scale_calibration_node_counts
                ),
            ),
            curriculum,
            condition_seeds=conditions,
            counterfactual_keep_reward=counterfactual_keep_reward,
            return_scale_by_node_count=return_scale_by_node_count,
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
