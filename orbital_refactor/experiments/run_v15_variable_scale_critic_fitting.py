from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from experiments.run_v15_variable_scale_critic_audit import _load_model
from experiments.training.variable_scale_critic_fitting import (
    CriticFittingConfiguration,
    fit_frozen_actor_critic,
)
from experiments.variable_scale_topology_curriculum import (
    VariableScaleTopologyCurriculum,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Test Critic learnability while keeping the Actor frozen."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-conditions", type=int, nargs="+", required=True)
    parser.add_argument("--validation-conditions", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--randomize-walker-initialization", action="store_true")
    arguments = parser.parse_args(argv)

    checkpoint = torch.load(
        arguments.ppo_checkpoint, map_location="cpu", weights_only=False,
    )
    source_configuration = checkpoint["configuration"]
    calibration = tuple(source_configuration.get(
        "critic_scale_calibration_node_counts", ()
    ))
    model = _load_model(
        arguments.warm_start, checkpoint["warm_model_state_dict"],
        critic_timestamp_horizon=source_configuration.get(
            "critic_timestamp_horizon"
        ),
        critic_scale_calibration_node_counts=calibration,
    )
    curriculum = replace(
        VariableScaleTopologyCurriculum(),
        randomize_walker_initialization=(
            arguments.randomize_walker_initialization
        ),
    )
    fit_configuration = CriticFittingConfiguration(
        learning_rate=arguments.learning_rate,
        epochs=arguments.epochs,
        minibatch_size=arguments.minibatch_size,
        weight_decay=arguments.weight_decay,
    )
    summary = fit_frozen_actor_critic(
        model, curriculum,
        training_condition_seeds=tuple(arguments.training_conditions),
        validation_condition_seeds=tuple(arguments.validation_conditions),
        configuration=fit_configuration,
        counterfactual_keep_reward=bool(
            source_configuration.get("counterfactual_keep_reward", False)
        ),
        return_scale_by_node_count=tuple(
            (int(node), float(scale)) for node, scale in source_configuration.get(
                "return_scale_by_node_count", ()
            )
        ),
    )
    payload = {
        "audit_role": "frozen_actor_critic_only_learnability_gate",
        "source_checkpoint": str(arguments.ppo_checkpoint),
        "walker_initialization_randomized": bool(
            arguments.randomize_walker_initialization
        ),
        "configuration": asdict(fit_configuration),
        **summary,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    torch.save({
        "role": payload["audit_role"],
        "model_state_dict": model.state_dict(),
        "configuration": payload["configuration"],
    }, arguments.output.with_suffix(".pt"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
