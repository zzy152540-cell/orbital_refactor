from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.run_v15_variable_scale_critic_audit import _load_model
from experiments.variable_scale_topology_curriculum import VariableScaleTopologyCurriculum
from experiments.variable_scale_value_feature_audit import (
    audit_variable_scale_value_features,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Audit held-out return predictability of current Critic features."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-conditions", type=int, nargs="+", required=True)
    parser.add_argument("--test-conditions", type=int, nargs="+", required=True)
    parser.add_argument(
        "--ridge-penalties", type=float, nargs="+",
        default=(0.1, 1.0, 10.0, 100.0),
    )
    arguments = parser.parse_args(argv)
    checkpoint = torch.load(arguments.ppo_checkpoint, map_location="cpu",
                            weights_only=False)
    configuration = checkpoint["configuration"]
    calibration = tuple(configuration.get(
        "critic_scale_calibration_node_counts", ()
    ))
    model = _load_model(
        arguments.warm_start, checkpoint["warm_model_state_dict"],
        critic_timestamp_horizon=configuration.get("critic_timestamp_horizon"),
        critic_scale_calibration_node_counts=calibration,
    )
    summary = audit_variable_scale_value_features(
        model, VariableScaleTopologyCurriculum(),
        training_condition_seeds=tuple(arguments.training_conditions),
        test_condition_seeds=tuple(arguments.test_conditions),
        counterfactual_keep_reward=bool(
            configuration.get("counterfactual_keep_reward", False)
        ),
        return_scale_by_node_count=tuple(
            (int(node), float(scale)) for node, scale in configuration.get(
                "return_scale_by_node_count", ()
            )
        ),
        ridge_penalties=tuple(arguments.ridge_penalties),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2, sort_keys=True),
                                encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
