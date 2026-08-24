from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.graph_action_gnn import torch_snapshot_action_group
from experiments.run_v15_stratified_ppo_pilot import _evaluate_model
from experiments.topology_ppo import (
    TopologyActorCritic,
    build_warm_started_actor_critic,
)
from experiments.topology_ppo_stage1 import (
    build_stage1_environment,
    five_node_stratified_physical_ppo_configuration,
)
from experiments.topology_snapshot_counterfactual import (
    build_online_snapshot_action_tensor,
)


def _random_model(configuration, state_dict):
    environment = build_stage1_environment(configuration)
    state = environment.reset(seed=0, condition_seed=200)
    group = torch_snapshot_action_group(
        build_online_snapshot_action_tensor(state)[0]
    )
    model = TopologyActorCritic(
        node_feature_count=group.node_features.shape[1],
        candidate_edge_feature_count=group.candidate_edge_features.shape[1],
        measurement_feature_count=group.measurement_features.shape[1],
        action_feature_count=group.action_features.shape[1],
        global_feature_count=len(state.policy_tensor.global_feature_names),
        hidden_size=32,
        message_passing_steps=2,
        explicit_action_pairing=False,
    )
    model.load_state_dict(state_dict)
    return model


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Run the one-time frozen V15 formal confirmation."
    )
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--policy-bundles", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    condition_seeds = tuple(range(240, 248))
    noise_seeds = tuple(range(4))
    records = []
    for policy_seed, path in enumerate(arguments.policy_bundles):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        configuration = five_node_stratified_physical_ppo_configuration(
            policy_seed=policy_seed,
        )
        warm_model = build_warm_started_actor_critic(
            arguments.warm_start, reset_type_head=False,
        )
        warm_model.load_state_dict(bundle["warm_model_state_dict"])
        random_model = _random_model(
            configuration, bundle["random_model_state_dict"]
        )
        records.append({
            "policy_seed": policy_seed,
            "bundle": str(path),
            "random_init": _evaluate_model(
                configuration, random_model,
                condition_seeds=condition_seeds, noise_seeds=noise_seeds,
            ),
            "warm_start": _evaluate_model(
                configuration, warm_model,
                condition_seeds=condition_seeds, noise_seeds=noise_seeds,
            ),
        })
    summary = {
        "condition_seeds": condition_seeds,
        "noise_seeds": noise_seeds,
        "decoder": "hierarchical_type_then_member_mode",
        "records": records,
    }
    for name in ("random_init", "warm_start"):
        values = [record[name] for record in records]
        summary[f"{name}_aggregate"] = {
            "mean_rmse_improvement": float(np.mean([
                value["mean_rmse_improvement"] for value in values
            ])),
            "improved_episode_count": int(sum(
                value["improved_episode_count"] for value in values
            )),
            "total_episode_count": int(sum(
                value["episode_count"] for value in values
            )),
            "worst_rmse_improvement": float(min(
                value["worst_rmse_improvement"] for value in values
            )),
            "fallbacks": float(sum(value["fallbacks"] for value in values)),
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return arguments.output


if __name__ == "__main__":
    main()
