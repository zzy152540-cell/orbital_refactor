from __future__ import annotations

import argparse
from pathlib import Path

from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_ppo_stage1 import (
    build_stage1_environment,
    five_node_stage1_configuration,
)
from experiments.topology_snapshot_counterfactual import (
    build_noise_robust_topology_snapshot_tensor_dataset,
    save_topology_snapshot_tensor_dataset,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Collect a restartable five-node noise-robust snapshot shard."
    )
    parser.add_argument("--condition-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--noise-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=(0,))
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--episode-epochs", type=int, default=6)
    parser.add_argument("--decision-interval", type=int, default=2)
    parser.add_argument("--maximum-switches", type=int, default=1)
    parser.add_argument("--gain-std-penalty", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    maximum_epoch = max(arguments.epochs)
    if arguments.episode_epochs <= maximum_epoch:
        parser.error("--episode-epochs must exceed every decision epoch.")
    configuration = five_node_stage1_configuration(
        training_episodes=1,
        episode_epochs=arguments.episode_epochs,
        decision_interval_epochs=arguments.decision_interval,
        maximum_topology_switches_per_episode=arguments.maximum_switches,
    )
    dataset = build_noise_robust_topology_snapshot_tensor_dataset(
        build_stage1_environment(configuration),
        condition_seeds=arguments.condition_seeds,
        noise_seeds=arguments.noise_seeds,
        decision_epochs=arguments.epochs,
        baseline_policy=AlwaysKeepPolicy(),
        lookahead_steps=arguments.lookahead,
        gain_standard_deviation_penalty=arguments.gain_std_penalty,
    )
    path = save_topology_snapshot_tensor_dataset(dataset, arguments.output)
    print(
        f"saved {len(dataset.groups)} robust groups / "
        f"{sum(len(group.action_kinds) for group in dataset.groups)} actions "
        f"to {path}"
    )
    return path


if __name__ == "__main__":
    main()
