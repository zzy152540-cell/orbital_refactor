from __future__ import annotations

import argparse
from pathlib import Path

from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.topology_snapshot_counterfactual import (
    build_topology_snapshot_tensor_dataset,
    save_topology_snapshot_tensor_dataset,
)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        description="Collect one restartable V15 Walker snapshot dataset shard."
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--episode-epochs", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    maximum_epoch = max(arguments.epochs)
    episode_epochs = arguments.episode_epochs or (
        maximum_epoch + arguments.lookahead + 1
    )
    if episode_epochs <= maximum_epoch:
        parser.error("--episode-epochs must exceed every decision epoch.")
    environment = TopologyControlEnvironment(
        node_count=20, episode_epochs=episode_epochs,
        relative_modalities=("RADAR", "INFRARED", "OPTICAL"),
        scenario_type="walker_20_5_3", walker_maximum_range=7000e3,
        top_k_candidate_neighbors=arguments.top_k,
        minimum_topology_dwell_decisions=2,
    )
    dataset = build_topology_snapshot_tensor_dataset(
        environment, seeds=arguments.seeds,
        decision_epochs=arguments.epochs,
        baseline_policy=AlwaysKeepPolicy(),
        lookahead_steps=arguments.lookahead,
    )
    path = save_topology_snapshot_tensor_dataset(dataset, arguments.output)
    print(
        f"saved {len(dataset.groups)} groups / "
        f"{sum(len(group.action_kinds) for group in dataset.groups)} actions "
        f"to {path}"
    )
    return path


if __name__ == "__main__":
    main()
