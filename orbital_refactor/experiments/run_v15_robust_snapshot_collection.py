from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_ppo_stage1 import (
    build_stage1_environment,
    five_node_heterogeneous_link_configuration,
    five_node_randomized_physical_configuration,
    five_node_stage1_configuration,
    five_node_stratified_physical_configuration,
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
    parser.add_argument("--include-all-noise-observations", action="store_true")
    parser.add_argument(
        "--relative-modalities", nargs="+", default=("RANGE",),
        choices=("RANGE", "RANGE_RATE", "AZ_EL", "OPTICAL"),
    )
    parser.add_argument("--cann-policy-features", action="store_true")
    parser.add_argument(
        "--navigation-dropout-node-count", type=int,
        help=(
            "Override the sampled absolute-navigation dropout-node count; "
            "use zero for a no-dropout control shard."
        ),
    )
    parser.add_argument(
        "--initial-topology-types", nargs="+",
        choices=("chain", "ring", "star", "fully_connected"),
        help=(
            "Override the sampled initial-topology families; use "
            "fully_connected to collect remove-informative controls."
        ),
    )
    distribution = parser.add_mutually_exclusive_group()
    distribution.add_argument("--heterogeneous-links", action="store_true")
    distribution.add_argument(
        "--randomized-physical-scenarios", action="store_true"
    )
    distribution.add_argument(
        "--stratified-physical-scenarios", action="store_true"
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    maximum_epoch = max(arguments.epochs)
    if (
        maximum_epoch * arguments.decision_interval
        >= arguments.episode_epochs
    ):
        parser.error(
            "Every --epochs decision index times --decision-interval must "
            "remain below --episode-epochs."
        )
    configuration_factory = five_node_stage1_configuration
    if arguments.heterogeneous_links:
        configuration_factory = five_node_heterogeneous_link_configuration
    if arguments.randomized_physical_scenarios:
        configuration_factory = five_node_randomized_physical_configuration
    if arguments.stratified_physical_scenarios:
        configuration_factory = five_node_stratified_physical_configuration
    configuration = configuration_factory(
        training_episodes=1,
        episode_epochs=arguments.episode_epochs,
        decision_interval_epochs=arguments.decision_interval,
        maximum_topology_switches_per_episode=arguments.maximum_switches,
    )
    distribution_changes = {}
    if arguments.navigation_dropout_node_count is not None:
        if arguments.navigation_dropout_node_count < 0:
            parser.error("--navigation-dropout-node-count cannot be negative.")
        distribution_changes["navigation_dropout_node_count"] = (
            arguments.navigation_dropout_node_count
        )
    if arguments.initial_topology_types is not None:
        if len(set(arguments.initial_topology_types)) != len(
            arguments.initial_topology_types
        ):
            parser.error("--initial-topology-types must be unique.")
        distribution_changes["initial_topology_types"] = tuple(
            arguments.initial_topology_types
        )
    if distribution_changes:
        configuration = replace(
            configuration,
            scenario_distribution=replace(
                configuration.scenario_distribution,
                **distribution_changes,
            ),
        )
    dataset = build_noise_robust_topology_snapshot_tensor_dataset(
        build_stage1_environment(
            configuration,
            relative_modalities=arguments.relative_modalities,
            cann_policy_features=arguments.cann_policy_features,
        ),
        condition_seeds=arguments.condition_seeds,
        noise_seeds=arguments.noise_seeds,
        decision_epochs=arguments.epochs,
        baseline_policy=AlwaysKeepPolicy(),
        lookahead_steps=arguments.lookahead,
        gain_standard_deviation_penalty=arguments.gain_std_penalty,
        include_all_noise_observations=(
            arguments.include_all_noise_observations
        ),
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
