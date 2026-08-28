from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from experiments.training.graph_action_gnn import torch_snapshot_action_group
from experiments.topology_control_baselines import AlwaysKeepPolicy
from experiments.topology_control_environment import TopologyControlEnvironment
from experiments.training.topology_ppo import (
    PPOUpdateResult,
    TopologyActorCritic,
    TopologyRollout,
    build_warm_started_actor_critic,
    collect_topology_rollout,
    prepare_topology_rollout,
    update_topology_ppo,
)
from experiments.topology_snapshot_counterfactual import (
    build_online_snapshot_action_tensor,
    evaluate_topology_action_snapshot,
)


@dataclass(frozen=True)
class Stage0Configuration:
    training_episodes: int = 40
    episode_epochs: int = 6
    environment_seed: int = 0
    policy_seed: int = 0
    learning_rate: float = 3.0e-4
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    communication_penalty: float = 0.0
    topology_switch_penalty: float = 0.0


@dataclass(frozen=True)
class Stage0EpisodeDiagnostic:
    episode: int
    task_return: float
    penalized_return: float
    transmitted_messages: float
    topology_switches: float
    initial_action_id: int
    update: PPOUpdateResult


@dataclass(frozen=True)
class Stage0TrainingResult:
    model: TopologyActorCritic
    oracle_initial_action_id: int
    initial_policy_action_id: int
    final_policy_action_id: int
    diagnostics: tuple[Stage0EpisodeDiagnostic, ...]


@dataclass(frozen=True)
class Stage0SeedComparison:
    policy_seed: int
    initialization: str
    initial_action_id: int
    final_action_id: int
    oracle_action_id: int
    first_sustained_success_episode: int | None
    final_window_success_rate: float


@dataclass(frozen=True)
class Stage0ComparisonResult:
    records: tuple[Stage0SeedComparison, ...]

    def records_for(self, initialization: str):
        return tuple(
            record for record in self.records
            if record.initialization == initialization
        )


def build_stage0_environment(*, episode_epochs: int = 6):
    """One decision followed by a fixed RANGE-only filtering evaluation window."""

    return TopologyControlEnvironment(
        node_count=3, episode_epochs=episode_epochs,
        relative_modalities=("RANGE",),
        decision_interval_epochs=episode_epochs,
    )


def train_stage0_ppo(
    configuration: Stage0Configuration = Stage0Configuration(),
    *,
    warm_start_checkpoint: str | None = None,
) -> Stage0TrainingResult:
    """Train PPO without exposing counterfactual labels to policy updates."""

    if configuration.training_episodes < 1:
        raise ValueError("Stage 0 requires at least one training episode.")
    torch.manual_seed(configuration.policy_seed)
    environment = build_stage0_environment(
        episode_epochs=configuration.episode_epochs
    )
    initial_state = environment.reset(seed=configuration.environment_seed)
    initial_group, _ = build_online_snapshot_action_tensor(initial_state)
    initial_group = torch_snapshot_action_group(initial_group)
    model = (
        build_warm_started_actor_critic(
            warm_start_checkpoint,
            node_feature_count=initial_group.node_features.shape[1],
        )
        if warm_start_checkpoint is not None
        else TopologyActorCritic(
            node_feature_count=initial_group.node_features.shape[1],
            candidate_edge_feature_count=(
                initial_group.candidate_edge_features.shape[1]
            ),
            measurement_feature_count=initial_group.measurement_features.shape[1],
            action_feature_count=initial_group.action_features.shape[1],
            global_feature_count=len(initial_state.policy_tensor.global_feature_names),
            hidden_size=32, message_passing_steps=2,
            explicit_action_pairing=False,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    initial_action = _initial_policy_action(environment, model, configuration)
    oracle_action = _oracle_initial_action(environment, configuration)
    generator = torch.Generator().manual_seed(configuration.policy_seed + 1000)
    diagnostics = []
    for episode in range(configuration.training_episodes):
        rollout = collect_topology_rollout(
            environment, model, seed=configuration.environment_seed,
            generator=generator,
        )
        training_rollout = _penalize_rollout(rollout, configuration)
        update = update_topology_ppo(
            model, optimizer,
            prepare_topology_rollout(
                training_rollout, gamma=configuration.gamma,
                gae_lambda=configuration.gae_lambda,
            ),
            update_epochs=configuration.update_epochs,
            entropy_coefficient=configuration.entropy_coefficient,
        )
        costs = rollout.cost_matrix.sum(dim=0)
        diagnostics.append(Stage0EpisodeDiagnostic(
            episode=episode,
            task_return=float(rollout.rewards.sum().item()),
            penalized_return=float(training_rollout.rewards.sum().item()),
            transmitted_messages=float(costs[0].item()),
            topology_switches=float(costs[4].item()),
            initial_action_id=rollout.transitions[0].environment_action_id,
            update=update,
        ))
    return Stage0TrainingResult(
        model=model, oracle_initial_action_id=oracle_action,
        initial_policy_action_id=initial_action,
        final_policy_action_id=_initial_policy_action(
            environment, model, configuration
        ),
        diagnostics=tuple(diagnostics),
    )


def compare_stage0_initializations(
    *, policy_seeds=range(3), training_episodes: int = 80,
    warm_start_checkpoint: str,
    success_window: int = 20,
) -> Stage0ComparisonResult:
    """Compare random and warm-start Actors under identical PPO budgets."""

    seeds = tuple(int(seed) for seed in policy_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Stage 0 comparison seeds must be unique and nonempty.")
    if success_window < 1 or success_window > training_episodes:
        raise ValueError("Stage 0 success window must fit inside training.")
    records = []
    for seed in seeds:
        configuration = Stage0Configuration(
            training_episodes=training_episodes, policy_seed=seed,
            learning_rate=1.0e-3, entropy_coefficient=0.005,
        )
        for initialization, checkpoint in (
            ("random", None), ("warm_start", warm_start_checkpoint),
        ):
            result = train_stage0_ppo(
                configuration, warm_start_checkpoint=checkpoint
            )
            actions = tuple(
                item.initial_action_id for item in result.diagnostics
            )
            oracle = result.oracle_initial_action_id
            records.append(Stage0SeedComparison(
                policy_seed=seed, initialization=initialization,
                initial_action_id=result.initial_policy_action_id,
                final_action_id=result.final_policy_action_id,
                oracle_action_id=oracle,
                first_sustained_success_episode=_first_sustained_success(
                    actions, oracle, success_window
                ),
                final_window_success_rate=float(np.mean(
                    np.asarray(actions[-success_window:]) == oracle
                )),
            ))
    return Stage0ComparisonResult(tuple(records))


def _oracle_initial_action(environment, configuration):
    records = evaluate_topology_action_snapshot(
        environment, seed=configuration.environment_seed, decision_epoch=0,
        baseline_policy=AlwaysKeepPolicy(), lookahead_steps=1,
    )
    return max(records, key=lambda record: (
        record.position_rmse_reduction_vs_keep, -record.action_id
    )).action_id


def _initial_policy_action(environment, model, configuration):
    state = environment.reset(seed=configuration.environment_seed)
    snapshot, action_ids = build_online_snapshot_action_tensor(state)
    with torch.no_grad():
        selected = model(torch_snapshot_action_group(snapshot)).distribution.mode()
    return int(action_ids[int(selected.item())])


def _penalize_rollout(rollout, configuration):
    transitions = tuple(replace(
        transition,
        reward=(
            transition.reward
            - configuration.communication_penalty * transition.costs[0]
            - configuration.topology_switch_penalty * transition.costs[4]
        ),
    ) for transition in rollout.transitions)
    return TopologyRollout(transitions, rollout.final_value)


def _first_sustained_success(actions, oracle_action, window):
    for start in range(len(actions) - window + 1):
        if all(action == oracle_action for action in actions[start:start + window]):
            return start
    return None
