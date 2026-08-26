from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from experiments.graph_action_gnn import torch_snapshot_action_group
from experiments.topology_ppo import (
    PPOUpdateResult,
    TopologyActorCritic,
    TopologyRollout,
    build_warm_started_actor_critic,
    collect_topology_rollout,
    combine_prepared_topology_rollouts,
    prepare_topology_rollout,
    update_topology_ppo,
)
from experiments.topology_ppo_stage1 import (
    Stage1PenaltyWeights,
    build_stage1_environment,
)
from experiments.topology_snapshot_counterfactual import (
    build_online_snapshot_action_tensor,
)
from experiments.variable_scale_topology_curriculum import (
    VariableScaleTopologyCurriculum,
)


ACTION_KINDS = ("keep", "add", "swap", "remove")
AUDITED_COUNTERFACTUAL_RETURN_SCALES = (
    (5, 0.024638556129628262),
    (10, 0.015423466948068319),
    (20, 0.007450872577417267),
)


@dataclass(frozen=True)
class VariableScalePPOConfiguration:
    curriculum: VariableScaleTopologyCurriculum = VariableScaleTopologyCurriculum()
    training_episodes: int = 20
    rollout_batch_episodes: int = 20
    training_condition_seed_offset: int = 400
    training_condition_seed_count: int = 20
    environment_seed_count: int = 4
    policy_seed: int = 0
    learning_rate: float = 1.0e-4
    update_epochs: int = 4
    minibatch_size: int = 32
    target_kl: float | None = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    explicit_action_pairing: bool = True
    critic_timestamp_horizon: float | None = None
    counterfactual_keep_reward: bool = False
    return_scale_by_node_count: tuple[tuple[int, float], ...] = ()
    critic_scale_calibration_node_counts: tuple[int, ...] = ()
    critic_weight_decay: float = 0.0
    penalty_weights: Stage1PenaltyWeights = Stage1PenaltyWeights()


@dataclass(frozen=True)
class VariableScaleEpisodeDiagnostic:
    episode: int
    condition_seed: int
    environment_seed: int
    node_count: int
    task_return: float
    absolute_task_return: float
    penalized_return: float
    unnormalized_penalized_return: float
    final_position_rmse: float
    transmitted_messages_per_node_epoch: float
    resynchronizations_per_node: float
    topology_switches: float
    fallback_count: float
    action_kind_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class VariableScaleBatchDiagnostic:
    batch_start: int
    batch_end: int
    episode_count_by_node_count: tuple[tuple[int, int], ...]
    transition_count_by_node_count: tuple[tuple[int, int], ...]
    update: PPOUpdateResult


@dataclass(frozen=True)
class VariableScaleTrainingResult:
    model: TopologyActorCritic
    diagnostics: tuple[VariableScaleEpisodeDiagnostic, ...]
    batch_diagnostics: tuple[VariableScaleBatchDiagnostic, ...]


def train_variable_scale_topology_ppo(
    configuration: VariableScalePPOConfiguration = VariableScalePPOConfiguration(),
    *,
    warm_start_checkpoint: str | None = None,
    reset_warm_start_type_head: bool = False,
    training_checkpoint: str | Path | None = None,
    resume_training_checkpoint: str | Path | None = None,
    stop_after_batches: int | None = None,
) -> VariableScaleTrainingResult:
    """Train one shared Actor-Critic from mixed 5/10/20-node rollouts."""

    _validate_configuration(configuration)
    if stop_after_batches is not None and stop_after_batches <= 0:
        raise ValueError("stop_after_batches must be positive when enabled.")
    torch.manual_seed(configuration.policy_seed)
    first_condition = configuration.training_condition_seed_offset
    first_environment = build_stage1_environment(
        configuration.curriculum.configuration_for_condition(first_condition)
    )
    first_state = first_environment.reset(
        seed=0, condition_seed=first_condition,
    )
    snapshot, _ = build_online_snapshot_action_tensor(first_state)
    group = torch_snapshot_action_group(snapshot)
    model = (
        build_warm_started_actor_critic(
            warm_start_checkpoint,
            node_feature_count=group.node_features.shape[1],
            reset_type_head=reset_warm_start_type_head,
            critic_timestamp_horizon=configuration.critic_timestamp_horizon,
            critic_scale_calibration_node_counts=(
                configuration.critic_scale_calibration_node_counts
            ),
        )
        if warm_start_checkpoint is not None
        else TopologyActorCritic(
            node_feature_count=group.node_features.shape[1],
            candidate_edge_feature_count=group.candidate_edge_features.shape[1],
            measurement_feature_count=group.measurement_features.shape[1],
            action_feature_count=group.action_features.shape[1],
            global_feature_count=len(first_state.policy_tensor.global_feature_names),
            hidden_size=32,
            message_passing_steps=2,
            explicit_action_pairing=configuration.explicit_action_pairing,
            critic_timestamp_horizon=configuration.critic_timestamp_horizon,
            critic_scale_calibration_node_counts=(
                configuration.critic_scale_calibration_node_counts
            ),
        )
    )
    if model.actor.explicit_action_pairing != configuration.explicit_action_pairing:
        raise ValueError(
            "Warm-start and random-init Actor structures must use the same "
            "explicit-action-pairing setting."
        )
    actor_parameters = list(model.actor.parameters())
    critic_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("actor.")
    ]
    optimizer = torch.optim.Adam((
        {"params": actor_parameters, "weight_decay": 0.0},
        {
            "params": critic_parameters,
            "weight_decay": configuration.critic_weight_decay,
        },
    ), lr=configuration.learning_rate)
    generator = torch.Generator().manual_seed(configuration.policy_seed + 2900)
    diagnostics = []
    batch_diagnostics = []
    first_episode = 0
    if resume_training_checkpoint is not None:
        checkpoint = torch.load(
            Path(resume_training_checkpoint), map_location="cpu", weights_only=True,
        )
        if checkpoint.get("role") != "variable_scale_ppo_training_checkpoint":
            raise ValueError("Not a variable-scale PPO training checkpoint.")
        if checkpoint.get("configuration") != asdict(configuration):
            raise ValueError("Training checkpoint configuration does not match.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        generator.set_state(checkpoint["generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        first_episode = int(checkpoint["next_episode"])
        diagnostics = [
            VariableScaleEpisodeDiagnostic(**item)
            for item in checkpoint["episode_diagnostics"]
        ]
        batch_diagnostics = [
            VariableScaleBatchDiagnostic(
                batch_start=item["batch_start"],
                batch_end=item["batch_end"],
                episode_count_by_node_count=tuple(
                    tuple(value) for value in item["episode_count_by_node_count"]
                ),
                transition_count_by_node_count=tuple(
                    tuple(value) for value in item["transition_count_by_node_count"]
                ),
                update=PPOUpdateResult(**item["update"]),
            )
            for item in checkpoint["batch_diagnostics"]
        ]
        if first_episode > configuration.training_episodes:
            raise ValueError("Training checkpoint exceeds the episode budget.")
        if first_episode % configuration.rollout_batch_episodes:
            raise ValueError("Training checkpoint does not end on a batch boundary.")
    completed_batches = 0
    for batch_start in range(
        first_episode, configuration.training_episodes,
        configuration.rollout_batch_episodes,
    ):
        prepared_rollouts = []
        metadata = []
        batch_end = min(
            configuration.training_episodes,
            batch_start + configuration.rollout_batch_episodes,
        )
        for episode in range(batch_start, batch_end):
            condition_seed = (
                configuration.training_condition_seed_offset
                + episode % configuration.training_condition_seed_count
            )
            environment_seed = episode % configuration.environment_seed_count
            episode_configuration = (
                configuration.curriculum.configuration_for_condition(
                    condition_seed,
                )
            )
            environment = build_stage1_environment(episode_configuration)
            rollout = collect_topology_rollout(
                environment, model,
                seed=environment_seed,
                condition_seed=condition_seed,
                generator=generator,
                counterfactual_keep_reward=(
                    configuration.counterfactual_keep_reward
                ),
            )
            penalized = apply_variable_scale_penalties(
                rollout,
                node_count=episode_configuration.node_count,
                decision_interval_epochs=(
                    episode_configuration.decision_interval_epochs
                ),
                weights=configuration.penalty_weights,
                return_scale=dict(
                    configuration.return_scale_by_node_count
                ).get(episode_configuration.node_count, 1.0),
            )
            prepared = prepare_topology_rollout(
                penalized,
                gamma=configuration.gamma,
                gae_lambda=configuration.gae_lambda,
                normalize_advantages=False,
            )
            prepared_rollouts.append(prepared)
            metadata.append((
                episode, condition_seed, environment_seed,
                episode_configuration.node_count,
                episode_configuration.decision_interval_epochs,
                rollout, penalized, float(environment._metrics()[0]),
            ))
        combined = combine_prepared_topology_rollouts(tuple(prepared_rollouts))
        update = update_topology_ppo(
            model, optimizer, combined,
            update_epochs=configuration.update_epochs,
            entropy_coefficient=configuration.entropy_coefficient,
            target_kl=configuration.target_kl,
            minibatch_size=configuration.minibatch_size,
            generator=generator,
        )
        episode_counts = Counter(item[3] for item in metadata)
        transition_counts = Counter()
        for item in metadata:
            transition_counts[item[3]] += len(item[5].transitions)
        batch_diagnostics.append(VariableScaleBatchDiagnostic(
            batch_start=batch_start,
            batch_end=batch_end,
            episode_count_by_node_count=tuple(sorted(episode_counts.items())),
            transition_count_by_node_count=tuple(sorted(transition_counts.items())),
            update=update,
        ))
        for (
            episode, condition_seed, environment_seed, node_count,
            decision_interval_epochs, rollout, penalized, final_rmse,
        ) in metadata:
            costs = rollout.cost_matrix.sum(dim=0)
            kinds = Counter(
                ACTION_KINDS[int(
                    transition.group.action_kind_index[
                        transition.action_index
                    ].item()
                )]
                for transition in rollout.transitions
            )
            diagnostics.append(VariableScaleEpisodeDiagnostic(
                episode=episode,
                condition_seed=condition_seed,
                environment_seed=environment_seed,
                node_count=node_count,
                task_return=float(rollout.rewards.sum().item()),
                absolute_task_return=float(sum(
                    transition.absolute_reward
                    if transition.absolute_reward is not None
                    else transition.reward
                    for transition in rollout.transitions
                )),
                penalized_return=float(penalized.rewards.sum().item()),
                unnormalized_penalized_return=float(
                    rollout.rewards.sum().item()
                    - configuration.penalty_weights.communication * costs[0]
                    / (node_count * decision_interval_epochs)
                    - configuration.penalty_weights.topology_switch * costs[4]
                    - configuration.penalty_weights.resynchronization
                    * costs[3] / node_count
                ),
                final_position_rmse=final_rmse,
                transmitted_messages_per_node_epoch=float(
                    costs[0] / (node_count * decision_interval_epochs)
                ),
                resynchronizations_per_node=float(costs[3] / node_count),
                topology_switches=float(costs[4]),
                fallback_count=float(costs[5]),
                action_kind_counts=tuple(sorted(kinds.items())),
            ))
        completed_batches += 1
        if training_checkpoint is not None:
            _save_training_checkpoint(
                training_checkpoint,
                configuration=configuration,
                next_episode=batch_end,
                model=model,
                optimizer=optimizer,
                generator=generator,
                diagnostics=diagnostics,
                batch_diagnostics=batch_diagnostics,
            )
        if stop_after_batches is not None and completed_batches >= stop_after_batches:
            break
    return VariableScaleTrainingResult(
        model=model,
        diagnostics=tuple(diagnostics),
        batch_diagnostics=tuple(batch_diagnostics),
    )


def _save_training_checkpoint(
    path: str | Path, *, configuration: VariableScalePPOConfiguration,
    next_episode: int, model: TopologyActorCritic,
    optimizer: torch.optim.Optimizer, generator: torch.Generator,
    diagnostics: list[VariableScaleEpisodeDiagnostic],
    batch_diagnostics: list[VariableScaleBatchDiagnostic],
) -> None:
    """Atomically persist all state needed at the next rollout-batch boundary."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({
        "role": "variable_scale_ppo_training_checkpoint",
        "configuration": asdict(configuration),
        "next_episode": next_episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "episode_diagnostics": [asdict(item) for item in diagnostics],
        "batch_diagnostics": [asdict(item) for item in batch_diagnostics],
    }, temporary)
    temporary.replace(destination)


def apply_variable_scale_penalties(
    rollout: TopologyRollout,
    *,
    node_count: int,
    decision_interval_epochs: int,
    weights: Stage1PenaltyWeights,
    return_scale: float = 1.0,
) -> TopologyRollout:
    """Apply fleet-size-normalized communication and resync penalties."""

    if node_count < 1 or decision_interval_epochs < 1 or return_scale <= 0.0:
        raise ValueError("Penalty normalization requires positive scale values.")
    transitions = tuple(replace(
        transition,
        reward=(
            transition.reward
            - weights.communication * transition.costs[0]
            / (node_count * decision_interval_epochs)
            - weights.topology_switch * transition.costs[4]
            - weights.resynchronization * transition.costs[3] / node_count
        ) / return_scale,
    ) for transition in rollout.transitions)
    return TopologyRollout(transitions, rollout.final_value)


def _validate_configuration(configuration):
    configuration.curriculum.validate()
    positive = (
        configuration.training_episodes,
        configuration.rollout_batch_episodes,
        configuration.training_condition_seed_count,
        configuration.environment_seed_count,
        configuration.learning_rate,
        configuration.update_epochs,
        configuration.minibatch_size,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Variable-scale PPO budgets must be positive.")
    if configuration.rollout_batch_episodes > configuration.training_episodes:
        raise ValueError("Rollout batch cannot exceed the training episode budget.")
    if configuration.target_kl is not None and configuration.target_kl <= 0.0:
        raise ValueError("PPO target KL must be positive when enabled.")
    if (
        configuration.critic_timestamp_horizon is not None
        and configuration.critic_timestamp_horizon <= 0.0
    ):
        raise ValueError("Critic timestamp horizon must be positive when enabled.")
    scales = configuration.return_scale_by_node_count
    if scales:
        if tuple(sorted(node for node, _ in scales)) != (5, 10, 20):
            raise ValueError("Return scales must define 5, 10, and 20 nodes once.")
        if any(scale <= 0.0 for _, scale in scales):
            raise ValueError("Return scales must be positive.")
    calibration_counts = configuration.critic_scale_calibration_node_counts
    if calibration_counts and tuple(sorted(calibration_counts)) != (5, 10, 20):
        raise ValueError("Critic scale calibration must define 5, 10, and 20 nodes.")
    if configuration.critic_weight_decay < 0.0:
        raise ValueError("Critic weight decay must be nonnegative.")
