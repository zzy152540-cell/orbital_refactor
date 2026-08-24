from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from experiments.graph_action_gnn import torch_snapshot_action_group
from experiments.topology_control_baselines import (
    AlwaysKeepPolicy,
    InformationGreedyPolicy,
    run_topology_control_baseline_episode,
)
from experiments.topology_control_environment import (
    CompactFleetScenarioDistribution,
    TopologyControlEnvironment,
)
from experiments.topology_ppo import (
    PPOUpdateResult,
    TopologyActorCritic,
    TopologyRollout,
    build_warm_started_actor_critic,
    combine_prepared_topology_rollouts,
    collect_topology_rollout,
    prepare_topology_rollout,
    update_topology_ppo,
)
from experiments.topology_snapshot_counterfactual import (
    build_online_snapshot_action_tensor,
)


@dataclass(frozen=True)
class Stage1CostNormalization:
    transmitted_messages: float = 4.0
    topology_switch: float = 1.0
    resynchronization: float = 2.0


@dataclass(frozen=True)
class Stage1PenaltyWeights:
    communication: float = 0.0025
    topology_switch: float = 0.001
    resynchronization: float = 0.001


@dataclass(frozen=True)
class Stage1Configuration:
    node_count: int = 3
    scenario_type: str = "compact_fleet"
    walker_maximum_range: float = 7000e3
    walker_plane_count: int = 5
    walker_phasing: int = 3
    top_k_candidate_neighbors: int | None = None
    training_episodes: int = 40
    episode_epochs: int = 12
    decision_interval_epochs: int = 2
    minimum_topology_dwell_decisions: int = 1
    maximum_topology_switches_per_episode: int | None = None
    environment_seed_count: int = 8
    rollout_batch_episodes: int = 8
    minibatch_size: int = 16
    environment_seed_offset: int = 0
    condition_seed_offset: int | None = None
    condition_seed_count: int | None = None
    policy_seed: int = 0
    learning_rate: float = 3.0e-4
    update_epochs: int = 4
    target_kl: float | None = 0.02
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    cost_normalization: Stage1CostNormalization = Stage1CostNormalization()
    penalty_weights: Stage1PenaltyWeights = Stage1PenaltyWeights()
    scenario_distribution: CompactFleetScenarioDistribution = (
        CompactFleetScenarioDistribution()
    )


@dataclass(frozen=True)
class Stage1SeedSplit:
    training: tuple[int, ...] = tuple(range(64))
    validation: tuple[int, ...] = tuple(range(100, 116))
    test: tuple[int, ...] = tuple(range(200, 216))

    def validate(self) -> None:
        groups = (self.training, self.validation, self.test)
        if any(not group or len(set(group)) != len(group) for group in groups):
            raise ValueError("Stage 1 seed groups must be unique and nonempty.")
        if any(set(left) & set(right) for index, left in enumerate(groups)
               for right in groups[index + 1:]):
            raise ValueError("Stage 1 train/validation/test seeds must be disjoint.")


@dataclass(frozen=True)
class RandomizedPhysicalConditionSplit:
    """Frozen condition seeds for the five-node physical curriculum.

    Conditions 112--119 were inspected during curriculum pre-scan and are
    therefore explicitly quarantined from fitting and confirmation.
    """

    prescan: tuple[int, ...] = tuple(range(112, 120))
    training: tuple[int, ...] = tuple(range(200, 224))
    selection: tuple[int, ...] = tuple(range(224, 232))
    development: tuple[int, ...] = tuple(range(232, 240))
    formal_confirmation: tuple[int, ...] = tuple(range(240, 248))

    def validate(self) -> None:
        groups = (
            self.prescan,
            self.training,
            self.selection,
            self.development,
            self.formal_confirmation,
        )
        if any(not group or len(set(group)) != len(group) for group in groups):
            raise ValueError("Physical condition groups must be unique and nonempty.")
        if any(
            set(left) & set(right)
            for index, left in enumerate(groups)
            for right in groups[index + 1:]
        ):
            raise ValueError("Physical condition groups must be mutually disjoint.")


FIVE_NODE_STAGE1_DISTRIBUTION = CompactFleetScenarioDistribution(
    packet_loss_range=(0.0, 0.2),
    communication_delay_range=(0.0, 2.0),
    navigation_dropout_node_count=1,
    initial_topology_types=("chain", "ring", "star"),
)

FIVE_NODE_HETEROGENEOUS_LINK_DISTRIBUTION = CompactFleetScenarioDistribution(
    packet_loss_range=(0.0, 0.2),
    communication_delay_range=(0.0, 2.0),
    navigation_dropout_node_count=1,
    initial_topology_types=("chain", "ring", "star"),
    link_condition_mode="undirected_independent",
)

FIVE_NODE_RANDOMIZED_PHYSICAL_DISTRIBUTION = CompactFleetScenarioDistribution(
    packet_loss_range=(0.0, 0.2),
    communication_delay_range=(0.0, 2.0),
    navigation_dropout_node_count=1,
    initial_topology_types=("chain", "ring", "star"),
    link_condition_mode="undirected_independent",
    physical_scenario_families=(
        "compact_along_track",
        "differential_along_track",
        "two_plane_cluster",
    ),
)

FIVE_NODE_STRATIFIED_PHYSICAL_DISTRIBUTION = replace(
    FIVE_NODE_RANDOMIZED_PHYSICAL_DISTRIBUTION,
    physical_family_assignment_mode="seed_cycle",
)


def five_node_stage1_configuration(**changes) -> Stage1Configuration:
    """Return the frozen five-node distribution baseline for PPO pilots."""

    baseline = Stage1Configuration(
        node_count=5,
        top_k_candidate_neighbors=2,
        scenario_distribution=FIVE_NODE_STAGE1_DISTRIBUTION,
    )
    return replace(baseline, **changes)


def five_node_heterogeneous_link_configuration(**changes) -> Stage1Configuration:
    """Return a link-informative condition distribution without changing V15 I/O."""

    baseline = five_node_stage1_configuration(
        scenario_distribution=FIVE_NODE_HETEROGENEOUS_LINK_DISTRIBUTION,
    )
    return replace(baseline, **changes)


def five_node_randomized_physical_configuration(**changes) -> Stage1Configuration:
    """Return the first geometry- and link-randomized five-node curriculum."""

    baseline = five_node_stage1_configuration(
        scenario_distribution=FIVE_NODE_RANDOMIZED_PHYSICAL_DISTRIBUTION,
    )
    return replace(baseline, **changes)


def five_node_stratified_physical_configuration(**changes) -> Stage1Configuration:
    """Return the frozen, family-balanced physical training curriculum."""

    baseline = five_node_stage1_configuration(
        scenario_distribution=FIVE_NODE_STRATIFIED_PHYSICAL_DISTRIBUTION,
    )
    return replace(baseline, **changes)


def five_node_stratified_physical_ppo_configuration(
    **changes,
) -> Stage1Configuration:
    """Return the frozen PPO budget for the 24-by-4 physical curriculum."""

    baseline = five_node_stratified_physical_configuration(
        training_episodes=96,
        episode_epochs=6,
        decision_interval_epochs=2,
        environment_seed_count=4,
        environment_seed_offset=0,
        condition_seed_offset=200,
        condition_seed_count=24,
        rollout_batch_episodes=32,
        minibatch_size=16,
        update_epochs=4,
        policy_seed=0,
        learning_rate=1.0e-4,
        maximum_topology_switches_per_episode=1,
    )
    return replace(baseline, **changes)


def five_node_robust_ppo_configuration(**changes) -> Stage1Configuration:
    """Return the accepted low-variance PPO pilot configuration."""

    baseline = five_node_stage1_configuration(
        training_episodes=64,
        episode_epochs=6,
        decision_interval_epochs=2,
        environment_seed_count=8,
        condition_seed_offset=40,
        condition_seed_count=4,
        rollout_batch_episodes=32,
        minibatch_size=16,
        update_epochs=4,
        policy_seed=0,
        learning_rate=1.0e-4,
        maximum_topology_switches_per_episode=1,
    )
    return replace(baseline, **changes)


@dataclass(frozen=True)
class Stage1ActionKindDiagnostic:
    action_kind: str
    transition_count: int
    actor_transition_count: int
    mean_advantage: float
    positive_advantage_fraction: float
    mean_task_reward: float
    mean_penalized_reward: float
    mean_transmitted_messages: float
    mean_resynchronization_count: float
    mean_topology_switch: float


@dataclass(frozen=True)
class Stage1EpisodeDiagnostic:
    episode: int
    environment_seed: int
    condition_seed: int
    task_return: float
    penalized_return: float
    final_position_rmse: float
    transmitted_messages: float
    dropped_messages: float
    replay_count: float
    resynchronization_count: float
    topology_switches: float
    type_entropy: float
    conditional_entropy: float
    initial_type_probabilities: tuple[float, ...]
    action_kind_diagnostics: tuple[Stage1ActionKindDiagnostic, ...]
    update: PPOUpdateResult


@dataclass(frozen=True)
class Stage1BatchActionDiagnostic:
    action_kind: str
    transition_count: int
    actor_transition_count: int
    mean_normalized_advantage: float
    positive_normalized_advantage_fraction: float


@dataclass(frozen=True)
class Stage1BatchDiagnostic:
    batch_start: int
    batch_end: int
    type_probabilities_before_update: tuple[float, ...]
    type_probabilities_after_update: tuple[float, ...]
    action_diagnostics: tuple[Stage1BatchActionDiagnostic, ...]
    update: PPOUpdateResult


@dataclass(frozen=True)
class Stage1TrainingResult:
    model: TopologyActorCritic
    diagnostics: tuple[Stage1EpisodeDiagnostic, ...]
    batch_diagnostics: tuple[Stage1BatchDiagnostic, ...] = ()


@dataclass(frozen=True)
class Stage1EvaluationRecord:
    policy_name: str
    environment_seed: int
    task_return: float
    penalized_return: float
    final_position_rmse: float
    transmitted_messages: float
    dropped_messages: float
    replay_count: float
    resynchronization_count: float
    topology_switches: float
    fallback_count: float


@dataclass(frozen=True)
class Stage1EvaluationResult:
    records: tuple[Stage1EvaluationRecord, ...]

    def records_for(self, policy_name: str):
        return tuple(
            record for record in self.records if record.policy_name == policy_name
        )


@dataclass(frozen=True)
class Stage1PenaltySensitivity:
    scale: float
    mean_penalized_return_by_policy: tuple[tuple[str, float], ...]
    best_policy: str


@dataclass(frozen=True)
class Stage1TrainingSeedRecord:
    policy_seed: int
    initialization: str
    mean_final_position_rmse: float
    mean_task_return: float
    mean_penalized_return: float
    mean_transmitted_messages: float
    mean_topology_switches: float
    mean_resynchronization_count: float
    beats_keep_rmse: bool
    beats_keep_penalized_return: bool


def build_stage1_environment(configuration: Stage1Configuration):
    return TopologyControlEnvironment(
        node_count=configuration.node_count,
        episode_epochs=configuration.episode_epochs,
        decision_interval_epochs=configuration.decision_interval_epochs,
        relative_modalities=("RANGE",),
        minimum_topology_dwell_decisions=(
            configuration.minimum_topology_dwell_decisions
        ),
        maximum_topology_switches_per_episode=(
            configuration.maximum_topology_switches_per_episode
        ),
        randomize_stage1_conditions=True,
        top_k_candidate_neighbors=configuration.top_k_candidate_neighbors,
        compact_scenario_distribution=configuration.scenario_distribution,
        scenario_type=configuration.scenario_type,
        walker_maximum_range=configuration.walker_maximum_range,
        walker_plane_count=configuration.walker_plane_count,
        walker_phasing=configuration.walker_phasing,
    )


def train_stage1_ppo(
    configuration: Stage1Configuration = Stage1Configuration(),
    *, warm_start_checkpoint: str | None = None,
    reset_warm_start_type_head: bool = True,
) -> Stage1TrainingResult:
    """Train multi-decision PPO over reproducible randomized fault episodes."""

    _validate_configuration(configuration)
    torch.manual_seed(configuration.policy_seed)
    environment = build_stage1_environment(configuration)
    initial_condition_seed = (
        configuration.environment_seed_offset
        if configuration.condition_seed_offset is None
        else configuration.condition_seed_offset
    )
    state = environment.reset(
        seed=configuration.environment_seed_offset,
        condition_seed=initial_condition_seed,
    )
    snapshot, _ = build_online_snapshot_action_tensor(state)
    group = torch_snapshot_action_group(snapshot)
    model = (
        build_warm_started_actor_critic(
            warm_start_checkpoint,
            node_feature_count=group.node_features.shape[1],
            reset_type_head=reset_warm_start_type_head,
        )
        if warm_start_checkpoint is not None
        else TopologyActorCritic(
            node_feature_count=group.node_features.shape[1],
            candidate_edge_feature_count=group.candidate_edge_features.shape[1],
            measurement_feature_count=group.measurement_features.shape[1],
            action_feature_count=group.action_features.shape[1],
            global_feature_count=len(state.policy_tensor.global_feature_names),
            hidden_size=32, message_passing_steps=2,
            explicit_action_pairing=False,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    generator = torch.Generator().manual_seed(configuration.policy_seed + 2000)
    diagnostics, batch_diagnostics = [], []
    for batch_start in range(
        0, configuration.training_episodes,
        configuration.rollout_batch_episodes,
    ):
        batch = []
        batch_metadata = []
        batch_end = min(
            configuration.training_episodes,
            batch_start + configuration.rollout_batch_episodes,
        )
        for episode in range(batch_start, batch_end):
            environment_seed = (
                configuration.environment_seed_offset
                + episode % configuration.environment_seed_count
            )
            condition_seed = _stage1_condition_seed(configuration, episode)
            rollout = collect_topology_rollout(
                environment, model, seed=environment_seed,
                condition_seed=condition_seed, generator=generator,
            )
            penalized = apply_stage1_penalties(rollout, configuration)
            prepared = prepare_topology_rollout(
                penalized, gamma=configuration.gamma,
                gae_lambda=configuration.gae_lambda,
                normalize_advantages=False,
            )
            batch.append(prepared)
            batch_metadata.append((
                episode, environment_seed, condition_seed, rollout, penalized,
                prepared,
                float(environment._metrics()[0]),
            ))
        combined = combine_prepared_topology_rollouts(tuple(batch))
        probabilities_before = _stage1_mean_type_probabilities(model, combined)
        action_diagnostics = _stage1_batch_action_diagnostics(combined)
        update = update_topology_ppo(
            model, optimizer,
            combined,
            update_epochs=configuration.update_epochs,
            entropy_coefficient=configuration.entropy_coefficient,
            target_kl=configuration.target_kl,
            minibatch_size=configuration.minibatch_size,
            generator=generator,
        )
        batch_diagnostics.append(Stage1BatchDiagnostic(
            batch_start=batch_start,
            batch_end=batch_end,
            type_probabilities_before_update=probabilities_before,
            type_probabilities_after_update=_stage1_mean_type_probabilities(
                model, combined,
            ),
            action_diagnostics=action_diagnostics,
            update=update,
        ))
        for (
            episode, environment_seed, condition_seed, rollout, penalized,
            prepared,
            final_rmse
        ) in batch_metadata:
            costs = rollout.cost_matrix.sum(dim=0)
            entropies = np.asarray([
                (transition.type_entropy, transition.conditional_entropy)
                for transition in rollout.transitions
            ])
            initial_types = model(
                rollout.transitions[0].group
            ).distribution.type_probabilities.detach().cpu().numpy()
            diagnostics.append(Stage1EpisodeDiagnostic(
                episode=episode, environment_seed=environment_seed,
                condition_seed=condition_seed,
                task_return=float(rollout.rewards.sum().item()),
                penalized_return=float(penalized.rewards.sum().item()),
                final_position_rmse=final_rmse,
                transmitted_messages=float(costs[0]), dropped_messages=float(costs[1]),
                replay_count=float(costs[2]), resynchronization_count=float(costs[3]),
                topology_switches=float(costs[4]),
                type_entropy=float(entropies[:, 0].mean()),
                conditional_entropy=float(entropies[:, 1].mean()),
                initial_type_probabilities=tuple(float(value) for value in initial_types),
                action_kind_diagnostics=_stage1_action_kind_diagnostics(
                    rollout, penalized, prepared,
                ),
                update=update,
            ))
    return Stage1TrainingResult(
        model=model,
        diagnostics=tuple(diagnostics),
        batch_diagnostics=tuple(batch_diagnostics),
    )


def _stage1_mean_type_probabilities(model, prepared):
    indices = [
        index for index, allowed in enumerate(prepared.actor_mask)
        if bool(allowed)
    ]
    if not indices:
        return (1.0, 0.0, 0.0, 0.0)
    with torch.no_grad():
        probabilities = torch.stack(tuple(
            model(prepared.transitions[index].group).distribution.type_probabilities
            for index in indices
        ))
    return tuple(float(value) for value in probabilities.mean(dim=0))


def _stage1_batch_action_diagnostics(prepared):
    names = ("keep", "add", "swap", "remove")
    records = []
    for kind_index, name in enumerate(names):
        indices = [
            index for index, transition in enumerate(prepared.transitions)
            if int(transition.group.action_kind_index[
                transition.action_index
            ].item()) == kind_index
        ]
        if not indices:
            continue
        advantages = prepared.advantages[indices].detach().cpu().numpy()
        records.append(Stage1BatchActionDiagnostic(
            action_kind=name,
            transition_count=len(indices),
            actor_transition_count=sum(
                bool(prepared.actor_mask[index]) for index in indices
            ),
            mean_normalized_advantage=float(np.mean(advantages)),
            positive_normalized_advantage_fraction=float(np.mean(
                advantages > 0.0
            )),
        ))
    return tuple(records)


def _stage1_action_kind_diagnostics(rollout, penalized, prepared):
    """Summarize causal rewards, costs, and GAE by executed action type."""

    if not (
        len(rollout.transitions)
        == len(penalized.transitions)
        == len(prepared.transitions)
        == len(prepared.advantages)
    ):
        raise ValueError("Stage 1 diagnostic rollouts must align.")
    names = ("keep", "add", "swap", "remove")
    records = []
    for kind_index, name in enumerate(names):
        indices = [
            index for index, transition in enumerate(rollout.transitions)
            if int(transition.group.action_kind_index[
                transition.action_index
            ].item()) == kind_index
        ]
        if not indices:
            continue
        advantages = prepared.advantages[indices].detach().cpu().numpy()
        costs = np.asarray([
            rollout.transitions[index].costs for index in indices
        ], dtype=float)
        records.append(Stage1ActionKindDiagnostic(
            action_kind=name,
            transition_count=len(indices),
            actor_transition_count=sum(
                bool(prepared.actor_mask[index]) for index in indices
            ),
            mean_advantage=float(np.mean(advantages)),
            positive_advantage_fraction=float(np.mean(advantages > 0.0)),
            mean_task_reward=float(np.mean([
                rollout.transitions[index].reward for index in indices
            ])),
            mean_penalized_reward=float(np.mean([
                penalized.transitions[index].reward for index in indices
            ])),
            mean_transmitted_messages=float(np.mean(costs[:, 0])),
            mean_resynchronization_count=float(np.mean(costs[:, 3])),
            mean_topology_switch=float(np.mean(costs[:, 4])),
        ))
    return tuple(records)


def apply_stage1_penalties(rollout, configuration):
    scales, weights = configuration.cost_normalization, configuration.penalty_weights
    transitions = tuple(replace(
        transition,
        reward=(
            transition.reward
            - weights.communication
            * transition.costs[0] / scales.transmitted_messages
            - weights.topology_switch
            * transition.costs[4] / scales.topology_switch
            - weights.resynchronization
            * transition.costs[3] / scales.resynchronization
        ),
    ) for transition in rollout.transitions)
    return TopologyRollout(transitions, rollout.final_value)


def evaluate_stage1_policies(
    configuration: Stage1Configuration,
    *, test_seeds,
    random_model: TopologyActorCritic,
    warm_start_model: TopologyActorCritic,
) -> Stage1EvaluationResult:
    """Evaluate all policies deterministically on identical unseen conditions."""

    seeds = tuple(int(seed) for seed in test_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Stage 1 test seeds must be unique and nonempty.")
    training_seeds = {
        configuration.environment_seed_offset + index
        for index in range(configuration.environment_seed_count)
    }
    if training_seeds.intersection(seeds):
        raise ValueError("Stage 1 evaluation seeds must be unseen during training.")
    records = []
    for seed in seeds:
        for policy in (AlwaysKeepPolicy(), InformationGreedyPolicy()):
            summary = run_topology_control_baseline_episode(
                build_stage1_environment(configuration), policy, seed=seed,
            )
            costs = summary.cumulative_costs
            records.append(_evaluation_record(
                policy.name, seed, summary.cumulative_reward,
                summary.final_position_rmse, costs, configuration,
            ))
        for name, model in (
            ("ppo_random_init", random_model),
            ("ppo_warm_start", warm_start_model),
        ):
            environment = build_stage1_environment(configuration)
            rollout = collect_topology_rollout(
                environment, model, seed=seed, deterministic=True,
            )
            costs = rollout.cost_matrix.sum(dim=0)
            records.append(Stage1EvaluationRecord(
                policy_name=name, environment_seed=seed,
                task_return=float(rollout.rewards.sum()),
                penalized_return=float(
                    apply_stage1_penalties(rollout, configuration).rewards.sum()
                ),
                final_position_rmse=float(environment._metrics()[0]),
                transmitted_messages=float(costs[0]),
                dropped_messages=float(costs[1]), replay_count=float(costs[2]),
                resynchronization_count=float(costs[3]),
                topology_switches=float(costs[4]), fallback_count=float(costs[5]),
            ))
    return Stage1EvaluationResult(tuple(records))


def scan_stage1_penalty_sensitivity(
    evaluation: Stage1EvaluationResult,
    configuration: Stage1Configuration,
    *, scales=(0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
) -> tuple[Stage1PenaltySensitivity, ...]:
    """Re-score fixed trajectories without rerunning or changing policies."""

    requested = tuple(float(scale) for scale in scales)
    if not requested or any(not np.isfinite(scale) or scale < 0.0
                            for scale in requested):
        raise ValueError("Penalty sensitivity scales must be finite/nonnegative.")
    names = tuple(sorted({record.policy_name for record in evaluation.records}))
    results = []
    for scale in requested:
        values = []
        for name in names:
            records = evaluation.records_for(name)
            scores = tuple(
                record.task_return
                - scale * _record_penalty(record, configuration)
                for record in records
            )
            values.append((name, float(np.mean(scores))))
        best = max(values, key=lambda item: (item[1], item[0]))[0]
        results.append(Stage1PenaltySensitivity(scale, tuple(values), best))
    return tuple(results)


def compare_stage1_training_seeds(
    base_configuration: Stage1Configuration,
    *, policy_seeds, test_seeds, warm_start_checkpoint: str,
) -> tuple[Stage1TrainingSeedRecord, ...]:
    """Train paired initializations and evaluate each on identical unseen seeds."""

    seeds = tuple(int(seed) for seed in policy_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Stage 1 policy seeds must be unique and nonempty.")
    records = []
    for seed in seeds:
        configuration = replace(base_configuration, policy_seed=seed)
        random_result = train_stage1_ppo(configuration)
        warm_result = train_stage1_ppo(
            configuration, warm_start_checkpoint=warm_start_checkpoint,
        )
        evaluation = evaluate_stage1_policies(
            configuration, test_seeds=test_seeds,
            random_model=random_result.model,
            warm_start_model=warm_result.model,
        )
        keep = evaluation.records_for("always_keep")
        keep_rmse = float(np.mean([item.final_position_rmse for item in keep]))
        keep_penalized = float(np.mean([
            item.penalized_return for item in keep
        ]))
        for initialization, policy_name in (
            ("random", "ppo_random_init"),
            ("warm_start", "ppo_warm_start"),
        ):
            selected = evaluation.records_for(policy_name)
            means = {
                name: float(np.mean([getattr(item, name) for item in selected]))
                for name in (
                    "final_position_rmse", "task_return", "penalized_return",
                    "transmitted_messages", "topology_switches",
                    "resynchronization_count",
                )
            }
            records.append(Stage1TrainingSeedRecord(
                policy_seed=seed, initialization=initialization,
                mean_final_position_rmse=means["final_position_rmse"],
                mean_task_return=means["task_return"],
                mean_penalized_return=means["penalized_return"],
                mean_transmitted_messages=means["transmitted_messages"],
                mean_topology_switches=means["topology_switches"],
                mean_resynchronization_count=means["resynchronization_count"],
                beats_keep_rmse=(
                    means["final_position_rmse"] < keep_rmse - 1.0e-6
                ),
                beats_keep_penalized_return=(
                    means["penalized_return"] > keep_penalized + 1.0e-6
                ),
            ))
    return tuple(records)


def _evaluation_record(name, seed, task_return, rmse, costs, configuration):
    scales, weights = configuration.cost_normalization, configuration.penalty_weights
    penalty = (
        weights.communication * costs.transmitted_messages
        / scales.transmitted_messages
        + weights.topology_switch * costs.topology_switch
        / scales.topology_switch
        + weights.resynchronization * costs.resynchronization_count
        / scales.resynchronization
    )
    return Stage1EvaluationRecord(
        policy_name=name, environment_seed=seed, task_return=float(task_return),
        penalized_return=float(task_return - penalty),
        final_position_rmse=float(rmse),
        transmitted_messages=costs.transmitted_messages,
        dropped_messages=costs.dropped_messages,
        replay_count=costs.replay_count,
        resynchronization_count=costs.resynchronization_count,
        topology_switches=costs.topology_switch,
        fallback_count=costs.action_fallback,
    )


def _record_penalty(record, configuration):
    scales, weights = configuration.cost_normalization, configuration.penalty_weights
    return (
        weights.communication * record.transmitted_messages
        / scales.transmitted_messages
        + weights.topology_switch * record.topology_switches
        / scales.topology_switch
        + weights.resynchronization * record.resynchronization_count
        / scales.resynchronization
    )


def _validate_configuration(configuration):
    positive = (
        configuration.training_episodes, configuration.episode_epochs,
        configuration.decision_interval_epochs,
        configuration.environment_seed_count,
        configuration.rollout_batch_episodes,
        configuration.minibatch_size,
        configuration.cost_normalization.transmitted_messages,
        configuration.cost_normalization.topology_switch,
        configuration.cost_normalization.resynchronization,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Stage 1 horizons, seed count, and cost scales must be positive.")
    if configuration.scenario_type == "compact_fleet":
        if configuration.node_count not in {3, 5}:
            raise ValueError("Stage 1 compact-fleet training supports 3 or 5 nodes.")
    elif configuration.scenario_type == "walker_delta":
        if (
            configuration.node_count < 2
            or configuration.walker_plane_count < 1
            or configuration.node_count % configuration.walker_plane_count
        ):
            raise ValueError(
                "Stage 1 Walker training requires nodes divisible by planes."
            )
    else:
        raise ValueError("Unsupported Stage 1 scenario type.")
    if configuration.target_kl is not None and configuration.target_kl <= 0.0:
        raise ValueError("Stage 1 target KL must be positive when enabled.")
    if (
        configuration.top_k_candidate_neighbors is not None
        and configuration.top_k_candidate_neighbors < 0
    ):
        raise ValueError("Stage 1 Top-K candidate count cannot be negative.")
    if (
        configuration.maximum_topology_switches_per_episode is not None
        and configuration.maximum_topology_switches_per_episode < 0
    ):
        raise ValueError("Stage 1 topology-switch budget cannot be negative.")
    if (
        configuration.condition_seed_count is not None
        and configuration.condition_seed_count <= 0
    ):
        raise ValueError("Stage 1 condition-seed count must be positive.")
    if (
        (configuration.condition_seed_offset is None)
        != (configuration.condition_seed_count is None)
    ):
        raise ValueError(
            "Stage 1 condition-seed offset and count must be set together."
        )
    configuration.scenario_distribution.validate(configuration.node_count)
    weights = configuration.penalty_weights
    if min(weights.communication, weights.topology_switch, weights.resynchronization) < 0:
        raise ValueError("Stage 1 penalty weights must be nonnegative.")


def _stage1_condition_seed(configuration, episode):
    if configuration.condition_seed_offset is None:
        return (
            configuration.environment_seed_offset
            + episode % configuration.environment_seed_count
        )
    return (
        configuration.condition_seed_offset
        + (episode // configuration.environment_seed_count)
        % configuration.condition_seed_count
    )
