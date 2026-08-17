from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as functional

from experiments.graph_action_gnn import (
    GraphActionValueNetwork,
    TorchGraphActionGroup,
)


@dataclass(frozen=True)
class HierarchicalActionDistribution:
    """Masked P(action kind) P(action | kind) for one graph snapshot."""

    action_log_probabilities: Tensor
    type_probabilities: Tensor
    type_entropy: Tensor
    conditional_entropy: Tensor

    @property
    def entropy(self) -> Tensor:
        return self.type_entropy + self.conditional_entropy

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        probabilities = self.action_log_probabilities.exp()
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(0)

    def mode(self) -> Tensor:
        return self.action_log_probabilities.argmax()

    def log_prob(self, action_index: Tensor | int) -> Tensor:
        return self.action_log_probabilities[action_index]


@dataclass(frozen=True)
class PPOLoss:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


@dataclass(frozen=True)
class ActorCriticOutput:
    distribution: HierarchicalActionDistribution
    value: Tensor


@dataclass(frozen=True)
class TopologyRolloutTransition:
    """One causal PPO transition; no truth state is stored in policy inputs."""

    group: TorchGraphActionGroup
    action_index: int
    environment_action_id: int
    old_log_probability: float
    value: float
    reward: float
    costs: tuple[float, ...]
    terminated: bool
    truncated: bool
    type_entropy: float
    conditional_entropy: float


@dataclass(frozen=True)
class TopologyRollout:
    transitions: tuple[TopologyRolloutTransition, ...]
    final_value: float

    @property
    def rewards(self) -> Tensor:
        return torch.tensor(
            [transition.reward for transition in self.transitions],
            dtype=torch.float32,
        )

    @property
    def cost_matrix(self) -> Tensor:
        if not self.transitions:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.tensor(
            [transition.costs for transition in self.transitions],
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class PreparedTopologyRollout:
    transitions: tuple[TopologyRolloutTransition, ...]
    old_log_probabilities: Tensor
    advantages: Tensor
    returns: Tensor
    actor_mask: Tensor


@dataclass(frozen=True)
class PPOUpdateResult:
    epochs_run: int
    final_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    stopped_early: bool
    transition_count: int
    actor_transition_count: int
    advantage_mean: float
    advantage_standard_deviation: float
    positive_advantage_fraction: float
    explained_variance_before_update: float


class TopologyActorCritic(nn.Module):
    """Warm-start-compatible hierarchical Actor with an independent Critic."""

    def __init__(
        self, *, node_feature_count: int, candidate_edge_feature_count: int,
        measurement_feature_count: int, action_feature_count: int,
        global_feature_count: int, hidden_size: int = 64,
        message_passing_steps: int = 2, explicit_action_pairing: bool = True,
    ) -> None:
        super().__init__()
        self.global_feature_count = int(global_feature_count)
        self.actor = GraphActionValueNetwork(
            node_feature_count=node_feature_count,
            candidate_edge_feature_count=candidate_edge_feature_count,
            measurement_feature_count=measurement_feature_count,
            action_feature_count=action_feature_count,
            hidden_size=hidden_size,
            message_passing_steps=message_passing_steps,
            explicit_action_pairing=explicit_action_pairing,
        )
        critic_input = (
            2 * node_feature_count + 2 * candidate_edge_feature_count
            + global_feature_count
        )
        self.critic = nn.Sequential(
            nn.Linear(critic_input, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, group: TorchGraphActionGroup) -> ActorCriticOutput:
        actor = self.actor(group)
        if actor.type_logits is None or group.action_kind_index is None:
            raise ValueError("PPO requires hierarchical online action groups.")
        legal_mask = torch.ones_like(actor.utility, dtype=torch.bool)
        distribution = hierarchical_action_distribution(
            actor.type_logits, actor.utility, group.action_kind_index, legal_mask,
        )
        if group.action_features.shape[1] < self.global_feature_count:
            raise ValueError("Online action features omit required action fields.")
        global_features = (
            group.action_features[0, -self.global_feature_count:]
            if self.global_feature_count else group.action_features.new_empty((0,))
        )
        critic_features = torch.cat((
            _mean_or_zeros(group.node_features),
            _max_or_zeros(group.node_features),
            _mean_or_zeros(group.candidate_edge_features),
            _max_or_zeros(group.candidate_edge_features),
            global_features,
        ))
        return ActorCriticOutput(
            distribution=distribution,
            value=self.critic(critic_features).squeeze(0),
        )

    def load_warm_start_actor(self, state_dict: dict[str, Tensor]) -> None:
        """Load the existing supervised GraphActionValueNetwork exactly."""

        self.actor.load_state_dict(state_dict)


def build_warm_started_actor_critic(
    checkpoint_path: str | Path,
    *, node_feature_count: int | None = None, reset_type_head: bool = False,
) -> TopologyActorCritic:
    """Build PPO Actor/Critic while preserving a supervised hierarchical Actor."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    configuration = checkpoint["configuration"]
    if configuration.get("loss_mode") != "hierarchical":
        raise ValueError("PPO warm start requires a hierarchical checkpoint.")
    checkpoint_node_features = int(checkpoint["node_feature_count"])
    requested_node_features = int(
        node_feature_count
        if node_feature_count is not None else checkpoint_node_features
    )
    if requested_node_features < checkpoint_node_features:
        raise ValueError("Warm-start node schema cannot remove checkpoint fields.")
    model = TopologyActorCritic(
        node_feature_count=requested_node_features,
        candidate_edge_feature_count=int(checkpoint["edge_feature_count"]),
        measurement_feature_count=int(checkpoint["edge_feature_count"]),
        action_feature_count=int(checkpoint["action_feature_count"]),
        global_feature_count=int(checkpoint["global_feature_count"]),
        hidden_size=int(configuration["hidden_size"]),
        message_passing_steps=int(configuration["message_passing_steps"]),
        explicit_action_pairing=bool(
            configuration.get("explicit_action_pairing", False)
        ),
    )
    actor_state = checkpoint["model_state_dict"]
    if requested_node_features != checkpoint_node_features:
        actor_state = _expand_node_encoder_input(
            actor_state, requested_node_features
        )
    model.load_warm_start_actor(actor_state)
    if reset_type_head:
        nn.init.zeros_(model.actor.type_head.weight)
        nn.init.zeros_(model.actor.type_head.bias)
    return model


def collect_topology_rollout(
    environment,
    model: TopologyActorCritic,
    *,
    seed: int,
    condition_seed: int | None = None,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> TopologyRollout:
    """Collect one episode from the existing truth-safe topology environment."""

    from experiments.graph_action_gnn import torch_snapshot_action_group
    from experiments.topology_snapshot_counterfactual import (
        build_online_snapshot_action_tensor,
    )

    state = (
        environment.reset(seed=seed)
        if condition_seed is None
        else environment.reset(seed=seed, condition_seed=condition_seed)
    )
    transitions = []
    while True:
        snapshot, action_ids = build_online_snapshot_action_tensor(state)
        group = torch_snapshot_action_group(snapshot)
        with torch.no_grad():
            output = model(group)
            selected = (
                output.distribution.mode() if deterministic
                else output.distribution.sample(generator=generator)
            )
        action_index = int(selected.item())
        environment_action_id = int(action_ids[action_index])
        step = environment.step(environment_action_id)
        costs = step.constraint_costs
        transitions.append(TopologyRolloutTransition(
            group=group, action_index=action_index,
            environment_action_id=environment_action_id,
            old_log_probability=float(
                output.distribution.log_prob(selected).item()
            ),
            value=float(output.value.item()), reward=float(step.reward),
            costs=tuple(float(value) for value in (
                costs.transmitted_messages, costs.dropped_messages,
                costs.replay_count, costs.resynchronization_count,
                costs.topology_switch, costs.action_fallback,
            )),
            terminated=bool(step.terminated), truncated=bool(step.truncated),
            type_entropy=float(output.distribution.type_entropy.item()),
            conditional_entropy=float(
                output.distribution.conditional_entropy.item()
            ),
        ))
        state = step.state
        if step.terminated or step.truncated:
            break
    final_value = 0.0
    if transitions and transitions[-1].truncated:
        snapshot, _ = build_online_snapshot_action_tensor(state)
        with torch.no_grad():
            final_value = float(
                model(torch_snapshot_action_group(snapshot)).value.item()
            )
    return TopologyRollout(tuple(transitions), final_value)


def prepare_topology_rollout(
    rollout: TopologyRollout,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> PreparedTopologyRollout:
    """Attach GAE and value targets to one chronological environment rollout."""

    if not rollout.transitions:
        raise ValueError("Cannot prepare an empty topology rollout.")
    values = torch.tensor(
        [transition.value for transition in rollout.transitions],
        dtype=torch.float32,
    )
    next_values = torch.cat((
        values[1:], torch.tensor([rollout.final_value], dtype=values.dtype)
    ))
    terminated = torch.tensor(
        [transition.terminated for transition in rollout.transitions],
        dtype=torch.bool,
    )
    advantages, returns = generalized_advantage_estimate(
        rollout.rewards, values, next_values, terminated,
        gamma=gamma, gae_lambda=gae_lambda,
    )
    if normalize_advantages and len(advantages) > 1:
        scale = advantages.std(unbiased=False)
        advantages = (advantages - advantages.mean()) / scale.clamp_min(1e-8)
    return PreparedTopologyRollout(
        transitions=rollout.transitions,
        old_log_probabilities=torch.tensor(
            [transition.old_log_probability for transition in rollout.transitions],
            dtype=torch.float32,
        ),
        advantages=advantages,
        returns=returns,
        actor_mask=torch.tensor([
            len(transition.group.action_features) > 1
            for transition in rollout.transitions
        ], dtype=torch.bool),
    )


def combine_prepared_topology_rollouts(
    prepared_rollouts: tuple[PreparedTopologyRollout, ...],
    *, normalize_advantages: bool = True,
) -> PreparedTopologyRollout:
    """Combine complete episodes without allowing GAE across boundaries."""

    if not prepared_rollouts:
        raise ValueError("At least one prepared rollout is required.")
    combined = PreparedTopologyRollout(
        transitions=tuple(
            transition for rollout in prepared_rollouts
            for transition in rollout.transitions
        ),
        old_log_probabilities=torch.cat(tuple(
            rollout.old_log_probabilities for rollout in prepared_rollouts
        )),
        advantages=torch.cat(tuple(
            rollout.advantages for rollout in prepared_rollouts
        )),
        returns=torch.cat(tuple(rollout.returns for rollout in prepared_rollouts)),
        actor_mask=torch.cat(tuple(
            rollout.actor_mask for rollout in prepared_rollouts
        )),
    )
    advantages = combined.advantages
    if normalize_advantages and len(advantages) > 1:
        advantages = (
            advantages - advantages.mean()
        ) / advantages.std(unbiased=False).clamp_min(1e-8)
    return PreparedTopologyRollout(
        combined.transitions, combined.old_log_probabilities,
        advantages, combined.returns, combined.actor_mask,
    )


def update_topology_ppo(
    model: TopologyActorCritic,
    optimizer: torch.optim.Optimizer,
    prepared: PreparedTopologyRollout,
    *,
    update_epochs: int = 4,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    maximum_gradient_norm: float = 0.5,
    target_kl: float | None = 0.02,
    minibatch_size: int | None = None,
    generator: torch.Generator | None = None,
) -> PPOUpdateResult:
    """Update on variable-size graph transitions and report PPO diagnostics."""

    if update_epochs < 1 or maximum_gradient_norm <= 0.0:
        raise ValueError("PPO epochs and maximum gradient norm must be positive.")
    if not prepared.transitions:
        raise ValueError("Cannot update PPO from an empty rollout.")
    if target_kl is not None and target_kl <= 0.0:
        raise ValueError("PPO target KL must be positive when enabled.")
    expected = len(prepared.transitions)
    if any(len(values) != expected for values in (
        prepared.old_log_probabilities, prepared.advantages, prepared.returns,
        prepared.actor_mask,
    )):
        raise ValueError("Prepared PPO tensors must align with transitions.")

    with torch.no_grad():
        initial_values = torch.stack(tuple(
            model(transition.group).value
            for transition in prepared.transitions
        ))
        explained_variance = _explained_variance(
            prepared.returns, initial_values
        )
    actor_advantages = prepared.advantages[prepared.actor_mask]
    if minibatch_size is not None and minibatch_size < 1:
        raise ValueError("PPO minibatch size must be positive when enabled.")
    final = None
    gradient_norm = 0.0
    stopped_early = False
    for epoch in range(update_epochs):
        order = torch.randperm(expected, generator=generator)
        size = minibatch_size or expected
        for start in range(0, expected, size):
            indices = order[start:start + size]
            actor_indices = indices[prepared.actor_mask[indices]]
            outputs = {
                int(index): model(prepared.transitions[int(index)].group)
                for index in indices
            }
            predicted_values = torch.stack(tuple(
                outputs[int(index)].value for index in indices
            ))
            value_loss = functional.mse_loss(
                predicted_values, prepared.returns[indices]
            )
            if len(actor_indices):
                actor_outputs = tuple(
                    outputs[int(index)] for index in actor_indices
                )
                new_log_probabilities = torch.stack(tuple(
                    output.distribution.log_prob(
                        prepared.transitions[int(index)].action_index
                    )
                    for output, index in zip(actor_outputs, actor_indices)
                ))
                actor_values = torch.stack(tuple(
                    output.value for output in actor_outputs
                ))
                entropies = torch.stack(tuple(
                    output.distribution.entropy for output in actor_outputs
                ))
                actor_loss = clipped_ppo_loss(
                    new_log_probabilities,
                    prepared.old_log_probabilities[actor_indices],
                    prepared.advantages[actor_indices], actor_values,
                    prepared.returns[actor_indices], entropies,
                    clip_ratio=clip_ratio, value_coefficient=0.0,
                    entropy_coefficient=entropy_coefficient,
                )
                if (
                    target_kl is not None
                    and actor_loss.approximate_kl.item() > target_kl
                ):
                    final = PPOLoss(
                        total=actor_loss.total + value_coefficient * value_loss,
                        policy=actor_loss.policy, value=value_loss,
                        entropy=actor_loss.entropy,
                        approximate_kl=actor_loss.approximate_kl,
                        clip_fraction=actor_loss.clip_fraction,
                    )
                    stopped_early = True
                    break
                policy_loss = actor_loss.policy
                entropy = actor_loss.entropy
                approximate_kl = actor_loss.approximate_kl
                clip_fraction = actor_loss.clip_fraction
                actor_objective = actor_loss.total
            else:
                zero = value_loss.detach() * 0.0
                policy_loss = entropy = approximate_kl = clip_fraction = zero
                actor_objective = value_loss * 0.0
            total = actor_objective + value_coefficient * value_loss
            final = PPOLoss(
                total=total, policy=policy_loss, value=value_loss,
                entropy=entropy, approximate_kl=approximate_kl,
                clip_fraction=clip_fraction,
            )
            optimizer.zero_grad(set_to_none=True)
            final.total.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), maximum_gradient_norm
            ).item())
            optimizer.step()
        if stopped_early:
            break
    if final is None:  # pragma: no cover - guarded by update_epochs validation
        raise RuntimeError("PPO update produced no loss.")
    return PPOUpdateResult(
        epochs_run=epoch + 1, final_loss=float(final.total.detach().item()),
        policy_loss=float(final.policy.detach().item()),
        value_loss=float(final.value.detach().item()),
        entropy=float(final.entropy.detach().item()),
        approximate_kl=float(final.approximate_kl.item()),
        clip_fraction=float(final.clip_fraction.item()),
        gradient_norm=gradient_norm, stopped_early=stopped_early,
        transition_count=expected,
        actor_transition_count=int(prepared.actor_mask.sum().item()),
        advantage_mean=float(actor_advantages.mean().item()),
        advantage_standard_deviation=float(
            actor_advantages.std(unbiased=False).item()
        ),
        positive_advantage_fraction=float(
            (actor_advantages > 0.0).float().mean().item()
        ),
        explained_variance_before_update=explained_variance,
    )


def hierarchical_action_distribution(
    type_logits: Tensor,
    conditional_action_logits: Tensor,
    action_kind_index: Tensor,
    legal_mask: Tensor,
) -> HierarchicalActionDistribution:
    """Build a normalized distribution without assigning mass to illegal actions.

    Every legal action belongs to exactly one action kind. Kinds with no legal
    action are masked before the type softmax. Conditional logits are normalized
    independently inside each available kind.
    """

    if type_logits.ndim != 1 or conditional_action_logits.ndim != 1:
        raise ValueError("Hierarchical policy logits must be one-dimensional.")
    if action_kind_index.shape != conditional_action_logits.shape:
        raise ValueError("Every action must have one action-kind index.")
    if legal_mask.shape != conditional_action_logits.shape:
        raise ValueError("The legal mask must align with action logits.")
    if action_kind_index.dtype != torch.long:
        raise ValueError("Action-kind indices must use torch.long.")
    legal_mask = legal_mask.to(dtype=torch.bool)
    if not torch.any(legal_mask):
        raise ValueError("At least one legal action is required.")
    if torch.any(action_kind_index < 0) or torch.any(
        action_kind_index >= len(type_logits)
    ):
        raise ValueError("Action-kind index lies outside the type logits.")

    type_available = torch.zeros_like(type_logits, dtype=torch.bool)
    type_available.scatter_(0, action_kind_index[legal_mask], True)
    masked_type_logits = type_logits.masked_fill(~type_available, -torch.inf)
    type_log_probabilities = functional.log_softmax(masked_type_logits, dim=0)
    type_probabilities = type_log_probabilities.exp()

    action_log_probabilities = torch.full_like(
        conditional_action_logits, -torch.inf
    )
    conditional_entropy_by_type = torch.zeros_like(type_logits)
    for kind in range(len(type_logits)):
        members = legal_mask & (action_kind_index == kind)
        if not torch.any(members):
            continue
        conditional_log_probabilities = functional.log_softmax(
            conditional_action_logits[members], dim=0
        )
        action_log_probabilities[members] = (
            type_log_probabilities[kind] + conditional_log_probabilities
        )
        conditional_probabilities = conditional_log_probabilities.exp()
        conditional_entropy_by_type[kind] = -torch.sum(
            conditional_probabilities * conditional_log_probabilities
        )

    type_entropy = -torch.sum(
        type_probabilities[type_available]
        * type_log_probabilities[type_available]
    )
    conditional_entropy = torch.sum(
        type_probabilities * conditional_entropy_by_type
    )
    return HierarchicalActionDistribution(
        action_log_probabilities=action_log_probabilities,
        type_probabilities=type_probabilities,
        type_entropy=type_entropy,
        conditional_entropy=conditional_entropy,
    )


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute GAE while bootstrapping across truncation but not termination."""

    if not (
        rewards.ndim == values.ndim == next_values.ndim == terminated.ndim == 1
        and rewards.shape == values.shape == next_values.shape == terminated.shape
    ):
        raise ValueError("GAE inputs must be aligned one-dimensional tensors.")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("GAE gamma and lambda must lie in [0, 1].")
    continuation = 1.0 - terminated.to(dtype=values.dtype)
    deltas = rewards + gamma * next_values * continuation - values
    advantages = torch.zeros_like(deltas)
    accumulator = torch.zeros((), dtype=deltas.dtype, device=deltas.device)
    for index in range(len(deltas) - 1, -1, -1):
        accumulator = (
            deltas[index]
            + gamma * gae_lambda * continuation[index] * accumulator
        )
        advantages[index] = accumulator
    return advantages, advantages + values


def clipped_ppo_loss(
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    predicted_values: Tensor,
    returns: Tensor,
    entropies: Tensor,
    *,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> PPOLoss:
    """Compute the PPO actor/value objective for one flattened minibatch."""

    inputs = (
        new_log_probabilities, old_log_probabilities, advantages,
        predicted_values, returns, entropies,
    )
    if any(value.ndim != 1 for value in inputs) or len({value.shape for value in inputs}) != 1:
        raise ValueError("PPO loss inputs must be aligned one-dimensional tensors.")
    if clip_ratio < 0.0 or value_coefficient < 0.0 or entropy_coefficient < 0.0:
        raise ValueError("PPO coefficients must be nonnegative.")
    ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = functional.mse_loss(predicted_values, returns)
    entropy = entropies.mean()
    total = (
        policy_loss + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    with torch.no_grad():
        log_ratio = new_log_probabilities - old_log_probabilities
        approximate_kl = ((ratios - 1.0) - log_ratio).mean()
        clip_fraction = ((ratios - 1.0).abs() > clip_ratio).float().mean()
    return PPOLoss(
        total=total, policy=policy_loss, value=value_loss, entropy=entropy,
        approximate_kl=approximate_kl, clip_fraction=clip_fraction,
    )


def _mean_or_zeros(values: Tensor) -> Tensor:
    if values.shape[0]:
        return values.mean(dim=0)
    return values.new_zeros((values.shape[1],))


def _max_or_zeros(values: Tensor) -> Tensor:
    if values.shape[0]:
        return values.max(dim=0).values
    return values.new_zeros((values.shape[1],))


def _expand_node_encoder_input(state_dict, requested_count):
    state = dict(state_dict)
    key = "node_encoder.0.weight"
    existing = state[key]
    expanded = existing.new_zeros((existing.shape[0], requested_count))
    if existing.shape[1] == 29 and requested_count == 31:
        # v15.0 inserted navigation availability before the eight legacy
        # estimator metrics and their masks. Preserve every legacy column.
        expanded[:, :13] = existing[:, :13]
        expanded[:, 14:22] = existing[:, 13:21]
        expanded[:, 23:31] = existing[:, 21:29]
    else:
        expanded[:, :existing.shape[1]] = existing
    state[key] = expanded
    return state


def _explained_variance(targets: Tensor, predictions: Tensor) -> float:
    target_variance = torch.var(targets, unbiased=False)
    if target_variance.item() <= 1e-12:
        return 0.0
    residual_variance = torch.var(targets - predictions, unbiased=False)
    return float((1.0 - residual_variance / target_variance).item())
