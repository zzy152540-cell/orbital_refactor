from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol
from pathlib import Path

import numpy as np

from experiments.action_graph_features import action_graph_metrics

from experiments.topology_control_environment import (
    TopologyConstraintCosts,
    TopologyControlEnvironment,
    TopologyEnvironmentState,
)


class EnvironmentPolicy(Protocol):
    name: str

    def select_action(self, state: TopologyEnvironmentState) -> int: ...


@dataclass(frozen=True)
class AlwaysKeepPolicy:
    name: str = "always_keep"

    def select_action(self, state: TopologyEnvironmentState) -> int:
        return 0


class RandomLegalPolicy:
    name = "random_legal"

    def __init__(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: TopologyEnvironmentState) -> int:
        legal = np.flatnonzero(state.action_space.legal_mask)
        return int(self._rng.choice(legal))


class HierarchicalGNNPolicy:
    """Deployable hierarchical GNN policy loaded from a supervised checkpoint."""

    name = "hierarchical_gnn"

    def __init__(self, checkpoint_path: str | Path) -> None:
        import torch
        from experiments.graph_action_gnn import GraphActionValueNetwork

        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True
        )
        configuration = checkpoint["configuration"]
        if configuration.get("loss_mode") != "hierarchical":
            raise ValueError("GNN policy requires a hierarchical checkpoint.")
        self._model = GraphActionValueNetwork(
            node_feature_count=int(checkpoint["node_feature_count"]),
            candidate_edge_feature_count=int(checkpoint["edge_feature_count"]),
            measurement_feature_count=int(checkpoint["edge_feature_count"]),
            action_feature_count=int(checkpoint["action_feature_count"]),
            hidden_size=int(configuration["hidden_size"]),
            message_passing_steps=int(configuration["message_passing_steps"]),
            explicit_action_pairing=bool(
                configuration.get("explicit_action_pairing", False)
            ),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._torch = torch

    def select_action(self, state: TopologyEnvironmentState) -> int:
        from experiments.graph_action_gnn import (
            _select_snapshot_action,
            torch_snapshot_action_group,
        )
        from experiments.topology_snapshot_counterfactual import (
            build_online_snapshot_action_tensor,
        )

        group, action_ids = build_online_snapshot_action_tensor(state)
        values = torch_snapshot_action_group(group)
        with self._torch.no_grad():
            output = self._model(values)
        selected = _select_snapshot_action(
            output, values, output.utility.detach().cpu().numpy()
        )
        return int(action_ids[selected])


@dataclass(frozen=True)
class InformationGreedyPolicy:
    """Cost-aware graph-information baseline using no truth-state features."""

    communication_risk_weight: float = 1.0
    topology_churn_weight: float = 1.0
    name: str = "cost_aware_information_greedy"

    def select_action(self, state: TopologyEnvironmentState) -> int:
        edge_by_nodes = {
            edge.nodes: edge for edge in state.observation.candidate_edges
        }
        stale_removals = [
            action for action, allowed in zip(
                state.action_space.actions, state.action_space.legal_mask
            )
            if allowed and action.kind == "remove"
            and all(not edge_by_nodes[edge].geometrically_visible
                    for edge in action.removed_edges)
        ]
        if stale_removals:
            return min(stale_removals, key=lambda action: action.action_id).action_id
        scored = []
        for action, allowed in zip(
            state.action_space.actions, state.action_space.legal_mask
        ):
            if not allowed:
                continue
            graph = action_graph_metrics(state.observation, action.topology)
            information = (
                graph.normalized_information_rank
                + graph.algebraic_connectivity
                + 0.1 * graph.minimum_positive_information_eigenvalue
                + 0.01 * graph.information_log_pseudodeterminant
            )
            risk = _action_risk(action, edge_by_nodes)
            churn = len(action.added_edges) + len(action.removed_edges)
            scored.append((
                information - self.communication_risk_weight * risk
                - self.topology_churn_weight * churn,
                -churn, -action.action_id, action.action_id,
            ))
        return max(scored)[-1] if scored else 0


class ShortHorizonOraclePolicy:
    """Simulation-only one-decision lookahead; never a deployable policy."""

    name = "short_horizon_oracle"

    def __init__(
        self, environment: TopologyControlEnvironment, *,
        lookahead_steps: int = 1,
        communication_cost_weight: float = 0.0,
        switch_cost_weight: float = 0.0,
        resynchronization_cost_weight: float = 0.0,
    ) -> None:
        if lookahead_steps < 1:
            raise ValueError("Oracle lookahead must be at least one step.")
        self.environment = environment
        self.lookahead_steps = int(lookahead_steps)
        self.communication_cost_weight = float(communication_cost_weight)
        self.switch_cost_weight = float(switch_cost_weight)
        self.resynchronization_cost_weight = float(
            resynchronization_cost_weight
        )

    def select_action(self, state: TopologyEnvironmentState) -> int:
        candidates = []
        for action, allowed in zip(
            state.action_space.actions, state.action_space.legal_mask
        ):
            if not allowed:
                continue
            branch = deepcopy(self.environment)
            result = branch.step(action.action_id)
            utility = self._step_utility(result)
            for _ in range(1, self.lookahead_steps):
                if result.terminated or result.truncated:
                    break
                result = branch.step(0)
                utility += self._step_utility(result)
            candidates.append((utility, -action.action_id, action.action_id))
        return max(candidates)[-1] if candidates else 0

    def _step_utility(self, result) -> float:
        return (
            result.reward
            - self.communication_cost_weight
            * result.constraint_costs.transmitted_messages
            - self.switch_cost_weight
            * result.constraint_costs.topology_switch
            - self.resynchronization_cost_weight
            * result.constraint_costs.resynchronization_count
        )


@dataclass(frozen=True)
class LowChurnObservablePolicy:
    """Prefer keep; otherwise minimize observable link risk and topology churn."""

    maximum_packet_loss: float = 0.5
    maximum_delay: float = 10.0
    name: str = "low_churn_observable"

    def select_action(self, state: TopologyEnvironmentState) -> int:
        space = state.action_space
        if bool(space.legal_mask[0]):
            return 0
        edge_by_nodes = {
            edge.nodes: edge for edge in state.observation.candidate_edges
        }
        legal = [
            action for action, allowed in zip(space.actions, space.legal_mask)
            if allowed
        ]
        if not legal:
            return 0
        return min(legal, key=lambda action: (
            _action_risk(action, edge_by_nodes),
            len(action.added_edges) + len(action.removed_edges),
            action.action_id,
        )).action_id


@dataclass(frozen=True)
class BaselineEpisodeSummary:
    policy_name: str
    seed: int
    step_count: int
    cumulative_reward: float
    final_position_rmse: float
    final_worst_node_position_rmse: float
    final_mean_covariance_logdet: float
    cumulative_costs: TopologyConstraintCosts
    selected_action_kind_counts: tuple[tuple[str, int], ...]
    fallback_reason_counts: tuple[tuple[str, int], ...]


def run_topology_control_baseline_episode(
    environment: TopologyControlEnvironment,
    policy: EnvironmentPolicy,
    *, seed: int,
) -> BaselineEpisodeSummary:
    """Run one policy without exposing truth-based reward to action selection."""

    state = environment.reset(seed=seed)
    rewards = []
    costs = []
    kinds = Counter()
    fallback_reasons = Counter()
    last = None
    while True:
        action_id = policy.select_action(state)
        last = environment.step(action_id)
        rewards.append(last.reward)
        costs.append(last.constraint_costs)
        kinds[last.action_resolution.executed_action.kind] += 1
        if last.action_resolution.used_fallback:
            fallback_reasons[last.action_resolution.reason or "unknown"] += 1
        state = last.state
        if last.terminated or last.truncated:
            break
    diagnostics = dict(last.diagnostics)
    return BaselineEpisodeSummary(
        policy_name=policy.name,
        seed=int(seed), step_count=len(rewards),
        cumulative_reward=float(sum(rewards)),
        final_position_rmse=diagnostics["position_rmse"],
        final_worst_node_position_rmse=(
            diagnostics["worst_node_position_rmse"]
        ),
        final_mean_covariance_logdet=diagnostics["mean_covariance_logdet"],
        cumulative_costs=_sum_costs(costs),
        selected_action_kind_counts=tuple(sorted(kinds.items())),
        fallback_reason_counts=tuple(sorted(fallback_reasons.items())),
    )


def _action_risk(action, edge_by_nodes):
    return sum(
        edge_by_nodes[edge].packet_loss_rate
        + edge_by_nodes[edge].delay / 10.0
        for edge in action.added_edges
    )


def _sum_costs(values):
    names = tuple(TopologyConstraintCosts.__dataclass_fields__)
    return TopologyConstraintCosts(**{
        name: float(sum(getattr(value, name) for value in values))
        for name in names
    })
