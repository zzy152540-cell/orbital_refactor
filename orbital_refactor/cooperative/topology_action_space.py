from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cooperative.topology_policy import (
    GraphObservation,
    TopologyAction,
    UndirectedEdge,
    validate_deployment_graph_observation,
)


@dataclass(frozen=True)
class StructuredTopologyAction:
    action_id: int
    kind: str
    topology: TopologyAction
    added_edges: tuple[UndirectedEdge, ...] = ()
    removed_edges: tuple[UndirectedEdge, ...] = ()


@dataclass(frozen=True)
class TopologyActionSpace:
    actions: tuple[StructuredTopologyAction, ...]
    legal_mask: np.ndarray
    rejection_reason_by_action: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not self.actions or self.actions[0].kind != "keep":
            raise ValueError("Topology action space requires keep at index zero.")
        if self.legal_mask.shape != (len(self.actions),):
            raise ValueError("Topology legal mask shape does not match actions.")
        if len(self.rejection_reason_by_action) != len(self.actions):
            raise ValueError("Topology rejection reasons do not match actions.")
        if not bool(self.legal_mask[0]):
            raise ValueError("Keep must always be a legal topology action.")


@dataclass(frozen=True)
class ResolvedTopologyAction:
    requested_action_id: int | None
    executed_action: StructuredTopologyAction
    used_fallback: bool
    reason: str | None = None


EdgeRiskGate = Callable[[object], bool]


def build_topology_action_space(
    observation: GraphObservation,
    *, edge_risk_gate: EdgeRiskGate | None = None,
    cooldown_remaining: int = 0,
    allow_emergency_invisible_removal: bool = True,
    eligible_addition_edges: tuple[UndirectedEdge, ...] | None = None,
) -> TopologyActionSpace:
    """Enumerate keep/add/remove and only connectivity-safe swap actions."""

    validate_deployment_graph_observation(observation)
    if cooldown_remaining < 0:
        raise ValueError("Topology cooldown cannot be negative.")
    node_ids = tuple(node.node_id for node in observation.nodes)
    candidates = tuple(edge.nodes for edge in observation.candidate_edges)
    candidate_set = set(candidates)
    baseline = tuple(sorted(observation.previous_active_edges))
    baseline_set = set(baseline)
    if baseline_set - candidate_set:
        raise ValueError("Current topology contains a non-candidate edge.")
    if not _connected(node_ids, baseline_set):
        raise ValueError("Current topology must be connected.")
    by_edge = {edge.nodes: edge for edge in observation.candidate_edges}
    eligible = (
        candidate_set - baseline_set
        if eligible_addition_edges is None
        else set(eligible_addition_edges)
    )
    if eligible - (candidate_set - baseline_set):
        raise ValueError(
            "Eligible addition edges must be inactive candidate edges."
        )
    actions = [StructuredTopologyAction(
        action_id=0,
        kind="keep",
        topology=TopologyAction("v15_keep", baseline),
    )]
    legal = [True]
    reasons: list[str | None] = [None]
    for added in sorted(eligible):
        add_topology = tuple(sorted(baseline_set | {added}))
        actions.append(StructuredTopologyAction(
            action_id=len(actions), kind="add",
            topology=TopologyAction("v15_add", add_topology),
            added_edges=(added,),
        ))
        allowed, reason = _edge_allowed(by_edge[added], edge_risk_gate)
        if cooldown_remaining > 0:
            allowed, reason = False, "minimum_dwell_time"
        legal.append(allowed)
        reasons.append(reason)
        for removed in baseline:
            swapped = (baseline_set | {added}) - {removed}
            if not _connected(node_ids, swapped):
                continue
            actions.append(StructuredTopologyAction(
                action_id=len(actions), kind="swap",
                topology=TopologyAction("v15_swap", tuple(sorted(swapped))),
                added_edges=(added,), removed_edges=(removed,),
            ))
            legal.append(allowed)
            reasons.append(reason)
    for removed in baseline:
        reduced = baseline_set - {removed}
        if not _connected(node_ids, reduced):
            continue
        actions.append(StructuredTopologyAction(
            action_id=len(actions), kind="remove",
            topology=TopologyAction("v15_remove", tuple(sorted(reduced))),
            removed_edges=(removed,),
        ))
        emergency = (
            allow_emergency_invisible_removal
            and not by_edge[removed].geometrically_visible
        )
        allowed = cooldown_remaining == 0 or emergency
        legal.append(allowed)
        reasons.append(None if allowed else "minimum_dwell_time")
    mask = np.asarray(legal, dtype=bool)
    mask.setflags(write=False)
    return TopologyActionSpace(
        actions=tuple(actions), legal_mask=mask,
        rejection_reason_by_action=tuple(reasons),
    )


def resolve_topology_action(
    action_space: TopologyActionSpace,
    requested_action_id: int | None,
) -> ResolvedTopologyAction:
    """Resolve a policy output, falling back safely to keep when invalid."""

    if not isinstance(requested_action_id, (int, np.integer)):
        return ResolvedTopologyAction(
            requested_action_id=None, executed_action=action_space.actions[0],
            used_fallback=True, reason="invalid_action_id",
        )
    action_id = int(requested_action_id)
    if action_id < 0 or action_id >= len(action_space.actions):
        return ResolvedTopologyAction(
            requested_action_id=action_id,
            executed_action=action_space.actions[0],
            used_fallback=True, reason="action_id_out_of_range",
        )
    if not bool(action_space.legal_mask[action_id]):
        return ResolvedTopologyAction(
            requested_action_id=action_id,
            executed_action=action_space.actions[0],
            used_fallback=True,
            reason=action_space.rejection_reason_by_action[action_id]
            or "masked_action",
        )
    return ResolvedTopologyAction(
        requested_action_id=action_id,
        executed_action=action_space.actions[action_id],
        used_fallback=False,
    )


def _edge_allowed(edge, risk_gate):
    if not edge.geometrically_visible:
        return False, "not_geometrically_visible"
    if not edge.communication_available:
        return False, "communication_unavailable"
    if risk_gate is not None and not bool(risk_gate(edge)):
        return False, "risk_gate_rejected"
    return True, None


def _connected(node_ids, edges):
    if len(node_ids) == 1:
        return True
    adjacency = {node: [] for node in node_ids}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    visited, pending = set(), [node_ids[0]]
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency[node])
    return len(visited) == len(node_ids)
