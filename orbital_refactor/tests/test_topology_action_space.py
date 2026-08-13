from dataclasses import replace

import numpy as np

from cooperative.network_schmidt_orchestrator import NetworkSchmidtOrchestrator
from cooperative.online_graph_observation import build_online_graph_observation
from cooperative.topology import fully_connected_topology
from cooperative.topology_action_space import (
    build_topology_action_space,
    resolve_topology_action,
)


def _observation():
    nodes = ("a", "b", "c")
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node={
            "a": np.zeros(6), "b": np.ones(6), "c": np.arange(6),
        },
        initial_covariance_by_node={node: np.eye(6) for node in nodes},
        topology=fully_connected_topology(nodes),
    )
    observation = build_online_graph_observation(orchestrator)
    return replace(
        observation,
        previous_active_edges=(("a", "b"), ("b", "c")),
        estimation_dependency_edges=(("a", "b"), ("b", "c")),
    )


def test_runtime_action_space_is_stable_for_tree_baseline():
    space = build_topology_action_space(_observation())
    assert tuple(action.kind for action in space.actions) == (
        "keep", "add", "swap", "swap",
    )
    assert tuple(action.action_id for action in space.actions) == (0, 1, 2, 3)
    assert tuple(space.legal_mask) == (True, True, True, True)
    assert all(action.kind != "remove" for action in space.actions)


def test_runtime_action_space_can_remove_only_redundant_edges():
    observation = replace(
        _observation(),
        previous_active_edges=(("a", "b"), ("a", "c"), ("b", "c")),
    )
    space = build_topology_action_space(observation)
    assert tuple(action.kind for action in space.actions) == (
        "keep", "remove", "remove", "remove",
    )
    assert tuple(space.legal_mask) == (True, True, True, True)
    assert all(len(action.topology.active_edges) == 2
               for action in space.actions[1:])


def test_cooldown_allows_only_emergency_invisible_removal():
    base = _observation()
    observation = replace(
        base,
        previous_active_edges=(("a", "b"), ("a", "c"), ("b", "c")),
        candidate_edges=tuple(
            replace(edge, geometrically_visible=False)
            if edge.nodes == ("a", "c") else edge
            for edge in base.candidate_edges
        ),
    )
    space = build_topology_action_space(observation, cooldown_remaining=2)
    legal = [action for action, allowed in zip(space.actions, space.legal_mask)
             if allowed]
    assert tuple(action.kind for action in legal) == ("keep", "remove")
    assert legal[1].removed_edges == (("a", "c"),)


def test_action_mask_separates_catalog_from_current_link_availability():
    observation = _observation()
    edges = tuple(
        replace(edge, communication_available=False)
        if edge.nodes == ("a", "c") else edge
        for edge in observation.candidate_edges
    )
    space = build_topology_action_space(replace(observation, candidate_edges=edges))
    assert tuple(space.legal_mask) == (True, False, False, False)
    assert space.rejection_reason_by_action[1:] == (
        "communication_unavailable", "communication_unavailable",
        "communication_unavailable",
    )


def test_masked_and_invalid_policy_outputs_fall_back_to_keep():
    observation = _observation()
    space = build_topology_action_space(
        observation, edge_risk_gate=lambda edge: edge.packet_loss_rate < 0.0,
    )
    masked = resolve_topology_action(space, 1)
    assert masked.used_fallback
    assert masked.executed_action.kind == "keep"
    assert masked.reason == "risk_gate_rejected"
    invalid = resolve_topology_action(space, 99)
    assert invalid.used_fallback
    assert invalid.reason == "action_id_out_of_range"
    keep = resolve_topology_action(space, 0)
    assert not keep.used_fallback
    assert keep.executed_action.kind == "keep"
