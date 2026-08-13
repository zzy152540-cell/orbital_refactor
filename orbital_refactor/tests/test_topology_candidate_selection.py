from dataclasses import replace

from cooperative.topology_candidate_selection import (
    select_top_k_addition_edges,
)
from tests.test_topology_action_space import _observation


def test_top_k_zero_keeps_no_addition_edges():
    assert select_top_k_addition_edges(
        _observation(), top_k_per_node=0,
    ) == ()


def test_top_k_selects_feasible_nearest_edges_deterministically():
    observation = _observation()
    edges = tuple(
        replace(edge, distance={
            ("a", "b"): 1.0, ("a", "c"): 2.0, ("b", "c"): 3.0,
        }[edge.nodes])
        for edge in observation.candidate_edges
    )
    selected = select_top_k_addition_edges(
        replace(observation, candidate_edges=edges), top_k_per_node=1,
    )
    assert selected == (("a", "c"),)


def test_top_k_excludes_invisible_and_unavailable_edges():
    observation = _observation()
    edges = tuple(
        replace(edge, geometrically_visible=False)
        if edge.nodes == ("a", "c") else edge
        for edge in observation.candidate_edges
    )
    assert select_top_k_addition_edges(
        replace(observation, candidate_edges=edges), top_k_per_node=1,
    ) == ()
