import numpy as np

from experiments.counterfactual_physical_scenarios import (
    build_three_satellite_counterfactual_scenarios,
)
from experiments.short_horizon_topology_counterfactual import (
    run_short_horizon_topology_counterfactual,
)


def test_physical_scenarios_change_geometry_but_preserve_node_schema():
    scenarios = build_three_satellite_counterfactual_scenarios()

    assert tuple(value.scenario_id for value in scenarios) == (
        "compact_equal_orbit",
        "medium_differential",
        "wide_differential",
    )
    assert all(
        set(value.truth_initial_state_by_node())
        == {"sat_01", "sat_02", "sat_03"}
        for value in scenarios
    )
    compact, _, wide = scenarios
    compact_states = compact.truth_initial_state_by_node()
    wide_states = wide.truth_initial_state_by_node()
    compact_distance = np.linalg.norm(
        compact_states["sat_01"][:3] - compact_states["sat_03"][:3]
    )
    wide_distance = np.linalg.norm(
        wide_states["sat_01"][:3] - wide_states["sat_03"][:3]
    )
    assert wide_distance > 3.0 * compact_distance


def test_explicit_physical_scenario_changes_counterfactual_decision_graph():
    compact, _, wide = build_three_satellite_counterfactual_scenarios()
    arguments = dict(
        node_count=3, seed=0, future_seed=100,
        decision_epoch=1, horizon_epochs=1,
        relative_modalities=("RANGE", "RANGE_RATE", "AZ_EL"),
    )
    compact_result = run_short_horizon_topology_counterfactual(
        **arguments,
        truth_initial_state_by_node=compact.truth_initial_state_by_node(),
    )
    wide_result = run_short_horizon_topology_counterfactual(
        **arguments,
        truth_initial_state_by_node=wide.truth_initial_state_by_node(),
    )
    compact_distances = tuple(
        edge.distance for edge in compact_result.decision_observation.candidate_edges
    )
    wide_distances = tuple(
        edge.distance for edge in wide_result.decision_observation.candidate_edges
    )

    assert not np.allclose(compact_distances, wide_distances)
