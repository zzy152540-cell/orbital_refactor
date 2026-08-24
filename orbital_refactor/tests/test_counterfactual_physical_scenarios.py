import numpy as np

from experiments.counterfactual_physical_scenarios import (
    FIVE_NODE_PHYSICAL_FAMILIES,
    build_three_satellite_counterfactual_scenarios,
    sample_five_satellite_physical_scenario,
)
from orbital_core.constants import R_EARTH
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


def test_randomized_five_node_physical_scenario_is_seed_reproducible():
    first = sample_five_satellite_physical_scenario(12)
    repeated = sample_five_satellite_physical_scenario(12)

    assert first == repeated
    assert first.node_count == 5
    assert set(first.truth_initial_state_by_node()) == {
        "sat_01", "sat_02", "sat_03", "sat_04", "sat_05",
    }
    assert first.semi_major_axis * (1.0 - first.eccentricity) > R_EARTH
    assert all(np.isfinite(state).all()
               for state in first.truth_initial_state_by_node().values())


def test_seed_cycle_assigns_physical_families_without_reward_selection():
    families = tuple(
        sample_five_satellite_physical_scenario(
            seed, family_assignment_mode="seed_cycle",
        ).family
        for seed in range(6)
    )
    assert families == 2 * FIVE_NODE_PHYSICAL_FAMILIES


def test_randomized_five_node_families_span_geometry_and_orbit_planes():
    compact = sample_five_satellite_physical_scenario(
        3, families=("compact_along_track",),
    )
    differential = sample_five_satellite_physical_scenario(
        3, families=("differential_along_track",),
    )
    multi_plane = sample_five_satellite_physical_scenario(
        3, families=("two_plane_cluster",),
    )

    assert compact.along_track_spacing < differential.along_track_spacing
    assert compact.semi_major_axis_step <= 200.0
    assert differential.semi_major_axis_step >= 200.0
    assert {plane for _, plane in compact.plane_index_by_node} == {0}
    assert {plane for _, plane in multi_plane.plane_index_by_node} == {0, 1}
    assert multi_plane.raan_separation > 0.0
    compact_states = compact.truth_initial_state_by_node()
    multi_plane_states = multi_plane.truth_initial_state_by_node()
    assert not np.allclose(
        np.stack(tuple(compact_states.values())),
        np.stack(tuple(multi_plane_states.values())),
    )
