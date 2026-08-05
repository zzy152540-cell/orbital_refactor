import numpy as np
import pytest

from experiments.v14_walker_geometry_audit import (
    run_v14_walker_geometry_audit,
    run_v14_walker_static_topology_scan,
)
from orbital_core.constants import R_EARTH
from scenarios.walker_scenario import (
    WalkerDeltaConfig,
    generate_walker_delta_scenario,
)


def test_walker_20_5_1_has_expected_planes_slots_and_phase():
    config = WalkerDeltaConfig(
        total_satellites=20, plane_count=5, phasing=1,
        semi_major_axis=R_EARTH + 700e3, eccentricity=0.0,
        inclination=np.deg2rad(53.0),
    )
    scenario = generate_walker_delta_scenario(
        timestamps=np.array([0.0, 10.0]), config=config,
    )

    assert len(scenario.node_ids) == 20
    assert config.satellites_per_plane == 4
    assert scenario.node_ids[0] == "sat_p01_s01"
    assert scenario.node_ids[-1] == "sat_p05_s04"
    first = scenario.elements_by_node["sat_p01_s01"]
    next_plane = scenario.elements_by_node["sat_p02_s01"]
    next_slot = scenario.elements_by_node["sat_p01_s02"]
    assert np.isclose(next_plane.raan - first.raan, 2.0 * np.pi / 5.0)
    assert np.isclose(next_plane.true_anomaly - first.true_anomaly, 2.0 * np.pi / 20.0)
    assert np.isclose(next_slot.true_anomaly - first.true_anomaly, 2.0 * np.pi / 4.0)
    assert all(history.shape == (2, 6) for history in scenario.truth_state_history_by_node.values())


def test_walker_geometry_audit_reports_physical_los_graph():
    result = run_v14_walker_geometry_audit(duration=60.0, dt=30.0)

    assert len(result.scenario.node_ids) == 20
    assert result.minimum_initial_pair_range > 0.0
    assert result.maximum_initial_pair_range > result.minimum_initial_pair_range
    assert 0 < result.minimum_visible_directed_edges <= result.maximum_visible_directed_edges
    assert 0.0 < result.visibility_summary.overall.visibility_rate < 1.0
    assert sum(result.persistent_component_sizes) == 20
    assert result.minimum_persistent_node_degree >= 0
    assert result.maximum_persistent_node_degree >= result.minimum_persistent_node_degree
    assert result.maximum_instantaneous_component_count >= 1
    assert result.minimum_largest_instantaneous_component >= 1


def test_walker_static_scan_checks_selected_plane_phasing_pairs():
    result = run_v14_walker_static_topology_scan(
        plane_counts=(4, 5),
        phasing_values_by_plane={4: (0, 1), 5: (0,)},
        duration=60.0, dt=30.0,
    )

    assert set(result.result_by_plane_and_phasing) == {(4, 0), (4, 1), (5, 0)}
    assert set(result.persistent_connected_candidates) <= set(
        result.instantaneously_connected_candidates
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"total_satellites": 20, "plane_count": 6}, "divisible"),
        ({"total_satellites": 20, "plane_count": 5, "phasing": 5}, "phasing"),
    ],
)
def test_walker_rejects_invalid_t_p_f(kwargs, message):
    defaults = dict(
        total_satellites=20, plane_count=5, phasing=1,
        semi_major_axis=R_EARTH + 700e3, eccentricity=0.0,
        inclination=np.deg2rad(53.0),
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        WalkerDeltaConfig(**defaults)
