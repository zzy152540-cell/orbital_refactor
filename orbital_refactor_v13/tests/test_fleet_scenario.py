import numpy as np

from cooperative.topology import chain_topology
from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import (
    DifferentialOrbitOffset,
    generate_differential_orbit_fleet_scenario,
    generate_fleet_scenario,
)
from scenarios.measurement_visibility import (
    VisibilityConfig,
    generate_inter_satellite_observation_opportunities,
    summarize_observation_opportunities,
)


def test_fleet_scenario_is_symmetric_and_stacks_in_node_order():
    timestamps = np.array([0.0, 2.0, 4.0])
    base = keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0)
    scenario = generate_fleet_scenario(
        timestamps=timestamps,
        initial_state_by_node={
            "sat_01": base,
            "sat_02": base + np.array([30.0, 10.0, 0.0, 0.0, -0.01, 0.0]),
            "sat_03": base + np.array([-20.0, 15.0, 5.0, 0.01, 0.0, 0.0]),
        },
    )

    assert scenario.node_ids == ("sat_01", "sat_02", "sat_03")
    assert scenario.stacked_state_history().shape == (3, 18)
    assert set(scenario.truth_state_history_by_node) == set(scenario.node_ids)
    for node_id, trajectory in scenario.trajectories.items():
        assert trajectory.satellite_id == node_id
        assert trajectory.state_history_eci.shape == (3, 6)
        assert trajectory.q_eci2rtn_history.shape == (3, 4)


def test_differential_orbit_scenario_crosses_range_visibility_boundary():
    timestamps = np.arange(0.0, 121.0, 2.0)
    scenario = generate_differential_orbit_fleet_scenario(
        timestamps=timestamps,
        base_semi_major_axis=R_EARTH + 700e3,
        eccentricity=0.001,
        inclination=np.deg2rad(23.0),
        raan=0.0,
        argument_of_perigee=0.0,
        base_true_anomaly=0.0,
        offset_by_node={
            "sat_a": DifferentialOrbitOffset(),
            "sat_b": DifferentialOrbitOffset(
                semi_major_axis=2000.0, true_anomaly=-0.0006,
            ),
        },
    )
    opportunities = generate_inter_satellite_observation_opportunities(
        timestamps=timestamps,
        truth_state_history_by_node=scenario.truth_state_history_by_node,
        candidate_topology=chain_topology(list(scenario.node_ids)),
        visibility_by_modality={
            "RANGE": VisibilityConfig(maximum_range=5000.0)
        },
    )
    summary = summarize_observation_opportunities(opportunities)

    assert opportunities[0].visibility.visible
    assert not opportunities[-1].visibility.visible
    assert opportunities[-1].visibility.reason == "range_exceeded"
    assert summary.visible_directed_edge_count_by_timestamp[0.0] == 2
    assert summary.visible_directed_edge_count_by_timestamp[120.0] == 0
    assert summary.overall.rejection_counts["range_exceeded"] > 0
