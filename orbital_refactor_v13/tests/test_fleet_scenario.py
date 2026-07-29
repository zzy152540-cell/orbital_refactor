import numpy as np

from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import generate_fleet_scenario


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
