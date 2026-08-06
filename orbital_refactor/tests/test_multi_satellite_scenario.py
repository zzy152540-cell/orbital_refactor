import numpy as np

from orbital_core.constants import R_EARTH
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.multi_satellite_scenario import generate_cooperative_scenario


def test_multi_satellite_scenario_generates_consistent_relative_histories():
    timestamps = np.arange(0.0, 31.0, 10.0)
    target = keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0)
    observers = {
        "sat_01": keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.001),
        "sat_02": keplerian_to_eci(R_EARTH + 701e3, 0.001, 0.3, 0.001, 0.0, -0.001),
    }
    scenario = generate_cooperative_scenario(
        timestamps=timestamps,
        target_id="target",
        target_initial_state_eci=target,
        observer_initial_states_eci=observers,
    )
    assert set(scenario.observer_trajectories) == {"sat_01", "sat_02"}
    for node_id, trajectory in scenario.observer_trajectories.items():
        np.testing.assert_allclose(
            scenario.relative_state_eci_by_node[node_id],
            scenario.target_trajectory.state_history_eci - trajectory.state_history_eci,
        )
        assert scenario.relative_state_spri_by_node[node_id].shape == (4, 6)
        assert trajectory.q_eci2pri_history.shape == (4, 4)
