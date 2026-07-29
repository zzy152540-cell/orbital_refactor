import numpy as np

from orbital_core.inter_satellite_model import inter_satellite_jacobians
from cooperative.topology import fully_connected_topology
from interfaces.data_objects import InterSatelliteObservation
from orbital_core.constants import R_EARTH
from orbital_core.measurements import measure_relative_range, measure_relative_range_rate
from orbital_core.orbit_elements import keplerian_to_eci
from pipelines.fleet_centralized import run_fleet_centralized_filter
from scenarios.fleet_scenario import generate_fleet_scenario


def _three_satellite_case():
    timestamps = np.arange(0.0, 11.0, 2.0)
    base = keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0)
    truth_initials = {
        "sat_01": base + np.array([0.0, -40.0, 0.0, 0.02, 0.0, 0.0]),
        "sat_02": base + np.array([50.0, 10.0, 20.0, 0.0, -0.02, 0.01]),
        "sat_03": base + np.array([-30.0, 35.0, -10.0, -0.01, 0.02, 0.0]),
    }
    scenario = generate_fleet_scenario(
        timestamps=timestamps, initial_state_by_node=truth_initials
    )
    errors = {
        "sat_01": np.array([15.0, -8.0, 5.0, 0.01, 0.0, 0.0]),
        "sat_02": np.array([-12.0, 10.0, -6.0, -0.01, 0.01, 0.0]),
        "sat_03": np.array([5.0, -5.0, 8.0, 0.0, -0.01, 0.01]),
    }
    estimates = {
        node_id: truth_initials[node_id] + errors[node_id]
        for node_id in scenario.node_ids
    }
    covariances = {
        node_id: np.diag([30.0, 30.0, 30.0, 0.1, 0.1, 0.1]) ** 2
        for node_id in scenario.node_ids
    }
    topology = fully_connected_topology(list(scenario.node_ids))
    observations = []
    for source in topology.node_ids:
        for target in topology.neighbors(source):
            for index, timestamp in enumerate(timestamps):
                state_i = scenario.trajectories[source].state_history_eci[index]
                state_j = scenario.trajectories[target].state_history_eci[index]
                observations.extend(
                    [
                        InterSatelliteObservation(
                            timestamp=float(timestamp),
                            source_node_id=source,
                            target_node_id=target,
                            modality="RANGE",
                            measurement=np.array([measure_relative_range(state_i, state_j)]),
                            covariance=np.array([[1.0]]),
                            confidence=1.0,
                            valid_flag=True,
                        ),
                        InterSatelliteObservation(
                            timestamp=float(timestamp),
                            source_node_id=source,
                            target_node_id=target,
                            modality="RANGE_RATE",
                            measurement=np.array([
                                measure_relative_range_rate(state_i, state_j)
                            ]),
                            covariance=np.array([[1e-4]]),
                            confidence=1.0,
                            valid_flag=True,
                        ),
                    ]
                )
    return scenario, estimates, covariances, observations


def test_range_rate_jacobians_are_antisymmetric():
    state_i = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    state_j = state_i + np.array([100.0, 20.0, -10.0, 0.2, -0.1, 0.05])
    for modality in ("RANGE", "RANGE_RATE"):
        h_i, h_j = inter_satellite_jacobians(
            state_i, state_j, modality=modality
        )
        np.testing.assert_allclose(h_i, -h_j)


def test_centralized_three_satellite_filter_returns_18_state_history():
    scenario, estimates, covariances, observations = _three_satellite_case()
    history = run_fleet_centralized_filter(
        timestamps=scenario.timestamps,
        initial_state_by_node=estimates,
        initial_covariance_by_node=covariances,
        inter_satellite_observations=observations,
        node_ids=scenario.node_ids,
        process_noise_acceleration=1e-8,
    )

    assert history.stacked_state_history.shape == (6, 18)
    assert history.stacked_covariance_history.shape == (6, 18, 18)
    assert set(history.state_history_by_node) == set(scenario.node_ids)
    assert len(history.nis_history[0]) == 12
    assert np.all(np.isfinite(history.stacked_state_history))
    assert np.all(np.linalg.eigvalsh(history.stacked_covariance_history[-1]) > -1e-6)
    cross_covariance = history.stacked_covariance_history[-1, :6, 6:12]
    assert np.linalg.norm(cross_covariance) > 0.0
