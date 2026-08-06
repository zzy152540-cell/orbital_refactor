import numpy as np

from cooperative.fleet_filter_runner import run_fleet_filter
from cooperative.topology import chain_topology
from interfaces.data_objects import InterSatelliteObservation
from orbital_core.constants import R_EARTH
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.orbit_elements import keplerian_to_eci


def _case():
    timestamps = np.array([0.0, 1.0, 2.0])
    base = keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0)
    truth = {
        "sat_01": propagate_absolute_orbit(base, timestamps),
        "sat_02": propagate_absolute_orbit(
            base + np.array([30.0, 0.0, 0.0, 0.0, -0.02, 0.0]),
            timestamps,
        ),
    }
    initial_state = {
        "sat_01": truth["sat_01"][0] + np.array([5.0, 0.0, 0.0, 0.01, 0.0, 0.0]),
        "sat_02": truth["sat_02"][0] + np.array([-5.0, 0.0, 0.0, -0.01, 0.0, 0.0]),
    }
    initial_covariance = {
        node_id: np.diag([20.0, 20.0, 20.0, 0.05, 0.05, 0.05]) ** 2
        for node_id in initial_state
    }
    observations = []
    for index, timestamp in enumerate(timestamps):
        for source, target in [("sat_01", "sat_02"), ("sat_02", "sat_01")]:
            observations.append(
                InterSatelliteObservation(
                    timestamp=float(timestamp),
                    source_node_id=source,
                    target_node_id=target,
                    modality="RANGE",
                    measurement=np.array([
                        measure_relative_range(truth[source][index], truth[target][index])
                    ]),
                    covariance=np.array([[1.0]]),
                    confidence=1.0,
                    valid_flag=True,
                )
            )
            observations.append(
                InterSatelliteObservation(
                    timestamp=float(timestamp),
                    source_node_id=source,
                    target_node_id=target,
                    modality="AZ_EL",
                    measurement=measure_relative_az_el(truth[source][index], truth[target][index]),
                    covariance=np.diag([1e-4, 1e-4]),
                    confidence=1.0,
                    valid_flag=True,
                )
            )
            observations.append(
                InterSatelliteObservation(
                    timestamp=float(timestamp),
                    source_node_id=source,
                    target_node_id=target,
                    modality="RANGE_RATE",
                    measurement=np.array([
                        measure_relative_range_rate(truth[source][index], truth[target][index])
                    ]),
                    covariance=np.array([[0.01]]),
                    confidence=1.0,
                    valid_flag=True,
                )
            )
    return timestamps, truth, initial_state, initial_covariance, observations


def test_run_fleet_filter_returns_closed_loop_histories():
    timestamps, _truth, initial_state, initial_covariance, observations = _case()
    result = run_fleet_filter(
        timestamps=timestamps,
        initial_state_by_node=initial_state,
        initial_covariance_by_node=initial_covariance,
        topology=chain_topology(["sat_01", "sat_02"]),
        inter_satellite_observations=observations,
        process_noise_acceleration_std=1e-8,
        inter_satellite_gate_enable=True,
        inter_satellite_gate_threshold=9.21,
        consensus_iterations=1,
        ci_grid_points=11,
    )

    assert set(result.node_ids) == {"sat_01", "sat_02"}
    assert result.state_history_by_node["sat_01"].shape == (3, 6)
    assert result.covariance_history_by_node["sat_01"].shape == (3, 6, 6)
    assert result.communication_stats.attempted_report_count == 6
    assert result.communication_stats.received_report_count == 6
    assert "sat_02:BLOCK" in result.inter_satellite_nis_history_by_node["sat_01"][0]
    assert "sat_02:AZ_EL" in result.inter_satellite_nis_history_by_node["sat_01"][0]
    assert len(result.iteration_weight_history_by_node["sat_01"][0]) == 1
    assert result.node_weight_history_by_node["sat_01"][0] == {"sat_01": 1.0}


def test_run_fleet_filter_rejects_ci_between_different_satellite_states():
    timestamps, _truth, initial_state, initial_covariance, _observations = _case()
    try:
        run_fleet_filter(
            timestamps=timestamps,
            initial_state_by_node=initial_state,
            initial_covariance_by_node=initial_covariance,
            topology=chain_topology(["sat_01", "sat_02"]),
            enable_state_consensus=True,
            consensus_iterations=2,
            ci_grid_points=11,
        )
    except ValueError as exc:
        assert "different physical satellites" in str(exc)
    else:
        raise AssertionError("Expected invalid cross-satellite state CI to be rejected.")


def test_run_fleet_filter_rejects_mismatched_nodes():
    timestamps, _truth, initial_state, initial_covariance, _observations = _case()
    initial_state.pop("sat_02")
    try:
        run_fleet_filter(
            timestamps=timestamps,
            initial_state_by_node=initial_state,
            initial_covariance_by_node=initial_covariance,
            topology=chain_topology(["sat_01", "sat_02"]),
        )
    except ValueError as exc:
        assert "initial_state_by_node" in str(exc)
    else:
        raise AssertionError("Expected node mismatch to be rejected.")
