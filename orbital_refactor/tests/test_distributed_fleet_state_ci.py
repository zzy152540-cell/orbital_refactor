import numpy as np

from cooperative.fleet_state_ci_runner import run_distributed_fleet_state_ci
from cooperative.topology import fully_connected_topology
from interfaces.attitude_data_objects import AttitudeEstimate
from interfaces.data_objects import AbsolutePositionObservation, InterSatelliteObservation
from orbital_core.constants import R_EARTH
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)
from orbital_core.orbit_elements import keplerian_to_eci
from scenarios.fleet_scenario import generate_fleet_scenario


def _case():
    timestamps = np.array([0.0, 2.0, 4.0])
    base = keplerian_to_eci(R_EARTH + 700e3, 0.001, 0.3, 0.0, 0.0, 0.0)
    initial_truth = {
        "sat_01": base + np.array([0.0, -30.0, 0.0, 0.01, 0.0, 0.0]),
        "sat_02": base + np.array([40.0, 10.0, 10.0, 0.0, -0.01, 0.0]),
        "sat_03": base + np.array([-20.0, 25.0, -5.0, -0.01, 0.01, 0.0]),
    }
    scenario = generate_fleet_scenario(
        timestamps=timestamps, initial_state_by_node=initial_truth
    )
    estimates = {
        "sat_01": initial_truth["sat_01"] + np.array([10.0, -5.0, 3.0, 0.0, 0.0, 0.0]),
        "sat_02": initial_truth["sat_02"] + np.array([-8.0, 7.0, -4.0, 0.0, 0.0, 0.0]),
        "sat_03": initial_truth["sat_03"] + np.array([5.0, -6.0, 8.0, 0.0, 0.0, 0.0]),
    }
    covariances = {
        node_id: np.diag([20.0, 20.0, 20.0, 0.1, 0.1, 0.1]) ** 2
        for node_id in scenario.node_ids
    }
    topology = fully_connected_topology(list(scenario.node_ids))
    relative = []
    for source in topology.node_ids:
        for target in topology.neighbors(source):
            for index, timestamp in enumerate(timestamps):
                state_i = scenario.trajectories[source].state_history_eci[index]
                state_j = scenario.trajectories[target].state_history_eci[index]
                relative.extend(
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
    anchors = [
        AbsolutePositionObservation(
            timestamp=float(timestamp),
            satellite_id="sat_01",
            measurement_eci=scenario.trajectories["sat_01"].state_history_eci[index, :3],
            covariance=np.eye(3),
            confidence=1.0,
            valid_flag=True,
        )
        for index, timestamp in enumerate(timestamps)
    ]
    return scenario, estimates, covariances, topology, relative, anchors


def test_distributed_fleet_ci_fuses_identical_18_state_semantics():
    scenario, estimates, covariances, topology, relative, anchors = _case()
    history = run_distributed_fleet_state_ci(
        timestamps=scenario.timestamps,
        initial_state_by_node=estimates,
        initial_covariance_by_node=covariances,
        topology=topology,
        inter_satellite_observations=relative,
        absolute_position_observations=anchors,
        node_ids=scenario.node_ids,
        process_noise_acceleration=1e-8,
        ci_grid_points=11,
    )

    assert history.node_ids == scenario.node_ids
    assert history.communication_stats.attempted_report_count == 18
    assert history.communication_stats.received_report_count == 18
    for node_id in scenario.node_ids:
        assert history.local_stacked_state_history_by_node[node_id].shape == (3, 18)
        assert history.local_stacked_covariance_history_by_node[node_id].shape == (3, 18, 18)
        assert history.physical_state_history_by_node[node_id].shape == (3, 6)
        assert history.pre_ci_stacked_state_history_by_node[node_id].shape == (3, 18)
        assert history.pre_ci_stacked_covariance_history_by_node[node_id].shape == (3, 18, 18)
        assert history.pre_ci_physical_state_history_by_node[node_id].shape == (3, 6)
        assert set(history.node_weight_history_by_node[node_id][0]) == set(scenario.node_ids)
        assert np.all(
            np.linalg.eigvalsh(
                history.local_stacked_covariance_history_by_node[node_id][-1]
            )
            > -1e-6
        )


def test_distributed_fleet_ci_accepts_body_angles_with_epoch_attitude():
    scenario, estimates, covariances, topology, _, anchors = _case()
    source = "sat_01"
    target = "sat_02"
    state_i = scenario.trajectories[source].state_history_eci[0]
    state_j = scenario.trajectories[target].state_history_eci[0]
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    observation = InterSatelliteObservation(
        timestamp=0.0,
        source_node_id=source,
        target_node_id=target,
        modality="AZ_EL",
        measurement=measure_relative_az_el(
            state_i,
            state_j,
            frame="BODY",
            quaternion_i2b_wxyz=quaternion,
        ),
        covariance=np.diag([1.0e-4, 1.0e-4]),
        confidence=1.0,
        valid_flag=True,
    )
    attitude = AttitudeEstimate(
        timestamp=0.0,
        satellite_id=source,
        quaternion_i2b_wxyz=quaternion,
        angular_velocity_body=np.zeros(3),
        gyro_bias=np.zeros(3),
        error_covariance=np.diag([1.0e-4] * 3 + [1.0e-6] * 6),
    )

    history = run_distributed_fleet_state_ci(
        timestamps=scenario.timestamps,
        initial_state_by_node=estimates,
        initial_covariance_by_node=covariances,
        topology=topology,
        inter_satellite_observations=[observation],
        absolute_position_observations=anchors,
        attitude_estimates=[attitude],
        node_ids=scenario.node_ids,
        process_noise_acceleration=1.0e-8,
        ci_grid_points=5,
        frame_by_modality={"AZ_EL": "BODY"},
    )

    assert "sat_01->sat_02:AZ_EL" in history.nis_history_by_node[source][0]
    assert np.all(
        np.isfinite(history.local_stacked_state_history_by_node[source])
    )


def test_distributed_fleet_ci_supports_loss_and_delay():
    scenario, estimates, covariances, topology, relative, anchors = _case()
    history = run_distributed_fleet_state_ci(
        timestamps=scenario.timestamps,
        initial_state_by_node=estimates,
        initial_covariance_by_node=covariances,
        topology=topology,
        inter_satellite_observations=relative,
        absolute_position_observations=anchors,
        node_ids=scenario.node_ids,
        packet_loss_rate_by_node={"sat_01": 1.0},
        delay_by_node={"sat_02": 2.0},
        ci_grid_points=11,
    )

    assert history.communication_stats.dropped_report_count == 6
    assert history.communication_stats.pending_report_count >= 0
    assert np.all(
        np.isfinite(history.local_stacked_state_history_by_node["sat_03"])
    )
