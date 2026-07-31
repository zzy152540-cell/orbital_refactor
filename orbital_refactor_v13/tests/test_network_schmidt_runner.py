import numpy as np

from cooperative.network_schmidt_runner import run_network_schmidt_filter
from cooperative.topology import chain_topology, fully_connected_topology
from interfaces.data_objects import ObservationMessage
from orbital_core.measurements import (
    measure_relative_az_el,
    measure_relative_range,
    measure_relative_range_rate,
)


def _case():
    timestamps = np.array([0.0, 1.0])
    base = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    states = {
        "sat_01": base + np.array([-1000.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        "sat_02": base.copy(),
        "sat_03": base + np.array([1200.0, -80.0, 30.0, 0.0, 0.0, 0.0]),
    }
    covariances = {node_id: np.eye(6) for node_id in states}
    observations = []
    for source, target in (("sat_01", "sat_02"), ("sat_02", "sat_03")):
        state_i, state_j = states[source], states[target]
        observations.extend(
            [
                ObservationMessage(
                    message_id=f"{source}-{target}-range",
                    observer_id=source,
                    target_id=target,
                    timestamp=0.0,
                    modality="RANGE",
                    measurement=np.array([measure_relative_range(state_i, state_j)]),
                    covariance=np.array([[1.0]]),
                ),
                ObservationMessage(
                    message_id=f"{source}-{target}-rate",
                    observer_id=source,
                    target_id=target,
                    timestamp=0.0,
                    modality="RANGE_RATE",
                    measurement=np.array([
                        measure_relative_range_rate(state_i, state_j)
                    ]),
                    covariance=np.array([[0.01]]),
                ),
                ObservationMessage(
                    message_id=f"{source}-{target}-angles",
                    observer_id=source,
                    target_id=target,
                    timestamp=0.0,
                    modality="AZ_EL",
                    measurement=measure_relative_az_el(
                        state_i, state_j, frame="RTN"
                    ),
                    covariance=np.eye(2) * 1e-4,
                    frame="RTN",
                ),
            ]
        )
    return timestamps, states, covariances, observations


def test_chain_network_builds_expected_local_schmidt_dimensions():
    timestamps, states, covariances, observations = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        process_noise_acceleration=0.0,
    )

    assert history.local_dimension_by_node == {
        "sat_01": 12,
        "sat_02": 18,
        "sat_03": 12,
    }
    assert set(history.active_cross_covariance_history_by_node["sat_02"]) == {
        "sat_01",
        "sat_03",
    }
    assert len(history.nis_history_by_node["sat_01"][0]) == 3
    assert len(history.nis_history_by_node["sat_02"][0]) == 3
    assert history.nis_history_by_node["sat_03"][0] == {}
    for node_id in history.node_ids:
        assert np.min(
            np.linalg.eigvalsh(history.joint_covariance_history_by_node[node_id][-1])
        ) >= -1e-8


def test_both_endpoint_routing_updates_the_target_local_filter_too():
    timestamps, states, covariances, observations = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        observation_usage="both_endpoints",
        process_noise_acceleration=0.0,
    )

    assert len(history.nis_history_by_node["sat_01"][0]) == 3
    assert len(history.nis_history_by_node["sat_02"][0]) == 6
    assert len(history.nis_history_by_node["sat_03"][0]) == 3


def test_fully_connected_network_uses_eighteen_dimensions_at_every_node():
    timestamps, states, covariances, _ = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=fully_connected_topology(list(states)),
        observation_messages=[],
        process_noise_acceleration=0.0,
    )

    assert history.local_dimension_by_node == {
        "sat_01": 18,
        "sat_02": 18,
        "sat_03": 18,
    }
    for node_id in history.node_ids:
        assert history.active_state_history_by_node[node_id].shape == (2, 6)
        assert history.active_covariance_history_by_node[node_id].shape == (2, 6, 6)


def test_synchronous_neighbor_refresh_modes_preserve_joint_psd():
    timestamps, states, covariances, observations = _case()
    for mode in ("safe_rescale", "zero_cross"):
        history = run_network_schmidt_filter(
            timestamps=timestamps,
            initial_state_by_node=states,
            initial_covariance_by_node=covariances,
            topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
            observation_messages=observations,
            process_noise_acceleration=0.0,
            consider_refresh_mode=mode,
        )
        for covariance_history in history.joint_covariance_history_by_node.values():
            assert min(np.linalg.eigvalsh(value).min() for value in covariance_history) >= -1e-8


def test_exact_transport_is_only_accepted_for_matching_provenance_baseline():
    timestamps, states, covariances, observations = _case()
    history = run_network_schmidt_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_01", "sat_02", "sat_03"]),
        observation_messages=observations,
        process_noise_acceleration=0.0,
        consider_refresh_mode="exact_if_compatible",
    )
    assert sum(history.refresh_diagnostics.values()) == 4
    assert history.refresh_diagnostics["accepted"] < 4
