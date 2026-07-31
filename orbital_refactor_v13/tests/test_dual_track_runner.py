import numpy as np

from cooperative.dual_track_runner import (
    run_dual_track_distributed_cooperative_filter,
)
from cooperative.recursive_cooperative_runner import (
    run_recursive_distributed_cooperative_filter,
)
from cooperative.topology import chain_topology
from interfaces.data_objects import ObservationMessage


def _case():
    timestamps = np.array([0.0, 1.0, 2.0])
    states = {
        "sat_a": np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        "sat_b": np.array([7.0e6 + 1000.0, 0.0, 0.0, 0.0, 7500.0, 0.0]),
    }
    covariances = {node_id: np.eye(6) for node_id in states}
    observations = [
        ObservationMessage(
            message_id=f"range-{timestamp}",
            physical_observation_id=f"physical-range-{timestamp}",
            observer_id="sat_a",
            target_id="sat_b",
            timestamp=float(timestamp),
            modality="RANGE",
            measurement=np.array([1001.0]),
            covariance=np.array([[1.0]]),
        )
        for timestamp in timestamps
    ]
    return timestamps, states, covariances, observations


def test_dual_track_private_path_matches_observer_only_filter():
    timestamps, states, covariances, observations = _case()
    topology = chain_topology(["sat_a", "sat_b"])
    private_only = run_recursive_distributed_cooperative_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=topology,
        observation_messages=observations,
        observation_usage="observer_only",
        process_noise_acceleration=0.0,
    )
    dual = run_dual_track_distributed_cooperative_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=topology,
        observation_messages=observations,
        process_noise_acceleration=0.0,
    )

    for node_id in topology.node_ids:
        np.testing.assert_allclose(
            dual.private_history.posterior_state_history_by_node[node_id],
            private_only.posterior_state_history_by_node[node_id],
        )
        np.testing.assert_allclose(
            dual.private_history.posterior_covariance_history_by_node[node_id],
            private_only.posterior_covariance_history_by_node[node_id],
        )


def test_dual_track_shared_observation_changes_only_cooperative_target_track():
    timestamps, states, covariances, observations = _case()
    dual = run_dual_track_distributed_cooperative_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_a", "sat_b"]),
        observation_messages=observations,
        process_noise_acceleration=0.0,
    )

    assert dual.private_history.used_observation_ids_by_node["sat_b"][0] == []
    assert dual.cooperative_history.used_observation_ids_by_node["sat_b"][0] == [
        "physical-range-0.0"
    ]
    assert not np.allclose(
        dual.cooperative_history.posterior_state_history_by_node["sat_b"],
        dual.private_history.posterior_state_history_by_node["sat_b"],
    )
