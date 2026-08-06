import numpy as np

from cooperative.message_transport import MessageChannel
from cooperative.recursive_cooperative_runner import (
    run_recursive_distributed_cooperative_filter,
)
from cooperative.temporal_alignment import propagate_state_covariance
from cooperative.topology import chain_topology
from interfaces.data_objects import ObservationMessage


def _inputs():
    timestamps = np.array([0.0, 1.0, 2.0])
    states = {
        "sat_a": np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        "sat_b": np.array([7.0e6 + 10.0, 0.0, 0.0, 7500.0, 7500.0, 0.0]),
    }
    covariances = {
        "sat_a": np.eye(6),
        "sat_b": 2.0 * np.eye(6),
    }
    observation = ObservationMessage(
        message_id="range-at-zero",
        observer_id="sat_a",
        target_id="sat_b",
        timestamp=0.0,
        modality="RANGE",
        measurement=np.array([11.0]),
        covariance=np.array([[0.25]]),
    )
    return timestamps, states, covariances, observation


def _run(*, observations=(), state_channel=None, observation_channel=None):
    timestamps, states, covariances, _ = _inputs()
    return run_recursive_distributed_cooperative_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_a", "sat_b"]),
        observation_messages=observations,
        state_channel=state_channel,
        observation_channel=observation_channel,
        process_noise_acceleration=0.0,
    )


def test_recursive_filter_propagates_previous_posterior_without_observations():
    result = _run()
    _, states, covariances, _ = _inputs()

    expected_state, expected_covariance = propagate_state_covariance(
        states["sat_a"],
        covariances["sat_a"],
        1.0,
        process_noise_acceleration=0.0,
    )
    np.testing.assert_allclose(
        result.predicted_state_history_by_node["sat_a"][1],
        expected_state,
    )
    np.testing.assert_allclose(
        result.predicted_covariance_history_by_node["sat_a"][1],
        expected_covariance,
    )
    np.testing.assert_allclose(
        result.posterior_state_history_by_node["sat_a"],
        result.predicted_state_history_by_node["sat_a"],
    )


def test_cooperative_posterior_becomes_next_epoch_prediction():
    observation = _inputs()[3]
    result = _run(observations=[observation])

    posterior_zero = result.posterior_state_history_by_node["sat_a"][0]
    covariance_zero = result.posterior_covariance_history_by_node["sat_a"][0]
    expected_state, expected_covariance = propagate_state_covariance(
        posterior_zero,
        covariance_zero,
        1.0,
        process_noise_acceleration=0.0,
    )
    np.testing.assert_allclose(
        result.predicted_state_history_by_node["sat_a"][1],
        expected_state,
    )
    np.testing.assert_allclose(
        result.predicted_covariance_history_by_node["sat_a"][1],
        expected_covariance,
    )
    assert result.used_observation_ids_by_node["sat_a"][0] == ["range-at-zero"]


def test_late_remote_observation_replay_matches_on_time_result():
    observation = _inputs()[3]
    on_time = _run(observations=[observation])
    delayed = _run(
        observations=[observation],
        observation_channel=MessageChannel(delay_by_source={"sat_a": 1.0}),
    )

    np.testing.assert_allclose(
        delayed.posterior_state_history_by_node["sat_b"],
        on_time.posterior_state_history_by_node["sat_b"],
    )
    np.testing.assert_allclose(
        delayed.posterior_covariance_history_by_node["sat_b"],
        on_time.posterior_covariance_history_by_node["sat_b"],
    )
    assert on_time.replayed_from_index_by_node["sat_b"] == [0, None, None]
    assert delayed.replayed_from_index_by_node["sat_b"] == [None, 0, None]


def test_delayed_state_causes_observation_to_wait_then_replay():
    observation = _inputs()[3]
    result = _run(
        observations=[observation],
        state_channel=MessageChannel(
            delay_by_source={"sat_a": 1.0, "sat_b": 1.0}
        ),
    )

    assert result.replayed_from_index_by_node["sat_a"] == [None, 0, None]
    assert result.replayed_from_index_by_node["sat_b"] == [None, 0, None]
    assert result.used_observation_ids_by_node["sat_a"][0] == ["range-at-zero"]
    assert result.used_observation_ids_by_node["sat_b"][0] == ["range-at-zero"]
    assert result.communication_stats.deferred_observation_count == 2


def test_physical_observation_is_used_once_despite_distinct_message_copies():
    observation = _inputs()[3]
    observation.physical_observation_id = "physical-range-zero"
    retransmission = ObservationMessage(
        message_id="range-at-zero-retransmission",
        physical_observation_id="physical-range-zero",
        observer_id=observation.observer_id,
        target_id=observation.target_id,
        timestamp=observation.timestamp,
        modality=observation.modality,
        measurement=observation.measurement.copy(),
        covariance=observation.covariance.copy(),
    )

    result = _run(
        observations=[observation, retransmission],
        observation_channel=MessageChannel(packet_loss_rate={"sat_a": 1.0}),
    )

    assert result.used_observation_ids_by_node["sat_a"][0] == [
        "physical-range-zero"
    ]
    assert len(result.nis_history_by_node["sat_a"][0]) == 1


def test_observer_only_usage_does_not_transmit_observation_to_target():
    observation = _inputs()[3]
    timestamps, states, covariances, _ = _inputs()
    result = run_recursive_distributed_cooperative_filter(
        timestamps=timestamps,
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["sat_a", "sat_b"]),
        observation_messages=[observation],
        observation_usage="observer_only",
        process_noise_acceleration=0.0,
    )

    assert result.used_observation_ids_by_node["sat_a"][0] == ["range-at-zero"]
    assert result.used_observation_ids_by_node["sat_b"][0] == []
    assert result.communication_stats.attempted_observation_count == 0
