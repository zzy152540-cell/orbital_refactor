import numpy as np

from cooperative.distributed_cooperative_runner import (
    run_distributed_cooperative_history,
)
from cooperative.message_transport import MessageChannel, TypedMessageBuffer
from cooperative.topology import chain_topology
from interfaces.data_objects import ObservationMessage
from orbital_core.dynamics import rk4_step_absolute


def _case(sample_count=2):
    timestamps = np.arange(sample_count, dtype=float)
    state_a = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    state_b = np.array([7.0e6 + 10.0, 0.0, 0.0, 0.0, 7500.0, 0.0])
    states = {
        "sat_a": np.tile(state_a, (sample_count, 1)),
        "sat_b": np.tile(state_b, (sample_count, 1)),
    }
    covariances = {
        "sat_a": np.tile(np.eye(6), (sample_count, 1, 1)),
        "sat_b": np.tile(np.eye(6), (sample_count, 1, 1)),
    }
    observation = ObservationMessage(
        message_id="a-to-b-range-0",
        observer_id="sat_a",
        target_id="sat_b",
        timestamp=0.0,
        modality="RANGE",
        measurement=np.array([11.0]),
        covariance=np.array([[0.25]]),
    )
    return timestamps, states, covariances, observation


def _run(*, state_channel=None, observation_channel=None, sample_count=2):
    timestamps, states, covariances, observation = _case(sample_count)
    return run_distributed_cooperative_history(
        timestamps=timestamps,
        state_history_by_node=states,
        covariance_history_by_node=covariances,
        topology=chain_topology(["sat_a", "sat_b"]),
        observation_messages=[observation],
        state_channel=state_channel,
        observation_channel=observation_channel,
    )


def test_both_observation_endpoints_update_without_cross_target_ci():
    result = _run()

    assert result.used_observation_ids_by_node["sat_a"][0] == [
        "a-to-b-range-0"
    ]
    assert result.used_observation_ids_by_node["sat_b"][0] == [
        "a-to-b-range-0"
    ]
    assert result.state_history_by_node["sat_a"][0, 0] < 7.0e6
    assert result.state_history_by_node["sat_b"][0, 0] > 7.0e6 + 10.0
    assert result.communication_stats.received_observation_count == 2


def test_observation_delay_affects_shared_copy_but_not_observer_local_copy():
    result = _run(
        observation_channel=MessageChannel(delay_by_source={"sat_a": 1.0})
    )

    assert result.used_observation_ids_by_node["sat_a"] == [
        ["a-to-b-range-0"],
        [],
    ]
    assert result.used_observation_ids_by_node["sat_b"] == [
        [],
        ["a-to-b-range-0"],
    ]
    assert result.state_history_by_node["sat_b"][0, 0] == 7.0e6 + 10.0
    propagated_without_observation = rk4_step_absolute(
        _case()[1]["sat_b"][0],
        1.0,
    )
    assert (
        result.state_history_by_node["sat_b"][1, 0]
        > propagated_without_observation[0]
    )


def test_observation_packet_loss_only_removes_remote_shared_copy():
    result = _run(
        observation_channel=MessageChannel(
            packet_loss_rate={"sat_a": 1.0},
            random_seed=3,
        )
    )

    assert result.used_observation_ids_by_node["sat_a"][0] == [
        "a-to-b-range-0"
    ]
    assert result.used_observation_ids_by_node["sat_b"] == [[], []]
    assert result.communication_stats.dropped_observation_count == 1


def test_observation_waits_until_neighbor_state_arrives():
    result = _run(
        state_channel=MessageChannel(
            delay_by_source={"sat_a": 1.0, "sat_b": 1.0}
        )
    )

    assert result.used_observation_ids_by_node["sat_a"] == [
        [],
        ["a-to-b-range-0"],
    ]
    assert result.used_observation_ids_by_node["sat_b"] == [
        [],
        ["a-to-b-range-0"],
    ]
    assert result.oosm_delay_history_by_node["sat_a"][1] == {
        "a-to-b-range-0": 1.0
    }
    assert result.oosm_delay_history_by_node["sat_b"][1] == {
        "a-to-b-range-0": 1.0
    }


def test_typed_buffer_orders_arrivals_and_suppresses_duplicates():
    _, _, _, first = _case()
    second = ObservationMessage(
        message_id="second",
        observer_id="sat_a",
        target_id="sat_b",
        timestamp=0.0,
        modality="RANGE",
        measurement=np.array([10.0]),
        covariance=np.array([[1.0]]),
        source_timestamp=0.0,
        arrival_timestamp=1.0,
    )
    first.arrival_timestamp = 2.0
    buffer = TypedMessageBuffer[ObservationMessage]()

    assert buffer.push(first)
    assert buffer.push(second)
    assert not buffer.push(second)
    assert [message.message_id for message in buffer.pop_available(1.0)] == [
        "second"
    ]
    assert [message.message_id for message in buffer.pop_available(2.0)] == [
        "a-to-b-range-0"
    ]
