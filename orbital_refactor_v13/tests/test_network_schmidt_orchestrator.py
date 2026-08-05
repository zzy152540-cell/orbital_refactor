import numpy as np

from cooperative.network_schmidt_orchestrator import (
    NetworkSchmidtOrchestrator,
    TransportSourceUpdate,
)
from cooperative.topology import chain_topology


def test_online_orchestrator_automatically_resynchronizes_long_suspension():
    states = {
        "a": np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        "b": np.array([7.001e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
    }
    covariances = {node: np.eye(6) for node in states}
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["a", "b"]),
        process_noise_acceleration=0.0,
        history_window=1.0, max_pinned_age=1.0,
    )

    def updates(timestamp):
        return {
            node: TransportSourceUpdate(
                state=value + np.array([timestamp, 0, 0, 0, 0, 0]),
                error_transition=np.eye(6) * 0.99,
                independent_process_noise=np.eye(6) * 0.01,
                information_ids=(f"{node}:{timestamp}",),
            )
            for node, value in states.items()
        }

    active = {"a": ("b",), "b": ("a",)}
    suspended = {"a": (), "b": ()}
    first = orchestrator.step(
        0.0, topology_version=0, active_neighbors_by_node=active,
        source_update_by_node=updates(0.0),
    )
    assert first.accepted_message_count == 2
    orchestrator.step(
        1.0, topology_version=1, active_neighbors_by_node=suspended,
        source_update_by_node=updates(1.0),
    )
    orchestrator.step(
        2.0, topology_version=1, active_neighbors_by_node=suspended,
        source_update_by_node=updates(2.0),
    )
    recovered = orchestrator.step(
        3.0, topology_version=2, active_neighbors_by_node=active,
        source_update_by_node=updates(3.0),
    )

    assert recovered.accepted_message_count == 2
    assert recovered.rejected_message_count == 0
    assert len(recovered.resynchronized_links) == 2
    assert all(
        ":resync:1" in lineage
        for _, _, lineage in recovered.resynchronized_links
    )
    assert all(
        session.link_by_neighbor[next(iter(session.link_by_neighbor))]
        .resynchronization_count == 1
        for session in orchestrator.sessions.values()
    )


def test_online_orchestrator_buffers_delay_and_reports_packet_loss():
    states = {
        "a": np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        "b": np.array([7.001e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
    }
    covariances = {node: np.eye(6) for node in states}
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["a", "b"]),
        process_noise_acceleration=0.0,
        communication_delay=2.0, packet_loss_rate=1.0,
    )
    active = {"a": ("b",), "b": ("a",)}
    updates = {
        node: TransportSourceUpdate(
            state=value, error_transition=np.eye(6),
            independent_process_noise=np.zeros((6, 6)),
            information_ids=(f"{node}:0",),
        )
        for node, value in states.items()
    }

    result = orchestrator.step(
        0.0, topology_version=0, active_neighbors_by_node=active,
        source_update_by_node=updates,
    )

    assert result.transmitted_message_count == 0
    assert result.dropped_message_count == 2
    assert result.accepted_message_count == 0
    assert result.protocol_rejected_message_count == 0


def test_stop_and_wait_keeps_only_one_delayed_message_per_link():
    states = {
        "a": np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        "b": np.array([7.001e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
    }
    covariances = {node: np.eye(6) for node in states}
    orchestrator = NetworkSchmidtOrchestrator(
        initial_state_by_node=states,
        initial_covariance_by_node=covariances,
        topology=chain_topology(["a", "b"]),
        process_noise_acceleration=0.0, communication_delay=4.0,
    )
    active = {"a": ("b",), "b": ("a",)}

    def updates(timestamp):
        return {
            node: TransportSourceUpdate(
                state=value, error_transition=np.eye(6),
                independent_process_noise=np.zeros((6, 6)),
                information_ids=(f"{node}:{timestamp}",),
            )
            for node, value in states.items()
        }

    first = orchestrator.step(
        0.0, topology_version=0, active_neighbors_by_node=active,
        source_update_by_node=updates(0.0),
    )
    second = orchestrator.step(
        2.0, topology_version=0, active_neighbors_by_node=active,
        source_update_by_node=updates(2.0),
    )
    delivered = orchestrator.step(
        4.0, topology_version=0, active_neighbors_by_node=active,
        source_update_by_node=updates(4.0),
    )

    assert first.transmitted_message_count == 2
    assert second.transmitted_message_count == 0
    assert delivered.accepted_message_count == 2
