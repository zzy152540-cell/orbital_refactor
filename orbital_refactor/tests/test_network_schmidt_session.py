import numpy as np

from cooperative.exact_transport_accumulator import ExactTransportAccumulator
from cooperative.link_lifecycle import LinkLifecycleState
from cooperative.multi_neighbor_schmidt import initialize_multi_neighbor_schmidt
from cooperative.network_schmidt_session import NetworkSchmidtSession


def test_session_closes_long_separation_with_receiver_baseline_and_new_lineage():
    active = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    neighbor = active + np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    covariance = np.eye(6)
    state = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="receiver",
        active_state=active, active_covariance=covariance,
        neighbor_state_by_id={"source": neighbor},
        neighbor_covariance_by_id={"source": covariance},
    )
    session = NetworkSchmidtSession(
        state, lineage_by_neighbor={"source": "source->receiver:0"},
        process_noise_acceleration=0.0, history_window=1.0,
        max_pinned_age=2.0,
    )
    initial = ExactTransportAccumulator(
        source_node_id="source", lineage_id="source->receiver:0",
        reference_timestamp=0.0, reference_state=neighbor,
        reference_covariance=covariance,
    )
    initial.append(
        timestamp=0.0, updated_state=neighbor,
        error_transition=np.eye(6),
        independent_process_noise=np.zeros((6, 6)),
        information_ids=("source:initial",),
    )
    assert session.step(
        0.0, state_messages=(initial.build_message(),)
    ).message_results[0].accepted

    session.step(3.0)
    assert session.link_by_neighbor[
        "source"
    ].state == LinkLifecycleState.RESYNC_REQUIRED
    assert not session.step(
        3.0, state_messages=(initial.build_message(),)
    ).message_results[0].accepted

    baseline = session.establish_resynchronized_link(
        "source", lineage_id="source->receiver:resync:1"
    )
    restarted = ExactTransportAccumulator(
        source_node_id="source", lineage_id=baseline.lineage_id,
        reference_timestamp=baseline.timestamp,
        reference_state=baseline.state_estimate,
        reference_covariance=baseline.covariance,
    )
    restarted.append(
        timestamp=3.0,
        updated_state=baseline.state_estimate + np.array([0.5, 0, 0, 0, 0, 0]),
        error_transition=np.eye(6) * 0.95,
        independent_process_noise=np.eye(6) * 0.01,
        information_ids=("source:resync-update",),
    )
    recovered = session.step(
        3.0, state_messages=(restarted.build_message(),)
    )

    assert recovered.message_results[0].accepted
    assert session.link_by_neighbor[
        "source"
    ].state == LinkLifecycleState.ACTIVE
    assert session.link_by_neighbor[
        "source"
    ].resynchronization_count == 1
