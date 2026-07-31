import numpy as np

from cooperative.exact_transport_accumulator import ExactTransportAccumulator
from cooperative.exact_transport_protocol import apply_exact_transport_state_message
from cooperative.multi_neighbor_schmidt import initialize_multi_neighbor_schmidt


def test_lost_update_is_recovered_by_next_cumulative_message():
    covariance = np.diag([4.0, 5.0, 6.0, 0.2, 0.3, 0.4])
    baseline = np.arange(6.0)
    receiver = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="a", active_state=np.zeros(6),
        active_covariance=np.eye(6), neighbor_state_by_id={"b": baseline},
        neighbor_covariance_by_id={"b": covariance},
    )
    accumulator = ExactTransportAccumulator(
        source_node_id="b", lineage_id="b:0", reference_timestamp=0.0,
        reference_state=baseline, reference_covariance=covariance,
    )
    t1 = np.diag([0.8, 0.9, 1.0, 1.0, 1.0, 1.0]); q1 = np.eye(6) * 0.1
    t2 = np.diag([0.7, 1.0, 0.9, 1.0, 1.0, 1.0]); q2 = np.eye(6) * 0.2
    accumulator.append(timestamp=1.0, updated_state=baseline + 1.0,
                       error_transition=t1, independent_process_noise=q1,
                       information_ids=("update-1",))
    _lost_message = accumulator.build_message()
    accumulator.append(timestamp=2.0, updated_state=baseline + 2.0,
                       error_transition=t2, independent_process_noise=q2,
                       information_ids=("update-2",))
    cumulative = accumulator.build_message()
    outcome = apply_exact_transport_state_message(
        receiver, cumulative, expected_lineage_id="b:0"
    )
    expected_transition = t2 @ t1
    expected_noise = t2 @ q1 @ t2.T + q2
    expected_covariance = expected_transition @ covariance @ expected_transition.T + expected_noise
    assert outcome.accepted
    assert np.allclose(outcome.state.neighbor_covariance("b"), expected_covariance)
    assert cumulative.information_ids == ("update-1", "update-2")
    accumulator.acknowledge(cumulative)
    assert np.allclose(accumulator.accumulated_transition, np.eye(6))
    assert np.allclose(accumulator.accumulated_process_noise, 0.0)


def test_out_of_order_old_message_cannot_overwrite_newer_acknowledged_state():
    covariance = np.eye(6) * 2.0
    baseline = np.zeros(6)
    receiver = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="a", active_state=np.zeros(6),
        active_covariance=np.eye(6), neighbor_state_by_id={"b": baseline},
        neighbor_covariance_by_id={"b": covariance},
    )
    accumulator = ExactTransportAccumulator(
        source_node_id="b", lineage_id="b:0", reference_timestamp=0.0,
        reference_state=baseline, reference_covariance=covariance,
    )
    accumulator.append(timestamp=1.0, updated_state=np.ones(6),
                       error_transition=np.eye(6) * 0.9,
                       independent_process_noise=np.eye(6) * 0.1)
    old_message = accumulator.build_message()
    accumulator.append(timestamp=2.0, updated_state=np.ones(6) * 2.0,
                       error_transition=np.eye(6) * 0.8,
                       independent_process_noise=np.eye(6) * 0.2)
    newer_message = accumulator.build_message()
    newer = apply_exact_transport_state_message(receiver, newer_message)
    assert newer.accepted
    stale = apply_exact_transport_state_message(newer.state, old_message)
    assert not stale.accepted
    assert stale.reason in {"reference_covariance_mismatch", "reference_mean_mismatch"}
