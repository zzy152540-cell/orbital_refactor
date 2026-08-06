from dataclasses import replace

import numpy as np

from cooperative.exact_transport_protocol import (
    apply_exact_transport_state_message,
    build_exact_transport_state_message,
)
from cooperative.message_transport import MessageChannel, TypedMessageBuffer
from cooperative.multi_neighbor_schmidt import initialize_multi_neighbor_schmidt


def _case():
    covariance = np.diag([4.0, 4.0, 4.0, 0.2, 0.2, 0.2])
    state = initialize_multi_neighbor_schmidt(
        timestamp=0.0, active_node_id="a", active_state=np.zeros(6),
        active_covariance=np.eye(6), neighbor_state_by_id={"b": np.ones(6)},
        neighbor_covariance_by_id={"b": covariance},
    )
    transition = np.diag([0.5, 0.6, 0.7, 1.0, 1.0, 1.0])
    message = build_exact_transport_state_message(
        source_node_id="b", timestamp=1.0, reference_timestamp=0.0,
        reference_state=np.ones(6), reference_covariance=covariance,
        updated_state=np.full(6, 2.0), error_transition=transition,
        independent_process_noise=np.eye(6) * 0.01, lineage_id="b:epoch0",
    )
    return state, message


def test_delayed_exact_transport_message_is_applied_when_baseline_still_matches():
    state, message = _case()
    channel = MessageChannel(delay_by_source={"b": 2.0}, random_seed=1)
    buffer = TypedMessageBuffer()
    buffer.push(channel.transmit(message))
    assert buffer.pop_available(2.9) == []
    received = buffer.pop_available(3.0)[0]
    result = apply_exact_transport_state_message(
        state, received, expected_lineage_id="b:epoch0"
    )
    assert result.accepted
    assert np.allclose(result.state.neighbor_covariance("b"), received.covariance)


def test_protocol_rejects_wrong_lineage_and_tampered_covariance():
    state, message = _case()
    assert apply_exact_transport_state_message(
        state, message, expected_lineage_id="other"
    ).reason == "lineage_mismatch"
    tampered = replace(message, covariance=message.covariance + np.eye(6))
    assert apply_exact_transport_state_message(
        state, tampered, expected_lineage_id="b:epoch0"
    ).reason == "advertised_covariance_mismatch"


def test_protocol_rejects_stale_message_after_local_baseline_changes():
    state, message = _case()
    covariance = state.joint_covariance.copy()
    covariance[6:, 6:] *= 1.1
    changed = replace(state, joint_covariance=covariance)
    result = apply_exact_transport_state_message(changed, message)
    assert not result.accepted
    assert result.reason == "reference_covariance_mismatch"


def test_packet_loss_prevents_state_message_delivery():
    _, message = _case()
    channel = MessageChannel(packet_loss_rate={"b": 1.0}, random_seed=2)
    assert channel.transmit(message) is None
