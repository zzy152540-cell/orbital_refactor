from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from cooperative.multi_neighbor_schmidt import MultiNeighborSchmidtState
from cooperative.schmidt_refresh import exact_transport_eligibility, refresh_consider_neighbor
from interfaces.data_objects import CovarianceTransportEvent, StateMessage

Array = np.ndarray


@dataclass(frozen=True)
class ExactTransportReceiveResult:
    state: MultiNeighborSchmidtState
    accepted: bool
    reason: str


def build_exact_transport_state_message(
    *, source_node_id: str, timestamp: float, reference_timestamp: float,
    reference_state: Array, reference_covariance: Array,
    updated_state: Array, error_transition: Array,
    independent_process_noise: Array, lineage_id: str,
    information_ids: tuple[str, ...] = (), quality_score: float = 1.0,
    transport_events: tuple[CovarianceTransportEvent, ...] = (),
) -> StateMessage:
    """Create a self-checking state message relative to a common baseline."""
    reference_covariance = _matrix(reference_covariance)
    transition = _matrix(error_transition)
    noise = _matrix(independent_process_noise)
    advertised_covariance = transition @ reference_covariance @ transition.T + noise
    return StateMessage(
        source_node_id=str(source_node_id), target_node_id=str(source_node_id),
        timestamp=float(timestamp), state_estimate=_vector(updated_state),
        covariance=0.5 * (advertised_covariance + advertised_covariance.T),
        quality_score=float(quality_score), source_timestamp=float(timestamp),
        information_ids=tuple(str(value) for value in information_ids),
        lineage_id=str(lineage_id), reference_timestamp=float(reference_timestamp),
        error_transition=transition, accumulated_process_noise=noise,
        reference_state_estimate=_vector(reference_state),
        reference_covariance=reference_covariance,
        transport_events=tuple(transport_events),
    )


def apply_exact_transport_state_message(
    state: MultiNeighborSchmidtState, message: StateMessage, *,
    expected_lineage_id: str | None = None,
) -> ExactTransportReceiveResult:
    """Validate provenance and transport one remote consider block exactly."""
    neighbor_id = str(message.target_node_id)
    if not message.valid_flag:
        return ExactTransportReceiveResult(state, False, "invalid_message")
    if neighbor_id != str(message.source_node_id) or neighbor_id not in state.neighbor_ids:
        return ExactTransportReceiveResult(state, False, "wrong_target")
    if expected_lineage_id is not None and message.lineage_id != expected_lineage_id:
        return ExactTransportReceiveResult(state, False, "lineage_mismatch")
    required = (
        message.reference_state_estimate, message.reference_covariance,
        message.error_transition, message.accumulated_process_noise,
    )
    if any(value is None for value in required):
        return ExactTransportReceiveResult(state, False, "missing_provenance")
    eligible, reason = exact_transport_eligibility(
        state, neighbor_id=neighbor_id,
        reference_covariance=message.reference_covariance,
        reference_mean=message.reference_state_estimate,
    )
    if not eligible:
        return ExactTransportReceiveResult(state, False, reason)
    expected = (
        _matrix(message.error_transition) @ _matrix(message.reference_covariance)
        @ _matrix(message.error_transition).T
        + _matrix(message.accumulated_process_noise)
    )
    if not np.allclose(expected, _matrix(message.covariance), rtol=1e-8, atol=1e-10):
        return ExactTransportReceiveResult(state, False, "advertised_covariance_mismatch")
    updated = refresh_consider_neighbor(
        state, neighbor_id=neighbor_id, neighbor_state=message.state_estimate,
        mode="exact_transport", error_transition=message.error_transition,
        independent_process_noise=message.accumulated_process_noise,
    )
    updated = replace(updated, timestamp=float(message.timestamp))
    return ExactTransportReceiveResult(updated, True, "accepted")


def _vector(value: Array) -> Array:
    return np.asarray(value, dtype=float).reshape(6).copy()


def _matrix(value: Array) -> Array:
    return np.asarray(value, dtype=float).reshape(6, 6).copy()
