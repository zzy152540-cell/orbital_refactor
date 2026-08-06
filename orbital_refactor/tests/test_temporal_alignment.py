import numpy as np
import pytest

from cooperative.cooperative_update import update_local_state
from cooperative.temporal_alignment import (
    align_state_message,
    apply_delayed_cooperative_update,
    propagate_state_covariance,
)
from interfaces.data_objects import ObservationMessage, StateMessage, TargetEstimate
from orbital_core.dynamics import rk4_step_absolute


def _local(timestamp=0.0):
    return TargetEstimate(
        estimator_node_id="sat_a",
        target_node_id="sat_a",
        timestamp=timestamp,
        state_estimate=np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        covariance=np.eye(6),
        quality_score=1.0,
    )


def _neighbor(timestamp=0.0):
    return StateMessage(
        source_node_id="sat_b",
        target_node_id="sat_b",
        timestamp=timestamp,
        state_estimate=np.array(
            [7.0e6 + 10.0, 0.0, 0.0, 0.0, 7500.0, 0.0]
        ),
        covariance=2.0 * np.eye(6),
        quality_score=1.0,
        source_timestamp=timestamp,
        arrival_timestamp=timestamp,
    )


def _observation(timestamp=0.0):
    return ObservationMessage(
        message_id=f"range-{timestamp}",
        observer_id="sat_a",
        target_id="sat_b",
        timestamp=timestamp,
        modality="RANGE",
        measurement=np.array([11.0]),
        covariance=np.array([[0.25]]),
    )


def test_align_state_message_uses_orbital_dynamics_and_preserves_source_time():
    message = _neighbor()
    aligned = align_state_message(
        message,
        1.0,
        process_noise_acceleration=0.0,
    )

    np.testing.assert_allclose(
        aligned.state_estimate,
        rk4_step_absolute(message.state_estimate, 1.0),
    )
    assert aligned.timestamp == 1.0
    assert aligned.source_timestamp == 0.0
    assert np.min(np.linalg.eigvalsh(aligned.covariance)) >= -1e-10


def test_state_alignment_rejects_backward_propagation():
    with pytest.raises(ValueError, match="backward"):
        align_state_message(_neighbor(timestamp=1.0), 0.0)


def test_zero_delay_temporal_update_matches_direct_update():
    direct = update_local_state(
        local_estimate=_local(),
        neighbor_state=_neighbor(),
        observation=_observation(),
    )
    temporal = apply_delayed_cooperative_update(
        local_estimate_at_observation=_local(),
        neighbor_state=_neighbor(),
        observation=_observation(),
        output_timestamp=0.0,
    )

    np.testing.assert_allclose(
        temporal.estimate.state_estimate,
        direct.estimate.state_estimate,
    )
    np.testing.assert_allclose(
        temporal.estimate.covariance,
        direct.estimate.covariance,
    )
    assert temporal.posterior_propagation_dt == 0.0


def test_delayed_update_occurs_at_measurement_time_then_propagates_forward():
    direct = update_local_state(
        local_estimate=_local(),
        neighbor_state=_neighbor(),
        observation=_observation(),
    )
    expected_state, expected_covariance = propagate_state_covariance(
        direct.estimate.state_estimate,
        direct.estimate.covariance,
        2.0,
        process_noise_acceleration=0.0,
    )

    delayed = apply_delayed_cooperative_update(
        local_estimate_at_observation=_local(),
        neighbor_state=_neighbor(),
        observation=_observation(),
        output_timestamp=2.0,
        process_noise_acceleration=0.0,
    )

    np.testing.assert_allclose(delayed.estimate.state_estimate, expected_state)
    np.testing.assert_allclose(delayed.estimate.covariance, expected_covariance)
    assert delayed.observation_timestamp == 0.0
    assert delayed.output_timestamp == 2.0
    assert delayed.posterior_propagation_dt == 2.0
