import numpy as np

from cooperative.cooperative_update import update_local_state
from cooperative.schmidt_consider import SchmidtState, schmidt_predict, schmidt_update
from interfaces.data_objects import ObservationMessage, StateMessage, TargetEstimate


def _state():
    return SchmidtState(
        timestamp=0.0,
        active_node_id="sat_a",
        consider_node_id="sat_b",
        active_state=np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        consider_state=np.array([7.0e6 + 1000.0, 0.0, 0.0, 0.0, 7500.0, 0.0]),
        active_covariance=np.eye(6),
        consider_covariance=2.0 * np.eye(6),
        cross_covariance=np.zeros((6, 6)),
    )


def _observation(message_id="range-0"):
    return ObservationMessage(
        message_id=message_id,
        physical_observation_id=message_id,
        observer_id="sat_a",
        target_id="sat_b",
        timestamp=0.0,
        modality="RANGE",
        measurement=np.array([1001.0]),
        covariance=np.array([[0.25]]),
    )


def test_first_schmidt_update_matches_zero_cross_consider_approximation():
    state = _state()
    schmidt = schmidt_update(state, _observation())
    approximate = update_local_state(
        local_estimate=TargetEstimate(
            estimator_node_id="sat_a",
            target_node_id="sat_a",
            timestamp=0.0,
            state_estimate=state.active_state,
            covariance=state.active_covariance,
            quality_score=1.0,
        ),
        neighbor_state=StateMessage(
            source_node_id="sat_b",
            target_node_id="sat_b",
            timestamp=0.0,
            state_estimate=state.consider_state,
            covariance=state.consider_covariance,
            quality_score=1.0,
        ),
        observation=_observation(),
    )

    np.testing.assert_allclose(
        schmidt.state.active_state,
        approximate.estimate.state_estimate,
    )
    np.testing.assert_allclose(
        schmidt.state.active_covariance,
        approximate.estimate.covariance,
    )
    np.testing.assert_allclose(
        schmidt.state.consider_covariance,
        state.consider_covariance,
    )


def test_schmidt_update_creates_cross_covariance_and_preserves_joint_psd():
    result = schmidt_update(_state(), _observation())

    assert np.linalg.norm(result.state.cross_covariance) > 0.0
    assert np.min(
        np.linalg.eigvalsh(result.state.augmented_covariance)
    ) >= -1e-10


def test_schmidt_prediction_propagates_cross_covariance():
    updated = schmidt_update(_state(), _observation()).state
    predicted = schmidt_predict(
        updated,
        1.0,
        process_noise_acceleration=0.0,
    )

    assert predicted.timestamp == 1.0
    assert np.linalg.norm(predicted.cross_covariance) > 0.0
    assert not np.allclose(predicted.cross_covariance, updated.cross_covariance)
    assert np.min(
        np.linalg.eigvalsh(predicted.augmented_covariance)
    ) >= -1e-8
