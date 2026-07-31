import numpy as np
import pytest

from cooperative.cooperative_update import update_local_state
from interfaces.data_objects import ObservationMessage, StateMessage, TargetEstimate
from orbital_core.inter_satellite_model import RelativeMeasurementModel


def _local(*, information_ids=()):
    return TargetEstimate(
        estimator_node_id="sat_a",
        target_node_id="sat_a",
        timestamp=0.0,
        state_estimate=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(6),
        quality_score=1.0,
        information_ids=information_ids,
    )


def _neighbor(covariance_scale=1.0):
    return StateMessage(
        source_node_id="sat_b",
        target_node_id="sat_b",
        timestamp=0.0,
        state_estimate=np.array([11.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        covariance=np.eye(6) * covariance_scale,
        quality_score=1.0,
    )


def _observation(*, measurement=11.0):
    return ObservationMessage(
        message_id="obs-001",
        observer_id="sat_a",
        target_id="sat_b",
        timestamp=0.0,
        modality="RANGE",
        measurement=np.array([measurement]),
        covariance=np.array([[0.25]]),
    )


def test_relative_measurement_model_unifies_prediction_and_jacobians():
    model = RelativeMeasurementModel("range")
    predicted = model.predict(_local().state_estimate, _neighbor().state_estimate)
    h_local, h_neighbor = model.jacobians(
        _local().state_estimate,
        _neighbor().state_estimate,
    )

    np.testing.assert_allclose(predicted, np.array([10.0]))
    np.testing.assert_allclose(h_local, -h_neighbor)
    np.testing.assert_allclose(h_local[0, :3], np.array([-1.0, 0.0, 0.0]))


def test_local_update_propagates_neighbor_uncertainty_and_records_observation():
    result = update_local_state(
        local_estimate=_local(),
        neighbor_state=_neighbor(covariance_scale=4.0),
        observation=_observation(),
    )

    np.testing.assert_allclose(
        result.effective_measurement_covariance,
        np.array([[4.25]]),
    )
    assert result.estimate.target_node_id == "sat_a"
    assert result.estimate.estimator_node_id == "sat_a"
    assert result.estimate.information_ids == ("obs-001",)
    assert result.estimate.state_estimate[0] < 1.0
    assert np.min(np.linalg.eigvalsh(result.estimate.covariance)) >= -1e-12


def test_larger_neighbor_uncertainty_weakens_local_correction():
    certain = update_local_state(
        local_estimate=_local(),
        neighbor_state=_neighbor(covariance_scale=0.01),
        observation=_observation(),
    )
    uncertain = update_local_state(
        local_estimate=_local(),
        neighbor_state=_neighbor(covariance_scale=100.0),
        observation=_observation(),
    )

    certain_change = abs(certain.estimate.state_estimate[0] - 1.0)
    uncertain_change = abs(uncertain.estimate.state_estimate[0] - 1.0)
    assert uncertain_change < certain_change


def test_local_update_accepts_observation_generated_by_neighbor():
    observation = _observation()
    observation.observer_id = "sat_b"
    observation.target_id = "sat_a"

    result = update_local_state(
        local_estimate=_local(),
        neighbor_state=_neighbor(),
        observation=observation,
    )

    assert result.estimate.state_estimate[0] < 1.0


def test_local_update_rejects_unrelated_observation_endpoints():
    observation = _observation()
    observation.observer_id = "sat_c"

    with pytest.raises(ValueError, match="endpoints"):
        update_local_state(
            local_estimate=_local(),
            neighbor_state=_neighbor(),
            observation=observation,
        )


def test_local_update_rejects_reused_observation():
    with pytest.raises(ValueError, match="already been used"):
        update_local_state(
            local_estimate=_local(information_ids=("obs-001",)),
            neighbor_state=_neighbor(),
            observation=_observation(),
        )
