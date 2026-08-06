import numpy as np

from interfaces.data_objects import AbsolutePositionObservation
from orbital_core.fleet_centralized_ekf import FleetCentralizedEKF
from orbital_core.metrics import compute_nees, compute_nees_history


def test_absolute_anchor_reduces_selected_position_covariance():
    filter_obj = FleetCentralizedEKF(["sat_01", "sat_02"])
    state = np.array([
        7e6, 0.0, 0.0, 0.0, 7500.0, 0.0,
        7e6 + 100.0, 0.0, 0.0, 0.0, 7500.0, 0.0,
    ])
    covariance = np.eye(12) * 100.0
    updated_state, updated_covariance, diagnostics = (
        filter_obj.update_absolute_positions(
            state,
            covariance,
            [
                AbsolutePositionObservation(
                    timestamp=0.0,
                    satellite_id="sat_01",
                    measurement_eci=state[:3] + np.array([1.0, 0.0, 0.0]),
                    covariance=np.eye(3),
                    confidence=1.0,
                    valid_flag=True,
                )
            ],
        )
    )

    assert updated_state[0] > state[0]
    assert updated_covariance[0, 0] < covariance[0, 0]
    assert updated_covariance[6, 6] == covariance[6, 6]
    assert "sat_01:ABS_POSITION" in diagnostics.nis_by_observation


def test_nees_helpers_match_identity_covariance():
    error = np.array([1.0, 2.0])
    assert compute_nees(error, np.eye(2)) == 5.0
    values = compute_nees_history(
        np.array([[1.0, 2.0], [2.0, 0.0]]),
        np.zeros((2, 2)),
        np.tile(np.eye(2), (2, 1, 1)),
    )
    np.testing.assert_allclose(values, np.array([5.0, 4.0]))
