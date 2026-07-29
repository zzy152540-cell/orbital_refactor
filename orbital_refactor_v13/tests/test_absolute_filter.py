import numpy as np

from orbital_core.absolute_filter import AbsoluteOrbitEKF
from orbital_core.dynamics import make_process_noise, propagate_absolute_orbit


def test_absolute_orbit_ekf_predict_matches_absolute_propagator():
    state = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    covariance = np.eye(6)
    ekf = AbsoluteOrbitEKF(process_noise=make_process_noise(1.0, 0.0))

    predicted_state, predicted_covariance = ekf.predict(state, covariance, 1.0)
    reference = propagate_absolute_orbit(state, np.array([0.0, 1.0]))[-1]

    np.testing.assert_allclose(predicted_state, reference, rtol=0.0, atol=1e-12)
    assert predicted_covariance.shape == (6, 6)
    assert np.allclose(predicted_covariance, predicted_covariance.T)


def test_absolute_orbit_ekf_rejects_nonpositive_dt():
    ekf = AbsoluteOrbitEKF(process_noise=np.eye(6))
    try:
        ekf.predict(np.zeros(6), np.eye(6), 0.0)
    except ValueError as exc:
        assert "dt" in str(exc)
    else:
        raise AssertionError("Expected nonpositive dt to be rejected.")
