import numpy as np

from orbital_core.coordinates import quat_to_dcm_wxyz
from orbital_core.dynamics import accel_two_body_j2, make_process_noise, rk4_step_rel


def test_identity_quaternion():
    dcm = quat_to_dcm_wxyz(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(dcm, np.eye(3))


def test_gravity_points_inward():
    a = accel_two_body_j2(np.array([7.0e6, 0.0, 0.0]))
    assert a[0] < 0.0
    assert abs(a[1]) < 1e-12


def test_process_noise_shape_and_symmetry():
    q = make_process_noise(1.0, 1e-4)
    assert q.shape == (6, 6)
    np.testing.assert_allclose(q, q.T)


def test_rk4_output_shape():
    x = np.array([100.0, 20.0, -10.0, 0.1, 0.0, 0.0])
    chief = np.array([7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    out = rk4_step_rel(x, chief, 1.0)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))
