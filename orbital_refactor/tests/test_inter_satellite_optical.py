import numpy as np
import pytest

from experiments.v14_exact_transport_scale_scan import _target_pointing_quaternion
from orbital_core.inter_satellite_model import RelativeMeasurementModel
from orbital_core.measurements import measure_relative_optical_uv


def test_body_optical_model_returns_normalized_image_coordinates():
    observer = np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    target = observer + np.array([1000.0, 20.0, -10.0, 0.0, 0.0, 0.0])
    quaternion = _target_pointing_quaternion(observer, target)

    measurement = measure_relative_optical_uv(
        observer, target, quaternion_i2b_wxyz=quaternion,
    )

    np.testing.assert_allclose(measurement, np.zeros(2), atol=1e-12)


def test_optical_relative_model_has_two_state_jacobians_and_nonangular_residual():
    observer = np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    target = observer + np.array([1000.0, 20.0, -10.0, 0.0, 0.0, 0.0])
    quaternion = _target_pointing_quaternion(observer, target)
    model = RelativeMeasurementModel("OPTICAL", "BODY")

    predicted = model.predict(
        observer, target, quaternion_i2b_wxyz=quaternion,
    )
    left, right = model.jacobians(
        observer, target, quaternion_i2b_wxyz=quaternion,
    )

    assert predicted.shape == (2,)
    assert left.shape == right.shape == (2, 6)
    np.testing.assert_allclose(left, -right, rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(
        model.residual(np.array([4.0, -4.0]), predicted),
        np.array([4.0, -4.0]),
    )


def test_optical_model_rejects_missing_attitude():
    observer = np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    target = observer + np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="quaternion"):
        RelativeMeasurementModel("OPTICAL", "BODY").predict(observer, target)
