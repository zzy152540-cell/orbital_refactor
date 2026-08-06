import numpy as np

from orbital_core.inter_satellite_model import RelativeMeasurementModel
from orbital_core.measurements import (
    measure_relative_range,
    measure_relative_range_rate,
)


def test_joint_radar_prediction_contains_range_and_range_rate():
    observer = np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    target = observer + np.array([1000.0, 20.0, -10.0, 0.5, -0.2, 0.1])

    predicted = RelativeMeasurementModel("RADAR").predict(observer, target)

    np.testing.assert_allclose(predicted, [
        measure_relative_range(observer, target),
        measure_relative_range_rate(observer, target),
    ])


def test_joint_radar_jacobian_stacks_scalar_component_jacobians():
    observer = np.array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    target = observer + np.array([1000.0, 20.0, -10.0, 0.5, -0.2, 0.1])
    joint = RelativeMeasurementModel("RADAR")
    range_model = RelativeMeasurementModel("RANGE")
    rate_model = RelativeMeasurementModel("RANGE_RATE")

    left, right = joint.jacobians(observer, target)
    range_left, range_right = range_model.jacobians(observer, target)
    rate_left, rate_right = rate_model.jacobians(observer, target)

    np.testing.assert_allclose(left, np.vstack((range_left, rate_left)))
    np.testing.assert_allclose(right, np.vstack((range_right, rate_right)))
    assert left.shape == right.shape == (2, 6)
