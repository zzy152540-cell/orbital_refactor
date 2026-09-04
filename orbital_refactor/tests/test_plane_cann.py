import numpy as np

from brain_inspired.line_cann import LineCANNConfig
from brain_inspired.plane_cann import PlaneCANN, PlaneCANNConfig


def _config():
    axis = LineCANNConfig(
        num_neurons=201, minimum_value=-10.0, maximum_value=10.0,
        tuning_width=0.5, cue_gain=0.5,
    )
    return PlaneCANNConfig(x_axis=axis, y_axis=axis)


def test_plane_cann_tracks_two_dimensional_velocity():
    cann = PlaneCANN(_config())
    cann.reset([1.0, -2.0])
    output = cann.step([0.2, 0.1], 5.0)
    assert np.allclose(output.decoded_position, [2.0, -1.5], atol=1e-8)
    assert output.neural_activity.shape == (201, 201)
    assert output.valid


def test_plane_cann_clamps_each_axis_without_wrapping():
    cann = PlaneCANN(_config())
    cann.reset([9.0, -9.0])
    output = cann.step([1.0, -1.0], 2.0)
    assert np.allclose(output.decoded_position, [10.0, -10.0], atol=1e-8)
    assert output.saturated_at_boundary


def test_compact_plane_cann_preserves_decoding_without_joint_activity():
    full = PlaneCANN()
    compact = PlaneCANN(PlaneCANNConfig(materialize_joint_activity=False))

    full_output = full.reset([0.2, -0.3])
    compact_output = compact.reset([0.2, -0.3])

    np.testing.assert_allclose(
        compact_output.decoded_position, full_output.decoded_position,
    )
    assert full_output.neural_activity.shape == (121, 121)
    assert compact_output.neural_activity.shape == (0, 0)
    assert compact_output.bump_concentration == full_output.bump_concentration
