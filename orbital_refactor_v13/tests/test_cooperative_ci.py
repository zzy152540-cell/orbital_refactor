import numpy as np

from cooperative.multi_node_ci import fuse_local_histories, relative_history_to_absolute


def test_relative_history_to_absolute_round_trip():
    observer = np.array([[1., 2., 3., 4., 5., 6.]])
    relative = np.array([[10., 20., 30., 1., 2., 3.]])
    np.testing.assert_allclose(relative_history_to_absolute(relative, observer), observer + relative)


def test_fuse_local_histories_uses_common_absolute_frame():
    timestamps = np.array([0.0, 1.0])
    truth = np.array([[100., 200., 300., 1., 2., 3.], [101., 202., 303., 1., 2., 3.]])
    observers = {
        "a": np.array([[10., 0., 0., 0., 0., 0.], [11., 0., 0., 0., 0., 0.]]),
        "b": np.array([[0., 20., 0., 0., 0., 0.], [0., 21., 0., 0., 0., 0.]]),
    }
    relative = {key: truth - value for key, value in observers.items()}
    covariances = {key: np.tile(np.eye(6)[None, :, :], (2, 1, 1)) for key in observers}
    result = fuse_local_histories(
        timestamps=timestamps,
        relative_state_history_by_node=relative,
        covariance_history_by_node=covariances,
        observer_state_history_by_node=observers,
        target_id="target",
    )
    np.testing.assert_allclose(result.state_history_eci, truth, atol=1e-10)
    assert all(abs(sum(weights.values()) - 1.0) < 1e-12 for weights in result.node_weight_history)
