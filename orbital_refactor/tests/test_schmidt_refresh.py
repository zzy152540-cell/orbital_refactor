import numpy as np

from cooperative.multi_neighbor_schmidt import MultiNeighborSchmidtState
from cooperative.schmidt_refresh import refresh_consider_neighbor


def _state():
    rng = np.random.default_rng(7)
    factor = rng.normal(size=(18, 18))
    covariance = factor @ factor.T + np.eye(18)
    return MultiNeighborSchmidtState(
        timestamp=0.0, active_node_id="a", neighbor_ids=("b", "c"),
        active_state=np.zeros(6),
        neighbor_state_by_id={"b": np.ones(6), "c": np.full(6, 2.0)},
        joint_covariance=covariance,
    )


def test_safe_rescale_replaces_marginal_and_preserves_joint_psd():
    state = _state()
    replacement = np.diag([100.0, 50.0, 20.0, 0.1, 0.2, 0.3])
    refreshed = refresh_consider_neighbor(
        state, neighbor_id="b", neighbor_state=np.arange(6.0),
        neighbor_covariance=replacement, mode="safe_rescale",
    )

    assert np.allclose(refreshed.neighbor_covariance("b"), replacement)
    assert np.allclose(refreshed.neighbor_state_by_id["b"], np.arange(6.0))
    keep = np.r_[0:6, 12:18]
    assert np.allclose(
        refreshed.joint_covariance[np.ix_(keep, keep)],
        state.joint_covariance[np.ix_(keep, keep)],
    )
    assert np.linalg.eigvalsh(refreshed.joint_covariance).min() >= -1e-8


def test_zero_cross_discards_only_target_cross_covariance():
    state = _state()
    refreshed = refresh_consider_neighbor(
        state, neighbor_id="b", neighbor_state=np.zeros(6),
        neighbor_covariance=np.eye(6) * 3.0, mode="zero_cross",
    )
    target = np.arange(6, 12)
    rest = np.r_[0:6, 12:18]
    assert np.allclose(refreshed.joint_covariance[np.ix_(rest, target)], 0.0)
    assert np.linalg.eigvalsh(refreshed.joint_covariance).min() >= -1e-8


def test_exact_transport_matches_full_linear_error_transform():
    state = _state()
    transition = np.diag([1.0, 1.1, 0.9, 0.8, 1.2, 1.0])
    noise = np.eye(6) * 0.25
    refreshed = refresh_consider_neighbor(
        state, neighbor_id="b", neighbor_state=np.zeros(6),
        mode="exact_transport", error_transition=transition,
        independent_process_noise=noise,
    )
    full_transition = np.eye(18)
    full_transition[6:12, 6:12] = transition
    expected = full_transition @ state.joint_covariance @ full_transition.T
    expected[6:12, 6:12] += noise
    assert np.allclose(refreshed.joint_covariance, expected)


def test_propagate_only_is_a_no_op():
    state = _state()
    assert refresh_consider_neighbor(
        state, neighbor_id="b", neighbor_state=np.zeros(6),
        mode="propagate_only",
    ) is state
