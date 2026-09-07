import numpy as np

from brain_inspired.navigation_brain_state import (
    FEATURE_NAMES,
    build_navigation_brain_states,
)
from brain_inspired.orbital_direction_runner import run_orbital_direction_states
from brain_inspired.orbital_rt_grid_runner import run_orbital_rt_grid_states
from orbital_core.dynamics import rk4_step_absolute


def _histories():
    initial = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    times = np.array([0.0, 2.0, 4.0])
    states = np.stack([
        initial, rk4_step_absolute(initial, 2.0),
        rk4_step_absolute(initial, 4.0),
    ])
    mapping = {"sat": states}
    direction = run_orbital_direction_states(
        timestamps=times, posterior_state_history_by_node=mapping,
    )
    rt = run_orbital_rt_grid_states(
        timestamps=times, posterior_state_history_by_node=mapping,
        reference_state_history_by_node=mapping,
    )
    return direction, rt


def test_navigation_brain_state_builds_wrap_safe_compact_features():
    direction, rt = _histories()
    history = build_navigation_brain_states(
        direction_by_node=direction, rt_by_node=rt,
    )["sat"]
    assert history.feature_matrix.shape == (3, len(FEATURE_NAMES))
    assert np.allclose(
        history.feature_matrix[:, 0] ** 2
        + history.feature_matrix[:, 1] ** 2,
        1.0,
    )
    assert history.valid.tolist() == [True, True, True]
    assert np.all(history.shadow_quality == 1.0)
    assert np.all(history.delayed_feedback_quality == 1.0)


def test_navigation_brain_state_rejects_node_mismatch():
    direction, rt = _histories()
    try:
        build_navigation_brain_states(
            direction_by_node=direction,
            rt_by_node={"other": next(iter(rt.values()))},
        )
    except ValueError as error:
        assert "identical nodes" in str(error)
    else:
        raise AssertionError("Expected navigation-state node mismatch rejection.")
