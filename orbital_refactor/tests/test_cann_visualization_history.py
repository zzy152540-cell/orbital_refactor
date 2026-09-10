import numpy as np

from brain_inspired.orbital_direction_runner import run_orbital_direction_states
from brain_inspired.orbital_rt_grid_runner import run_orbital_rt_grid_states


def _state_history():
    return np.array([
        [7.0e6, 0.0, 0.0, 0.0, 7500.0, 0.0],
        [6.99998e6, 15_000.0, 0.0, -16.0, 7499.98, 0.0],
    ])


def test_cann_runners_retain_activity_only_when_requested():
    times = np.array([0.0, 2.0])
    posterior = {"sat_01": _state_history()}
    direction_default = run_orbital_direction_states(
        timestamps=times, posterior_state_history_by_node=posterior,
    )["sat_01"]
    direction_visual = run_orbital_direction_states(
        timestamps=times, posterior_state_history_by_node=posterior,
        retain_activity=True,
    )["sat_01"]
    assert direction_default.neural_activity is None
    assert direction_visual.neural_activity.shape == (2, 180)

    rt_default = run_orbital_rt_grid_states(
        timestamps=times, posterior_state_history_by_node=posterior,
        reference_state_history_by_node=posterior,
    )["sat_01"]
    rt_visual = run_orbital_rt_grid_states(
        timestamps=times, posterior_state_history_by_node=posterior,
        reference_state_history_by_node=posterior, retain_activity=True,
    )["sat_01"]
    assert rt_default.radial_activity is None
    assert rt_default.along_track_activity is None
    assert rt_visual.radial_activity.shape == (2, 81)
    assert rt_visual.along_track_activity.shape == (2, 81)
