import numpy as np

from brain_inspired.orbital_rt_multiscale_runner import (
    run_orbital_rt_multiscale_states,
)
from orbital_core.dynamics import rk4_step_absolute


def test_multiscale_rt_runner_maintains_per_node_histories():
    initial = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    times = np.array([0.0, 2.0, 4.0])
    states = np.stack([
        initial,
        rk4_step_absolute(initial, 2.0),
        rk4_step_absolute(initial, 4.0),
    ])
    result = run_orbital_rt_multiscale_states(
        timestamps=times,
        posterior_state_history_by_node={"sat": states},
        reference_state_history_by_node={"sat": states},
        anchor_mask_by_node={"sat": np.array([False, True, True])},
    )["sat"]
    assert result.decoded_rt.shape == (3, 2)
    assert result.coarse_center_rt.shape == (3, 2)
    assert result.valid.tolist() == [True, True, True]
    assert result.cue_applied.tolist() == [False, True, True]
