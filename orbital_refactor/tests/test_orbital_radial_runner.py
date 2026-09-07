import numpy as np

from brain_inspired.orbital_radial_runner import run_orbital_radial_states
from orbital_core.dynamics import rk4_step_absolute


def test_radial_runner_is_causal_and_tracks_anchor_age():
    initial = np.array([7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0])
    times = np.array([0.0, 2.0, 4.0])
    states = np.stack([
        initial,
        rk4_step_absolute(initial, 2.0),
        rk4_step_absolute(initial, 4.0),
    ])
    history = run_orbital_radial_states(
        timestamps=times, posterior_state_history_by_node={"sat": states},
        anchor_mask_by_node={"sat": np.array([False, False, True])},
    )["sat"]
    assert history.valid.tolist() == [True, True, True]
    assert history.cue_applied.tolist() == [False, False, True]
    assert history.anchor_age.tolist() == [0.0, 2.0, 0.0]
    assert history.source_displacement.shape == (3,)


def test_radial_runner_keeps_nodes_independent():
    left = np.array([7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0])
    right = left.copy()
    right[0] += 1_000.0
    result = run_orbital_radial_states(
        timestamps=np.array([0.0]),
        posterior_state_history_by_node={
            "left": left[None, :], "right": right[None, :],
        },
    )
    assert result["left"].reference_radius != result["right"].reference_radius
