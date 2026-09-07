import numpy as np

from brain_inspired.orbital_rt_grid_runner import run_orbital_rt_grid_states
from orbital_core.dynamics import rk4_step_absolute


def test_rt_grid_runner_uses_explicit_reference_and_causal_anchor():
    reference0 = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    posterior0 = reference0.copy()
    posterior0[:2] += [100.0, 200.0]
    times = np.array([0.0, 2.0, 4.0])
    references = np.stack([
        reference0, rk4_step_absolute(reference0, 2.0),
        rk4_step_absolute(reference0, 4.0),
    ])
    posterior = np.stack([
        posterior0, rk4_step_absolute(posterior0, 2.0),
        rk4_step_absolute(posterior0, 4.0),
    ])
    history = run_orbital_rt_grid_states(
        timestamps=times,
        posterior_state_history_by_node={"sat": posterior},
        reference_state_history_by_node={"sat": references},
        anchor_mask_by_node={"sat": np.array([False, False, True])},
    )["sat"]
    assert history.source_rt.shape == (3, 2)
    assert history.valid.tolist() == [True, True, True]
    assert history.cue_applied.tolist() == [False, False, True]
    assert history.anchor_age.tolist() == [0.0, 2.0, 0.0]


def test_rt_grid_runner_rejects_reference_node_mismatch():
    state = np.array([7e6, 0.0, 0.0, 0.0, 7500.0, 0.0])
    try:
        run_orbital_rt_grid_states(
            timestamps=np.array([0.0]),
            posterior_state_history_by_node={"sat": state[None, :]},
            reference_state_history_by_node={"other": state[None, :]},
        )
    except ValueError as error:
        assert "identical nodes" in str(error)
    else:
        raise AssertionError("Expected reference-node mismatch rejection.")
