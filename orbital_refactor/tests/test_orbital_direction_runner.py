import numpy as np

from brain_inspired.orbital_direction_runner import (
    run_orbital_direction_states,
)
from orbital_core.dynamics import rk4_step_absolute


def _history():
    times = np.array([0.0, 2.0, 4.0])
    initial = np.array([7e6, 0.0, 0.0, 0.0, 7546.0, 0.0])
    values = [initial]
    for _ in times[1:]:
        values.append(rk4_step_absolute(values[-1], 2.0))
    return times, np.vstack(values)


def test_runner_records_prediction_before_same_epoch_anchor():
    times, posterior = _history()
    shifted = posterior.copy()
    shifted[1, :3] += np.array([0.0, 20_000.0, 0.0])
    result = run_orbital_direction_states(
        timestamps=times,
        posterior_state_history_by_node={"sat": shifted},
        anchor_mask_by_node={"sat": np.array([False, True, False])},
        anchor_confidence_by_node={"sat": np.array([1.0, 1.0, 1.0])},
    )["sat"]

    assert result.cue_applied.tolist() == [False, True, False]
    assert result.anchor_age.tolist() == [0.0, 0.0, 2.0]
    assert abs(result.anchored_residual[1]) < abs(
        result.prediction_residual[1]
    )


def test_runner_keeps_nodes_and_controls_isolated():
    times, posterior = _history()
    output = run_orbital_direction_states(
        timestamps=times,
        posterior_state_history_by_node={
            "left": posterior,
            "right": posterior.copy(),
        },
        anchor_mask_by_node={
            "left": np.array([False, True, False]),
            "right": np.zeros(3, dtype=bool),
        },
    )
    assert output["left"].cue_applied.tolist() == [False, True, False]
    assert not output["right"].cue_applied.any()
