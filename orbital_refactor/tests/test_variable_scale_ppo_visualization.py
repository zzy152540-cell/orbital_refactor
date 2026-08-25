import numpy as np

from experiments.variable_scale_ppo_visualization import _moving_average


def test_moving_average_preserves_episode_alignment():
    values = _moving_average((1.0, 2.0, 3.0, 4.0), 3)
    assert len(values) == 4
    assert np.isnan(values[:2]).all()
    assert values[2:].tolist() == [2.0, 3.0]
