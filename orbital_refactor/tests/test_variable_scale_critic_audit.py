import numpy as np

from experiments.variable_scale_critic_audit import (
    _discounted_returns,
    _value_summary,
)


def test_discounted_returns_do_not_depend_on_critic_values():
    np.testing.assert_allclose(
        _discounted_returns(np.asarray([1.0, 2.0, 3.0]), 0.5),
        (2.75, 3.5, 3.0),
    )


def test_value_summary_reports_bias_and_explained_variance():
    records = [
        {
            "action_kind": "keep",
            "monte_carlo_return": target,
            "predicted_value": target + 1.0,
            "gae_advantage": advantage,
        }
        for target, advantage in ((1.0, -1.0), (2.0, 1.0), (3.0, 2.0))
    ]
    summary = _value_summary(records)
    assert summary["mean_value_error"] == 1.0
    assert summary["value_rmse"] == 1.0
    assert summary["explained_variance"] == 1.0
    assert summary["positive_gae_fraction"] == 2.0 / 3.0
