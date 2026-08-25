import numpy as np

from experiments.variable_scale_value_feature_audit import _ridge_summary


def test_ridge_summary_recovers_held_out_linear_signal():
    training = [
        {"features": [value, 1.0], "target": 2.0 * value + 1.0}
        for value in range(8)
    ]
    test = [
        {"features": [value, 1.0], "target": 2.0 * value + 1.0}
        for value in range(8, 12)
    ]
    summary = _ridge_summary(training, test, None, 1.0e-8)
    assert summary["explained_variance"] > 0.999
    assert summary["correlation"] > 0.999
    assert np.isfinite(summary["rmse"])
