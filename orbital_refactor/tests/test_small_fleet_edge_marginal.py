import numpy as np
import pytest

from experiments.small_fleet_edge_marginal import (
    run_small_fleet_edge_marginal_experiment,
)


@pytest.mark.parametrize("node_count", (3, 5))
def test_small_fleet_edge_marginal_experiment_runs_fixed_counterfactuals(
    node_count,
):
    result = run_small_fleet_edge_marginal_experiment(
        node_count=node_count, duration=4.0, dt=2.0,
        relative_modalities=("RANGE",),
    )

    assert result.node_count == node_count
    assert len(result.baseline_action.active_edges) == node_count - 1
    assert len(result.edge_marginals) == node_count * (node_count - 1) // 2
    for value in result.edge_marginals:
        assert np.isfinite(value.covariance_trace_reduction)
        assert np.isfinite(value.covariance_logdet_reduction)
        assert np.isfinite(value.position_rmse_reduction)
        assert value.transmitted_message_cost > 0


def test_small_fleet_edge_marginal_is_reproducible():
    arguments = dict(
        node_count=3, duration=4.0, dt=2.0,
        relative_modalities=("RANGE",),
    )
    left = run_small_fleet_edge_marginal_experiment(**arguments)
    right = run_small_fleet_edge_marginal_experiment(**arguments)

    assert left == right


def test_small_fleet_edge_marginal_rejects_unsupported_size():
    with pytest.raises(ValueError, match="3 or 5"):
        run_small_fleet_edge_marginal_experiment(node_count=4)
