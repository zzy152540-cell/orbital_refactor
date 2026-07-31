import numpy as np

from experiments.v14_consistency import (
    run_v14_network_schmidt_monte_carlo,
    run_v14_range_consistency_monte_carlo,
)


def test_range_consistency_compares_three_information_sharing_strategies():
    result = run_v14_range_consistency_monte_carlo(
        seeds=2,
        duration=4.0,
        dt=2.0,
    )

    assert len(result.runs) == 10
    assert set(result.summary_by_strategy) == {
        "single_endpoint",
        "single_endpoint_schmidt",
        "shared",
        "shared_dual_track",
        "independent_reciprocal",
    }
    for strategy, summary in result.summary_by_strategy.items():
        assert summary.strategy == strategy
        assert summary.run_count == 2
        assert np.isfinite(summary.mean_position_rmse)
        assert np.isfinite(summary.mean_velocity_rmse)
        assert np.isfinite(summary.mean_nees)
        assert np.isfinite(summary.mean_nis)
        assert 0.0 <= summary.mean_nees_95_coverage <= 1.0
        assert 0.0 <= summary.mean_nis_95_coverage <= 1.0


def test_range_consistency_is_reproducible():
    left = run_v14_range_consistency_monte_carlo(
        seeds=1,
        duration=2.0,
        dt=2.0,
    )
    right = run_v14_range_consistency_monte_carlo(
        seeds=1,
        duration=2.0,
        dt=2.0,
    )

    assert left == right


def test_three_satellite_network_consistency_compares_schmidt_with_approximation():
    result = run_v14_network_schmidt_monte_carlo(
        seeds=2,
        duration=4.0,
        dt=2.0,
    )

    assert len(result.runs) == 4
    assert set(result.summary_by_strategy) == {
        "network_approximate",
        "network_schmidt",
    }
    for summary in result.summary_by_strategy.values():
        assert summary.run_count == 2
        assert np.isfinite(summary.mean_nees)
        assert np.isfinite(summary.mean_nis)
        assert 0.0 <= summary.mean_nees_95_coverage <= 1.0
