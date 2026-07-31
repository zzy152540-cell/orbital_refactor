import numpy as np

from experiments.neighbor_update_transport import run_neighbor_update_transport_monte_carlo


def test_exact_transport_is_consistent_after_private_neighbor_update():
    result = run_neighbor_update_transport_monte_carlo(samples=8000, seed=11)
    exact = result["exact_transport"]
    assert abs(exact.mean_nees - 6.0) < 0.25
    assert exact.nees_95_coverage > 0.93
    assert abs(exact.mean_nis - 3.0) < 0.15
    assert exact.minimum_covariance_eigenvalue >= -1e-9
    assert result["zero_cross"].mean_nees > exact.mean_nees
