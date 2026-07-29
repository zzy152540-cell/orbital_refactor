import numpy as np

from examples.run_v13_4_attitude_coupling import run_comparison


def test_attitude_coupling_comparison_is_reproducible_and_finite():
    first = run_comparison(duration=20.0)
    second = run_comparison(duration=20.0)

    assert first.mean_attitude_error_deg == second.mean_attitude_error_deg
    for label in (
        "truth_attitude",
        "mekf_with_covariance",
        "mekf_without_covariance",
    ):
        first_metrics = getattr(first, label)
        second_metrics = getattr(second, label)
        np.testing.assert_allclose(
            [
                first_metrics.position_rmse,
                first_metrics.velocity_rmse,
                first_metrics.mean_angle_nis,
                first_metrics.mean_orbit_nees,
            ],
            [
                second_metrics.position_rmse,
                second_metrics.velocity_rmse,
                second_metrics.mean_angle_nis,
                second_metrics.mean_orbit_nees,
            ],
        )
        assert np.all(
            np.isfinite(
                [
                    first_metrics.position_rmse,
                    first_metrics.velocity_rmse,
                    first_metrics.mean_angle_nis,
                    first_metrics.mean_orbit_nees,
                ]
            )
        )

    assert (
        first.mekf_with_covariance.mean_angle_nis
        < first.mekf_without_covariance.mean_angle_nis
    )
    assert first.mean_attitude_error_deg < 0.2
    assert abs(first.truth_attitude.mean_angle_nis - 2.0) < abs(
        first.mekf_without_covariance.mean_angle_nis - 2.0
    )
