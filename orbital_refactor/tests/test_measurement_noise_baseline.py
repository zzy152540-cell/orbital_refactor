import numpy as np

from examples.run_v13_1_baseline import build_case
from examples.run_v13_2_fleet_ci import build_anchor_observations


def test_gnss_anchor_noise_is_nonzero_reproducible_and_matches_covariance():
    scenario, *_ = build_case()
    first = build_anchor_observations(
        scenario,
        interval=5,
        position_sigma=2.0,
        random_seed=123,
    )
    second = build_anchor_observations(
        scenario,
        interval=5,
        position_sigma=2.0,
        random_seed=123,
    )

    truth = scenario.trajectories["sat_01"].state_history_eci
    residuals = []
    for observation_1, observation_2 in zip(first, second, strict=True):
        index = int(np.where(scenario.timestamps == observation_1.timestamp)[0][0])
        residuals.append(observation_1.measurement_eci - truth[index, :3])
        np.testing.assert_allclose(
            observation_1.measurement_eci,
            observation_2.measurement_eci,
        )
        np.testing.assert_allclose(observation_1.covariance, np.eye(3) * 4.0)
    assert np.linalg.norm(np.vstack(residuals)) > 0.0


def test_baseline_contains_rtn_angle_observations_with_radian_covariance():
    _scenario, _initial, _covariance, _topology, observations = build_case()
    angles = [observation for observation in observations if observation.modality == "AZ_EL"]

    assert angles
    expected_variance = np.deg2rad(0.02) ** 2
    for observation in angles:
        assert observation.measurement.shape == (2,)
        np.testing.assert_allclose(
            observation.covariance,
            np.eye(2) * expected_variance,
        )
        assert observation.metadata["frame"] == "RTN"
