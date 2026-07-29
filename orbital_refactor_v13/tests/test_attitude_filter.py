import numpy as np

from orbital_core.attitude import (
    attitude_error_angle_deg,
    quat_multiply_wxyz,
    quat_to_dcm_i2b,
    small_angle_quaternion_wxyz,
)
from orbital_core.attitude_filter import AttitudeGyroBiasMEKF
from scenarios.attitude_scenario import (
    generate_attitude_truth,
    simulate_gyro_observations,
    simulate_star_tracker_observations,
)


def test_wxyz_quaternion_convention_rotates_x_to_y_for_positive_z():
    quaternion = np.array([
        np.cos(np.pi / 4.0),
        0.0,
        0.0,
        np.sin(np.pi / 4.0),
    ])
    rotated = quat_to_dcm_i2b(quaternion) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_attitude_truth_and_sensor_noise_are_reproducible():
    truth = generate_attitude_truth(
        satellite_id="sat_01",
        timestamps=np.arange(0.0, 2.1, 0.1),
        initial_quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        initial_angular_velocity_body=np.deg2rad([0.1, -0.05, 0.2]),
        inertia=np.diag([1.0, 1.2, 0.8]),
    )
    first, first_bias = simulate_gyro_observations(
        truth,
        white_noise_std=1e-4,
        bias_random_walk_std=1e-5,
        initial_bias=np.array([2e-4, -1e-4, 3e-4]),
        random_seed=7,
    )
    second, second_bias = simulate_gyro_observations(
        truth,
        white_noise_std=1e-4,
        bias_random_walk_std=1e-5,
        initial_bias=np.array([2e-4, -1e-4, 3e-4]),
        random_seed=7,
    )

    np.testing.assert_allclose(
        np.vstack([item.angular_rate_body for item in first]),
        np.vstack([item.angular_rate_body for item in second]),
    )
    np.testing.assert_allclose(first_bias, second_bias)
    np.testing.assert_allclose(
        np.linalg.norm(truth.quaternion_i2b_wxyz, axis=1),
        np.ones(len(truth.timestamps)),
    )


def test_mekf_uses_gyro_and_star_tracker_to_reduce_attitude_error():
    timestamps = np.arange(0.0, 20.1, 0.1)
    inertia = np.diag([1.0, 1.2, 0.8])
    truth = generate_attitude_truth(
        satellite_id="sat_01",
        timestamps=timestamps,
        initial_quaternion_i2b_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        initial_angular_velocity_body=np.deg2rad([0.15, -0.08, 0.2]),
        inertia=inertia,
    )
    true_initial_bias = np.deg2rad([0.01, -0.008, 0.006])
    gyro, _bias_truth = simulate_gyro_observations(
        truth,
        white_noise_std=np.deg2rad(0.002),
        bias_random_walk_std=np.deg2rad(0.00005),
        initial_bias=true_initial_bias,
        random_seed=10,
    )
    star = simulate_star_tracker_observations(
        truth,
        update_interval=10,
        small_angle_noise_std=np.deg2rad(0.02),
        random_seed=11,
    )
    star_by_time = {item.timestamp: item for item in star}
    initial_error = small_angle_quaternion_wxyz(np.deg2rad([2.0, -1.0, 0.5]))
    initial_quaternion = quat_multiply_wxyz(
        initial_error, truth.quaternion_i2b_wxyz[0]
    )
    filter_obj = AttitudeGyroBiasMEKF(
        satellite_id="sat_01",
        quaternion_i2b_wxyz=initial_quaternion,
        angular_velocity_body=truth.angular_velocity_body[0],
        gyro_bias=np.zeros(3),
        covariance=np.diag(
            [
                *([np.deg2rad(5.0) ** 2] * 3),
                *([np.deg2rad(0.5) ** 2] * 3),
                *([np.deg2rad(0.05) ** 2] * 3),
            ]
        ),
        inertia=inertia,
        angular_acceleration_noise_std=np.deg2rad(0.001),
        gyro_bias_random_walk_std=np.deg2rad(0.00005),
    )
    initial_angle_error = attitude_error_angle_deg(
        filter_obj.quaternion_i2b_wxyz, truth.quaternion_i2b_wxyz[0]
    )
    for index in range(1, len(timestamps)):
        filter_obj.predict(float(timestamps[index] - timestamps[index - 1]))
        filter_obj.update_gyro(
            gyro[index].angular_rate_body,
            gyro[index].covariance,
        )
        star_observation = star_by_time.get(float(timestamps[index]))
        if star_observation is not None:
            filter_obj.update_star_tracker(
                star_observation.quaternion_i2b_wxyz,
                star_observation.covariance_small_angle,
            )

    final_angle_error = attitude_error_angle_deg(
        filter_obj.quaternion_i2b_wxyz,
        truth.quaternion_i2b_wxyz[-1],
    )
    assert final_angle_error < 0.1
    assert final_angle_error < initial_angle_error
    assert np.linalg.norm(filter_obj.gyro_bias - true_initial_bias) < np.deg2rad(0.01)
    assert np.all(np.linalg.eigvalsh(filter_obj.covariance) > -1e-10)
