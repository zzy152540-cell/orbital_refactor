from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.attitude_data_objects import GyroObservation, StarTrackerObservation
from orbital_core.attitude import (
    propagate_attitude,
    quat_multiply_wxyz,
    quat_normalize_wxyz,
    small_angle_quaternion_wxyz,
)

Array = np.ndarray


@dataclass(frozen=True)
class AttitudeTruthTrajectory:
    satellite_id: str
    timestamps: Array
    quaternion_i2b_wxyz: Array
    angular_velocity_body: Array


def generate_attitude_truth(
    *,
    satellite_id: str,
    timestamps: Array,
    initial_quaternion_i2b_wxyz: Array,
    initial_angular_velocity_body: Array,
    inertia: Array,
    torque: Array | None = None,
) -> AttitudeTruthTrajectory:
    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("timestamps cannot be empty.")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    quaternions = np.zeros((times.size, 4), dtype=float)
    angular_velocity = np.zeros((times.size, 3), dtype=float)
    quaternions[0] = quat_normalize_wxyz(initial_quaternion_i2b_wxyz)
    angular_velocity[0] = np.asarray(
        initial_angular_velocity_body, dtype=float
    ).reshape(3)
    for index in range(1, times.size):
        quaternions[index], angular_velocity[index] = propagate_attitude(
            quaternions[index - 1],
            angular_velocity[index - 1],
            float(times[index] - times[index - 1]),
            inertia,
            torque,
        )
    return AttitudeTruthTrajectory(
        satellite_id=str(satellite_id),
        timestamps=times.copy(),
        quaternion_i2b_wxyz=quaternions,
        angular_velocity_body=angular_velocity,
    )


def simulate_gyro_observations(
    truth: AttitudeTruthTrajectory,
    *,
    white_noise_std: float,
    bias_random_walk_std: float,
    initial_bias: Array | None = None,
    random_seed: int = 42,
) -> tuple[list[GyroObservation], Array]:
    if white_noise_std <= 0.0:
        raise ValueError("white_noise_std must be positive.")
    if bias_random_walk_std < 0.0:
        raise ValueError("bias_random_walk_std cannot be negative.")
    rng = np.random.default_rng(random_seed)
    bias = np.zeros((truth.timestamps.size, 3), dtype=float)
    if initial_bias is not None:
        bias[0] = np.asarray(initial_bias, dtype=float).reshape(3)
    for index in range(1, truth.timestamps.size):
        dt = float(truth.timestamps[index] - truth.timestamps[index - 1])
        bias[index] = (
            bias[index - 1]
            + rng.normal(0.0, bias_random_walk_std * np.sqrt(dt), size=3)
        )
    noise = rng.normal(
        0.0, white_noise_std, size=truth.angular_velocity_body.shape
    )
    covariance = np.eye(3) * white_noise_std**2
    observations = [
        GyroObservation(
            timestamp=float(timestamp),
            satellite_id=truth.satellite_id,
            angular_rate_body=truth.angular_velocity_body[index] + bias[index] + noise[index],
            covariance=covariance.copy(),
        )
        for index, timestamp in enumerate(truth.timestamps)
    ]
    return observations, bias


def simulate_star_tracker_observations(
    truth: AttitudeTruthTrajectory,
    *,
    update_interval: int,
    small_angle_noise_std: float,
    random_seed: int = 123,
) -> list[StarTrackerObservation]:
    if update_interval < 1:
        raise ValueError("update_interval must be at least one sample.")
    if small_angle_noise_std <= 0.0:
        raise ValueError("small_angle_noise_std must be positive.")
    rng = np.random.default_rng(random_seed)
    covariance = np.eye(3) * small_angle_noise_std**2
    observations = []
    for index in range(0, truth.timestamps.size, update_interval):
        error_quaternion = small_angle_quaternion_wxyz(
            rng.normal(0.0, small_angle_noise_std, size=3)
        )
        measurement = quat_normalize_wxyz(
            quat_multiply_wxyz(
                error_quaternion,
                truth.quaternion_i2b_wxyz[index],
            )
        )
        observations.append(
            StarTrackerObservation(
                timestamp=float(truth.timestamps[index]),
                satellite_id=truth.satellite_id,
                quaternion_i2b_wxyz=measurement,
                covariance_small_angle=covariance.copy(),
            )
        )
    return observations
