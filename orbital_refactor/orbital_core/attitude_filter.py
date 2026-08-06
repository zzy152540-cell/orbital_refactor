from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.attitude_data_objects import AttitudeEstimate
from orbital_core.attitude import (
    left_attitude_error_vector,
    propagate_attitude,
    quat_multiply_wxyz,
    quat_normalize_wxyz,
    quat_to_dcm_i2b,
    rigid_body_angular_acceleration,
    skew,
    small_angle_quaternion_wxyz,
)

Array = np.ndarray


@dataclass
class AttitudeGyroBiasMEKF:
    """MEKF with error state [dtheta, domega, db_gyro]."""

    satellite_id: str
    quaternion_i2b_wxyz: Array
    angular_velocity_body: Array
    gyro_bias: Array
    covariance: Array
    inertia: Array
    angular_acceleration_noise_std: float
    gyro_bias_random_walk_std: float
    torque: Array | None = None

    def __post_init__(self) -> None:
        self.quaternion_i2b_wxyz = quat_normalize_wxyz(
            self.quaternion_i2b_wxyz
        )
        self.angular_velocity_body = np.asarray(
            self.angular_velocity_body, dtype=float
        ).reshape(3)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=float).reshape(3)
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(9, 9)
        self.inertia = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        if self.angular_acceleration_noise_std < 0.0:
            raise ValueError("angular_acceleration_noise_std cannot be negative.")
        if self.gyro_bias_random_walk_std < 0.0:
            raise ValueError("gyro_bias_random_walk_std cannot be negative.")

    def predict(self, dt: float) -> None:
        previous_omega = self.angular_velocity_body.copy()
        self.quaternion_i2b_wxyz, self.angular_velocity_body = propagate_attitude(
            self.quaternion_i2b_wxyz,
            self.angular_velocity_body,
            dt,
            self.inertia,
            self.torque,
        )
        transition_rate = np.zeros((9, 9), dtype=float)
        transition_rate[:3, :3] = -skew(previous_omega)
        transition_rate[:3, 3:6] = np.eye(3)
        transition_rate[3:6, 3:6] = self._angular_velocity_jacobian(
            previous_omega
        )
        transition = np.eye(9) + transition_rate * dt
        process_noise = np.zeros((9, 9), dtype=float)
        acceleration_variance = self.angular_acceleration_noise_std**2
        process_noise[:3, :3] = acceleration_variance * dt**3 / 3.0 * np.eye(3)
        process_noise[:3, 3:6] = acceleration_variance * dt**2 / 2.0 * np.eye(3)
        process_noise[3:6, :3] = process_noise[:3, 3:6]
        process_noise[3:6, 3:6] = acceleration_variance * dt * np.eye(3)
        process_noise[6:9, 6:9] = (
            self.gyro_bias_random_walk_std**2 * dt * np.eye(3)
        )
        self.covariance = (
            transition @ self.covariance @ transition.T + process_noise
        )
        self.covariance = _symmetrize(self.covariance)

    def update_gyro(self, measurement: Array, measurement_covariance: Array) -> float:
        measurement = np.asarray(measurement, dtype=float).reshape(3)
        measurement_covariance = np.asarray(
            measurement_covariance, dtype=float
        ).reshape(3, 3)
        innovation = measurement - (
            self.angular_velocity_body + self.gyro_bias
        )
        jacobian = np.zeros((3, 9), dtype=float)
        jacobian[:, 3:6] = np.eye(3)
        jacobian[:, 6:9] = np.eye(3)
        return self._update(innovation, jacobian, measurement_covariance)

    def update_star_tracker(
        self,
        quaternion_measurement_i2b_wxyz: Array,
        small_angle_covariance: Array,
    ) -> float:
        innovation = left_attitude_error_vector(
            quaternion_measurement_i2b_wxyz,
            self.quaternion_i2b_wxyz,
        )
        jacobian = np.zeros((3, 9), dtype=float)
        jacobian[:, :3] = np.eye(3)
        return self._update(
            innovation,
            jacobian,
            np.asarray(small_angle_covariance, dtype=float).reshape(3, 3),
        )

    def update_body_vectors(
        self,
        measured_body_vectors: list[Array],
        reference_inertial_vectors: list[Array],
        measurement_covariance: Array,
    ) -> list[float]:
        if len(measured_body_vectors) != len(reference_inertial_vectors):
            raise ValueError("Measured and reference vector counts must match.")
        covariance = np.asarray(measurement_covariance, dtype=float).reshape(3, 3)
        nis_values = []
        for measured, reference in zip(
            measured_body_vectors, reference_inertial_vectors, strict=True
        ):
            measured = _unit_vector(measured)
            reference = _unit_vector(reference)
            predicted = quat_to_dcm_i2b(self.quaternion_i2b_wxyz) @ reference
            innovation = measured - predicted
            jacobian = np.zeros((3, 9), dtype=float)
            jacobian[:, :3] = -skew(predicted)
            nis_values.append(self._update(innovation, jacobian, covariance))
        return nis_values

    def estimate(self, timestamp: float) -> AttitudeEstimate:
        return AttitudeEstimate(
            timestamp=float(timestamp),
            satellite_id=str(self.satellite_id),
            quaternion_i2b_wxyz=self.quaternion_i2b_wxyz.copy(),
            angular_velocity_body=self.angular_velocity_body.copy(),
            gyro_bias=self.gyro_bias.copy(),
            error_covariance=self.covariance.copy(),
        )

    def _update(self, innovation: Array, jacobian: Array, noise: Array) -> float:
        innovation_covariance = jacobian @ self.covariance @ jacobian.T + noise
        innovation_covariance = (
            _symmetrize(innovation_covariance)
            + 1e-12 * np.eye(innovation_covariance.shape[0])
        )
        gain = (
            self.covariance
            @ jacobian.T
            @ np.linalg.pinv(innovation_covariance)
        )
        correction = gain @ innovation
        attitude_increment = small_angle_quaternion_wxyz(correction[:3])
        self.quaternion_i2b_wxyz = quat_normalize_wxyz(
            quat_multiply_wxyz(
                attitude_increment,
                self.quaternion_i2b_wxyz,
            )
        )
        self.angular_velocity_body += correction[3:6]
        self.gyro_bias += correction[6:9]
        residual = np.eye(9) - gain @ jacobian
        self.covariance = (
            residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        )
        self.covariance = _symmetrize(self.covariance)
        return float(
            innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation
        )

    def _angular_velocity_jacobian(self, angular_velocity: Array) -> Array:
        base = rigid_body_angular_acceleration(
            angular_velocity, self.inertia, self.torque
        )
        jacobian = np.zeros((3, 3), dtype=float)
        for index in range(3):
            step = 1e-7 * max(1.0, abs(float(angular_velocity[index])))
            perturbed = np.asarray(angular_velocity, dtype=float).copy()
            perturbed[index] += step
            jacobian[:, index] = (
                rigid_body_angular_acceleration(
                    perturbed, self.inertia, self.torque
                )
                - base
            ) / step
        return jacobian


def _unit_vector(vector: Array) -> Array:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("Direction vector must be nonzero.")
    return vector / norm


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)
