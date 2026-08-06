from __future__ import annotations

import numpy as np

Array = np.ndarray


def skew(vector: Array) -> Array:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def quat_normalize_wxyz(quaternion: Array) -> Array:
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-15:
        raise ValueError("Quaternion norm is too small.")
    return quaternion / norm


def quat_conjugate_wxyz(quaternion: Array) -> Array:
    w, x, y, z = quat_normalize_wxyz(quaternion)
    return np.array([w, -x, -y, -z], dtype=float)


def quat_multiply_wxyz(left: Array, right: Array) -> Array:
    w1, x1, y1, z1 = np.asarray(left, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(right, dtype=float).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def small_angle_quaternion_wxyz(rotation_vector: Array) -> Array:
    rotation_vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = np.linalg.norm(rotation_vector)
    if angle < 1e-12:
        return quat_normalize_wxyz(
            np.array([1.0, *(0.5 * rotation_vector)], dtype=float)
        )
    axis = rotation_vector / angle
    half_angle = 0.5 * angle
    return np.hstack([np.cos(half_angle), axis * np.sin(half_angle)])


def quat_to_dcm_i2b(quaternion_i2b: Array) -> Array:
    """Return the DCM that maps inertial vector components into body components."""

    w, x, y, z = quat_normalize_wxyz(quaternion_i2b)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def left_attitude_error_vector(reference: Array, estimate: Array) -> Array:
    """Small rotation dtheta satisfying q_ref ~= dq(dtheta) x q_est."""

    error = quat_multiply_wxyz(
        quat_normalize_wxyz(reference),
        quat_conjugate_wxyz(estimate),
    )
    error = quat_normalize_wxyz(error)
    if error[0] < 0.0:
        error = -error
    return 2.0 * error[1:]


def attitude_error_angle_deg(estimate: Array, truth: Array) -> float:
    error = quat_multiply_wxyz(
        quat_normalize_wxyz(estimate),
        quat_conjugate_wxyz(truth),
    )
    error = quat_normalize_wxyz(error)
    return float(np.rad2deg(2.0 * np.arccos(np.clip(abs(error[0]), 0.0, 1.0))))


def rigid_body_angular_acceleration(
    angular_velocity: Array,
    inertia: Array,
    torque: Array | None = None,
) -> Array:
    omega = np.asarray(angular_velocity, dtype=float).reshape(3)
    inertia = np.asarray(inertia, dtype=float).reshape(3, 3)
    applied_torque = (
        np.zeros(3, dtype=float)
        if torque is None
        else np.asarray(torque, dtype=float).reshape(3)
    )
    return np.linalg.solve(
        inertia,
        applied_torque - np.cross(omega, inertia @ omega),
    )


def rk4_angular_velocity(
    angular_velocity: Array,
    dt: float,
    inertia: Array,
    torque: Array | None = None,
) -> Array:
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    omega = np.asarray(angular_velocity, dtype=float).reshape(3)
    function = lambda value: rigid_body_angular_acceleration(value, inertia, torque)
    k1 = function(omega)
    k2 = function(omega + 0.5 * dt * k1)
    k3 = function(omega + 0.5 * dt * k2)
    k4 = function(omega + dt * k3)
    return omega + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def propagate_attitude(
    quaternion_i2b: Array,
    angular_velocity: Array,
    dt: float,
    inertia: Array,
    torque: Array | None = None,
) -> tuple[Array, Array]:
    omega_next = rk4_angular_velocity(angular_velocity, dt, inertia, torque)
    omega_average = 0.5 * (
        np.asarray(angular_velocity, dtype=float).reshape(3) + omega_next
    )
    increment = small_angle_quaternion_wxyz(omega_average * dt)
    quaternion_next = quat_normalize_wxyz(
        quat_multiply_wxyz(quaternion_i2b, increment)
    )
    return quaternion_next, omega_next
