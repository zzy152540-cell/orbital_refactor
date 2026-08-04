from __future__ import annotations

from typing import Callable

import numpy as np

from .attitude import quat_to_dcm_i2b
from .coordinates import build_rtn_quaternion, rotate_eci_to_pri, state_eci_to_spri


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def h_optical_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    r = state_eci_to_spri(x_rel_eci, q_eci2pri)[:3]
    if abs(r[2]) < 1e-12:
        raise ValueError("光学量测模型中 z 分量过小。")
    return np.array([r[0] / r[2], r[1] / r[2]], dtype=float)


def h_nn_position_eci(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    del q_eci2pri
    return np.asarray(x_rel_eci, dtype=float)[:3].copy()


def h_nn_position_velocity_eci(
    x_rel_eci: np.ndarray,
    q_eci2pri: np.ndarray,
) -> np.ndarray:
    del q_eci2pri
    return np.asarray(x_rel_eci, dtype=float).copy()


def h_nn_position_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    return rotate_eci_to_pri(np.asarray(x_rel_eci, dtype=float)[:3], q_eci2pri)


def h_nn_position_velocity_spri(
    x_rel_eci: np.ndarray,
    q_eci2pri: np.ndarray,
) -> np.ndarray:
    return state_eci_to_spri(x_rel_eci, q_eci2pri).copy()


def h_ir_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    r = state_eci_to_spri(x_rel_eci, q_eci2pri)[:3]
    rho_xy = np.hypot(r[0], r[1])
    return np.array([np.arctan2(r[1], r[0]), np.arctan2(r[2], rho_xy)], dtype=float)


def h_radar_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    state = state_eci_to_spri(x_rel_eci, q_eci2pri)
    r, v = state[:3], state[3:]
    rho = np.linalg.norm(r)
    if rho <= 0.0:
        raise ValueError("雷达量测模型中距离必须大于零。")
    return np.array([rho, float(r @ v) / rho], dtype=float)


def measure_relative_range(
    state_i: np.ndarray,
    state_j: np.ndarray,
    noise: float = 0.0,
) -> float:
    """Inter-satellite range measurement between two absolute ECI states."""

    position_i = np.asarray(state_i, dtype=float).reshape(6)[:3]
    position_j = np.asarray(state_j, dtype=float).reshape(6)[:3]
    return float(np.linalg.norm(position_j - position_i) + noise)


def measure_relative_range_rate(
    state_i: np.ndarray,
    state_j: np.ndarray,
    noise: float = 0.0,
) -> float:
    """Inter-satellite range-rate between two absolute ECI states."""

    state_i = np.asarray(state_i, dtype=float).reshape(6)
    state_j = np.asarray(state_j, dtype=float).reshape(6)
    relative_position = state_j[:3] - state_i[:3]
    relative_velocity = state_j[3:] - state_i[3:]
    range_value = np.linalg.norm(relative_position)
    if range_value <= 0.0:
        raise ValueError("Inter-satellite range must be positive.")
    return float(relative_position @ relative_velocity / range_value + noise)


def measure_relative_az_el(
    state_i: np.ndarray,
    state_j: np.ndarray,
    frame: str = "ECI",
    noise: np.ndarray | None = None,
    *,
    quaternion_i2b_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Inter-satellite azimuth/elevation from i to j.

    ``BODY`` measurements use the source satellite's inertial-to-body
    quaternion. The quaternion follows the project's explicit ``wxyz``
    convention.
    """

    state_i = np.asarray(state_i, dtype=float).reshape(6)
    state_j = np.asarray(state_j, dtype=float).reshape(6)
    relative_position = state_j[:3] - state_i[:3]
    if np.linalg.norm(relative_position) <= 0.0:
        raise ValueError("Inter-satellite relative position must be nonzero.")
    normalized_frame = str(frame).upper()
    if normalized_frame in {"RTN", "PRI", "SPRI"}:
        relative_position = rotate_eci_to_pri(
            relative_position,
            build_rtn_quaternion(state_i),
        )
    elif normalized_frame == "BODY":
        if quaternion_i2b_wxyz is None:
            raise ValueError(
                "BODY inter-satellite angles require quaternion_i2b_wxyz."
            )
        relative_position = (
            quat_to_dcm_i2b(quaternion_i2b_wxyz) @ relative_position
        )
    elif normalized_frame != "ECI":
        raise ValueError(f"Unsupported inter-satellite angle frame: {frame}")
    rho_xy = np.hypot(relative_position[0], relative_position[1])
    measurement = np.array(
        [
            np.arctan2(relative_position[1], relative_position[0]),
            np.arctan2(relative_position[2], rho_xy),
        ],
        dtype=float,
    )
    if noise is not None:
        measurement = measurement + np.asarray(noise, dtype=float).reshape(2)
    return measurement


def measure_relative_optical_uv(
    state_i: np.ndarray,
    state_j: np.ndarray,
    frame: str = "BODY",
    noise: np.ndarray | None = None,
    *,
    quaternion_i2b_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """Normalized pinhole image coordinates for a directed observation.

    The inter-satellite camera uses ``+X`` as its BODY boresight and returns
    ``[y/x, z/x]``. This is the same normalized-image-coordinate measurement
    family as the single-satellite ``[x/z, y/z]`` SPRI model, with an explicit
    camera-axis convention.
    """

    state_i = np.asarray(state_i, dtype=float).reshape(6)
    state_j = np.asarray(state_j, dtype=float).reshape(6)
    relative = state_j[:3] - state_i[:3]
    normalized_frame = str(frame).upper()
    if normalized_frame != "BODY":
        raise ValueError("Inter-satellite optical measurements require BODY frame.")
    if quaternion_i2b_wxyz is None:
        raise ValueError("BODY optical measurements require quaternion_i2b_wxyz.")
    relative_body = quat_to_dcm_i2b(quaternion_i2b_wxyz) @ relative
    depth = float(relative_body[0])
    if depth <= 1e-12:
        raise ValueError("Optical target must lie in front of the camera boresight.")
    measurement = np.array(
        [relative_body[1] / depth, relative_body[2] / depth], dtype=float
    )
    if noise is not None:
        measurement += np.asarray(noise, dtype=float).reshape(2)
    return measurement


def measurement_residual(z: np.ndarray, z_pred: np.ndarray, mode: str) -> np.ndarray:
    residual = np.asarray(z, dtype=float) - np.asarray(z_pred, dtype=float)
    if mode.lower() == "ir":
        residual = wrap_angle(residual)
    return residual


def numerical_measurement_jacobian(
    h: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    hx = np.asarray(h(x), dtype=float)
    jac = np.zeros((hx.size, x.size), dtype=float)
    for i in range(x.size):
        step = eps * max(1.0, abs(float(x[i])))
        xp = x.copy(); xp[i] += step
        xm = x.copy(); xm[i] -= step
        jac[:, i] = (np.asarray(h(xp)) - np.asarray(h(xm))) / (2.0 * step)
    return jac
