from __future__ import annotations

from typing import Callable

import numpy as np

from .constants import J2, MU_EARTH, R_EARTH


def finite_difference_velocity(r: np.ndarray, dt: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("r 必须为形状 (N, 3) 的数组。")
    if len(r) < 2:
        raise ValueError("至少需要两个位置样本。")
    if dt <= 0.0:
        raise ValueError("dt 必须大于零。")
    v = np.zeros_like(r)
    v[1:-1] = (r[2:] - r[:-2]) / (2.0 * dt)
    v[0] = (r[1] - r[0]) / dt
    v[-1] = (r[-1] - r[-2]) / dt
    return v


def accel_two_body_j2(
    r_eci: np.ndarray,
    mu: float = MU_EARTH,
    earth_radius: float = R_EARTH,
    j2: float = J2,
) -> np.ndarray:
    r_eci = np.asarray(r_eci, dtype=float).reshape(3)
    x, y, z = r_eci
    r2 = float(x * x + y * y + z * z)
    if r2 <= 0.0:
        raise ValueError("位置向量范数必须大于零。")
    r = np.sqrt(r2)
    a_2b = -mu * r_eci / (r2 * r)
    fac = 1.5 * j2 * mu * earth_radius**2 / r**5
    s = 5.0 * z * z / r2
    a_j2 = fac * np.array([x * (s - 1.0), y * (s - 1.0), z * (s - 3.0)])
    return a_2b + a_j2


def rel_dynamics_rhs(x_rel_eci: np.ndarray, chief_state_eci: np.ndarray) -> np.ndarray:
    x = np.asarray(x_rel_eci, dtype=float).reshape(6)
    chief = np.asarray(chief_state_eci, dtype=float).reshape(6)
    dr, dv = x[:3], x[3:]
    rc = chief[:3]
    rt = rc + dr
    return np.hstack([dv, accel_two_body_j2(rt) - accel_two_body_j2(rc)])


def compute_target_absolute_accel_model(
    x_rel_eci: np.ndarray,
    chief_state_eci: np.ndarray,
) -> np.ndarray:
    x = np.asarray(x_rel_eci, dtype=float).reshape(6)
    chief = np.asarray(chief_state_eci, dtype=float).reshape(6)
    return accel_two_body_j2(chief[:3] + x[:3])


def build_target_absolute_accel_history(
    x_hist_eci: np.ndarray,
    chief_hist_eci: np.ndarray,
) -> np.ndarray:
    x_hist = np.asarray(x_hist_eci, dtype=float)
    chief_hist = np.asarray(chief_hist_eci, dtype=float)
    if x_hist.shape != chief_hist.shape or x_hist.ndim != 2 or x_hist.shape[1] != 6:
        raise ValueError("状态历史必须均为形状 (N, 6) 的数组。")
    return np.vstack([
        compute_target_absolute_accel_model(x_hist[k], chief_hist[k])
        for k in range(len(x_hist))
    ])


def numerical_diff_accel_from_velocity(t: np.ndarray, v_hist: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    v_hist = np.asarray(v_hist, dtype=float)
    if v_hist.ndim != 2 or v_hist.shape[1] != 3 or len(t) != len(v_hist):
        raise ValueError("t 与 v_hist 尺寸不匹配。")
    a_hist = np.zeros_like(v_hist)
    for i in range(3):
        a_hist[:, i] = np.gradient(v_hist[:, i], t, edge_order=2)
    return a_hist


def rk4_step_rel(x_rel_eci: np.ndarray, chief_state_eci: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
        raise ValueError("dt 必须大于零。")
    x = np.asarray(x_rel_eci, dtype=float).reshape(6)
    chief = np.asarray(chief_state_eci, dtype=float).reshape(6)
    k1 = rel_dynamics_rhs(x, chief)
    k2 = rel_dynamics_rhs(x + 0.5 * dt * k1, chief)
    k3 = rel_dynamics_rhs(x + 0.5 * dt * k2, chief)
    k4 = rel_dynamics_rhs(x + dt * k3, chief)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def numerical_jacobian_discrete(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    fx = np.asarray(f(x), dtype=float)
    jac = np.zeros((fx.size, x.size), dtype=float)
    for i in range(x.size):
        step = eps * max(1.0, abs(float(x[i])))
        xp = x.copy(); xp[i] += step
        xm = x.copy(); xm[i] -= step
        jac[:, i] = (np.asarray(f(xp)) - np.asarray(f(xm))) / (2.0 * step)
    return jac


def make_process_noise(dt: float, q_acc: float) -> np.ndarray:
    if dt <= 0.0 or q_acc < 0.0:
        raise ValueError("dt 必须大于零且 q_acc 不能为负。")
    i3 = np.eye(3)
    return q_acc * np.block([
        [(dt**3 / 3.0) * i3, (dt**2 / 2.0) * i3],
        [(dt**2 / 2.0) * i3, dt * i3],
    ])


def absolute_dynamics_rhs(state_eci: np.ndarray) -> np.ndarray:
    """Continuous two-body + J2 dynamics for an absolute ECI state."""
    state = np.asarray(state_eci, dtype=float).reshape(6)
    return np.hstack((state[3:], accel_two_body_j2(state[:3])))


def rk4_step_absolute(state_eci: np.ndarray, dt: float) -> np.ndarray:
    """Advance an absolute ECI orbit by one RK4 step."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    state = np.asarray(state_eci, dtype=float).reshape(6)
    k1 = absolute_dynamics_rhs(state)
    k2 = absolute_dynamics_rhs(state + 0.5 * dt * k1)
    k3 = absolute_dynamics_rhs(state + 0.5 * dt * k2)
    k4 = absolute_dynamics_rhs(state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def propagate_absolute_orbit(initial_state_eci: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Propagate an absolute orbit at strictly increasing timestamps."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if timestamps.size == 0:
        raise ValueError("timestamps cannot be empty.")
    if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("timestamps must be strictly increasing.")
    history = np.zeros((timestamps.size, 6), dtype=float)
    history[0] = np.asarray(initial_state_eci, dtype=float).reshape(6)
    for index in range(1, timestamps.size):
        history[index] = rk4_step_absolute(
            history[index - 1], float(timestamps[index] - timestamps[index - 1])
        )
    return history
