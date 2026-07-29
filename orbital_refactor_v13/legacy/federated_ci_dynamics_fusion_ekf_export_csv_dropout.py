from pathlib import Path

# -*- coding: utf-8 -*-
"""
SHIRT 三模态联邦 + CI 轨道动力学滤波（Federated + Covariance Intersection）

说明：
1. 与 single_modal_dynamics_integrated_final.py 保持相同的数据读取、spri->ECI 预处理、
   二体+J2 相对动力学传播与量测模型；
2. 每个模态（opt/ir/rad）维护一个本地 EKF；
3. 每个时刻各本地先独立 predict / update（支持 NIS gate）；
4. 再将所有“本时刻有有效后验”的本地结果，用 CI 融合得到全局融合结果；
5. 当三模态同时有效时，采用三路同时 CI；当仅有两路有效时，采用两路 CI；
6. 可选 reset_feedback：是否把融合后的状态回灌到各本地滤波器（默认 False）；
7. 仅报告和可视化目标星绝对加速度误差。
"""

import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


CONFIG = {
    "metadata_path": r"E:\Satellite Datasets\shirt\roe2\metadata.json",
    "roe_path": r"E:\Satellite Datasets\shirt\roe2\roe2.json",
    "traj_name": "roe2",
    "save_preprocessed_npz": None,
    "random_seed_meas": 42,
    "random_seed_init": 2025,
    "do_plot": True,

    # ---- 量测噪声 ----
    "sigma_u": 2e-3,
    "sigma_v": 2e-3,
    "sigma_az_deg": 1.5,
    "sigma_el_deg": 1.5,
    "sigma_rho": 1.0,
    "sigma_rhodot": 0.02,

    # ---- 过程噪声 ----
    "q_acc": 1e-4,

    # ---- 初值 ----
    "pos_init_std": 10.0,
    "vel_init_std": 0.05,

    # ---- NIS gate ----
    "nis_gate_enable": True,
    "nis_gate_mode": "soft",     # "hard" / "soft"
    "nis_soft_scale": 20.0,
    "nis_gate_thresholds": {"opt": 16.0, "ir": 16.0, "rad": 16.0},

    # ---- 联邦 + CI ----
    "reset_feedback": True,      # 融合后是否回灌到各本地
    "ci_objective": "trace",      # "trace" / "logdet"
    "ci_grid_points": 41,        # CI 一维网格搜索密度

    "dropout_windows_opt": [], # (start_time, end_time)

    "dropout_windows_ir":  [],

    "dropout_windows_rad": [(0000,12000)],
    "export_csv_enable": True,
    "export_csv_path": None,          # None 时自动按 traj_name 命名
    "export_rmse_report_enable": True,
    "export_rmse_report_path": None,  # None 时自动按 traj_name 命名
}


MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0
J2 = 1.08262668e-3


# =========================
# 基础工具
# =========================
def load_json(json_path: str) -> Any:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_numpy_safe(x: Any, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def wrap_angle(a: np.ndarray):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compute_rmse(err: np.ndarray):
    rmse_dim = np.sqrt(np.mean(err**2, axis=0))
    rmse_norm = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    return rmse_dim, rmse_norm


# =========================
# 数据读取与预处理
# =========================
def build_time_axis(pSim: Dict[str, Any], n: int) -> np.ndarray:
    cam_step = float(pSim.get("cam_step", 1.0))
    return np.arange(n, dtype=float) * cam_step


def build_unified_dataset(metadata_path: str, roe_path: str, traj_name: str) -> Dict[str, Any]:
    metadata = load_json(metadata_path)
    roe_data = load_json(roe_path)

    filenames = [item["filename"] for item in roe_data]
    q_vbs2tango_true = to_numpy_safe([item["q_vbs2tango_true"] for item in roe_data], dtype=float)
    r_Vo2To_vbs_true = to_numpy_safe([item["r_Vo2To_vbs_true"] for item in roe_data], dtype=float)

    sim_info = metadata["pSim"]
    sAbsState = metadata["sAbsState"]
    tRelState = metadata["tRelState"]

    rv_scom2tcom_spri = to_numpy_safe(tRelState["rv_scom2tcom_spri"], dtype=float)
    r_scom2tcom_spri = rv_scom2tcom_spri[:, 0:3]
    v_scom2tcom_spri = rv_scom2tcom_spri[:, 3:6]

    common = {
        "traj_name": traj_name,
        "time_sec": build_time_axis(sim_info, len(filenames)),
        "filenames": filenames,
        "q_vbs2tango_true": q_vbs2tango_true,
        "r_Vo2To_vbs_true": r_Vo2To_vbs_true,
        "rv_scom2tcom_spri": rv_scom2tcom_spri,
        "r_scom2tcom_spri": r_scom2tcom_spri,
        "v_scom2tcom_spri": v_scom2tcom_spri,
        "q_spri2tpri": to_numpy_safe(tRelState["q_spri2tpri"], dtype=float),
        "w_tpri2spri_tpri": to_numpy_safe(tRelState["w_tpri2spri_tpri"], dtype=float),
        "rv_eci2com_eci": to_numpy_safe(sAbsState["rv_eci2com_eci"], dtype=float),
        "q_eci2pri": to_numpy_safe(sAbsState["q_eci2pri"], dtype=float),
        "w_pri": to_numpy_safe(sAbsState["w_pri"], dtype=float),
    }
    return {"sim_info": sim_info, "common": common}


def quat_to_dcm_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),         1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),         2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rotate_pri_to_eci(vec_pri: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    # 已验证：本数据下 q_eci2pri 数值上更适合按 pri->eci 主动旋转使用
    return quat_to_dcm_wxyz(q_eci2pri) @ vec_pri


def rotate_eci_to_pri(vec_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    return quat_to_dcm_wxyz(q_eci2pri).T @ vec_eci


def finite_difference_velocity(r: np.ndarray, dt: float) -> np.ndarray:
    v = np.zeros_like(r)
    v[1:-1] = (r[2:] - r[:-2]) / (2.0 * dt)
    v[0] = (r[1] - r[0]) / dt
    v[-1] = (r[-1] - r[-2]) / dt
    return v


def build_orbit_eci_dataset(metadata_path: str, roe_path: str, traj_name: str) -> Dict[str, Any]:
    dataset = build_unified_dataset(metadata_path, roe_path, traj_name)
    common = dataset["common"]
    t = common["time_sec"]
    dt = float(t[1] - t[0])

    r_rel_pri = common["r_scom2tcom_spri"]
    v_rel_pri = common["v_scom2tcom_spri"]
    q_eci2pri = common["q_eci2pri"]
    chief = common["rv_eci2com_eci"]

    n = len(t)
    r_rel_eci = np.zeros((n, 3), dtype=float)
    v_rel_eci = np.zeros((n, 3), dtype=float)
    for k in range(n):
        r_rel_eci[k] = rotate_pri_to_eci(r_rel_pri[k], q_eci2pri[k])
        v_rel_eci[k] = rotate_pri_to_eci(v_rel_pri[k], q_eci2pri[k])

    v_fd = finite_difference_velocity(r_rel_eci, dt)
    vel_consistency_rmse = float(np.sqrt(np.mean(np.sum((v_rel_eci - v_fd) ** 2, axis=1))))

    target = np.hstack([chief[:, 0:3] + r_rel_eci, chief[:, 3:6] + v_rel_eci])

    return {
        "time_sec": t,
        "rv_rel_eci_true": np.hstack([r_rel_eci, v_rel_eci]),
        "rv_eci2scom_eci_true": chief,
        "rv_eci2tcom_eci_true": target,
        "r_scom2tcom_spri": r_rel_pri,
        "v_scom2tcom_spri": v_rel_pri,
        "q_eci2pri": q_eci2pri,
        "preprocess_summary": {
            "quat_order": "wxyz",
            "quat_maps": "pri_to_eci",
            "vel_rule": "ignore_omega",
            "velocity_consistency_rmse_mps": vel_consistency_rmse,
        },
    }


# =========================
# 动力学模型
# =========================
def accel_two_body_j2(r_eci: np.ndarray) -> np.ndarray:
    x, y, z = r_eci
    r2 = x * x + y * y + z * z
    r = np.sqrt(max(r2, 1e-18))
    r3 = r2 * r

    a_2b = -MU_EARTH * r_eci / r3

    z2 = z * z
    fac = 1.5 * J2 * MU_EARTH * (R_EARTH ** 2) / (r ** 5)
    s = 5.0 * z2 / r2
    a_j2 = fac * np.array([
        x * (s - 1.0),
        y * (s - 1.0),
        z * (s - 3.0),
    ])
    return a_2b + a_j2


def rel_dynamics_rhs(x_rel_eci: np.ndarray, chief_state_eci: np.ndarray) -> np.ndarray:
    dr = x_rel_eci[0:3]
    dv = x_rel_eci[3:6]
    rc = chief_state_eci[0:3]
    rt = rc + dr

    a_c = accel_two_body_j2(rc)
    a_t = accel_two_body_j2(rt)
    a_rel = a_t - a_c
    return np.hstack([dv, a_rel])


def compute_target_absolute_accel_model(x_rel_eci: np.ndarray, chief_state_eci: np.ndarray) -> np.ndarray:
    rc = chief_state_eci[0:3]
    rt = rc + x_rel_eci[0:3]
    return accel_two_body_j2(rt)


def build_target_absolute_accel_history(x_hist_eci: np.ndarray, chief_hist_eci: np.ndarray) -> np.ndarray:
    a_hist = np.zeros((len(x_hist_eci), 3), dtype=float)
    for k in range(len(x_hist_eci)):
        a_hist[k] = compute_target_absolute_accel_model(x_hist_eci[k], chief_hist_eci[k])
    return a_hist


def numerical_diff_accel_from_velocity(t: np.ndarray, v_hist: np.ndarray) -> np.ndarray:
    a_hist = np.zeros_like(v_hist)
    for i in range(3):
        a_hist[:, i] = np.gradient(v_hist[:, i], t, edge_order=2)
    return a_hist


def rk4_step_rel(x_rel_eci: np.ndarray, chief_state_eci: np.ndarray, dt: float) -> np.ndarray:
    k1 = rel_dynamics_rhs(x_rel_eci, chief_state_eci)
    k2 = rel_dynamics_rhs(x_rel_eci + 0.5 * dt * k1, chief_state_eci)
    k3 = rel_dynamics_rhs(x_rel_eci + 0.5 * dt * k2, chief_state_eci)
    k4 = rel_dynamics_rhs(x_rel_eci + dt * k3, chief_state_eci)
    return x_rel_eci + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def numerical_jacobian_discrete(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = len(x)
    F = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        F[:, i] = (f(x + dx) - f(x - dx)) / (2.0 * eps)
    return F


def make_process_noise(dt: float, q_acc: float) -> np.ndarray:
    I3 = np.eye(3)
    Q = q_acc * np.block([
        [(dt**4 / 4.0) * I3, (dt**3 / 2.0) * I3],
        [(dt**3 / 2.0) * I3, (dt**2) * I3],
    ])
    return Q


# =========================
# 量测生成
# =========================
def gen_optical_measurements(r_true_spri, sigma_u=1e-3, sigma_v=1e-3, dropout_prob=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    x = r_true_spri[:, 0]
    y = r_true_spri[:, 1]
    z = r_true_spri[:, 2]
    eps = 1e-12
    u_true = x / np.maximum(z, eps)
    v_true = y / np.maximum(z, eps)
    z_opt = np.column_stack([
        u_true + rng.normal(0.0, sigma_u, size=len(z)),
        v_true + rng.normal(0.0, sigma_v, size=len(z)),
    ])
    valid = np.ones(len(z), dtype=bool)
    if dropout_prob > 0.0:
        valid = rng.random(len(z)) > dropout_prob
    return z_opt, valid


def gen_ir_measurements(r_true_spri, sigma_az, sigma_el, dropout_prob=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    x = r_true_spri[:, 0]
    y = r_true_spri[:, 1]
    z = r_true_spri[:, 2]
    az_true = np.arctan2(y, x)
    el_true = np.arctan2(z, np.sqrt(x**2 + y**2))
    z_ir = np.column_stack([
        az_true + rng.normal(0.0, sigma_az, size=len(x)),
        el_true + rng.normal(0.0, sigma_el, size=len(x)),
    ])
    valid = np.ones(len(x), dtype=bool)
    if dropout_prob > 0.0:
        valid = rng.random(len(x)) > dropout_prob
    return z_ir, valid


def gen_radar_measurements(r_true_spri, v_true_spri, sigma_rho, sigma_rhodot, dropout_prob=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    rho_true = np.linalg.norm(r_true_spri, axis=1)
    rhodot_true = np.sum(r_true_spri * v_true_spri, axis=1) / np.maximum(rho_true, 1e-12)
    z_rad = np.column_stack([
        rho_true + rng.normal(0.0, sigma_rho, size=len(rho_true)),
        rhodot_true + rng.normal(0.0, sigma_rhodot, size=len(rho_true)),
    ])
    valid = np.ones(len(rho_true), dtype=bool)
    if dropout_prob > 0.0:
        valid = rng.random(len(rho_true)) > dropout_prob
    return z_rad, valid

def apply_dropout_windows(valid: np.ndarray, t: np.ndarray, windows):
    """
    将 valid 按指定时间区间置为 False
    windows: [(t_start, t_end), ...]
    默认采用闭区间 [t_start, t_end]
    """
    valid_new = valid.copy()
    if windows is None:
        return valid_new

    for w in windows:
        if len(w) != 2:
            raise ValueError(f"dropout window 格式错误: {w}")
        t_start, t_end = float(w[0]), float(w[1])
        mask = (t >= t_start) & (t <= t_end)
        valid_new[mask] = False
    return valid_new

# =========================
# 观测模型
# =========================
def state_eci_to_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    r_spri = rotate_eci_to_pri(x_rel_eci[0:3], q_eci2pri)
    v_spri = rotate_eci_to_pri(x_rel_eci[3:6], q_eci2pri)
    return np.hstack([r_spri, v_spri])


def h_optical_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray):
    xs = state_eci_to_spri(x_rel_eci, q_eci2pri)
    px, py, pz = xs[0], xs[1], xs[2]
    eps = 1e-12
    return np.array([px / max(pz, eps), py / max(pz, eps)])


def h_ir_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray):
    xs = state_eci_to_spri(x_rel_eci, q_eci2pri)
    px, py, pz = xs[0], xs[1], xs[2]
    return np.array([
        np.arctan2(py, px),
        np.arctan2(pz, np.sqrt(px**2 + py**2))
    ])


def h_radar_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray):
    xs = state_eci_to_spri(x_rel_eci, q_eci2pri)
    px, py, pz, vx, vy, vz = xs
    rho = np.sqrt(px**2 + py**2 + pz**2)
    rho = max(rho, 1e-12)
    rhodot = (px * vx + py * vy + pz * vz) / rho
    return np.array([rho, rhodot])


def measurement_residual(z: np.ndarray, z_pred: np.ndarray, mode: str):
    y = z - z_pred
    if mode == "ir":
        y[0] = wrap_angle(np.array([y[0]]))[0]
        y[1] = wrap_angle(np.array([y[1]]))[0]
    return y


def numerical_measurement_jacobian(h, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = len(h(x))
    n = len(x)
    H = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        H[:, i] = (h(x + dx) - h(x - dx)) / (2.0 * eps)
    return H


# =========================
# 本地 EKF
# =========================
class LocalDynamicsEKF:
    def __init__(self, Q: np.ndarray, R: np.ndarray, mode_name: str,
                 gate_enable=False, gate_threshold=np.inf, gate_mode="soft", soft_scale=20.0):
        self.Q = Q
        self.R = R
        self.mode_name = mode_name
        self.gate_enable = gate_enable
        self.gate_threshold = gate_threshold
        self.gate_mode = gate_mode
        self.soft_scale = soft_scale

    def predict(self, x, P, chief_state_eci, dt):
        f = lambda xx: rk4_step_rel(xx, chief_state_eci, dt)
        x_pred = f(x)
        F = numerical_jacobian_discrete(f, x)
        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred

    def get_h(self, q_eci2pri: np.ndarray):
        if self.mode_name == "opt":
            return lambda xx: h_optical_spri(xx, q_eci2pri)
        if self.mode_name == "ir":
            return lambda xx: h_ir_spri(xx, q_eci2pri)
        if self.mode_name == "rad":
            return lambda xx: h_radar_spri(xx, q_eci2pri)
        raise ValueError(f"Unsupported mode_name: {self.mode_name}")

    def update(self, x_pred, P_pred, z, q_eci2pri):
        h = self.get_h(q_eci2pri)
        z_pred = h(x_pred)
        H = numerical_measurement_jacobian(h, x_pred)
        y = measurement_residual(z, z_pred, self.mode_name)
        R_eff = self.R.copy()

        S = H @ P_pred @ H.T + R_eff + 1e-9 * np.eye(H.shape[0])
        nis = float(y.T @ np.linalg.pinv(S) @ y)
        gated = False
        skipped = False

        if self.gate_enable and np.isfinite(self.gate_threshold) and nis > self.gate_threshold:
            gated = True
            if self.gate_mode == "hard":
                skipped = True
                return x_pred.copy(), P_pred.copy(), nis, gated, skipped
            if self.gate_mode == "soft":
                R_eff = self.soft_scale * self.R
                S = H @ P_pred @ H.T + R_eff + 1e-9 * np.eye(H.shape[0])

        K = (P_pred @ H.T) @ np.linalg.pinv(S)
        x_upd = x_pred + K @ y
        I = np.eye(len(x_pred))
        P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R_eff @ K.T
        return x_upd, P_upd, nis, gated, skipped


# =========================
# CI 融合
# =========================
def ci_objective(P: np.ndarray, mode: str = "trace") -> float:
    if mode == "trace":
        return float(np.trace(P))
    if mode == "logdet":
        sign, logdet = np.linalg.slogdet(P)
        if sign <= 0:
            return np.inf
        return float(logdet)
    raise ValueError(f"Unsupported CI objective: {mode}")


def ci_fuse_pair(x1, P1, x2, P2, objective="trace", grid_points=101) -> Tuple[np.ndarray, np.ndarray, float]:
    I1 = np.linalg.pinv(P1)
    I2 = np.linalg.pinv(P2)

    best_val = np.inf
    best = None
    omegas = np.linspace(0.0, 1.0, int(grid_points))
    for w in omegas:
        P_inv = w * I1 + (1.0 - w) * I2
        P = np.linalg.pinv(P_inv)
        val = ci_objective(P, objective)
        if val < best_val:
            x = P @ (w * I1 @ x1 + (1.0 - w) * I2 @ x2)
            best_val = val
            best = (x, P, float(w))
    return best


def ci_fuse_three(x1, P1, x2, P2, x3, P3, objective="trace", grid_points=31):
    """
    三路同时 CI 融合：
        P^{-1} = w1*P1^{-1} + w2*P2^{-1} + w3*P3^{-1}
        x = P * (w1*P1^{-1}x1 + w2*P2^{-1}x2 + w3*P3^{-1}x3)
    约束：
        w1,w2,w3 >= 0, 且 w1 + w2 + w3 = 1
    """
    I1 = np.linalg.pinv(P1)
    I2 = np.linalg.pinv(P2)
    I3 = np.linalg.pinv(P3)

    best_val = np.inf
    best = None

    omegas = np.linspace(0.0, 1.0, int(grid_points))
    for w1 in omegas:
        for w2 in omegas:
            w3 = 1.0 - w1 - w2
            if w3 < 0.0:
                continue

            P_inv = w1 * I1 + w2 * I2 + w3 * I3
            P = np.linalg.pinv(P_inv)
            val = ci_objective(P, objective)

            if val < best_val:
                x = P @ (w1 * I1 @ x1 + w2 * I2 @ x2 + w3 * I3 @ x3)
                best_val = val
                best = (x, P, np.array([w1, w2, w3], dtype=float))

    return best


def ci_fuse_posteriors(posterior_list, objective="trace", grid_points=31):
    """
    根据 posterior_list 中的有效模态数量自动选择 CI 方式：
    - 1 路：直接返回
    - 2 路：pairwise CI
    - 3 路：simultaneous CI

    输入:
        posterior_list = [(name, x, P), ...]
    返回:
        x_fused, P_fused, weight_dict
    """
    if len(posterior_list) == 0:
        raise ValueError("CI fusion 输入为空")

    if len(posterior_list) == 1:
        name, x, P = posterior_list[0]
        return x.copy(), P.copy(), {name: 1.0}

    if len(posterior_list) == 2:
        name1, x1, P1 = posterior_list[0]
        name2, x2, P2 = posterior_list[1]
        x_f, P_f, w = ci_fuse_pair(x1, P1, x2, P2, objective=objective, grid_points=grid_points)
        return x_f, P_f, {name1: float(w), name2: float(1.0 - w)}

    if len(posterior_list) == 3:
        posterior_dict = {name: (x, P) for name, x, P in posterior_list}
        x_opt, P_opt = posterior_dict["opt"]
        x_ir, P_ir = posterior_dict["ir"]
        x_rad, P_rad = posterior_dict["rad"]

        x_f, P_f, w_vec = ci_fuse_three(
            x_opt, P_opt,
            x_ir, P_ir,
            x_rad, P_rad,
            objective=objective,
            grid_points=grid_points,
        )
        return x_f, P_f, {
            "opt": float(w_vec[0]),
            "ir": float(w_vec[1]),
            "rad": float(w_vec[2]),
        }

    raise ValueError(f"当前只支持最多 3 路融合，收到 {len(posterior_list)} 路")


# =========================
# 联邦 + CI 主流程
# =========================
def run_federated_ci_filter(
    t, x_true_eci, chief_hist_eci, q_eci2pri_hist,
    z_opt, valid_opt, z_ir, valid_ir, z_rad, valid_rad,
    local_filters: Dict[str, LocalDynamicsEKF], x0, P0,
    reset_feedback=False, ci_objective_name="trace", ci_grid_points=101
):
    modes = ["opt", "ir", "rad"]
    N = len(t)

    x_local = {m: np.zeros((N, 6)) for m in modes}
    P_local = {m: np.zeros((N, 6, 6)) for m in modes}
    x_fused = np.zeros((N, 6))
    P_fused = np.zeros((N, 6, 6))

    for m in modes:
        x_local[m][0] = x0.copy()
        P_local[m][0] = P0.copy()
    x_fused[0] = x0.copy()
    P_fused[0] = P0.copy()

    nis_hist = {m: np.full(N, np.nan) for m in modes}
    gate_hist = {m: np.zeros(N, dtype=bool) for m in modes}
    stats = {m: {"accepted": 0, "rejected": 0, "skipped": 0} for m in modes}
    ci_weight_hist = []

    valid_map = {"opt": valid_opt, "ir": valid_ir, "rad": valid_rad}
    z_map = {"opt": z_opt, "ir": z_ir, "rad": z_rad}

    x_prev = {m: x0.copy() for m in modes}
    P_prev = {m: P0.copy() for m in modes}

    for k in range(1, N):
        dt = float(t[k] - t[k - 1])
        posterior_list = []

        for m in modes:
            ekf = local_filters[m]
            x_pred, P_pred = ekf.predict(x_prev[m], P_prev[m], chief_hist_eci[k - 1], dt)

            if not valid_map[m][k]:
                stats[m]["skipped"] += 1
                x_post, P_post = x_pred, P_pred
                nis = np.nan
                gated = False
                skipped = True
            else:
                x_post, P_post, nis, gated, skipped = ekf.update(
                    x_pred, P_pred, z_map[m][k], q_eci2pri_hist[k]
                )
                nis_hist[m][k] = nis
                gate_hist[m][k] = gated
                if skipped:
                    stats[m]["rejected"] += 1
                else:
                    stats[m]["accepted"] += 1
                    posterior_list.append((m, x_post, P_post))

            x_local[m][k] = x_post
            P_local[m][k] = P_post
            x_prev[m] = x_post
            P_prev[m] = P_post

        # CI 融合
        if len(posterior_list) == 0:
            x_fused[k] = x_fused[k - 1].copy()
            P_fused[k] = P_fused[k - 1].copy()
            ci_weight_hist.append(None)
        else:
            sort_key = {"opt": 0, "ir": 1, "rad": 2}
            posterior_list_sorted = sorted(posterior_list, key=lambda x: sort_key[x[0]])

            x_ci, P_ci, weight_dict = ci_fuse_posteriors(
                posterior_list_sorted,
                objective=ci_objective_name,
                grid_points=ci_grid_points
            )
            x_fused[k] = x_ci
            P_fused[k] = P_ci
            ci_weight_hist.append(weight_dict)

        # 是否回灌
        if reset_feedback:
            for m in modes:
                x_prev[m] = x_fused[k].copy()
                P_prev[m] = P_fused[k].copy()
                x_local[m][k] = x_fused[k].copy()
                P_local[m][k] = P_fused[k].copy()

    pos_err = x_fused[:, 0:3] - x_true_eci[:, 0:3]
    vel_err = x_fused[:, 3:6] - x_true_eci[:, 3:6]

    target_true_eci = np.zeros_like(x_true_eci)
    target_true_eci[:, 0:3] = chief_hist_eci[:, 0:3] + x_true_eci[:, 0:3]
    target_true_eci[:, 3:6] = chief_hist_eci[:, 3:6] + x_true_eci[:, 3:6]

    target_a_hat = build_target_absolute_accel_history(x_fused, chief_hist_eci)
    target_a_num = numerical_diff_accel_from_velocity(t, target_true_eci[:, 3:6])
    target_acc_err = target_a_hat - target_a_num

    return {
        "x_fused": x_fused,
        "P_fused": P_fused,
        "x_local": x_local,
        "P_local": P_local,
        "nis": nis_hist,
        "gate_hist": gate_hist,
        "valid_opt": np.asarray(valid_opt, dtype=bool),
        "valid_ir": np.asarray(valid_ir, dtype=bool),
        "valid_rad": np.asarray(valid_rad, dtype=bool),
        "pos_err": pos_err,
        "vel_err": vel_err,
        "target_acc_err": target_acc_err,
        "stats": stats,
        "ci_weight_hist": ci_weight_hist,
    }


# =========================
# 输出与可视化
# =========================
def print_metrics(name, res):
    pos_rmse_dim, pos_rmse_norm = compute_rmse(res["pos_err"])
    vel_rmse_dim, vel_rmse_norm = compute_rmse(res["vel_err"])
    acc_rmse_dim, acc_rmse_norm = compute_rmse(res["target_acc_err"])

    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"位置 RMSE 分量 [m]   : x={pos_rmse_dim[0]:.4f}, y={pos_rmse_dim[1]:.4f}, z={pos_rmse_dim[2]:.4f}")
    print(f"位置 RMSE 范数 [m]   : {pos_rmse_norm:.4f}")
    print(f"速度 RMSE 分量 [m/s] : vx={vel_rmse_dim[0]:.6f}, vy={vel_rmse_dim[1]:.6f}, vz={vel_rmse_dim[2]:.6f}")
    print(f"速度 RMSE 范数 [m/s] : {vel_rmse_norm:.6f}")
    print(f"目标星绝对加速度误差 RMSE 分量 [m/s²]: ax={acc_rmse_dim[0]:.6e}, ay={acc_rmse_dim[1]:.6e}, az={acc_rmse_dim[2]:.6e}")
    print(f"目标星绝对加速度误差 RMSE 范数 [m/s²]: {acc_rmse_norm:.6e}")
    for mode in ["opt", "ir", "rad"]:
        nis = res["nis"][mode]
        nis_valid = nis[~np.isnan(nis)]
        nis_mean = np.mean(nis_valid) if len(nis_valid) > 0 else np.nan
        nis_med = np.median(nis_valid) if len(nis_valid) > 0 else np.nan
        st = res["stats"][mode]
        print(f"[{mode}] NIS均值={nis_mean:.4f}, 中位数={nis_med:.4f}, accepted={st['accepted']}, rejected={st['rejected']}, skipped={st['skipped']}")
    print()


def plot_results(t, res, t_start_plot=500.0):
    """
    仅显示 t >= t_start_plot 之后的滤波结果
    例如：
        plot_results(t, res, t_start_plot=500.0)
        plot_results(t, res, t_start_plot=1000.0)
    """
    mask = t >= float(t_start_plot)
    if not np.any(mask):
        raise ValueError(f"t_start_plot={t_start_plot} 超出时间范围，当前最大时间为 {t[-1]:.3f} s")

    t_plot = t[mask]

    pos_err_plot = res["pos_err"][mask]
    vel_err_plot = res["vel_err"][mask]
    acc_err_plot = res["target_acc_err"][mask]

    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, np.linalg.norm(pos_err_plot, axis=1), label="federated+CI")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error Norm [m]")
    plt.title(f"Federated+CI Position Error Norm (t >= {t_start_plot:.0f} s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, np.linalg.norm(vel_err_plot, axis=1), label="federated+CI")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity Error Norm [m/s]")
    plt.title(f"Federated+CI Velocity Error Norm (t >= {t_start_plot:.0f} s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(t_plot, np.linalg.norm(acc_err_plot, axis=1), label="federated+CI")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration Error Norm [m/s²]")
    plt.title(f"Target Absolute Acceleration Error Norm (t >= {t_start_plot:.0f} s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for mode in ["opt", "ir", "rad"]:
        nis_plot = res["nis"][mode][mask]
        gate_plot = res["gate_hist"][mode][mask]

        plt.plot(t_plot, nis_plot, label=mode)
        if np.any(gate_plot):
            idx = np.where(gate_plot)[0]
            plt.scatter(t_plot[idx], nis_plot[idx], s=12, marker="x")

    plt.xlabel("Time [s]")
    plt.ylabel("NIS")
    plt.title(f"Local Filter Per-Modal NIS (t >= {t_start_plot:.0f} s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # -----------------------------
    # CI 权重可视化（统一三模态权重）
    # ci_weight_hist[k-1] 对应时刻 t[k]
    # 每个元素形如：
    #   None
    #   {"opt": 1.0}
    #   {"opt": w1, "ir": w2}
    #   {"opt": w1, "ir": w2, "rad": w3}
    # -----------------------------
    ci_weight_hist = res["ci_weight_hist"]

    w_opt = np.full(len(t), np.nan)
    w_ir = np.full(len(t), np.nan)
    w_rad = np.full(len(t), np.nan)

    for k, wk in enumerate(ci_weight_hist, start=1):
        if wk is None:
            continue
        if "opt" in wk:
            w_opt[k] = wk["opt"]
        if "ir" in wk:
            w_ir[k] = wk["ir"]
        if "rad" in wk:
            w_rad[k] = wk["rad"]

    w_opt_plot = w_opt[mask]
    w_ir_plot = w_ir[mask]
    w_rad_plot = w_rad[mask]

    plt.figure(figsize=(10, 6))
    has_curve = False

    if np.any(~np.isnan(w_opt_plot)):
        plt.plot(t_plot, w_opt_plot, label="w_opt")
        has_curve = True
    if np.any(~np.isnan(w_ir_plot)):
        plt.plot(t_plot, w_ir_plot, label="w_ir")
        has_curve = True
    if np.any(~np.isnan(w_rad_plot)):
        plt.plot(t_plot, w_rad_plot, label="w_rad")
        has_curve = True

    plt.xlabel("Time [s]")
    plt.ylabel("CI Weight")
    plt.title(f"CI Fusion Weights (t >= {t_start_plot:.0f} s)")
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    if has_curve:
        plt.legend()
    plt.tight_layout()

    plt.show()


# =========================
# 主函数
# =========================

def export_results_to_csv(save_path: str, t: np.ndarray, res: Dict[str, Any]):
    """导出联邦+CI 滤波时序结果，格式与集中式/单模态尽量一致。"""
    pos_err = np.asarray(res["pos_err"], dtype=float)
    vel_err = np.asarray(res["vel_err"], dtype=float)
    pos_err_norm = np.linalg.norm(pos_err, axis=1)
    vel_err_norm = np.linalg.norm(vel_err, axis=1)

    nis_opt = np.asarray(res["nis"]["opt"], dtype=float).reshape(-1)
    nis_ir = np.asarray(res["nis"]["ir"], dtype=float).reshape(-1)
    nis_rad = np.asarray(res["nis"]["rad"], dtype=float).reshape(-1)

    gate_opt = np.asarray(res["gate_hist"]["opt"], dtype=np.uint8).reshape(-1)
    gate_ir = np.asarray(res["gate_hist"]["ir"], dtype=np.uint8).reshape(-1)
    gate_rad = np.asarray(res["gate_hist"]["rad"], dtype=np.uint8).reshape(-1)

    valid_opt = np.asarray(res["valid_opt"], dtype=np.uint8).reshape(-1)
    valid_ir = np.asarray(res["valid_ir"], dtype=np.uint8).reshape(-1)
    valid_rad = np.asarray(res["valid_rad"], dtype=np.uint8).reshape(-1)

    w_opt = np.full(len(t), np.nan)
    w_ir = np.full(len(t), np.nan)
    w_rad = np.full(len(t), np.nan)
    for k, wk in enumerate(res["ci_weight_hist"], start=1):
        if wk is None:
            continue
        if "opt" in wk:
            w_opt[k] = wk["opt"]
        if "ir" in wk:
            w_ir[k] = wk["ir"]
        if "rad" in wk:
            w_rad[k] = wk["rad"]

    data_mat = np.column_stack([
        np.asarray(t, dtype=float).reshape(-1),
        pos_err[:, 0], pos_err[:, 1], pos_err[:, 2], pos_err_norm,
        vel_err[:, 0], vel_err[:, 1], vel_err[:, 2], vel_err_norm,
        nis_opt, nis_ir, nis_rad,
        gate_opt, gate_ir, gate_rad,
        valid_opt, valid_ir, valid_rad,
        w_opt, w_ir, w_rad,
    ])

    header = (
        "time_sec,"
        "pos_err_x,pos_err_y,pos_err_z,pos_err_norm,"
        "vel_err_x,vel_err_y,vel_err_z,vel_err_norm,"
        "nis_opt,nis_ir,nis_rad,"
        "gate_opt,gate_ir,gate_rad,"
        "valid_opt,valid_ir,valid_rad,"
        "ci_w_opt,ci_w_ir,ci_w_rad"
    )
    np.savetxt(save_path, data_mat, delimiter=",", header=header, comments="", fmt="%.10e")
    print(f"[INFO] 已导出联邦+CI 时序 CSV 文件: {save_path}")


def export_rmse_report_csv(save_path: str, traj_name: str, res: Dict[str, Any]):
    pos_rmse_dim, pos_rmse_norm = compute_rmse(np.asarray(res["pos_err"], dtype=float))
    vel_rmse_dim, vel_rmse_norm = compute_rmse(np.asarray(res["vel_err"], dtype=float))
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "traj_name,method,"
            "pos_rmse_x_m,pos_rmse_y_m,pos_rmse_z_m,pos_rmse_norm_m,"
            "vel_rmse_x_mps,vel_rmse_y_mps,vel_rmse_z_mps,vel_rmse_norm_mps\n"
        )
        f.write(
            f"{traj_name},federated_ci,"
            f"{pos_rmse_dim[0]:.10e},{pos_rmse_dim[1]:.10e},{pos_rmse_dim[2]:.10e},{pos_rmse_norm:.10e},"
            f"{vel_rmse_dim[0]:.10e},{vel_rmse_dim[1]:.10e},{vel_rmse_dim[2]:.10e},{vel_rmse_norm:.10e}\n"
        )
    print(f"[INFO] 已导出联邦+CI RMSE 报告文件: {save_path}")

def main():
    cfg = CONFIG
    for key in ["metadata_path", "roe_path"]:
        if not cfg.get(key):
            raise ValueError(f"CONFIG 中缺少必要路径: {key}")

    data = build_orbit_eci_dataset(cfg["metadata_path"], cfg["roe_path"], cfg["traj_name"])
    if cfg.get("save_preprocessed_npz"):
        np.savez(cfg["save_preprocessed_npz"], **data)
        print(f"[INFO] 已保存预处理结果到: {cfg['save_preprocessed_npz']}")

    print("[INFO] 预处理摘要:")
    for k, v in data["preprocess_summary"].items():
        print(f"  - {k}: {v}")

    t = data["time_sec"]
    x_true_eci = data["rv_rel_eci_true"]
    chief_hist_eci = data["rv_eci2scom_eci_true"]
    q_eci2pri_hist = data["q_eci2pri"]
    r_true_spri = data["r_scom2tcom_spri"]
    v_true_spri = data["v_scom2tcom_spri"]

    dt_nominal = float(np.median(np.diff(t)))
    print(f"[INFO] 样本帧数: {len(t)}")
    print(f"[INFO] 名义时间步长 dt: {dt_nominal:.3f} s")

    rng = np.random.default_rng(cfg["random_seed_meas"])
    sigma_u = float(cfg["sigma_u"])
    sigma_v = float(cfg["sigma_v"])
    sigma_az = np.deg2rad(float(cfg["sigma_az_deg"]))
    sigma_el = np.deg2rad(float(cfg["sigma_el_deg"]))
    sigma_rho = float(cfg["sigma_rho"])
    sigma_rhodot = float(cfg["sigma_rhodot"])

    z_opt, valid_opt = gen_optical_measurements(r_true_spri, sigma_u, sigma_v, 0.0, rng)
    z_ir, valid_ir = gen_ir_measurements(r_true_spri, sigma_az, sigma_el, 0.0, rng)
    z_rad, valid_rad = gen_radar_measurements(r_true_spri, v_true_spri, sigma_rho, sigma_rhodot, 0.0, rng)

    # 应用区间 dropout（按时间段置为模态缺失）
    valid_opt = apply_dropout_windows(valid_opt, t, cfg.get("dropout_windows_opt", []))
    valid_ir = apply_dropout_windows(valid_ir, t, cfg.get("dropout_windows_ir", []))
    valid_rad = apply_dropout_windows(valid_rad, t, cfg.get("dropout_windows_rad", []))

    print(f"[INFO] dropout_windows_opt={cfg.get('dropout_windows_opt', [])}")
    print(f"[INFO] dropout_windows_ir ={cfg.get('dropout_windows_ir', [])}")
    print(f"[INFO] dropout_windows_rad={cfg.get('dropout_windows_rad', [])}")

    Q = make_process_noise(dt_nominal, q_acc=float(cfg["q_acc"]))

    rng_init = np.random.default_rng(cfg["random_seed_init"])
    pos_init_std = float(cfg["pos_init_std"])
    vel_init_std = float(cfg["vel_init_std"])
    x0 = x_true_eci[0].copy()
    x0[0:3] += rng_init.normal(0.0, pos_init_std, size=3)
    x0[3:6] += rng_init.normal(0.0, vel_init_std, size=3)
    P0 = np.diag([pos_init_std**2] * 3 + [vel_init_std**2] * 3)

    R_opt = np.diag([sigma_u**2, sigma_v**2])
    R_ir = np.diag([sigma_az**2, sigma_el**2])
    R_rad = np.diag([sigma_rho**2, sigma_rhodot**2])

    gate_enable = bool(cfg.get("nis_gate_enable", False))
    gate_mode = str(cfg.get("nis_gate_mode", "hard"))
    soft_scale = float(cfg.get("nis_soft_scale", 10.0))
    gate_thresholds = cfg.get("nis_gate_thresholds", {})
    print(f"[INFO] NIS 门限: enable={gate_enable}, mode={gate_mode}, soft_scale={soft_scale}")
    print(f"[INFO] NIS thresholds: {gate_thresholds}")
    print(f"[INFO] reset_feedback={cfg['reset_feedback']}, ci_objective={cfg['ci_objective']}, ci_grid_points={cfg['ci_grid_points']}")

    local_filters = {
        "opt": LocalDynamicsEKF(Q, R_opt, "opt", gate_enable, float(gate_thresholds.get("opt", np.inf)), gate_mode, soft_scale),
        "ir": LocalDynamicsEKF(Q, R_ir, "ir", gate_enable, float(gate_thresholds.get("ir", np.inf)), gate_mode, soft_scale),
        "rad": LocalDynamicsEKF(Q, R_rad, "rad", gate_enable, float(gate_thresholds.get("rad", np.inf)), gate_mode, soft_scale),
    }

    res = run_federated_ci_filter(
        t, x_true_eci, chief_hist_eci, q_eci2pri_hist,
        z_opt, valid_opt, z_ir, valid_ir, z_rad, valid_rad,
        local_filters, x0, P0,
        reset_feedback=bool(cfg["reset_feedback"]),
        ci_objective_name=str(cfg["ci_objective"]),
        ci_grid_points=int(cfg["ci_grid_points"]),
    )

    print_metrics("三模态联邦 + CI 轨道动力学滤波结果", res)

    csv_path = cfg["export_csv_path"] if cfg.get("export_csv_path") else f"{cfg['traj_name']}_federated_ci_results.csv"
    rmse_path = cfg["export_rmse_report_path"] if cfg.get("export_rmse_report_path") else f"{cfg['traj_name']}_federated_ci_rmse_report.csv"

    if cfg.get("export_csv_enable", False):
        export_results_to_csv(csv_path, t, res)

    if cfg.get("export_rmse_report_enable", False):
        export_rmse_report_csv(rmse_path, cfg["traj_name"], res)

    if cfg.get("do_plot", True):
        plot_results(t, res, t_start_plot=500.0)


if __name__ == "__main__":
    main()

