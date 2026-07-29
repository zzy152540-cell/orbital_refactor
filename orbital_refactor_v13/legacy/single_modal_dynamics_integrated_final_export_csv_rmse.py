# -*- coding: utf-8 -*-
"""
SHIRT 三模态单模态轨道动力学滤波（收束版）

当前保留内容：
1. 数据读取与必要预处理（spri -> ECI）；
2. 二体 + J2 相对动力学差分传播；
3. 三模态单模态 EKF（opt / ir / rad）；
4. NIS soft/hard gate；
5. 目标星绝对加速度误差统计与可视化。

当前删去内容：
- IR 闭环诊断；
- 结构化 Q 试验代码；
- 相对加速度误差报告；
- 服务星绝对加速度误差报告；
- 其它中间诊断量。
"""

import json
import os
from typing import Any, Dict

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

    # 可视化设置
    "plot_start_time_sec": 0.0,          # 仅显示该时刻之后的曲线
    "plot_linewidth": 2.0,
    "plot_alpha": 0.95,
    "plot_nis_ylim": None,               # 例如 (0, 20)
    "plot_dpi": 120,
    "export_csv_enable": True,
    "export_csv_prefix": "roe2_single_modal_results",
    "export_rmse_report_enable": True,
    "export_rmse_report_path": "roe2_single_modal_rmse_report.csv",

    "nis_gate_enable": True,
    "nis_gate_mode": "soft",
    "nis_soft_scale": 20.0,
    "nis_gate_thresholds": {
        "opt": 16.0,
        "ir": 16.0,
        "rad": 16.0,
    },
}


MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0
J2 = 1.08262668e-3


def load_json(json_path: str) -> Any:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_numpy_safe(x: Any, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


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
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rotate_pri_to_eci(vec_pri: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
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
    return np.hstack([dv, a_t - a_c])


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
    return q_acc * np.block([
        [(dt**4 / 4.0) * I3, (dt**3 / 2.0) * I3],
        [(dt**3 / 2.0) * I3, (dt**2) * I3]
    ])


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


def wrap_angle(a: np.ndarray):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


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


def compute_rmse(err: np.ndarray):
    rmse_dim = np.sqrt(np.mean(err**2, axis=0))
    rmse_norm = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    return rmse_dim, rmse_norm


class DynamicsEKF:
    def __init__(self, Q, R, mode_name="generic", gate_enable=False, gate_threshold=np.inf,
                 gate_mode="hard", soft_scale=10.0):
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

    def _measurement_function(self, q_eci2pri):
        if self.mode_name == "opt":
            return lambda xx: h_optical_spri(xx, q_eci2pri)
        if self.mode_name == "ir":
            return lambda xx: h_ir_spri(xx, q_eci2pri)
        if self.mode_name == "rad":
            return lambda xx: h_radar_spri(xx, q_eci2pri)
        raise ValueError(f"Unsupported mode_name: {self.mode_name}")

    def update(self, x_pred, P_pred, z, q_eci2pri):
        h = self._measurement_function(q_eci2pri)
        z_pred = h(x_pred)
        H = numerical_measurement_jacobian(h, x_pred)
        y = measurement_residual(z, z_pred, self.mode_name)

        R_used = self.R.copy()
        S = H @ P_pred @ H.T + R_used + 1e-9 * np.eye(H.shape[0])
        nis = float(y.T @ np.linalg.pinv(S) @ y)
        gate_flag = False

        if self.gate_enable and np.isfinite(self.gate_threshold) and nis > self.gate_threshold:
            gate_flag = True
            if self.gate_mode == "hard":
                return x_pred.copy(), P_pred.copy(), nis, gate_flag
            if self.gate_mode == "soft":
                R_used = self.soft_scale * self.R
                S = H @ P_pred @ H.T + R_used + 1e-9 * np.eye(H.shape[0])
                nis = float(y.T @ np.linalg.pinv(S) @ y)

        K = (P_pred @ H.T) @ np.linalg.pinv(S)
        x_upd = x_pred + K @ y
        I = np.eye(len(x_pred))
        P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R_used @ K.T
        return x_upd, P_upd, nis, gate_flag


def run_single_modal_filter(t, x_true_eci, chief_hist_eci, q_eci2pri_hist, z_all, valid_all, ekf, x0, P0):
    N = len(t)
    x_hat = np.zeros((N, 6))
    P_hist = np.zeros((N, 6, 6))
    nis_hist = np.full(N, np.nan)
    gate_hist = np.zeros(N, dtype=bool)

    x = x0.copy()
    P = P0.copy()
    x_hat[0] = x
    P_hist[0] = P

    accepted = rejected = skipped = 0
    for k in range(1, N):
        dt = float(t[k] - t[k - 1])
        x_pred, P_pred = ekf.predict(x, P, chief_hist_eci[k - 1], dt)
        if valid_all[k]:
            x, P, nis, gated = ekf.update(x_pred, P_pred, z_all[k], q_eci2pri_hist[k])
            nis_hist[k] = nis
            gate_hist[k] = gated
            if gated and ekf.gate_enable and ekf.gate_mode == "hard":
                rejected += 1
            else:
                accepted += 1
        else:
            x, P = x_pred, P_pred
            skipped += 1
        x_hat[k] = x
        P_hist[k] = P

    pos_err = x_hat[:, 0:3] - x_true_eci[:, 0:3]
    vel_err = x_hat[:, 3:6] - x_true_eci[:, 3:6]

    target_true_eci = np.zeros_like(x_true_eci)
    target_true_eci[:, 0:3] = chief_hist_eci[:, 0:3] + x_true_eci[:, 0:3]
    target_true_eci[:, 3:6] = chief_hist_eci[:, 3:6] + x_true_eci[:, 3:6]

    target_a_hat = build_target_absolute_accel_history(x_hat, chief_hist_eci)
    target_a_num = numerical_diff_accel_from_velocity(t, target_true_eci[:, 3:6])
    target_acc_err = target_a_hat - target_a_num

    return {
        "x_hat": x_hat,
        "P_hist": P_hist,
        "nis": nis_hist,
        "gate_hist": gate_hist,
        "pos_err": pos_err,
        "vel_err": vel_err,
        "target_acc_err": target_acc_err,
        "stats": {"accepted": accepted, "rejected": rejected, "skipped": skipped},
    }


def print_metrics(name, res):
    pos_rmse_dim, pos_rmse_norm = compute_rmse(res["pos_err"])
    vel_rmse_dim, vel_rmse_norm = compute_rmse(res["vel_err"])
    acc_rmse_dim, acc_rmse_norm = compute_rmse(res["target_acc_err"])
    nis_valid = res["nis"][~np.isnan(res["nis"])]
    nis_mean = np.mean(nis_valid) if len(nis_valid) > 0 else np.nan
    nis_median = np.median(nis_valid) if len(nis_valid) > 0 else np.nan
    stats = res.get("stats", {})

    print("=" * 60)
    print(f"{name} 单模态动力学滤波结果")
    print("=" * 60)
    print(f"位置 RMSE 分量 [m]   : x={pos_rmse_dim[0]:.4f}, y={pos_rmse_dim[1]:.4f}, z={pos_rmse_dim[2]:.4f}")
    print(f"位置 RMSE 范数 [m]   : {pos_rmse_norm:.4f}")
    print(f"速度 RMSE 分量 [m/s] : vx={vel_rmse_dim[0]:.6f}, vy={vel_rmse_dim[1]:.6f}, vz={vel_rmse_dim[2]:.6f}")
    print(f"速度 RMSE 范数 [m/s] : {vel_rmse_norm:.6f}")
    print(f"目标星绝对加速度误差 RMSE 分量 [m/s²]: ax={acc_rmse_dim[0]:.6e}, ay={acc_rmse_dim[1]:.6e}, az={acc_rmse_dim[2]:.6e}")
    print(f"目标星绝对加速度误差 RMSE 范数 [m/s²]: {acc_rmse_norm:.6e}")
    print(f"NIS 均值             : {nis_mean:.4f}")
    print(f"更新统计             : accepted={stats.get('accepted', 0)}, rejected={stats.get('rejected', 0)}, skipped={stats.get('skipped', 0)}")
    print(f"NIS 中位数           : {nis_median:.4f}")
    print()



def setup_plot_style(plot_dpi: int = 120):
    """统一设置绘图风格，尽量接近简洁的论文/Matlab 风格。"""
    plt.rcParams.update({
        "figure.dpi": plot_dpi,
        "savefig.dpi": plot_dpi,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
    })


def _slice_from_time(t: np.ndarray, start_time_sec: float):
    """返回从指定时刻开始的切片掩码。"""
    start_time_sec = max(float(start_time_sec), float(t[0]))
    mask = t >= start_time_sec
    if not np.any(mask):
        mask[-1] = True
    return mask


def _beautify_axis(ax, xlabel: str, ylabel: str, title: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, which="major")
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    for spine in ax.spines.values():
        spine.set_alpha(0.85)

def plot_results(t, results_dict, start_time_sec=0.0, linewidth=2.0,
                 alpha=0.95, nis_ylim=None, plot_dpi=120):
    """
    可视化位置误差、速度误差和 NIS 曲线。
    - 不再绘制加速度曲线；
    - 支持仅显示 start_time_sec 之后的结果；
    - 统一绘图风格，便于论文插图使用。
    """
    setup_plot_style(plot_dpi=plot_dpi)
    mask = _slice_from_time(t, start_time_sec)
    t_plot = t[mask]

    fig1, ax1 = plt.subplots(figsize=(10.5, 6.0))
    for name, res in results_dict.items():
        y = np.linalg.norm(res["pos_err"], axis=1)[mask]
        ax1.plot(t_plot, y, label=name.upper(), linewidth=linewidth, alpha=alpha)
    _beautify_axis(ax1, "Time [s]", "Position Error Norm [m]",
                   f"Position Error Norm (t ≥ {start_time_sec:.1f} s)")
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10.5, 6.0))
    for name, res in results_dict.items():
        y = np.linalg.norm(res["vel_err"], axis=1)[mask]
        ax2.plot(t_plot, y, label=name.upper(), linewidth=linewidth, alpha=alpha)
    _beautify_axis(ax2, "Time [s]", "Velocity Error Norm [m/s]",
                   f"Velocity Error Norm (t ≥ {start_time_sec:.1f} s)")
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(10.5, 6.0))
    for name, res in results_dict.items():
        nis = res["nis"][mask]
        ax3.plot(t_plot, nis, label=name.upper(), linewidth=linewidth, alpha=alpha)

        gate_hist = res.get("gate_hist", None)
        if gate_hist is not None and np.any(gate_hist[mask]):
            idx_local = np.where(gate_hist[mask])[0]
            ax3.scatter(t_plot[idx_local], nis[idx_local], s=20, marker="x", label=f"{name.upper()} gated")
    if nis_ylim is not None:
        ax3.set_ylim(nis_ylim)
    _beautify_axis(ax3, "Time [s]", "NIS",
                   f"NIS Curves (t ≥ {start_time_sec:.1f} s)")
    fig3.tight_layout()

    plt.show()



def export_results_to_csv(prefix: str, t: np.ndarray, results_dict: Dict[str, Any]):
    """将常用绘图结果导出为多个 CSV 文件，便于 MATLAB 直接读取。"""
    t = np.asarray(t, dtype=float).reshape(-1)
    for name, res in results_dict.items():
        pos_err = np.asarray(res["pos_err"], dtype=float)
        vel_err = np.asarray(res["vel_err"], dtype=float)
        nis = np.asarray(res["nis"], dtype=float).reshape(-1)
        gate_hist = np.asarray(res["gate_hist"], dtype=np.uint8).reshape(-1)

        pos_err_norm = np.linalg.norm(pos_err, axis=1)
        vel_err_norm = np.linalg.norm(vel_err, axis=1)

        data_mat = np.column_stack([
            t,
            pos_err[:, 0], pos_err[:, 1], pos_err[:, 2],
            pos_err_norm,
            vel_err[:, 0], vel_err[:, 1], vel_err[:, 2],
            vel_err_norm,
            nis,
            gate_hist
        ])

        header = (
            "time_sec,"
            "pos_err_x,pos_err_y,pos_err_z,pos_err_norm,"
            "vel_err_x,vel_err_y,vel_err_z,vel_err_norm,"
            "nis,gate_hist"
        )
        save_path = f"{prefix}_{name}.csv"
        np.savetxt(save_path, data_mat, delimiter=",", header=header, comments="", fmt="%.10e")
        print(f"[INFO] 已导出 CSV 文件: {save_path}")


def export_rmse_report_csv(save_path: str, results_dict: Dict[str, Any]):
    """导出各模态位置/速度 RMSE 分量与范数统计，便于 MATLAB 画柱状图与生成报告。"""
    header = (
        "modality,"
        "pos_rmse_x_m,pos_rmse_y_m,pos_rmse_z_m,pos_rmse_norm_m,"
        "vel_rmse_x_mps,vel_rmse_y_mps,vel_rmse_z_mps,vel_rmse_norm_mps"
    )
    rows = []
    for name, res in results_dict.items():
        pos_rmse_dim, pos_rmse_norm = compute_rmse(np.asarray(res["pos_err"], dtype=float))
        vel_rmse_dim, vel_rmse_norm = compute_rmse(np.asarray(res["vel_err"], dtype=float))
        row = [
            name,
            pos_rmse_dim[0], pos_rmse_dim[1], pos_rmse_dim[2], pos_rmse_norm,
            vel_rmse_dim[0], vel_rmse_dim[1], vel_rmse_dim[2], vel_rmse_norm,
        ]
        rows.append(row)

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(
                f"{row[0]},"
                f"{row[1]:.10e},{row[2]:.10e},{row[3]:.10e},{row[4]:.10e},"
                f"{row[5]:.10e},{row[6]:.10e},{row[7]:.10e},{row[8]:.10e}\n"
            )
    print(f"[INFO] 已导出 RMSE 报告文件: {save_path}")

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
    sigma_u = 2e-3
    sigma_v = 2e-3
    sigma_az = np.deg2rad(1.5)
    sigma_el = np.deg2rad(1.5)
    sigma_rho = 1.0
    sigma_rhodot = 0.02

    z_opt, valid_opt = gen_optical_measurements(r_true_spri, sigma_u, sigma_v, 0.0, rng)
    z_ir, valid_ir = gen_ir_measurements(r_true_spri, sigma_az, sigma_el, 0.0, rng)
    z_rad, valid_rad = gen_radar_measurements(r_true_spri, v_true_spri, sigma_rho, sigma_rhodot, 0.0, rng)

    Q = make_process_noise(dt_nominal, q_acc=1e-4)

    rng_init = np.random.default_rng(cfg["random_seed_init"])
    pos_init_std = 20.0
    vel_init_std = 0.05
    x0 = x_true_eci[0].copy()
    x0[0:3] += rng_init.normal(0.0, pos_init_std, size=3)
    x0[3:6] += rng_init.normal(0.0, vel_init_std, size=3)
    P0 = np.diag([pos_init_std**2]*3 + [vel_init_std**2]*3)

    R_opt = np.diag([sigma_u**2, sigma_v**2])
    R_ir = np.diag([sigma_az**2, sigma_el**2])
    R_rad = np.diag([sigma_rho**2, sigma_rhodot**2])

    gate_enable = bool(cfg.get("nis_gate_enable", False))
    gate_mode = str(cfg.get("nis_gate_mode", "hard"))
    soft_scale = float(cfg.get("nis_soft_scale", 10.0))
    gate_thresholds = cfg.get("nis_gate_thresholds", {})

    print(f"[INFO] NIS 门限: enable={gate_enable}, mode={gate_mode}, soft_scale={soft_scale}")
    print(f"[INFO] NIS thresholds: {gate_thresholds}")

    results = {
        "opt": run_single_modal_filter(
            t, x_true_eci, chief_hist_eci, q_eci2pri_hist, z_opt, valid_opt,
            DynamicsEKF(Q, R_opt, mode_name="opt", gate_enable=gate_enable,
                        gate_threshold=float(gate_thresholds.get("opt", np.inf)),
                        gate_mode=gate_mode, soft_scale=soft_scale),
            x0, P0),
        "ir": run_single_modal_filter(
            t, x_true_eci, chief_hist_eci, q_eci2pri_hist, z_ir, valid_ir,
            DynamicsEKF(Q, R_ir, mode_name="ir", gate_enable=gate_enable,
                        gate_threshold=float(gate_thresholds.get("ir", np.inf)),
                        gate_mode=gate_mode, soft_scale=soft_scale),
            x0, P0),
        "rad": run_single_modal_filter(
            t, x_true_eci, chief_hist_eci, q_eci2pri_hist, z_rad, valid_rad,
            DynamicsEKF(Q, R_rad, mode_name="rad", gate_enable=gate_enable,
                        gate_threshold=float(gate_thresholds.get("rad", np.inf)),
                        gate_mode=gate_mode, soft_scale=soft_scale),
            x0, P0),
    }

    for name, res in results.items():
        print_metrics(name, res)

    if cfg.get("export_csv_enable", False):
        export_results_to_csv(
            prefix=str(cfg.get("export_csv_prefix", "single_modal_results")),
            t=t,
            results_dict=results,
        )

    if cfg.get("export_rmse_report_enable", False):
        export_rmse_report_csv(
            save_path=str(cfg.get("export_rmse_report_path", "single_modal_rmse_report.csv")),
            results_dict=results,
        )

    if cfg.get("do_plot", True):
        plot_results(
            t,
            results,
            start_time_sec=float(cfg.get("plot_start_time_sec", 0.0)),
            linewidth=float(cfg.get("plot_linewidth", 2.0)),
            alpha=float(cfg.get("plot_alpha", 0.95)),
            nis_ylim=cfg.get("plot_nis_ylim", None),
            plot_dpi=int(cfg.get("plot_dpi", 120)),
        )


if __name__ == "__main__":
    main()
