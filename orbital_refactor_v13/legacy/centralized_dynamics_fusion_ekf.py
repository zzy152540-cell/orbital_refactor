# -*- coding: utf-8 -*-
"""
SHIRT 三模态集中式轨道动力学滤波（Centralized EKF）

说明：
1. 保留与 single_modal_dynamics_integrated_final.py 一致的数据读取、spri->ECI 预处理、二体+J2 相对动力学传播；
2. 使用单一全局状态 x=[dr_eci,dv_eci]；
3. 每个时刻对可用模态（opt/ir/rad）先做各自 NIS 预检，再将通过门限的量测集中堆叠为一个联合量测做一次更新；
4. 支持 hard/soft gate；soft gate 时仅放大对应模态的 R；
5. 仅报告目标星绝对加速度误差。
"""

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

CONFIG = {
    "metadata_path": r"E:\Satellite Datasets\shirt\roe1\metadata.json",
    "roe_path": r"E:\Satellite Datasets\shirt\roe1\roe1.json",
    "traj_name": "roe1",
    "save_preprocessed_npz": None,
    "random_seed_meas": 42,
    "random_seed_init": 2025,
    "do_plot": True,
    "nis_gate_enable": True,
    "nis_gate_mode": "soft",
    "nis_soft_scale": 20.0,
    "nis_gate_thresholds": {"opt": 16.0, "ir": 16.0, "rad": 16.0},
    "dropout_windows_opt": [],  # [(start_time, end_time), ...]
    "dropout_windows_ir": [],
    "dropout_windows_rad": [],
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


def wrap_angle(a: np.ndarray):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compute_rmse(err: np.ndarray):
    rmse_dim = np.sqrt(np.mean(err**2, axis=0))
    rmse_norm = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    return rmse_dim, rmse_norm


def block_diag(mats: List[np.ndarray]) -> np.ndarray:
    if len(mats) == 0:
        return np.zeros((0, 0))
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols))
    r = c = 0
    for m in mats:
        rr, cc = m.shape
        out[r:r + rr, c:c + cc] = m
        r += rr
        c += cc
    return out


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
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
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
    a_j2 = fac * np.array([x * (s - 1.0), y * (s - 1.0), z * (s - 3.0)])
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
        [(dt**3 / 2.0) * I3, (dt**2) * I3],
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
    return np.array([np.arctan2(py, px), np.arctan2(pz, np.sqrt(px**2 + py**2))])


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


class CentralizedDynamicsEKF:
    def __init__(self, Q: np.ndarray, R_dict: Dict[str, np.ndarray], gate_enable=False,
                 gate_thresholds=None, gate_mode="soft", soft_scale=20.0):
        self.Q = Q
        self.R_dict = R_dict
        self.gate_enable = gate_enable
        self.gate_thresholds = gate_thresholds or {}
        self.gate_mode = gate_mode
        self.soft_scale = soft_scale

    def predict(self, x, P, chief_state_eci, dt):
        f = lambda xx: rk4_step_rel(xx, chief_state_eci, dt)
        x_pred = f(x)
        F = numerical_jacobian_discrete(f, x)
        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred

    def _get_h(self, mode: str, q_eci2pri: np.ndarray):
        if mode == "opt":
            return lambda xx: h_optical_spri(xx, q_eci2pri)
        if mode == "ir":
            return lambda xx: h_ir_spri(xx, q_eci2pri)
        if mode == "rad":
            return lambda xx: h_radar_spri(xx, q_eci2pri)
        raise ValueError(f"Unsupported mode: {mode}")

    def centralized_update(self, x_pred, P_pred, meas_dict: Dict[str, np.ndarray], q_eci2pri):
        accepted_modes, rejected_modes, skipped_modes = [], [], []
        nis_dict, gated_dict = {}, {}
        z_blocks, zpred_blocks, H_blocks, R_blocks, mode_dims = [], [], [], [], []

        for mode in ["opt", "ir", "rad"]:
            if mode not in meas_dict or meas_dict[mode] is None:
                skipped_modes.append(mode)
                continue

            z = meas_dict[mode]
            h = self._get_h(mode, q_eci2pri)
            z_pred = h(x_pred)
            H = numerical_measurement_jacobian(h, x_pred)
            y = measurement_residual(z, z_pred, mode)
            Rm = self.R_dict[mode].copy()
            S = H @ P_pred @ H.T + Rm + 1e-9 * np.eye(H.shape[0])
            nis = float(y.T @ np.linalg.pinv(S) @ y)
            nis_dict[mode] = nis
            th = float(self.gate_thresholds.get(mode, np.inf))
            gated = False
            if self.gate_enable and np.isfinite(th) and nis > th:
                gated = True
                if self.gate_mode == "hard":
                    rejected_modes.append(mode)
                    gated_dict[mode] = True
                    continue
                if self.gate_mode == "soft":
                    Rm = self.soft_scale * self.R_dict[mode]
            gated_dict[mode] = gated
            accepted_modes.append(mode)
            z_blocks.append(z)
            zpred_blocks.append(z_pred)
            H_blocks.append(H)
            R_blocks.append(Rm)
            mode_dims.append((mode, len(z)))

        if len(z_blocks) == 0:
            return x_pred.copy(), P_pred.copy(), nis_dict, gated_dict, accepted_modes, rejected_modes, skipped_modes

        z_stack = np.concatenate(z_blocks, axis=0)
        zpred_stack = np.concatenate(zpred_blocks, axis=0)
        H_stack = np.vstack(H_blocks)
        R_stack = block_diag(R_blocks)
        y_stack = z_stack - zpred_stack

        offset = 0
        for mode, dim in mode_dims:
            if mode == "ir":
                y_stack[offset] = wrap_angle(np.array([y_stack[offset]]))[0]
                y_stack[offset + 1] = wrap_angle(np.array([y_stack[offset + 1]]))[0]
            offset += dim

        S_stack = H_stack @ P_pred @ H_stack.T + R_stack + 1e-9 * np.eye(H_stack.shape[0])
        K = (P_pred @ H_stack.T) @ np.linalg.pinv(S_stack)
        x_upd = x_pred + K @ y_stack
        I = np.eye(len(x_pred))
        P_upd = (I - K @ H_stack) @ P_pred @ (I - K @ H_stack).T + K @ R_stack @ K.T
        return x_upd, P_upd, nis_dict, gated_dict, accepted_modes, rejected_modes, skipped_modes


def run_centralized_filter(t, x_true_eci, chief_hist_eci, q_eci2pri_hist,
                           z_opt, valid_opt, z_ir, valid_ir, z_rad, valid_rad,
                           ekf: CentralizedDynamicsEKF, x0, P0):
    N = len(t)
    x_hat = np.zeros((N, 6))
    P_hist = np.zeros((N, 6, 6))
    x_hat[0] = x0
    P_hist[0] = P0

    nis_hist = {"opt": np.full(N, np.nan), "ir": np.full(N, np.nan), "rad": np.full(N, np.nan)}
    gate_hist = {"opt": np.zeros(N, dtype=bool), "ir": np.zeros(N, dtype=bool), "rad": np.zeros(N, dtype=bool)}
    stats = {m: {"accepted": 0, "rejected": 0, "skipped": 0} for m in ["opt", "ir", "rad"]}

    x = x0.copy()
    P = P0.copy()
    for k in range(1, N):
        dt = float(t[k] - t[k - 1])
        x_pred, P_pred = ekf.predict(x, P, chief_hist_eci[k - 1], dt)

        meas_dict = {}
        if valid_opt[k]:
            meas_dict["opt"] = z_opt[k]
        else:
            stats["opt"]["skipped"] += 1
        if valid_ir[k]:
            meas_dict["ir"] = z_ir[k]
        else:
            stats["ir"]["skipped"] += 1
        if valid_rad[k]:
            meas_dict["rad"] = z_rad[k]
        else:
            stats["rad"]["skipped"] += 1

        x, P, nis_dict, gated_dict, accepted_modes, rejected_modes, _ = ekf.centralized_update(
            x_pred, P_pred, meas_dict, q_eci2pri_hist[k]
        )

        for mode in ["opt", "ir", "rad"]:
            if mode in nis_dict:
                nis_hist[mode][k] = nis_dict[mode]
            if mode in gated_dict:
                gate_hist[mode][k] = gated_dict[mode]
        for mode in accepted_modes:
            stats[mode]["accepted"] += 1
        for mode in rejected_modes:
            stats[mode]["rejected"] += 1

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
        "stats": stats,
    }


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


def plot_results(t, res):
    plt.figure(figsize=(10, 6))
    plt.plot(t, np.linalg.norm(res["pos_err"], axis=1), label="centralized")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error Norm [m]")
    plt.title("Centralized Position Error Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(t, np.linalg.norm(res["vel_err"], axis=1), label="centralized")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity Error Norm [m/s]")
    plt.title("Centralized Velocity Error Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    plt.plot(t, np.linalg.norm(res["target_acc_err"], axis=1), label="centralized")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration Error Norm [m/s²]")
    plt.title("Target Absolute Acceleration Error Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for mode in ["opt", "ir", "rad"]:
        plt.plot(t, res["nis"][mode], label=mode)
        gate_hist = res["gate_hist"][mode]
        if np.any(gate_hist):
            idx = np.where(gate_hist)[0]
            plt.scatter(t[idx], res["nis"][mode][idx], s=12, marker="x")
    plt.xlabel("Time [s]")
    plt.ylabel("NIS")
    plt.title("Centralized Filter Per-Modal NIS")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    # 应用区间 dropout（按时间段置为模态缺失）
    valid_opt = apply_dropout_windows(valid_opt, t, cfg.get("dropout_windows_opt", []))
    valid_ir = apply_dropout_windows(valid_ir, t, cfg.get("dropout_windows_ir", []))
    valid_rad = apply_dropout_windows(valid_rad, t, cfg.get("dropout_windows_rad", []))

    print(f"[INFO] dropout_windows_opt={cfg.get('dropout_windows_opt', [])}")
    print(f"[INFO] dropout_windows_ir ={cfg.get('dropout_windows_ir', [])}")
    print(f"[INFO] dropout_windows_rad={cfg.get('dropout_windows_rad', [])}")

    Q = make_process_noise(dt_nominal, q_acc=1e-4)
    rng_init = np.random.default_rng(cfg["random_seed_init"])
    pos_init_std = 20.0
    vel_init_std = 0.05
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

    centralized_ekf = CentralizedDynamicsEKF(
        Q=Q,
        R_dict={"opt": R_opt, "ir": R_ir, "rad": R_rad},
        gate_enable=gate_enable,
        gate_thresholds=gate_thresholds,
        gate_mode=gate_mode,
        soft_scale=soft_scale,
    )

    res = run_centralized_filter(
        t, x_true_eci, chief_hist_eci, q_eci2pri_hist,
        z_opt, valid_opt, z_ir, valid_ir, z_rad, valid_rad,
        centralized_ekf, x0, P0,
    )

    print_metrics("三模态集中式轨道动力学滤波结果", res)
    if cfg.get("do_plot", True):
        plot_results(t, res)


if __name__ == "__main__":
    main()
