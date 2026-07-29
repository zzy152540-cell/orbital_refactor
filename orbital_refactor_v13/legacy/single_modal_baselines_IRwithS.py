# -*- coding: utf-8 -*-
"""
基于 SHIRT 真值的三模态单模态滤波 baseline

功能：
1. 从导出的 SHIRT npz 文件中读取相对状态真值
2. 生成三种模态量测：
   - 光学：归一化像面 (u, v)
   - 红外：方位 / 俯仰 (az, el)
   - 雷达：距离 / 距离率 (rho, rhodot)
3. 分别构建三个单模态 EKF：
   - Optical EKF
   - Infrared EKF
   - Radar EKF
4. 输出：
   - 位置误差 / 速度误差
   - xyz 分量误差
   - RMSE
   - NIS
5. 绘图展示结果

说明：
- 当前仅估计相对平动状态 [x, y, z, vx, vy, vz]
- 姿态真值当前不参与滤波，只保留为后续扩展接口
- 动力学模型采用常速度 CV 模型，作为 baseline
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 数据读取
# =========================
def load_truth_from_npz(npz_path: str):
    """
    从之前导出的 npz 中读取时间、相对位置、相对速度真值
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到 npz 文件: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    time_sec = data["time_sec"]                    # (N,)
    r_true = data["r_scom2tcom_spri"]             # (N,3)
    v_true = data["v_scom2tcom_spri"]             # (N,3)

    # 可选读取姿态，后续扩展用
    q_true = data["q_spri2tpri"] if "q_spri2tpri" in data.files else None

    return time_sec, r_true, v_true, q_true


# =========================
# 2. 量测生成
# =========================
def gen_optical_measurements(
    r_true: np.ndarray,
    sigma_u: float = 1e-3,
    sigma_v: float = 1e-3,
    dropout_prob: float = 0.0,
    rng: np.random.Generator = None,
):
    """
    光学量测：归一化像面 (u, v)
    u = x / z
    v = y / z

    返回：
    - z_opt: (N,2)
    - valid: (N,) bool，是否有量测
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x = r_true[:, 0]
    y = r_true[:, 1]
    z = r_true[:, 2]

    eps = 1e-12
    u_true = x / np.maximum(z, eps)
    v_true = y / np.maximum(z, eps)

    u_meas = u_true + rng.normal(0.0, sigma_u, size=len(z))
    v_meas = v_true + rng.normal(0.0, sigma_v, size=len(z))

    z_opt = np.column_stack([u_meas, v_meas])

    valid = np.ones(len(z), dtype=bool)
    if dropout_prob > 0.0:
        valid = rng.random(len(z)) > dropout_prob

    return z_opt, valid


def gen_ir_measurements(
    r_true: np.ndarray,
    sigma_az: float,
    sigma_el: float,
    sigma_s_norm: float,
    K_s: float,
    s0: float,
    dropout_prob: float = 0.0,
    rng: np.random.Generator = None,
):
    """
    红外量测：方位 / 俯仰 / 归一化尺度 (az, el, s_norm)

    az = atan2(y, x)
    el = atan2(z, sqrt(x^2 + y^2))
    s = K_s / r
    s_norm = s / s0
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x = r_true[:, 0]
    y = r_true[:, 1]
    z = r_true[:, 2]

    rho = np.sqrt(x**2 + y**2 + z**2)
    rho = np.maximum(rho, 1e-12)

    az_true = np.arctan2(y, x)
    el_true = np.arctan2(z, np.sqrt(x**2 + y**2))
    s_true = K_s / rho
    s_norm_true = s_true / s0

    az_meas = az_true + rng.normal(0.0, sigma_az, size=len(x))
    el_meas = el_true + rng.normal(0.0, sigma_el, size=len(x))
    s_norm_meas = s_norm_true + rng.normal(0.0, sigma_s_norm, size=len(x))

    z_ir = np.column_stack([az_meas, el_meas, s_norm_meas])

    valid = np.ones(len(x), dtype=bool)
    if dropout_prob > 0.0:
        valid = rng.random(len(x)) > dropout_prob

    return z_ir, valid


def gen_radar_measurements(
    r_true: np.ndarray,
    v_true: np.ndarray,
    sigma_rho: float,
    sigma_rhodot: float,
    dropout_prob: float = 0.0,
    rng: np.random.Generator = None,
):
    """
    雷达量测：距离 / 距离率 (rho, rhodot)
    rho = ||r||
    rhodot = (r^T v) / ||r||
    """
    if rng is None:
        rng = np.random.default_rng(0)

    rho_true = np.linalg.norm(r_true, axis=1)
    rhodot_true = np.sum(r_true * v_true, axis=1) / np.maximum(rho_true, 1e-12)

    rho_meas = rho_true + rng.normal(0.0, sigma_rho, size=len(rho_true))
    rhodot_meas = rhodot_true + rng.normal(0.0, sigma_rhodot, size=len(rho_true))

    z_rad = np.column_stack([rho_meas, rhodot_meas])

    valid = np.ones(len(rho_true), dtype=bool)
    if dropout_prob > 0.0:
        valid = rng.random(len(rho_true)) > dropout_prob

    return z_rad, valid


# =========================
# 3. 动力学模型
# =========================
def make_cv_model(dt: float, q_acc: float):
    """
    常速度模型
    状态: [x, y, z, vx, vy, vz]^T
    """
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    F = np.block([
        [I3, dt * I3],
        [Z3, I3]
    ])

    Q = q_acc * np.block([
        [(dt**4 / 4.0) * I3, (dt**3 / 2.0) * I3],
        [(dt**3 / 2.0) * I3, (dt**2) * I3]
    ])
    return F, Q


# =========================
# 4. 三种观测函数与雅可比
# =========================
def h_optical(x: np.ndarray):
    px, py, pz = x[0], x[1], x[2]
    eps = 1e-12
    return np.array([px / max(pz, eps), py / max(pz, eps)])


def H_optical(x: np.ndarray):
    px, py, pz = x[0], x[1], x[2]
    eps = 1e-12
    pz = max(pz, eps)

    H = np.zeros((2, 6))
    H[0, 0] = 1.0 / pz
    H[0, 2] = -px / (pz**2)
    H[1, 1] = 1.0 / pz
    H[1, 2] = -py / (pz**2)
    return H


def h_ir(x: np.ndarray):
    px, py, pz = x[0], x[1], x[2]
    az = np.arctan2(py, px)
    el = np.arctan2(pz, np.sqrt(px**2 + py**2))
    return np.array([az, el])


def H_ir(x: np.ndarray):
    px, py, pz = x[0], x[1], x[2]

    rho_xy2 = px**2 + py**2
    rho_xy2 = max(rho_xy2, 1e-12)
    rho_xy = np.sqrt(rho_xy2)

    rho2 = px**2 + py**2 + pz**2
    rho2 = max(rho2, 1e-12)

    H = np.zeros((2, 6))

    # az = atan2(y, x)
    H[0, 0] = -py / rho_xy2
    H[0, 1] =  px / rho_xy2

    # el = atan2(z, sqrt(x^2+y^2))
    H[1, 0] = -px * pz / (rho_xy * rho2)
    H[1, 1] = -py * pz / (rho_xy * rho2)
    H[1, 2] =  rho_xy / rho2

    return H


def h_ir_scale_norm(x: np.ndarray, K_s: float, s0: float):
    """
    红外观测函数：
    z = [az, el, s_norm]^T
    s_norm = (K_s / r) / s0
    """
    px, py, pz = x[0], x[1], x[2]

    az = np.arctan2(py, px)
    el = np.arctan2(pz, np.sqrt(px**2 + py**2))

    rho = np.sqrt(px**2 + py**2 + pz**2)
    rho = max(rho, 1e-12)

    s = K_s / rho
    s_norm = s / s0

    return np.array([az, el, s_norm])


def H_ir_scale_norm(x: np.ndarray, K_s: float, s0: float):
    """
    红外观测雅可比：
    z = [az, el, s_norm]^T
    s_norm = (K_s / r) / s0
    """
    px, py, pz = x[0], x[1], x[2]

    rho_xy2 = px**2 + py**2
    rho_xy2 = max(rho_xy2, 1e-12)
    rho_xy = np.sqrt(rho_xy2)

    rho2 = px**2 + py**2 + pz**2
    rho2 = max(rho2, 1e-12)
    rho = np.sqrt(rho2)

    H = np.zeros((3, 6))

    # az = atan2(y, x)
    H[0, 0] = -py / rho_xy2
    H[0, 1] =  px / rho_xy2

    # el = atan2(z, sqrt(x^2+y^2))
    H[1, 0] = -px * pz / (rho_xy * rho2)
    H[1, 1] = -py * pz / (rho_xy * rho2)
    H[1, 2] =  rho_xy / rho2

    # s_norm = (K_s / r) / s0
    H[2, 0] = -(K_s / s0) * px / (rho**3)
    H[2, 1] = -(K_s / s0) * py / (rho**3)
    H[2, 2] = -(K_s / s0) * pz / (rho**3)

    return H


def h_radar(x: np.ndarray):
    px, py, pz, vx, vy, vz = x
    rho = np.sqrt(px**2 + py**2 + pz**2)
    rho = max(rho, 1e-12)
    rhodot = (px * vx + py * vy + pz * vz) / rho
    return np.array([rho, rhodot])


def H_radar(x: np.ndarray):
    px, py, pz, vx, vy, vz = x

    rho2 = px**2 + py**2 + pz**2
    rho2 = max(rho2, 1e-12)
    rho = np.sqrt(rho2)

    n = px * vx + py * vy + pz * vz

    H = np.zeros((2, 6))

    # rho
    H[0, 0] = px / rho
    H[0, 1] = py / rho
    H[0, 2] = pz / rho

    # rhodot
    H[1, 0] = (vx * rho2 - n * px) / (rho**3)
    H[1, 1] = (vy * rho2 - n * py) / (rho**3)
    H[1, 2] = (vz * rho2 - n * pz) / (rho**3)
    H[1, 3] = px / rho
    H[1, 4] = py / rho
    H[1, 5] = pz / rho

    return H


# =========================
# 5. 工具函数
# =========================
def wrap_angle(a: np.ndarray):
    """
    将角度包装到 [-pi, pi]
    """
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def measurement_residual(z: np.ndarray, z_pred: np.ndarray, mode: str):
    """
    根据不同模态计算创新
    """
    y = z - z_pred
    if mode == "ir":
        y[0] = wrap_angle(np.array([y[0]]))[0]
        y[1] = wrap_angle(np.array([y[1]]))[0]
    return y


def compute_rmse(err: np.ndarray):
    """
    err: (N, d)
    返回每维 RMSE 和总范数 RMSE
    """
    rmse_dim = np.sqrt(np.mean(err**2, axis=0))
    rmse_norm = np.sqrt(np.mean(np.sum(err**2, axis=1)))
    return rmse_dim, rmse_norm


# =========================
# 6. EKF
# =========================
class EKF:
    def __init__(self, F, Q, R, h_func, H_func, mode_name="generic", h_args=None):
        self.F = F
        self.Q = Q
        self.R = R
        self.h_func = h_func
        self.H_func = H_func
        self.mode_name = mode_name
        self.h_args = h_args if h_args is not None else {}

    def predict(self, x, P):
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z):
        z_pred = self.h_func(x_pred, **self.h_args)
        H = self.H_func(x_pred, **self.h_args)

        y = measurement_residual(z, z_pred, self.mode_name)

        S = H @ P_pred @ H.T + self.R
        S = S + 1e-9 * np.eye(S.shape[0])

        K = (P_pred @ H.T) @ np.linalg.pinv(S)

        x_upd = x_pred + K @ y

        I = np.eye(len(x_pred))
        P_upd = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ self.R @ K.T

        nis = float(y.T @ np.linalg.pinv(S) @ y)

        return x_upd, P_upd, y, S, K, nis


# =========================
# 7. 单模态滤波主循环
# =========================
def run_single_modal_filter(
    t: np.ndarray,
    r_true: np.ndarray,
    v_true: np.ndarray,
    z_all: np.ndarray,
    valid_all: np.ndarray,
    ekf: EKF,
    x0: np.ndarray,
    P0: np.ndarray,
    nis_gate: float = None,
):
    """
    对单一模态执行 EKF
    可选：
    - nis_gate: NIS 门限，若超过则跳过更新
    """
    N = len(t)

    x_hat = np.zeros((N, 6))
    P_hist = np.zeros((N, 6, 6))
    nis_hist = np.full(N, np.nan)

    x = x0.copy()
    P = P0.copy()

    x_hat[0] = x
    P_hist[0] = P

    accepted_count = 0
    rejected_count = 0
    skipped_count = 0

    for k in range(1, N):
        x_pred, P_pred = ekf.predict(x, P)

        if valid_all[k]:
            do_update = True

            # 光学模态：前向深度保护
            if ekf.mode_name == "opt":
                if x_pred[2] <= 0.5:
                    do_update = False

            if do_update:
                try:
                    x_upd, P_upd, _, _, _, nis = ekf.update(x_pred, P_pred, z_all[k])

                    # 无论是否通过门限，都先记录原始 NIS
                    nis_hist[k] = nis

                    if (nis_gate is not None) and (nis > nis_gate):
                        x, P = x_pred, P_pred
                        rejected_count += 1
                    else:
                        x, P = x_upd, P_upd
                        accepted_count += 1

                except np.linalg.LinAlgError:
                    x, P = x_pred, P_pred
                    skipped_count += 1
            else:
                x, P = x_pred, P_pred
                skipped_count += 1
        else:
            x, P = x_pred, P_pred
            skipped_count += 1

        x_hat[k] = x
        P_hist[k] = P

    pos_err = x_hat[:, 0:3] - r_true
    vel_err = x_hat[:, 3:6] - v_true

    print(f"[{ekf.mode_name}] accepted = {accepted_count}, rejected = {rejected_count}, skipped = {skipped_count}")

    return {
        "x_hat": x_hat,
        "P_hist": P_hist,
        "nis": nis_hist,
        "pos_err": pos_err,
        "vel_err": vel_err,
    }


# =========================
# 8. 绘图
# =========================
def plot_error_curves(t, results_dict):
    """
    画位置误差范数和速度误差范数
    """
    plt.figure(figsize=(10, 6))
    for name, res in results_dict.items():
        pos_norm = np.linalg.norm(res["pos_err"], axis=1)
        plt.plot(t, pos_norm, label=f"{name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Position Error Norm [m]")
    plt.title("Position Error Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    for name, res in results_dict.items():
        vel_norm = np.linalg.norm(res["vel_err"], axis=1)
        plt.plot(t, vel_norm, label=f"{name}")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity Error Norm [m/s]")
    plt.title("Velocity Error Norm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_xyz_errors(t, results_dict):
    """
    每个模态单独画 xyz 位置误差
    """
    for name, res in results_dict.items():
        err = res["pos_err"]
        plt.figure(figsize=(10, 6))
        plt.plot(t, err[:, 0], label="x error")
        plt.plot(t, err[:, 1], label="y error")
        plt.plot(t, err[:, 2], label="z error")
        plt.xlabel("Time [s]")
        plt.ylabel("Position Error [m]")
        plt.title(f"{name} Position Component Errors")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()


def plot_nis(t, results_dict):
    """
    绘制 NIS 曲线
    """
    plt.figure(figsize=(10, 6))
    for name, res in results_dict.items():
        plt.plot(t, res["nis"], label=name)
    plt.xlabel("Time [s]")
    plt.ylabel("NIS")
    plt.title("NIS Curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_rmse_bar(results_dict):
    """
    绘制位置/速度 RMSE 柱状图
    """
    names = []
    pos_rmse = []
    vel_rmse = []

    for name, res in results_dict.items():
        _, pos_norm_rmse = compute_rmse(res["pos_err"])
        _, vel_norm_rmse = compute_rmse(res["vel_err"])
        names.append(name)
        pos_rmse.append(pos_norm_rmse)
        vel_rmse.append(vel_norm_rmse)

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, pos_rmse, width=width, label="Position RMSE [m]")
    plt.bar(x + width / 2, vel_rmse, width=width, label="Velocity RMSE [m/s]")
    plt.xticks(x, names)
    plt.title("RMSE Comparison")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()


# =========================
# 9. 打印指标
# =========================
def print_metrics(name, res):
    pos_rmse_dim, pos_rmse_norm = compute_rmse(res["pos_err"])
    vel_rmse_dim, vel_rmse_norm = compute_rmse(res["vel_err"])

    nis_valid = res["nis"][~np.isnan(res["nis"])]
    nis_mean = np.mean(nis_valid) if len(nis_valid) > 0 else np.nan
    nis_median = np.median(nis_valid) if len(nis_valid) > 0 else np.nan

    print("=" * 60)
    print(f"{name} 单模态滤波结果")
    print("=" * 60)
    print(f"位置 RMSE 分量 [m]   : x={pos_rmse_dim[0]:.4f}, y={pos_rmse_dim[1]:.4f}, z={pos_rmse_dim[2]:.4f}")
    print(f"位置 RMSE 范数 [m]   : {pos_rmse_norm:.4f}")
    print(f"速度 RMSE 分量 [m/s] : vx={vel_rmse_dim[0]:.6f}, vy={vel_rmse_dim[1]:.6f}, vz={vel_rmse_dim[2]:.6f}")
    print(f"速度 RMSE 范数 [m/s] : {vel_rmse_norm:.6f}")
    print(f"NIS 均值             : {nis_mean:.4f}")
    print(f"NIS 中位数           : {nis_median:.4f}")
    print()


# =========================
# 10. 主函数
# =========================
def main():
    # =====================
    # 路径设置
    # =====================
    npz_path = r"E:\Python Files\Thesis_Code\shirt_separate_output\roe1\roe1_common_fields.npz"
    # 如果你要跑 roe2，就改成对应文件：
    # npz_path = r"E:\Python Files\Thesis_Code\shirt_separate_output\roe2\roe2_common_fields.npz"

    # =====================
    # 读取真值
    # =====================
    t, r_true, v_true, q_true = load_truth_from_npz(npz_path)
    N = len(t)
    dt = float(np.median(np.diff(t)))

    print(f"[INFO] 样本帧数: {N}")
    print(f"[INFO] 时间步长 dt: {dt:.3f} s")

    x_true = np.hstack([r_true, v_true])  # (N,6)

    # =====================
    # 生成三种量测
    # =====================
    rng = np.random.default_rng(42)

    # 光学噪声
    sigma_u = 2e-3
    sigma_v = 2e-3
    
    # 红外噪声（角度 + 归一化尺度）
    sigma_az = np.deg2rad(0.8)
    sigma_el = np.deg2rad(0.8)

    rho_true = np.linalg.norm(r_true, axis=1)
    rho_median = np.median(rho_true)

    # 让原始尺度中值约为 100
    K_s = 100.0 * rho_median
    s_true_check = K_s / rho_true
    s0 = np.median(s_true_check)

    s_norm_true_check = s_true_check / s0

    print("s_true min    =", s_true_check.min())
    print("s_true max    =", s_true_check.max())
    print("s_true median =", np.median(s_true_check))

    print("s_norm min    =", s_norm_true_check.min())
    print("s_norm max    =", s_norm_true_check.max())
    print("s_norm median =", np.median(s_norm_true_check))

    # 归一化尺度噪声：先取 5% ~ 10% 中值
    sigma_s_norm = 0.08
    print("sigma_s_norm =", sigma_s_norm)

    # 雷达噪声
    sigma_rho = 1.0       # m
    sigma_rhodot = 0.02   # m/s

    z_opt, valid_opt = gen_optical_measurements(
        r_true, sigma_u=sigma_u, sigma_v=sigma_v, dropout_prob=0.0, rng=rng
    )
    
    z_ir, valid_ir = gen_ir_measurements(
    r_true, sigma_az=sigma_az, sigma_el=sigma_el, sigma_s_norm=sigma_s_norm, K_s=K_s, s0=s0, dropout_prob=0.0, rng=rng
    )

    z_rad, valid_rad = gen_radar_measurements(
        r_true, v_true, sigma_rho=sigma_rho, sigma_rhodot=sigma_rhodot, dropout_prob=0.0, rng=rng
    )

    # =====================
    # 构建统一动力学
    # =====================
    q_acc = 1e-4
    F, Q = make_cv_model(dt, q_acc=q_acc)

    # =====================
    # 初值设置：真值 + 扰动
    # =====================
    rng_init = np.random.default_rng(2025)

    pos_init_std = 20.0     # m
    vel_init_std = 0.05     # m/s

    x0 = x_true[0].copy()
    x0[0:3] += rng_init.normal(0.0, pos_init_std, size=3)
    x0[3:6] += rng_init.normal(0.0, vel_init_std, size=3)

    P0 = np.diag([
        pos_init_std**2, pos_init_std**2, pos_init_std**2,
        vel_init_std**2, vel_init_std**2, vel_init_std**2
    ])

    # =====================
    # 构建三个 EKF
    # =====================
    R_opt = np.diag([sigma_u**2, sigma_v**2])
    R_ir = np.diag([sigma_az**2, sigma_el**2, sigma_s_norm**2])
    R_rad = np.diag([sigma_rho**2, sigma_rhodot**2])

    ekf_opt = EKF(F, Q, R_opt, h_optical, H_optical, mode_name="opt")
    
    ekf_ir = EKF(F, Q, R_ir, h_ir_scale_norm, H_ir_scale_norm, mode_name="ir", h_args={"K_s": K_s, "s0": s0})

    ekf_rad = EKF(F, Q, R_rad, h_radar, H_radar, mode_name="rad")

    # =====================
    # 分别运行单模态滤波
    # =====================
    nis_gate_opt = 9.21   # 自由度2
    nis_gate_ir = 1e7    # 自由度2
    nis_gate_rad = 9.21   # 雷达也是2维量测，这里可先同样设置

    res_opt = run_single_modal_filter(
    t, r_true, v_true, z_opt, valid_opt, ekf_opt, x0, P0, nis_gate=nis_gate_opt
    )
    res_ir = run_single_modal_filter(
    t, r_true, v_true, z_ir, valid_ir, ekf_ir, x0, P0, nis_gate=nis_gate_ir
    )
    res_rad = run_single_modal_filter(
    t, r_true, v_true, z_rad, valid_rad, ekf_rad, x0, P0, nis_gate=nis_gate_rad
    )

    results_dict = {
        "Optical-EKF": res_opt,
        "Infrared-EKF": res_ir,
        "Radar-EKF": res_rad,
    }

    # =====================
    # 打印指标
    # =====================
    for name, res in results_dict.items():
        print_metrics(name, res)

    # =====================
    # 绘图
    # =====================
    plot_error_curves(t, results_dict)
    plot_xyz_errors(t, results_dict)
    plot_nis(t, results_dict)
    plot_rmse_bar(results_dict)

    plt.show()


if __name__ == "__main__":
    main()