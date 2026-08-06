from __future__ import annotations

import numpy as np


def quat_to_dcm_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("四元数范数必须大于零。")
    q = q / norm
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rotate_pri_to_eci(vec_pri: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    return quat_to_dcm_wxyz(q_eci2pri) @ np.asarray(vec_pri, dtype=float)


def rotate_eci_to_pri(vec_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    return quat_to_dcm_wxyz(q_eci2pri).T @ np.asarray(vec_eci, dtype=float)


def state_eci_to_spri(x_rel_eci: np.ndarray, q_eci2pri: np.ndarray) -> np.ndarray:
    x = np.asarray(x_rel_eci, dtype=float).reshape(6)
    return np.hstack([
        rotate_eci_to_pri(x[:3], q_eci2pri),
        rotate_eci_to_pri(x[3:], q_eci2pri),
    ])


def dcm_to_quat_wxyz(dcm: np.ndarray) -> np.ndarray:
    """Convert a proper direction-cosine matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(dcm, dtype=float).reshape(3, 3)
    u, _, vh = np.linalg.svd(matrix)
    matrix = u @ vh
    if np.linalg.det(matrix) < 0.0:
        u[:, -1] *= -1.0
        matrix = u @ vh
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
                      (matrix[0, 2] - matrix[2, 0]) / s,
                      (matrix[1, 0] - matrix[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            q = np.array([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                          (matrix[0, 1] + matrix[1, 0]) / s,
                          (matrix[0, 2] + matrix[2, 0]) / s])
        elif index == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            q = np.array([(matrix[0, 2] - matrix[2, 0]) / s,
                          (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                          (matrix[1, 2] + matrix[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            q = np.array([(matrix[1, 0] - matrix[0, 1]) / s,
                          (matrix[0, 2] + matrix[2, 0]) / s,
                          (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def build_rtn_quaternion(state_eci: np.ndarray) -> np.ndarray:
    """Build the local RTN/PRI frame quaternion used by this project.

    The returned quaternion's DCM maps PRI components to ECI components, matching
    ``rotate_pri_to_eci`` and the existing historical interface.
    """
    state = np.asarray(state_eci, dtype=float).reshape(6)
    radial = state[:3] / np.linalg.norm(state[:3])
    normal_raw = np.cross(state[:3], state[3:])
    normal = normal_raw / np.linalg.norm(normal_raw)
    transverse = np.cross(normal, radial)
    pri_to_eci = np.column_stack((radial, transverse, normal))
    return dcm_to_quat_wxyz(pri_to_eci)


def build_rtn_quaternion_history(state_history_eci: np.ndarray) -> np.ndarray:
    history = np.asarray(state_history_eci, dtype=float)
    if history.ndim != 2 or history.shape[1] != 6:
        raise ValueError("state_history_eci must have shape (N, 6).")
    return np.vstack([build_rtn_quaternion(state) for state in history])


def state_history_eci_to_spri(state_history_eci: np.ndarray, q_history: np.ndarray) -> np.ndarray:
    states = np.asarray(state_history_eci, dtype=float)
    quaternions = np.asarray(q_history, dtype=float)
    if states.ndim != 2 or states.shape[1] != 6 or quaternions.shape != (len(states), 4):
        raise ValueError("State and quaternion histories have inconsistent shapes.")
    return np.vstack([state_eci_to_spri(states[k], quaternions[k]) for k in range(len(states))])
