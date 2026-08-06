from __future__ import annotations

import numpy as np


def compute_rmse(error: np.ndarray) -> float:
    error = np.asarray(error, dtype=float)
    if error.size == 0:
        raise ValueError("误差数组不能为空。")
    if error.ndim == 1:
        return float(np.sqrt(np.mean(error**2)))
    return float(np.sqrt(np.mean(np.sum(error**2, axis=1))))


def compute_nees(error: np.ndarray, covariance: np.ndarray) -> float:
    """Normalized estimation error squared for one state/covariance pair."""

    error = np.asarray(error, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (error.size, error.size):
        raise ValueError("covariance shape must match the error dimension.")
    return float(error.T @ np.linalg.pinv(covariance) @ error)


def compute_nees_history(
    estimate_history: np.ndarray,
    truth_history: np.ndarray,
    covariance_history: np.ndarray,
) -> np.ndarray:
    estimates = np.asarray(estimate_history, dtype=float)
    truth = np.asarray(truth_history, dtype=float)
    covariances = np.asarray(covariance_history, dtype=float)
    if estimates.shape != truth.shape or estimates.ndim != 2:
        raise ValueError("estimate and truth histories must have matching shape (N, D).")
    if covariances.shape != (estimates.shape[0], estimates.shape[1], estimates.shape[1]):
        raise ValueError("covariance history must have shape (N, D, D).")
    return np.array(
        [
            compute_nees(estimates[index] - truth[index], covariances[index])
            for index in range(estimates.shape[0])
        ],
        dtype=float,
    )
