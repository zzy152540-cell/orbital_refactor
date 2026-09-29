from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.measurements import measure_relative_range, measure_relative_range_rate

from .data_contracts import IODObservation


def refine_initial_state(
    observations: tuple[IODObservation, ...],
    *,
    epoch: float,
    initial_state_eci: np.ndarray,
    robust_loss: str = "soft_l1",
    outlier_threshold_sigma: float = 8.0,
):
    valid = tuple(item for item in observations if item.valid_flag)
    times = np.array(sorted({float(item.timestamp) for item in valid}))
    if times.size < 2 or times[0] < epoch:
        raise ValueError("IOD batch requires at least two epochs at or after the reference epoch.")
    time_index = {value: index for index, value in enumerate(times)}

    def residual_for(state, selected):
        propagated = propagate_absolute_orbit(
            state, np.concatenate(([0.0], times - epoch))[1:] if times[0] > epoch else times - epoch,
        )
        parts = []
        for item in selected:
            target = propagated[time_index[float(item.timestamp)]]
            observer = item.observer_state_eci
            if item.modality == "RADAR":
                predicted = np.array([
                    measure_relative_range(observer, target),
                    measure_relative_range_rate(observer, target),
                ])
                parts.append(_whiten(item.measurement - predicted, item.covariance))
            else:
                relative = target[:3] - observer[:3]
                predicted = relative / np.linalg.norm(relative)
                angular_sigma = np.sqrt(float(np.trace(item.covariance) / 2.0))
                parts.append(np.cross(predicted, item.measurement) / angular_sigma)
        return np.concatenate(parts)

    def solve(selected, initial):
        return least_squares(
            lambda state: residual_for(state, selected),
            initial,
            loss=robust_loss,
            x_scale=np.array([1e6, 1e6, 1e6, 1e3, 1e3, 1e3]),
            max_nfev=200,
        )

    result = solve(valid, np.asarray(initial_state_eci, dtype=float).reshape(6))
    rejected = _outlier_indices(result.fun, valid, outlier_threshold_sigma)
    used = tuple(item for index, item in enumerate(valid) if index not in rejected)
    if rejected and len(used) >= 4:
        result = solve(used, result.x)
    jacobian = np.asarray(result.jac, dtype=float)
    information = jacobian.T @ jacobian
    rank = int(np.linalg.matrix_rank(information))
    condition = float(np.linalg.cond(information)) if rank == 6 else float("inf")
    covariance = np.linalg.pinv(information)
    covariance = 0.5 * (covariance + covariance.T)
    return result, covariance, {
        "jacobianRank": rank,
        "informationConditionNumber": condition,
        "residualRms": float(np.sqrt(np.mean(result.fun ** 2))),
        "residualCount": int(result.fun.size),
        "functionEvaluations": int(result.nfev),
        "usedObservationCount": len(used),
        "rejectedObservationCount": len(rejected),
        "rejectedObservationIds": [
            str(valid[index].metadata.get(
                "sourceMessageId", f"observation-{index}",
            ))
            for index in sorted(rejected)
        ],
    }


def _whiten(residual, covariance):
    return np.linalg.solve(np.linalg.cholesky(covariance), residual)


def _outlier_indices(residual, observations, threshold):
    rejected = set()
    offset = 0
    for index, observation in enumerate(observations):
        dimension = 2 if observation.modality == "RADAR" else 3
        value = np.asarray(residual[offset:offset + dimension], dtype=float)
        if np.linalg.norm(value) > float(threshold) * np.sqrt(dimension):
            rejected.add(index)
        offset += dimension
    return rejected
