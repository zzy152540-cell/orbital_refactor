from __future__ import annotations

import numpy as np

from interfaces.data_objects import NodeReport
from orbital_core.dynamics import accel_two_body_j2


def _rk4_absolute(state: np.ndarray, dt: float) -> np.ndarray:
    """Propagate absolute ECI state with two-body+J2 dynamics."""
    def rhs(x):
        r = x[:3]
        v = x[3:]
        a = accel_two_body_j2(r)
        return np.hstack([v, a])

    x = np.asarray(state, dtype=float).reshape(6)
    if dt <= 0:
        return x.copy()

    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


def align_report_time(report: NodeReport, current_time: float) -> NodeReport:
    """
    Propagate delayed state from source timestamp to current fusion epoch.

    Uses the same absolute orbital dynamics model as the simulator
    (two-body + J2) instead of a constant velocity approximation.
    """
    source_time = (
        report.source_timestamp
        if report.source_timestamp is not None
        else report.timestamp
    )

    dt = float(current_time - source_time)

    if dt <= 0:
        return report

    state = _rk4_absolute(report.state_estimate, dt)

    # First-order covariance propagation. The state transition approximation
    # is sufficient for validating asynchronous fusion; EKF-level STM can
    # replace this in future versions.
    F = np.eye(6)
    F[:3, 3:] = dt * np.eye(3)
    covariance = F @ report.covariance @ F.T

    return NodeReport(
        node_id=report.node_id,
        target_id=report.target_id,
        timestamp=float(current_time),
        state_estimate=state,
        covariance=covariance,
        quality_score=report.quality_score,
        health_status=report.health_status,
        communication_delay=report.communication_delay,
        valid_flag=report.valid_flag,
        source_timestamp=source_time,
        arrival_timestamp=report.arrival_timestamp,
    )
