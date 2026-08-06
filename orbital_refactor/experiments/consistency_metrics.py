from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.network_filter_metrics import NEES_95_DOF6, NIS_95_DOF1
from experiments.summary_statistics import interval_coverage
from orbital_core.metrics import compute_nees_history, compute_rmse


@dataclass(frozen=True)
class FleetConsistencyMetrics:
    position_rmse: float
    velocity_rmse: float
    mean_nees: float
    nees_95_coverage: float
    mean_nis: float
    nis_95_coverage: float


def fleet_consistency_metrics(*, truth, estimates, covariances, nis_history):
    """Compute common fleet accuracy and consistency metrics."""

    position_errors = []
    velocity_errors = []
    nees_values = []
    nis_values = []
    for node_id in truth:
        error = estimates[node_id] - truth[node_id]
        position_errors.append(error[:, :3])
        velocity_errors.append(error[:, 3:])
        nees_values.extend(compute_nees_history(
            estimates[node_id], truth[node_id], covariances[node_id]
        ))
        nis_values.extend(
            value for epoch in nis_history[node_id] for value in epoch.values()
        )
    nees = np.asarray(nees_values, dtype=float)
    nis = np.asarray(nis_values, dtype=float)
    return FleetConsistencyMetrics(
        position_rmse=compute_rmse(np.vstack(position_errors)),
        velocity_rmse=compute_rmse(np.vstack(velocity_errors)),
        mean_nees=float(np.mean(nees)),
        nees_95_coverage=interval_coverage(nees, NEES_95_DOF6),
        mean_nis=float(np.mean(nis)) if nis.size else float("nan"),
        nis_95_coverage=interval_coverage(nis, NIS_95_DOF1),
    )
