from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.summary_statistics import interval_coverage
from orbital_core.metrics import compute_nees_history, compute_rmse

NEES_95_DOF6 = (1.2373442458, 14.4493753354)


@dataclass(frozen=True)
class SchmidtArchitectureSummary:
    architecture: str
    run_count: int
    mean_position_rmse: float
    mean_velocity_rmse: float
    mean_nees: float
    mean_nees_95_coverage: float
    mean_position_covariance_trace: float
    mean_position_rmse_by_node: dict[str, float]
    mean_runtime_seconds: float


def history_metrics(history, truth, runtime_seconds):
    """Compute architecture metrics from a network Schmidt history."""

    return array_metrics(
        history.active_state_history_by_node,
        history.active_covariance_history_by_node,
        truth,
        runtime_seconds=runtime_seconds,
    )


def array_metrics(
    states, covariances, truth, *, runtime_seconds, sample_mask=None,
):
    """Compute fleet metrics from node-indexed state/covariance arrays."""

    position_errors = []
    velocity_errors = []
    nees = []
    position_traces = []
    position_by_node = {}
    if sample_mask is None:
        sample_mask = slice(None)
    for node in truth:
        error = (states[node] - truth[node])[sample_mask]
        position_errors.append(error[:, :3])
        velocity_errors.append(error[:, 3:])
        position_by_node[node] = compute_rmse(error[:, :3])
        nees.extend(compute_nees_history(
            states[node][sample_mask],
            truth[node][sample_mask],
            covariances[node][sample_mask],
        ))
        position_traces.extend(np.trace(
            covariances[node][sample_mask, :3, :3], axis1=1, axis2=2
        ))
    nees_array = np.asarray(nees, dtype=float)
    return {
        "position_rmse": compute_rmse(np.vstack(position_errors)),
        "velocity_rmse": compute_rmse(np.vstack(velocity_errors)),
        "nees": float(np.mean(nees_array)),
        "nees_coverage": interval_coverage(nees_array, NEES_95_DOF6),
        "position_trace": float(np.mean(position_traces)),
        "position_by_node": position_by_node,
        "runtime_seconds": float(runtime_seconds),
    }


def experiment_phase_masks(
    timestamps, *, absolute_navigation_dropout_windows,
    communication_outage_windows_by_directed_link,
    absolute_navigation_dropout_windows_by_node,
    topology_inactive_windows_by_undirected_edge,
):
    """Return pre-fault, fault, and recovery masks for configured windows."""

    dropout_windows = list(absolute_navigation_dropout_windows)
    dropout_windows.extend(
        window
        for windows in (absolute_navigation_dropout_windows_by_node or {}).values()
        for window in windows
    )
    dropout_windows.extend(
        window
        for windows in (topology_inactive_windows_by_undirected_edge or {}).values()
        for window in windows
    )
    dropout_windows.extend(
        window
        for windows in (communication_outage_windows_by_directed_link or {}).values()
        for window in windows
    )
    if not dropout_windows:
        return {}
    timestamps = np.asarray(timestamps, dtype=float)
    first_start = min(float(start) for start, _ in dropout_windows)
    last_end = max(float(end) for _, end in dropout_windows)
    masks = {
        "pre_dropout": timestamps < first_start,
        "dropout": np.asarray([
            any(start <= timestamp <= end for start, end in dropout_windows)
            for timestamp in timestamps
        ], dtype=bool),
        "post_recovery": timestamps > last_end,
    }
    return {name: mask for name, mask in masks.items() if np.any(mask)}


def aggregate_architecture_metrics(architecture, values):
    """Aggregate repeated-run metrics for one estimator architecture."""

    nodes = tuple(values[0]["position_by_node"])
    return SchmidtArchitectureSummary(
        architecture=architecture,
        run_count=len(values),
        mean_position_rmse=float(np.mean([v["position_rmse"] for v in values])),
        mean_velocity_rmse=float(np.mean([v["velocity_rmse"] for v in values])),
        mean_nees=float(np.mean([v["nees"] for v in values])),
        mean_nees_95_coverage=float(np.mean([
            v["nees_coverage"] for v in values
        ])),
        mean_position_covariance_trace=float(np.mean([
            v["position_trace"] for v in values
        ])),
        mean_position_rmse_by_node={
            node: float(np.mean([
                v["position_by_node"][node] for v in values
            ]))
            for node in nodes
        },
        mean_runtime_seconds=float(np.mean([
            v["runtime_seconds"] for v in values
        ])),
    )
