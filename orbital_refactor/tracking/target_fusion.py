from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from orbital_core.ci_fusion import ci_fuse_posteriors

from .data_contracts import GlobalTargetEstimate, TargetNodeReport


def group_reports_by_target(
    reports: Iterable[TargetNodeReport],
) -> dict[str, list[TargetNodeReport]]:
    """Group valid node reports without ever crossing physical targets."""

    grouped: dict[str, list[TargetNodeReport]] = defaultdict(list)
    for report in reports:
        if report.valid_flag:
            grouped[report.target_id].append(report)
    return {
        target_id: sorted(values, key=lambda item: item.observer_id)
        for target_id, values in sorted(grouped.items())
    }


def fuse_target_reports(
    reports: Iterable[TargetNodeReport],
    *,
    objective: str = "trace",
    grid_points: int = 31,
) -> GlobalTargetEstimate:
    """Fuse reports for exactly one target and one aligned epoch."""

    valid = [report for report in reports if report.valid_flag]
    if not valid:
        raise ValueError("At least one valid target report is required.")
    target_ids = {report.target_id for report in valid}
    track_ids = {report.track_id for report in valid}
    timestamps = {report.timestamp for report in valid}
    observers = [report.observer_id for report in valid]
    if len(target_ids) != 1:
        raise ValueError("CI inputs must estimate the same physical target_id.")
    if len(track_ids) != 1:
        raise ValueError("CI inputs must belong to the same track_id.")
    if len(timestamps) != 1:
        raise ValueError("CI inputs must be aligned to the same timestamp.")
    if len(set(observers)) != len(observers):
        raise ValueError("Only one report per observer is allowed in one target fusion epoch.")
    if len(valid) > 3:
        raise ValueError("The current simultaneous CI core supports at most three observers.")

    ordered = sorted(valid, key=lambda item: item.observer_id)
    fusion = ci_fuse_posteriors(
        [
            (report.observer_id, report.state_eci, report.covariance_eci)
            for report in ordered
        ],
        objective=objective,
        grid_points=grid_points,
    )
    return GlobalTargetEstimate(
        target_id=ordered[0].target_id,
        track_id=ordered[0].track_id,
        timestamp=ordered[0].timestamp,
        state_eci=fusion.state,
        covariance_eci=fusion.covariance,
        contributing_observer_ids=tuple(report.observer_id for report in ordered),
        node_weights=fusion.weights,
        valid_flag=bool(np.all(np.isfinite(fusion.state))),
    )
