from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .data_contracts import MultiTargetOutput, TargetNodeReport
from .target_fusion import fuse_target_reports, group_reports_by_target


def run_cooperative_target_fusion(
    *,
    scene_id: str,
    timestamp: float,
    reports: Iterable[TargetNodeReport],
    expected_target_ids: Iterable[str] | None = None,
    objective: str = "trace",
    grid_points: int = 31,
) -> MultiTargetOutput:
    """Fuse one aligned epoch of observer reports independently per target.

    This is the first target-centric orchestration boundary. It deliberately
    consumes already-computed absolute J2000 posteriors; sensor filtering and
    track prediction remain separate stages.
    """

    epoch = float(timestamp)
    if not np.isfinite(epoch):
        raise ValueError("timestamp must be finite.")
    values = list(reports)
    mismatched = [report for report in values if not np.isclose(report.timestamp, epoch)]
    if mismatched:
        raise ValueError("Every report must be aligned to the requested fusion timestamp.")

    grouped = group_reports_by_target(values)
    expected = (
        None
        if expected_target_ids is None
        else tuple(dict.fromkeys(str(value).strip() for value in expected_target_ids))
    )
    if expected is not None:
        if any(not target_id for target_id in expected):
            raise ValueError("expected_target_ids must contain non-empty identifiers.")
        unexpected = set(grouped) - set(expected)
        if unexpected:
            raise ValueError(
                f"Reports contain unexpected target IDs: {sorted(unexpected)}"
            )

    estimates = {
        target_id: fuse_target_reports(
            target_reports,
            objective=objective,
            grid_points=grid_points,
        )
        for target_id, target_reports in grouped.items()
    }
    return MultiTargetOutput(
        scene_id=scene_id,
        timestamp=epoch,
        estimates_by_target=estimates,
    )
