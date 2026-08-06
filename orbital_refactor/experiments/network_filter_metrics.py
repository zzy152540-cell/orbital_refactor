from __future__ import annotations

import numpy as np

from experiments.summary_statistics import interval_coverage
from orbital_core.metrics import compute_nees_history, compute_rmse

NEES_95_DOF6 = (1.2373442458, 14.4493753354)
NIS_95_DOF1 = (0.0009820691, 5.0238861873)
NIS_95_DOF2 = (0.0506356159, 7.3777589082)


def modality_from_information_id(information_id: str) -> str:
    """Recover the relative-measurement modality from an information ID."""

    if ":radar:" in information_id:
        return "RADAR"
    if ":range_rate:" in information_id:
        return "RANGE_RATE"
    if ":az_el:" in information_id:
        return "AZ_EL"
    if ":infrared:" in information_id:
        return "INFRARED"
    if ":optical:" in information_id:
        return "OPTICAL"
    return "RANGE"


def nis_interval(modality: str) -> tuple[float, float]:
    """Return the 95-percent NIS interval for a scalar or 2-D modality."""

    return (
        NIS_95_DOF2
        if modality in {"RADAR", "AZ_EL", "INFRARED", "OPTICAL"}
        else NIS_95_DOF1
    )


def modality_aware_nis_coverage(values_by_modality) -> float:
    """Pool NIS coverage while respecting each modality's dimension."""

    covered = 0
    count = 0
    for modality, values in values_by_modality.items():
        array = np.asarray(values)
        lower, upper = nis_interval(modality)
        covered += int(np.count_nonzero((array >= lower) & (array <= upper)))
        count += int(array.size)
    return covered / count if count else float("nan")


def network_history_metrics(history, truth, transmitted_count, run_seconds):
    """Summarize accuracy, consistency, refresh, and replay diagnostics."""

    position = []
    velocity = []
    nees = []
    nis = []
    minimum = float("inf")
    failures = 0
    nis_by_modality: dict[str, list[float]] = {}
    for node in history.node_ids:
        error = history.active_state_history_by_node[node] - truth[node]
        position.append(error[:, :3])
        velocity.append(error[:, 3:])
        nees.extend(compute_nees_history(
            history.active_state_history_by_node[node],
            truth[node],
            history.active_covariance_history_by_node[node],
        ))
        for epoch in history.nis_history_by_node[node]:
            for information_id, value in epoch.items():
                if ":absolute:" in information_id:
                    continue
                nis.append(value)
                nis_by_modality.setdefault(
                    modality_from_information_id(information_id), []
                ).append(value)
        for covariance in history.joint_covariance_history_by_node[node]:
            value = float(np.linalg.eigvalsh(covariance).min())
            minimum = min(minimum, value)
            failures += int(value < -1e-8)
    nees = np.asarray(nees)
    nis = np.asarray(nis)
    accepted = int(history.refresh_diagnostics.get("accepted", 0))
    rejected = sum(
        value for key, value in history.refresh_diagnostics.items()
        if key != "accepted"
    )
    rejection_counts = {
        key: int(value)
        for key, value in history.refresh_diagnostics.items()
        if key != "accepted" and value
    }
    replay_stats = list(history.replay_performance_by_node.values())
    performance = {
        "total_replay_seconds": sum(
            value.total_replay_seconds for value in replay_stats
        ),
        "replay_count": sum(value.replay_count for value in replay_stats),
        "batch_count": sum(value.batch_count for value in replay_stats),
        "maximum_replay_seconds": max(
            (value.maximum_replay_seconds for value in replay_stats), default=0.0
        ),
        "maximum_replay_span": max(
            (value.maximum_replay_span for value in replay_stats), default=0.0
        ),
        "total_replay_span": sum(
            value.total_replay_span for value in replay_stats
        ),
        "maximum_batch_size": max(
            (value.maximum_batch_size for value in replay_stats), default=0
        ),
        "replayed_remote_events": sum(
            value.replayed_remote_events for value in replay_stats
        ),
        "replayed_observations": sum(
            value.replayed_observations for value in replay_stats
        ),
        "fallback_count": sum(value.fallback_count for value in replay_stats),
        "maximum_remote_event_count": max(
            (value.maximum_remote_event_count for value in replay_stats), default=0
        ),
        "maximum_observation_count": max(
            (value.maximum_observation_count for value in replay_stats), default=0
        ),
        "maximum_checkpoint_count": max(
            (value.maximum_checkpoint_count for value in replay_stats), default=0
        ),
        "maximum_posterior_state_count": max(
            (value.maximum_posterior_state_count for value in replay_stats), default=0
        ),
        "maximum_pinned_checkpoint_count": max(
            (value.maximum_pinned_checkpoint_count for value in replay_stats), default=0
        ),
        "maximum_resync_required_count": max(
            (value.maximum_resync_required_count for value in replay_stats), default=0
        ),
        "maximum_retained_journal_count": max(
            (value.maximum_retained_journal_count for value in replay_stats), default=0
        ),
    }
    return (
        compute_rmse(np.vstack(position)),
        compute_rmse(np.vstack(velocity)),
        float(np.mean(nees)),
        interval_coverage(nees, NEES_95_DOF6),
        float(np.mean(nis)),
        modality_aware_nis_coverage(nis_by_modality),
        minimum,
        accepted,
        transmitted_count,
        rejected,
        failures,
        rejection_counts,
        float(run_seconds),
        performance,
        {key: float(np.mean(value)) for key, value in nis_by_modality.items()},
        {
            key: interval_coverage(value, nis_interval(key))
            for key, value in nis_by_modality.items()
        },
    )
