from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.network_filter_metrics import modality_from_information_id
from orbital_core.metrics import compute_nees_history, compute_rmse


@dataclass(frozen=True)
class DynamicVisibilityRunMetrics:
    pre_transition_position_rmse: float
    post_transition_position_rmse: float
    final_position_error: float
    mean_nees: float
    mean_nis: float
    mean_nis_by_modality: dict[str, float]
    transition_nis: float | None
    transition_nis_by_modality: dict[str, float]
    accepted_messages: int
    rejected_messages: int
    psd_failure_count: int
    pre_transition_velocity_rmse: float
    post_transition_velocity_rmse: float
    final_velocity_error: float


def dynamic_visibility_run_metrics(history, truth, transition_timestamp):
    """Summarize one dynamic-visibility filter run around its transition."""

    position_errors = {"pre": [], "post": [], "final": []}
    velocity_errors = {"pre": [], "post": [], "final": []}
    nees = []
    nis = []
    nis_by_modality: dict[str, list[float]] = {}
    psd_failures = 0
    pre_mask = history.timestamps < transition_timestamp
    post_mask = ~pre_mask
    transition_index = int(np.flatnonzero(
        np.isclose(history.timestamps, transition_timestamp)
    )[0])
    for node_id in history.node_ids:
        error = history.active_state_history_by_node[node_id] - truth[node_id]
        position_errors["pre"].append(error[pre_mask, :3])
        position_errors["post"].append(error[post_mask, :3])
        position_errors["final"].append(float(np.linalg.norm(error[-1, :3])))
        velocity_errors["pre"].append(error[pre_mask, 3:])
        velocity_errors["post"].append(error[post_mask, 3:])
        velocity_errors["final"].append(float(np.linalg.norm(error[-1, 3:])))
        nees.extend(compute_nees_history(
            history.active_state_history_by_node[node_id], truth[node_id],
            history.active_covariance_history_by_node[node_id],
        ))
        for epoch in history.nis_history_by_node[node_id]:
            for information_id, value in epoch.items():
                nis.append(value)
                modality = modality_from_information_id(information_id)
                nis_by_modality.setdefault(modality, []).append(value)
        for covariance in history.joint_covariance_history_by_node[node_id]:
            psd_failures += int(np.linalg.eigvalsh(covariance).min() < -1e-8)
    accepted = int(history.refresh_diagnostics.get("accepted", 0))
    rejected = sum(
        value for key, value in history.refresh_diagnostics.items()
        if key != "accepted"
    )
    transition_values = []
    transition_by_modality: dict[str, list[float]] = {}
    for epochs in history.nis_history_by_node.values():
        for information_id, value in epochs[transition_index].items():
            transition_values.append(value)
            modality = modality_from_information_id(information_id)
            transition_by_modality.setdefault(modality, []).append(value)
    return DynamicVisibilityRunMetrics(
        pre_transition_position_rmse=compute_rmse(np.vstack(position_errors["pre"])),
        post_transition_position_rmse=compute_rmse(np.vstack(position_errors["post"])),
        final_position_error=float(np.mean(position_errors["final"])),
        mean_nees=float(np.mean(nees)),
        mean_nis=float(np.mean(nis)),
        mean_nis_by_modality={
            key: float(np.mean(values)) for key, values in nis_by_modality.items()
        },
        transition_nis=(
            float(np.mean(transition_values)) if transition_values else None
        ),
        transition_nis_by_modality={
            key: float(np.mean(values))
            for key, values in transition_by_modality.items()
        },
        accepted_messages=accepted,
        rejected_messages=rejected,
        psd_failure_count=psd_failures,
        pre_transition_velocity_rmse=compute_rmse(
            np.vstack(velocity_errors["pre"])
        ),
        post_transition_velocity_rmse=compute_rmse(
            np.vstack(velocity_errors["post"])
        ),
        final_velocity_error=float(np.mean(velocity_errors["final"])),
    )
