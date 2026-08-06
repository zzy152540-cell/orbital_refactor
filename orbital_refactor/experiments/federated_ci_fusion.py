from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_core.ci_fusion import ci_fuse_posteriors
from orbital_core.dynamics import (
    make_process_noise,
    numerical_jacobian_discrete,
    rk4_step_absolute,
)
from orbital_core.measurement_semantics import PHYSICAL_SENSOR_MODALITIES


@dataclass(frozen=True)
class FederatedCIFusionHistory:
    state_by_node: dict[str, np.ndarray]
    covariance_by_node: dict[str, np.ndarray]
    weights_by_node: dict[str, list[dict[str, float]]]
    modality_validity_by_node: dict[str, list[dict[str, bool]]]
    weight_samples_by_modality: dict[str, list[float]]
    weight_samples_by_node_and_modality: dict[
        str, dict[str, list[float]]
    ]
    exclusion_count_by_modality: dict[str, int]
    prediction_only_exclusion_count_by_modality: dict[str, int]
    all_modalities_unavailable_count: int


def fuse_local_schmidt_histories(
    *, timestamps, node_ids, local_histories, hard_thresholds,
    ci_objective, ci_grid_points, process_noise_acceleration,
) -> FederatedCIFusionHistory:
    """Fuse same-satellite local modality posteriors without feedback."""

    states = {
        node: np.zeros((len(timestamps), 6), dtype=float) for node in node_ids
    }
    covariances = {
        node: np.zeros((len(timestamps), 6, 6), dtype=float)
        for node in node_ids
    }
    weights_by_node = {node: [] for node in node_ids}
    validity_by_node = {node: [] for node in node_ids}
    weight_samples = {
        modality: [] for modality in PHYSICAL_SENSOR_MODALITIES
    }
    weight_samples_by_node = {
        node: {
            modality: [] for modality in PHYSICAL_SENSOR_MODALITIES
        }
        for node in node_ids
    }
    exclusion_counts = {
        modality: 0 for modality in PHYSICAL_SENSOR_MODALITIES
    }
    prediction_only_counts = {
        modality: 0 for modality in PHYSICAL_SENSOR_MODALITIES
    }
    all_unavailable_count = 0
    for node in node_ids:
        for index in range(len(timestamps)):
            participating = []
            for modality in PHYSICAL_SENSOR_MODALITIES:
                threshold = hard_thresholds.get(modality)
                epoch_nis = _modality_nis(
                    local_histories[modality], node, index, modality
                )
                if not epoch_nis:
                    prediction_only_counts[modality] += 1
                    continue
                if not any(
                    threshold is None or value <= threshold
                    for value in epoch_nis.values()
                ):
                    exclusion_counts[modality] += 1
                    continue
                participating.append((
                    modality,
                    local_histories[modality]
                    .active_state_history_by_node[node][index],
                    local_histories[modality]
                    .active_covariance_history_by_node[node][index],
                ))
            if not participating:
                all_unavailable_count += 1
                _apply_navigation_or_prediction_fallback(
                    states=states,
                    covariances=covariances,
                    node=node,
                    index=index,
                    timestamps=timestamps,
                    local_histories=local_histories,
                    process_noise_acceleration=process_noise_acceleration,
                )
                weights_by_node[node].append({})
                validity_by_node[node].append({
                    modality: False
                    for modality in PHYSICAL_SENSOR_MODALITIES
                })
                continue
            fusion = ci_fuse_posteriors(
                participating,
                objective=ci_objective,
                grid_points=ci_grid_points,
            )
            states[node][index] = fusion.state
            covariances[node][index] = fusion.covariance
            weights_by_node[node].append(dict(fusion.weights))
            validity_by_node[node].append({
                modality: any(
                    item[0] == modality for item in participating
                )
                for modality in PHYSICAL_SENSOR_MODALITIES
            })
            for modality in PHYSICAL_SENSOR_MODALITIES:
                weight = fusion.weights.get(modality, 0.0)
                weight_samples[modality].append(weight)
                weight_samples_by_node[node][modality].append(weight)
    return FederatedCIFusionHistory(
        state_by_node=states,
        covariance_by_node=covariances,
        weights_by_node=weights_by_node,
        modality_validity_by_node=validity_by_node,
        weight_samples_by_modality=weight_samples,
        weight_samples_by_node_and_modality=weight_samples_by_node,
        exclusion_count_by_modality=exclusion_counts,
        prediction_only_exclusion_count_by_modality=prediction_only_counts,
        all_modalities_unavailable_count=all_unavailable_count,
    )


def _modality_nis(history, node, index, modality):
    return {
        information_id: value
        for information_id, value in history.nis_history_by_node[
            node
        ][index].items()
        if history.modality_history_by_node[node][index].get(
            information_id
        ) == modality
    }


def _apply_navigation_or_prediction_fallback(
    *, states, covariances, node, index, timestamps, local_histories,
    process_noise_acceleration,
):
    navigation_track = local_histories[PHYSICAL_SENSOR_MODALITIES[0]]
    navigation_ids = {
        information_id
        for information_id, tracked_modality in (
            navigation_track.modality_history_by_node[node][index].items()
        )
        if tracked_modality == "ABSOLUTE_POSITION"
    }
    if navigation_ids:
        states[node][index] = navigation_track.active_state_history_by_node[
            node
        ][index]
        covariances[node][index] = (
            navigation_track.active_covariance_history_by_node[node][index]
        )
        return
    if index == 0:
        states[node][index] = navigation_track.active_state_history_by_node[
            node
        ][index]
        covariances[node][index] = (
            navigation_track.active_covariance_history_by_node[node][index]
        )
        return
    delta_time = float(timestamps[index] - timestamps[index - 1])
    previous_state = states[node][index - 1]
    transition = numerical_jacobian_discrete(
        lambda value: rk4_step_absolute(value, delta_time), previous_state
    )
    states[node][index] = rk4_step_absolute(previous_state, delta_time)
    covariances[node][index] = (
        transition @ covariances[node][index - 1] @ transition.T
        + make_process_noise(delta_time, process_noise_acceleration)
    )
