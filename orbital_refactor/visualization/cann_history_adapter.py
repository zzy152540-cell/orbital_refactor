"""Build read-only navigation-cell display histories from filter posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from brain_inspired.navigation_brain_state import build_navigation_brain_states
from brain_inspired.navigation_place_cells import (
    NavigationPlaceCellConfig,
    build_navigation_place_cell_histories,
)
from brain_inspired.orbital_direction_runner import run_orbital_direction_states
from brain_inspired.orbital_direction_state import OrbitalDirectionConfig
from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
from brain_inspired.orbital_rt_grid_runner import run_orbital_rt_grid_states
from brain_inspired.ring_cann import RingCANNConfig
from interfaces.data_objects import AbsolutePositionObservation
from orbital_core.dynamics import rk4_step_absolute
from visualization.data_contract import (
    VisualCANNSnapshot,
    VisualNavigationState,
)


@dataclass(frozen=True)
class CANNVisualizationHistory:
    navigation_by_epoch: tuple[tuple[VisualNavigationState, ...], ...]
    cann_by_epoch: tuple[tuple[VisualCANNSnapshot, ...], ...]


def build_cann_visualization_history(
    *, history, initial_state_by_node: Mapping[str, np.ndarray],
    absolute_position_observations: Iterable[AbsolutePositionObservation] = (),
) -> CANNVisualizationHistory:
    """Create direction, RT and place-cell display state after filtering."""
    times = np.asarray(history.timestamps, dtype=float)
    nodes = tuple(history.node_ids)
    posterior = history.active_state_history_by_node
    frames = {
        node: OrbitalPlaneFrame.from_state_eci(initial_state_by_node[node])
        for node in nodes
    }
    masks = _absolute_navigation_masks(
        nodes, times, absolute_position_observations,
    )
    confidence = {
        node: _posterior_anchor_confidence(
            history.active_covariance_history_by_node[node]
        )
        for node in nodes
    }
    direction = run_orbital_direction_states(
        timestamps=times, posterior_state_history_by_node=posterior,
        frame_by_node=frames, anchor_mask_by_node=masks,
        anchor_confidence_by_node=confidence,
        config=OrbitalDirectionConfig(
            ring=RingCANNConfig(num_neurons=90, internal_dt=0.002),
        ),
        retain_activity=True,
    )
    references = {
        node: _propagate_reference(initial_state_by_node[node], times)
        for node in nodes
    }
    rt = run_orbital_rt_grid_states(
        timestamps=times, posterior_state_history_by_node=posterior,
        reference_state_history_by_node=references,
        anchor_mask_by_node=masks, anchor_confidence_by_node=confidence,
        retain_activity=True,
    )
    combined = build_navigation_brain_states(
        direction_by_node=direction, rt_by_node=rt,
    )
    place_config = NavigationPlaceCellConfig()
    place = build_navigation_place_cell_histories(
        navigation_by_node=combined, config=place_config,
    )

    navigation_by_epoch, cann_by_epoch = [], []
    for index, timestamp in enumerate(times):
        navigation_items, cann_items = [], []
        for node in nodes:
            direction_item = direction[node]
            rt_item = rt[node]
            place_item = place[node]
            valid = bool(direction_item.valid[index] and rt_item.valid[index])
            anchor_age = float(max(
                direction_item.anchor_age[index], rt_item.anchor_age[index],
            ))
            navigation_items.append(VisualNavigationState(
                node_id=node,
                absolute_navigation_available=bool(masks[node][index]),
                last_absolute_navigation_timestamp=_last_available_timestamp(
                    times, masks[node], index,
                ),
                anchor_age=anchor_age,
                orbital_phase=float(direction_item.anchored_phase[index]),
                radial_along_track=rt_item.anchored_rt[index],
                status="VALID" if valid else "INVALID",
                metrics={
                    "direction_residual_rad": float(
                        direction_item.anchored_residual[index]
                    ),
                    "radial_residual_m": float(
                        rt_item.anchored_residual_rt[index, 0]
                    ),
                    "along_track_residual_m": float(
                        rt_item.anchored_residual_rt[index, 1]
                    ),
                },
            ))
            cann_items.extend((
                VisualCANNSnapshot(
                    node_id=node, representation="DIRECTION_RING",
                    available=True,
                    activity=direction_item.neural_activity[index],
                    decoded_value=np.array([
                        direction_item.anchored_phase[index]
                    ]),
                    concentration=float(
                        direction_item.bump_concentration[index]
                    ),
                    bump_width=float(direction_item.bump_width[index]),
                    anchor_age=float(direction_item.anchor_age[index]),
                    cue_applied=bool(direction_item.cue_applied[index]),
                ),
                VisualCANNSnapshot(
                    node_id=node, representation="RT_LINE_PAIR",
                    available=True,
                    activity=np.stack((
                        rt_item.radial_activity[index],
                        rt_item.along_track_activity[index],
                    )),
                    decoded_value=rt_item.anchored_rt[index],
                    anchor_age=float(rt_item.anchor_age[index]),
                    cue_applied=bool(rt_item.cue_applied[index]),
                    status=(
                        "BOUNDARY_SATURATED"
                        if rt_item.saturated_at_boundary[index] else "OK"
                    ),
                    metrics={
                        "anchor_rejected": bool(
                            rt_item.anchor_rejected[index]
                        ),
                        "reference_rebased": bool(np.any(
                            rt_item.reference_rebased[index]
                        )),
                    },
                ),
                VisualCANNSnapshot(
                    node_id=node, representation="PLACE_CELL_RT",
                    available=True,
                    activity=place_item.activity[index].reshape(
                        place_config.phase_cell_count,
                        len(place_config.radial_centers_m),
                        len(place_config.along_track_centers_m),
                    ).sum(axis=0),
                    decoded_value=np.concatenate((
                        [place_item.decoded_phase[index]],
                        place_item.decoded_rt[index],
                    )),
                    concentration=float(place_item.peak_activity[index]),
                    status=(
                        "BOUNDARY_SATURATED"
                        if place_item.boundary_saturated[index] else "OK"
                    ),
                    metrics={
                        "normalized_entropy": float(
                            place_item.normalized_entropy[index]
                        ),
                    },
                ),
            ))
        navigation_by_epoch.append(tuple(navigation_items))
        cann_by_epoch.append(tuple(cann_items))
    return CANNVisualizationHistory(
        navigation_by_epoch=tuple(navigation_by_epoch),
        cann_by_epoch=tuple(cann_by_epoch),
    )


def _absolute_navigation_masks(nodes, times, observations):
    masks = {node: np.zeros(times.size, dtype=bool) for node in nodes}
    for observation in observations:
        if observation.satellite_id not in masks or not observation.valid_flag:
            continue
        matches = np.flatnonzero(np.isclose(
            times, float(observation.timestamp), atol=1e-9, rtol=0.0,
        ))
        if matches.size == 1:
            masks[observation.satellite_id][int(matches[0])] = True
    return masks


def _posterior_anchor_confidence(covariance_history):
    covariance = np.asarray(covariance_history, dtype=float)
    position_sigma = np.sqrt(np.maximum(
        0.0,
        np.trace(covariance[:, :3, :3], axis1=1, axis2=2) / 3.0,
    ))
    return np.clip(10.0 / (10.0 + position_sigma), 0.0, 1.0)


def _propagate_reference(initial_state, times):
    values = [np.asarray(initial_state, dtype=float).copy()]
    for index in range(1, times.size):
        values.append(rk4_step_absolute(
            values[-1], float(times[index] - times[index - 1]),
        ))
    return np.asarray(values)


def _last_available_timestamp(times, mask, index):
    available = np.flatnonzero(mask[:index + 1])
    return None if available.size == 0 else float(times[int(available[-1])])
