from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from interfaces.data_objects import Observation
from orbital_core.coordinates import rotate_eci_to_pri
from tracking.data_contracts import TargetInitialState
from tracking.initial_orbit.tracking_handoff import build_tracking_handoff
from tracking.sequence_runner import KnownTargetSequenceHistory, run_known_target_sequence

from .data_contracts import AssociationResult, UnlabeledIODObservation
from .pipeline import MultiTargetAssociationPipeline


@dataclass(frozen=True)
class AutonomousTrackingSequenceResult:
    initialized_states_by_target: Mapping[str, TargetInitialState]
    association_by_epoch: Mapping[float, AssociationResult]
    tracking_history: KnownTargetSequenceHistory


def run_unlabeled_tracking_sequence(
    *,
    scene_id: str,
    observations,
    observer_state_by_epoch: Mapping[tuple[str, float], np.ndarray],
    q_eci2pri_by_epoch: Mapping[tuple[str, float], np.ndarray],
    association_pipeline: MultiTargetAssociationPipeline | None = None,
    max_coast_epochs: int = 3,
) -> AutonomousTrackingSequenceResult:
    """Run target-free association, IOD, filtering, CI and lifecycle offline."""

    values = tuple(observations)
    if any(not isinstance(item, UnlabeledIODObservation) for item in values):
        raise TypeError("Autonomous sequence requires unlabeled IOD observations.")
    pipeline = association_pipeline or MultiTargetAssociationPipeline()
    by_epoch = defaultdict(list)
    for item in values:
        by_epoch[item.timestamp].append(item)
    initialized: dict[str, TargetInitialState] = {}
    associations = {}
    tracking_observations: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    tracking_times = set()

    for timestamp in sorted(by_epoch):
        update = pipeline.step(by_epoch[timestamp], initialized)
        associations[timestamp] = update.association
        initialized.update(update.autonomous_iod.initialized_states_by_target)
        for target_id, target_observations in update.observations_by_target.items():
            tracking_times.add(float(timestamp))
            for item in target_observations:
                key = (item.observer_id, float(timestamp))
                if key not in q_eci2pri_by_epoch:
                    raise ValueError(f"Missing observer SPRI attitude for {key!r}.")
                tracking_observations[(item.observer_id, target_id)].append(
                    _tracking_observation(item, target_id, q_eci2pri_by_epoch[key])
                )

    times = np.array(sorted(tracking_times), dtype=float)
    if not initialized:
        raise ValueError("No target completed autonomous initial orbit determination.")
    if times.size < 2:
        raise ValueError("At least two post-IOD tracking epochs are required.")
    observers = tuple(sorted({key[0] for key in observer_state_by_epoch}))
    observer_histories = {}
    quaternion_histories = {}
    for observer_id in observers:
        state_values = []
        quaternion_values = []
        for timestamp in times:
            key = (observer_id, float(timestamp))
            if key not in observer_state_by_epoch or key not in q_eci2pri_by_epoch:
                raise ValueError(f"Missing observer runtime state or attitude for {key!r}.")
            state_values.append(observer_state_by_epoch[key])
            quaternion_values.append(q_eci2pri_by_epoch[key])
        observer_histories[observer_id] = np.asarray(state_values, dtype=float)
        quaternion_histories[observer_id] = np.asarray(quaternion_values, dtype=float)
    observations_by_link = {
        (observer_id, target_id): tracking_observations.get(
            (observer_id, target_id), [],
        )
        for observer_id in observers
        for target_id in initialized
    }
    if any(not items for items in observations_by_link.values()):
        empty = sorted(key for key, items in observations_by_link.items() if not items)
        raise ValueError(f"Every observer-target link needs a tracking observation; empty={empty}.")
    handoff = build_tracking_handoff(
        initialized_states=initialized,
        timestamps=times,
        observer_state_history_by_id=observer_histories,
        q_eci2pri_history_by_id=quaternion_histories,
        observations_by_link=observations_by_link,
    )
    history = run_known_target_sequence(
        scene_id=scene_id,
        module_inputs=handoff.module_inputs,
        initial_target_states=handoff.target_initial_states,
        max_coast_epochs=max_coast_epochs,
    )
    return AutonomousTrackingSequenceResult(
        initialized_states_by_target=dict(initialized),
        association_by_epoch=dict(associations),
        tracking_history=history,
    )


def _tracking_observation(item, target_id, q_eci2pri):
    source_modality = str(item.metadata.get("sourceModality", item.modality)).upper()
    metadata = {
        **item.metadata,
        "observation_id": item.metadata.get("sourceMessageId"),
        "association_target_id": target_id,
    }
    if item.modality == "RADAR":
        measurement = item.measurement
        modality = "RADAR"
        valid = item.valid_flag
    else:
        line_spri = rotate_eci_to_pri(item.measurement, q_eci2pri)
        modality = source_modality if source_modality in {"OPTICAL", "INFRARED"} else "INFRARED"
        if modality == "OPTICAL":
            valid = bool(item.valid_flag and line_spri[2] > 1e-12)
            measurement = line_spri[:2] / line_spri[2] if valid else np.zeros(2)
        else:
            valid = item.valid_flag
            measurement = np.array([
                np.arctan2(line_spri[1], line_spri[0]),
                np.arctan2(line_spri[2], np.linalg.norm(line_spri[:2])),
            ])
    return Observation(
        timestamp=item.timestamp,
        observer_id=item.observer_id,
        target_id=target_id,
        modality=modality,
        source_type="RAW_SENSOR",
        measurement=np.asarray(measurement, dtype=float),
        covariance=np.asarray(item.covariance, dtype=float),
        confidence=float(item.metadata.get("confidence", 1.0)),
        frame="SPRI",
        valid_flag=bool(valid),
        metadata=metadata,
    )

