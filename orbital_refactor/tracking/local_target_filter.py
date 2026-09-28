from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from interfaces.data_objects import ModuleInput, Observation
from interfaces.state_awareness_module import StateAwarenessModule
from orbital_core.quality import quality_score_from_covariance

from .data_contracts import TargetNodeReport


def run_local_target_filter(
    module_input: ModuleInput,
    *,
    track_id: str | None = None,
) -> TargetNodeReport:
    """Run the existing single-observer filter and emit an absolute target report.

    The legacy filter state is relative to ``chief_state_history_eci``. This
    adapter is the explicit boundary that converts that posterior into the
    absolute J2000 state required by target-level cooperative fusion.
    """

    return run_local_target_history(module_input, track_id=track_id)[-1]


def run_local_target_history(
    module_input: ModuleInput,
    *,
    track_id: str | None = None,
) -> tuple[TargetNodeReport, ...]:
    """Return one absolute target report for every local filter epoch."""

    observations = list(module_input.sensor_measurements)
    observer_ids = {str(item.observer_id) for item in observations}
    if len(observer_ids) != 1:
        raise ValueError(
            "One local target filter requires observations from exactly one observer_id."
        )
    observer_id = next(iter(observer_ids))
    target_id = str(module_input.initial_state.target_id)
    if any(str(item.target_id) != target_id for item in observations):
        raise ValueError("One local target filter requires exactly one target_id.")

    runtime = module_input.config.get("runtime", {})
    configured_node = str(
        runtime.get("node_id", module_input.config.get("node_id", observer_id))
    )
    if configured_node != observer_id:
        raise ValueError("runtime.node_id must match the observation observer_id.")
    chief_history = np.asarray(runtime.get("chief_state_history_eci"), dtype=float)
    timestamps = np.asarray(runtime.get("timestamps"), dtype=float).reshape(-1)
    if chief_history.shape != (timestamps.size, 6) or timestamps.size == 0:
        raise ValueError(
            "runtime.chief_state_history_eci must align with non-empty runtime.timestamps."
        )

    history = StateAwarenessModule().run_history(module_input)
    relative_values = getattr(history, "fused_state_history", None)
    if relative_values is None:
        relative_values = history.state_history
    covariance_values = getattr(history, "fused_covariance_history", None)
    if covariance_values is None:
        covariance_values = history.covariance_history
    relative_states = np.asarray(relative_values, dtype=float)
    covariances = np.asarray(covariance_values, dtype=float)
    valid_by_modality = {
        name: np.asarray(values, dtype=bool)
        for name, values in history.measurement_valid_history.items()
    }
    report_track_id = target_id if track_id is None else str(track_id)
    identifiers_by_timestamp = _measurement_ids_by_timestamp(observations)
    reports = []
    for index, timestamp in enumerate(timestamps):
        weights = _modality_weights(history, valid_by_modality, index)
        quality = quality_score_from_covariance(covariances[index])
        confidence = float(np.clip(quality / (1.0 + quality), 0.0, 1.0))
        reports.append(TargetNodeReport(
            observer_id=observer_id,
            target_id=target_id,
            track_id=report_track_id,
            timestamp=float(timestamp),
            state_eci=chief_history[index] + relative_states[index],
            covariance_eci=covariances[index],
            quality_score=confidence,
            valid_flag=any(values[index] for values in valid_by_modality.values()),
            modality_weights=weights,
            used_measurement_ids=identifiers_by_timestamp.get(float(timestamp), ()),
        ))
    return tuple(reports)


def _measurement_ids_by_timestamp(
    observations: Iterable[Observation],
) -> dict[float, tuple[str, ...]]:
    grouped: dict[float, list[str]] = {}
    for observation in observations:
        if not observation.valid_flag:
            continue
        identifier = observation.metadata.get(
            "physical_observation_id",
            observation.metadata.get("measurement_id"),
        )
        if identifier is not None:
            grouped.setdefault(float(observation.timestamp), []).append(str(identifier))
    return {
        timestamp: tuple(dict.fromkeys(values))
        for timestamp, values in grouped.items()
    }


def _modality_weights(history, valid_by_modality, index):
    ci_history = getattr(history, "ci_weight_history", None)
    if ci_history is not None:
        return dict(ci_history[index] or {})
    active = [name for name, values in valid_by_modality.items() if values[index]]
    return (
        {name: 1.0 / len(active) for name in active}
        if active else {}
    )
