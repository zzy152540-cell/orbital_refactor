from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from interfaces.data_objects import ModuleInput, Observation
from interfaces.state_awareness_module import StateAwarenessModule

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
    result = history.final_fusion_result(
        node_id=observer_id,
        target_id=target_id,
    )
    absolute_state = chief_history[-1] + np.asarray(result.state_estimate, dtype=float)
    return TargetNodeReport(
        observer_id=observer_id,
        target_id=target_id,
        track_id=target_id if track_id is None else track_id,
        timestamp=float(result.timestamp),
        state_eci=absolute_state,
        covariance_eci=result.covariance,
        quality_score=result.confidence_level,
        valid_flag=True,
        modality_weights=result.modality_weights,
        used_measurement_ids=_measurement_ids(observations),
    )


def _measurement_ids(observations: Iterable[Observation]) -> tuple[str, ...]:
    values = []
    for observation in observations:
        if not observation.valid_flag:
            continue
        identifier = observation.metadata.get(
            "physical_observation_id",
            observation.metadata.get("measurement_id"),
        )
        if identifier is not None:
            values.append(str(identifier))
    return tuple(dict.fromkeys(values))
