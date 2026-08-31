from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from interfaces.data_objects import (
    AbnormalEvent,
    FusionStatus,
    ModuleOutput,
    RuntimeStatus,
    SingleFusionResult,
    StateOutput,
)
from orbital_core.centralized_filter import CentralizedDynamicsEKF
from orbital_core.dynamics import build_target_absolute_accel_history
from orbital_core.quality import quality_score_from_covariance

Array = np.ndarray


@dataclass
class CentralizedHistory:
    timestamps: Array
    state_history: Array
    covariance_history: Array
    acceleration_history: Array
    measurement_valid_history: dict[str, Array]
    nis_history: dict[str, Array]
    gate_history: dict[str, Array]
    statistics: dict[str, dict[str, int]]
    abnormal_events: list[AbnormalEvent]
    processing_time: float
    cann_sidecar_history: Any | None = None

    def final_fusion_result(self, *, node_id: str, target_id: str) -> SingleFusionResult:
        index = len(self.timestamps) - 1
        valid_flags = {
            modality: bool(flags[index])
            for modality, flags in self.measurement_valid_history.items()
        }
        active = [name for name, valid in valid_flags.items() if valid]
        weights = ({name: 1.0 / len(active) for name in active} if active else {})
        quality = quality_score_from_covariance(self.covariance_history[index])
        confidence = float(np.clip(quality / (1.0 + quality), 0.0, 1.0))
        return SingleFusionResult(
            node_id=node_id,
            target_id=target_id,
            timestamp=float(self.timestamps[index]),
            state_estimate=self.state_history[index].copy(),
            acceleration=self.acceleration_history[index].copy(),
            covariance=self.covariance_history[index].copy(),
            modality_weights=weights,
            modality_valid_flags=valid_flags,
            confidence_level=confidence,
        )

    def to_module_output(self, *, node_id: str, target_id: str) -> ModuleOutput:
        result = self.final_fusion_result(node_id=node_id, target_id=target_id)
        active = [name for name, valid in result.modality_valid_flags.items() if valid]
        return ModuleOutput(
            state_output=StateOutput(
                timestamp=result.timestamp,
                target_id=target_id,
                position_estimate=result.state_estimate[:3].copy(),
                velocity_estimate=result.state_estimate[3:].copy(),
                acceleration_estimate=result.acceleration.copy(),
                covariance=result.covariance.copy(),
                valid_flag=True,
                confidence_level=result.confidence_level,
            ),
            fusion_status=FusionStatus(
                modality_weights=result.modality_weights,
                modality_valid_flags=result.modality_valid_flags,
                active_nodes=[node_id],
            ),
            abnormal_events=list(self.abnormal_events),
            runtime_status=RuntimeStatus(
                processing_time=self.processing_time,
                observation_count=sum(int(np.count_nonzero(v)) for v in self.measurement_valid_history.values()),
                active_modality_count=len(active),
                active_node_count=1,
                status="OK" if active else "PREDICTION_ONLY",
            ),
        )


def run_centralized_filter(
    *,
    timestamps: Array,
    chief_state_history_eci: Array,
    q_eci2pri_history: Array,
    measurements_by_modality: Mapping[str, Array],
    valid_flags_by_modality: Mapping[str, Array],
    ekf: CentralizedDynamicsEKF,
    initial_state: Array,
    initial_covariance: Array,
    node_id: str = "node_0",
    target_id: str = "target_0",
) -> CentralizedHistory:
    start = perf_counter()
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    chief = np.asarray(chief_state_history_eci, dtype=float)
    quaternions = np.asarray(q_eci2pri_history, dtype=float)
    count = timestamps.size
    if count == 0:
        raise ValueError("At least one timestamp is required.")
    if chief.shape != (count, 6) or quaternions.shape != (count, 4):
        raise ValueError("Runtime histories have inconsistent shapes.")
    modalities = list(measurements_by_modality)
    if set(modalities) != set(valid_flags_by_modality):
        raise ValueError("Measurement and validity modalities must match.")

    states = np.zeros((count, 6), dtype=float)
    covariances = np.zeros((count, 6, 6), dtype=float)
    states[0] = np.asarray(initial_state, dtype=float).reshape(6)
    covariances[0] = np.asarray(initial_covariance, dtype=float).reshape(6, 6)
    nis = {name: np.full(count, np.nan) for name in modalities}
    gates = {name: np.zeros(count, dtype=bool) for name in modalities}
    statistics = {name: {"accepted": 0, "rejected": 0, "skipped": 0} for name in modalities}
    events: list[AbnormalEvent] = []

    for index in range(1, count):
        dt = float(timestamps[index] - timestamps[index - 1])
        if dt <= 0:
            raise ValueError("timestamps must be strictly increasing.")
        predicted_state, predicted_covariance = ekf.predict(
            states[index - 1], covariances[index - 1], chief[index - 1], dt
        )
        current: dict[str, Array | None] = {}
        for modality in modalities:
            if bool(valid_flags_by_modality[modality][index]):
                current[modality] = measurements_by_modality[modality][index]
            else:
                current[modality] = None
                statistics[modality]["skipped"] += 1
                events.append(AbnormalEvent(
                    timestamp=float(timestamps[index]), event_type="MODALITY_MISSING",
                    severity="WARNING", description=f"{modality} measurement unavailable.",
                    node_id=node_id, target_id=target_id, modality=modality,
                ))
        state, covariance, diagnostics = ekf.centralized_update(
            predicted_state, predicted_covariance, current, quaternions[index]
        )
        for modality, value in diagnostics.nis_by_modality.items():
            nis[modality][index] = value
        for modality, value in diagnostics.gated_by_modality.items():
            gates[modality][index] = value
        for modality in diagnostics.accepted_modalities:
            statistics[modality]["accepted"] += 1
            if diagnostics.gated_by_modality.get(modality, False):
                events.append(AbnormalEvent(
                    timestamp=float(timestamps[index]), event_type="OBSERVATION_DOWNWEIGHTED",
                    severity="WARNING", description=f"{modality} observation downweighted by NIS gate.",
                    node_id=node_id, target_id=target_id, modality=modality,
                ))
        for modality in diagnostics.rejected_modalities:
            statistics[modality]["rejected"] += 1
            events.append(AbnormalEvent(
                timestamp=float(timestamps[index]), event_type="OBSERVATION_REJECTED",
                severity="ERROR", description=f"{modality} observation rejected by NIS gate.",
                node_id=node_id, target_id=target_id, modality=modality,
            ))
        states[index] = state
        covariances[index] = covariance

    accelerations = build_target_absolute_accel_history(states, chief)
    return CentralizedHistory(
        timestamps=timestamps,
        state_history=states,
        covariance_history=covariances,
        acceleration_history=accelerations,
        measurement_valid_history={k: np.asarray(v, dtype=bool) for k, v in valid_flags_by_modality.items()},
        nis_history=nis,
        gate_history=gates,
        statistics=statistics,
        abnormal_events=events,
        processing_time=perf_counter() - start,
    )
