from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from interfaces.data_objects import (
    AbnormalEvent,
    FusionStatus,
    LocalEstimate,
    ModuleOutput,
    RuntimeStatus,
    SingleFusionResult,
    StateOutput,
)
from orbital_core.ci_fusion import ci_fuse_posteriors
from orbital_core.dynamics import build_target_absolute_accel_history
from orbital_core.filters import LocalDynamicsEKF
from orbital_core.quality import quality_score_from_covariance
from orbital_core.measurement_integrity import INTEGRITY_PREDICTION_ONLY


Array = np.ndarray
_MODALITY_ORDER = {"opt": 0, "nn": 0, "ir": 1, "rad": 2}


@dataclass(frozen=True)
class FederatedCIHistory:
    timestamps: Array
    fused_state_history: Array
    fused_covariance_history: Array
    fused_acceleration_history: Array
    local_state_history: dict[str, Array]
    local_covariance_history: dict[str, Array]
    local_output_valid_history: dict[str, Array]
    measurement_valid_history: dict[str, Array]
    nis_history: dict[str, Array]
    gate_history: dict[str, Array]
    ci_weight_history: list[dict[str, float] | None]
    statistics: dict[str, dict[str, int]]
    abnormal_events: list[AbnormalEvent]
    processing_time: float
    processed_nis_history: dict[str, Array]
    measurement_covariance_scale_history: dict[str, Array]
    integrity_status_history: dict[str, tuple[str, ...]]
    consecutive_anomaly_history: dict[str, Array]
    cann_sidecar_history: Any | None = None

    def final_local_estimates(self, *, node_id: str, target_id: str) -> list[LocalEstimate]:
        estimates: list[LocalEstimate] = []
        final_index = len(self.timestamps) - 1
        for modality in self.local_state_history:
            nis = self.nis_history[modality][final_index]
            quality = quality_score_from_covariance(
                self.local_covariance_history[modality][final_index],
                nis=float(nis) if np.isfinite(nis) else None,
                expected_nis=None,
            )
            estimates.append(
                LocalEstimate(
                    modality=modality,
                    timestamp=float(self.timestamps[final_index]),
                    state_estimate=self.local_state_history[modality][final_index].copy(),
                    acceleration=self.fused_acceleration_history[final_index].copy(),
                    covariance=self.local_covariance_history[modality][final_index].copy(),
                    quality_score=quality,
                    valid_flag=bool(self.local_output_valid_history[modality][final_index]),
                    node_id=node_id,
                    target_id=target_id,
                )
            )
        return estimates

    def final_fusion_result(self, *, node_id: str, target_id: str) -> SingleFusionResult:
        final_index = len(self.timestamps) - 1
        final_weights = self.ci_weight_history[-1] or {}
        valid_flags = {
            modality: bool(flags[final_index])
            for modality, flags in self.measurement_valid_history.items()
        }
        confidence = _confidence_from_covariance_and_weights(
            self.fused_covariance_history[final_index], final_weights
        )
        return SingleFusionResult(
            node_id=node_id,
            target_id=target_id,
            timestamp=float(self.timestamps[final_index]),
            state_estimate=self.fused_state_history[final_index].copy(),
            acceleration=self.fused_acceleration_history[final_index].copy(),
            covariance=self.fused_covariance_history[final_index].copy(),
            modality_weights=dict(final_weights),
            modality_valid_flags=valid_flags,
            confidence_level=confidence,
        )

    def to_module_output(self, *, node_id: str, target_id: str) -> ModuleOutput:
        result = self.final_fusion_result(node_id=node_id, target_id=target_id)
        active_modalities = [
            name for name, valid in result.modality_valid_flags.items() if valid
        ]
        state_output = StateOutput(
            timestamp=result.timestamp,
            target_id=result.target_id,
            position_estimate=result.state_estimate[:3].copy(),
            velocity_estimate=result.state_estimate[3:6].copy(),
            acceleration_estimate=result.acceleration.copy(),
            covariance=result.covariance.copy(),
            valid_flag=True,
            confidence_level=result.confidence_level,
        )
        fusion_status = FusionStatus(
            modality_weights=result.modality_weights,
            modality_valid_flags=result.modality_valid_flags,
            active_nodes=[node_id],
        )
        runtime_status = RuntimeStatus(
            processing_time=self.processing_time,
            observation_count=sum(
                int(np.count_nonzero(flags))
                for flags in self.measurement_valid_history.values()
            ),
            active_modality_count=len(active_modalities),
            active_node_count=1,
            status="OK" if active_modalities else "PREDICTION_ONLY",
        )
        return ModuleOutput(
            state_output=state_output,
            fusion_status=fusion_status,
            abnormal_events=list(self.abnormal_events),
            runtime_status=runtime_status,
        )


def run_federated_ci_filter(
    *,
    timestamps: Array,
    chief_state_history_eci: Array,
    q_eci2pri_history: Array,
    measurements_by_modality: Mapping[str, Array],
    valid_flags_by_modality: Mapping[str, Array],
    local_filters: Mapping[str, LocalDynamicsEKF],
    initial_state: Array,
    initial_covariance: Array,
    reset_feedback: bool = False,
    ci_objective: str = "trace",
    ci_grid_points: int = 101,
    node_id: str = "node_0",
    target_id: str = "target_0",
) -> FederatedCIHistory:
    """Run the legacy-compatible federated EKF + CI workflow.

    The implementation intentionally preserves the original numerical behavior:
    missing or hard-rejected measurements are not included in CI; when every
    modality is unavailable the previous fused posterior is held; optional
    feedback overwrites all local posteriors with the fused posterior.
    """
    start_time = perf_counter()
    timestamps = np.asarray(timestamps, dtype=float)
    chief_history = np.asarray(chief_state_history_eci, dtype=float)
    q_history = np.asarray(q_eci2pri_history, dtype=float)
    sample_count = len(timestamps)
    if sample_count < 1:
        raise ValueError("At least one sample is required.")
    if chief_history.shape != (sample_count, 6):
        raise ValueError("chief_state_history_eci must have shape (N, 6).")
    if len(q_history) != sample_count:
        raise ValueError("q_eci2pri_history length must match timestamps.")

    modalities = _validate_modal_inputs(
        sample_count,
        measurements_by_modality,
        valid_flags_by_modality,
        local_filters,
    )
    initial_state = np.asarray(initial_state, dtype=float).reshape(6)
    initial_covariance = np.asarray(initial_covariance, dtype=float).reshape(6, 6)

    local_states = {name: np.zeros((sample_count, 6)) for name in modalities}
    local_covariances = {name: np.zeros((sample_count, 6, 6)) for name in modalities}
    local_output_valid = {name: np.ones(sample_count, dtype=bool) for name in modalities}
    nis_history = {name: np.full(sample_count, np.nan) for name in modalities}
    processed_nis_history = {
        name: np.full(sample_count, np.nan) for name in modalities
    }
    covariance_scale_history = {
        name: np.ones(sample_count) for name in modalities
    }
    integrity_status_history = {
        name: [INTEGRITY_PREDICTION_ONLY] * sample_count for name in modalities
    }
    consecutive_anomaly_history = {
        name: np.zeros(sample_count, dtype=int) for name in modalities
    }
    consecutive_anomalies = {name: 0 for name in modalities}
    gate_history = {name: np.zeros(sample_count, dtype=bool) for name in modalities}
    statistics = {
        name: {"accepted": 0, "rejected": 0, "skipped": 0}
        for name in modalities
    }

    fused_states = np.zeros((sample_count, 6))
    fused_covariances = np.zeros((sample_count, 6, 6))
    for name in modalities:
        local_states[name][0] = initial_state
        local_covariances[name][0] = initial_covariance
    fused_states[0] = initial_state
    fused_covariances[0] = initial_covariance

    previous_states = {name: initial_state.copy() for name in modalities}
    previous_covariances = {name: initial_covariance.copy() for name in modalities}
    ci_weight_history: list[dict[str, float] | None] = [None]
    abnormal_events: list[AbnormalEvent] = []

    for index in range(1, sample_count):
        dt = float(timestamps[index] - timestamps[index - 1])
        if dt <= 0.0:
            raise ValueError("timestamps must be strictly increasing.")
        posterior_list: list[tuple[str, Array, Array]] = []
        reference_filter = local_filters[modalities[0]]
        predicted_fused_state, predicted_fused_covariance = (
            reference_filter.predict(
                fused_states[index - 1], fused_covariances[index - 1],
                chief_history[index - 1], dt,
            )
        )

        for modality in modalities:
            ekf = local_filters[modality]
            predicted_state, predicted_covariance = ekf.predict(
                previous_states[modality],
                previous_covariances[modality],
                chief_history[index - 1],
                dt,
            )
            measurement_is_valid = bool(valid_flags_by_modality[modality][index])
            if not measurement_is_valid:
                state, covariance = predicted_state, predicted_covariance
                local_output_valid[modality][index] = True
                statistics[modality]["skipped"] += 1
                consecutive_anomalies[modality] = 0
                abnormal_events.append(
                    AbnormalEvent(
                        timestamp=float(timestamps[index]),
                        event_type="MODALITY_MISSING",
                        severity="WARNING",
                        description=f"{modality} measurement unavailable; prediction-only output used.",
                        node_id=node_id,
                        target_id=target_id,
                        modality=modality,
                    )
                )
            else:
                state, covariance, diagnostics = ekf.update(
                    predicted_state,
                    predicted_covariance,
                    measurements_by_modality[modality][index],
                    q_history[index],
                )
                nis_history[modality][index] = diagnostics.nis
                processed_nis_history[modality][index] = (
                    diagnostics.integrity.processed_nis
                )
                covariance_scale_history[modality][index] = (
                    diagnostics.integrity.measurement_covariance_scale
                )
                integrity_status_history[modality][index] = (
                    diagnostics.integrity.status
                )
                consecutive_anomalies[modality] = (
                    consecutive_anomalies[modality] + 1
                    if diagnostics.integrity.anomalous else 0
                )
                consecutive_anomaly_history[modality][index] = (
                    consecutive_anomalies[modality]
                )
                gate_history[modality][index] = diagnostics.gated
                local_output_valid[modality][index] = not diagnostics.skipped
                if diagnostics.skipped:
                    statistics[modality]["rejected"] += 1
                    abnormal_events.append(
                        AbnormalEvent(
                            timestamp=float(timestamps[index]),
                            event_type="OBSERVATION_REJECTED",
                            severity="ERROR",
                            description=(
                                f"{modality} observation rejected by hard NIS gate "
                                f"(NIS={diagnostics.nis:.6g})."
                            ),
                            node_id=node_id,
                            target_id=target_id,
                            modality=modality,
                        )
                    )
                else:
                    statistics[modality]["accepted"] += 1
                    posterior_list.append((modality, state, covariance))
                    if diagnostics.gated:
                        abnormal_events.append(
                            AbnormalEvent(
                                timestamp=float(timestamps[index]),
                                event_type="OBSERVATION_DOWNWEIGHTED",
                                severity="WARNING",
                                description=(
                                    f"{modality} observation exceeded soft NIS gate "
                                    f"(NIS={diagnostics.nis:.6g})."
                                ),
                                node_id=node_id,
                                target_id=target_id,
                                modality=modality,
                            )
                        )

            local_states[modality][index] = state
            local_covariances[modality][index] = covariance
            previous_states[modality] = state
            previous_covariances[modality] = covariance

        if not posterior_list:
            fused_states[index] = predicted_fused_state
            fused_covariances[index] = predicted_fused_covariance
            ci_weight_history.append(None)
            abnormal_events.append(
                AbnormalEvent(
                    timestamp=float(timestamps[index]),
                    event_type="ALL_MODALITIES_UNAVAILABLE",
                    severity="ERROR",
                    description=(
                        "No accepted modality posterior; fused result propagated "
                        "with relative-orbit dynamics."
                    ),
                    node_id=node_id,
                    target_id=target_id,
                )
            )
        else:
            posterior_list.sort(key=lambda item: _MODALITY_ORDER.get(item[0], 99))
            fusion = ci_fuse_posteriors(
                posterior_list,
                objective=ci_objective,
                grid_points=ci_grid_points,
            )
            fused_states[index] = fusion.state
            fused_covariances[index] = fusion.covariance
            ci_weight_history.append(dict(fusion.weights))

        if reset_feedback:
            for modality in modalities:
                previous_states[modality] = fused_states[index].copy()
                previous_covariances[modality] = fused_covariances[index].copy()
                local_states[modality][index] = fused_states[index]
                local_covariances[modality][index] = fused_covariances[index]

    acceleration_history = build_target_absolute_accel_history(
        fused_states, chief_history
    )
    processing_time = perf_counter() - start_time
    return FederatedCIHistory(
        timestamps=timestamps.copy(),
        fused_state_history=fused_states,
        fused_covariance_history=fused_covariances,
        fused_acceleration_history=acceleration_history,
        local_state_history=local_states,
        local_covariance_history=local_covariances,
        local_output_valid_history=local_output_valid,
        measurement_valid_history={
            name: np.asarray(valid_flags_by_modality[name], dtype=bool).copy()
            for name in modalities
        },
        nis_history=nis_history,
        gate_history=gate_history,
        ci_weight_history=ci_weight_history,
        statistics=statistics,
        abnormal_events=abnormal_events,
        processing_time=processing_time,
        processed_nis_history=processed_nis_history,
        measurement_covariance_scale_history=covariance_scale_history,
        integrity_status_history={
            name: tuple(values)
            for name, values in integrity_status_history.items()
        },
        consecutive_anomaly_history=consecutive_anomaly_history,
    )


def _validate_modal_inputs(
    sample_count: int,
    measurements_by_modality: Mapping[str, Array],
    valid_flags_by_modality: Mapping[str, Array],
    local_filters: Mapping[str, LocalDynamicsEKF],
) -> list[str]:
    modalities = list(local_filters)
    if not modalities:
        raise ValueError("At least one local filter is required.")
    if set(modalities) != set(measurements_by_modality):
        raise ValueError("measurements_by_modality keys must match local_filters keys.")
    if set(modalities) != set(valid_flags_by_modality):
        raise ValueError("valid_flags_by_modality keys must match local_filters keys.")
    for modality in modalities:
        if len(measurements_by_modality[modality]) != sample_count:
            raise ValueError(f"Measurement length mismatch for modality {modality}.")
        if len(valid_flags_by_modality[modality]) != sample_count:
            raise ValueError(f"Valid-flag length mismatch for modality {modality}.")
        expected_mode = modality
        if local_filters[modality].mode_name != expected_mode:
            raise ValueError(
                f"Filter mode mismatch for {modality}: expected {expected_mode}, "
                f"got {local_filters[modality].mode_name}."
            )
    return modalities


def _confidence_from_covariance_and_weights(
    covariance: Array,
    weights: Mapping[str, float],
) -> float:
    covariance_score = quality_score_from_covariance(covariance)
    if not weights:
        return float(np.clip(covariance_score, 0.0, 1.0))
    weight_array = np.asarray(list(weights.values()), dtype=float)
    concentration = float(np.sum(weight_array**2))
    diversity = 1.0 / max(concentration * len(weight_array), 1.0)
    return float(np.clip(0.8 * covariance_score + 0.2 * diversity, 0.0, 1.0))
