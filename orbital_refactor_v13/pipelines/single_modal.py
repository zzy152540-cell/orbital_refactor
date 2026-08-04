from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interfaces.data_objects import LocalEstimate
from orbital_core.dynamics import (
    build_target_absolute_accel_history,
    numerical_diff_accel_from_velocity,
)
from orbital_core.filters import LocalDynamicsEKF
from orbital_core.quality import quality_score_from_covariance
from orbital_core.measurement_integrity import INTEGRITY_PREDICTION_ONLY


Array = np.ndarray


@dataclass(frozen=True)
class SingleModalHistory:
    state_history: Array
    covariance_history: Array
    acceleration_history: Array
    nis_history: Array
    gate_history: Array
    valid_history: Array
    statistics: dict[str, int]
    processed_nis_history: Array
    measurement_covariance_scale_history: Array
    integrity_status_history: tuple[str, ...]
    consecutive_anomaly_history: Array

    def final_estimate(
        self,
        *,
        modality: str,
        timestamp: float,
        node_id: str,
        target_id: str,
    ) -> LocalEstimate:
        final_nis = float(self.nis_history[-1]) if np.isfinite(self.nis_history[-1]) else None
        quality = quality_score_from_covariance(
            self.covariance_history[-1], nis=final_nis, expected_nis=None
        )
        return LocalEstimate(
            modality=modality,
            timestamp=float(timestamp),
            state_estimate=self.state_history[-1].copy(),
            acceleration=self.acceleration_history[-1].copy(),
            covariance=self.covariance_history[-1].copy(),
            quality_score=quality,
            valid_flag=bool(self.valid_history[-1]),
            node_id=node_id,
            target_id=target_id,
        )


def run_single_modal_filter(
    *,
    timestamps: Array,
    chief_state_history_eci: Array,
    q_eci2pri_history: Array,
    measurements: Array,
    measurement_valid_flags: Array,
    ekf: LocalDynamicsEKF,
    initial_state: Array,
    initial_covariance: Array,
) -> SingleModalHistory:
    timestamps = np.asarray(timestamps, dtype=float)
    sample_count = len(timestamps)
    if sample_count < 1:
        raise ValueError("At least one sample is required.")

    state_history = np.zeros((sample_count, 6))
    covariance_history = np.zeros((sample_count, 6, 6))
    nis_history = np.full(sample_count, np.nan)
    processed_nis_history = np.full(sample_count, np.nan)
    covariance_scale_history = np.ones(sample_count)
    integrity_status_history = [INTEGRITY_PREDICTION_ONLY] * sample_count
    consecutive_anomaly_history = np.zeros(sample_count, dtype=int)
    gate_history = np.zeros(sample_count, dtype=bool)
    output_valid_history = np.ones(sample_count, dtype=bool)

    state = np.asarray(initial_state, dtype=float).reshape(6).copy()
    covariance = np.asarray(initial_covariance, dtype=float).reshape(6, 6).copy()
    state_history[0] = state
    covariance_history[0] = covariance

    accepted = rejected = skipped = 0
    consecutive_anomalies = 0
    for index in range(1, sample_count):
        dt = float(timestamps[index] - timestamps[index - 1])
        predicted_state, predicted_covariance = ekf.predict(
            state,
            covariance,
            chief_state_history_eci[index - 1],
            dt,
        )
        if bool(measurement_valid_flags[index]):
            state, covariance, diagnostics = ekf.update(
                predicted_state,
                predicted_covariance,
                measurements[index],
                q_eci2pri_history[index],
            )
            nis_history[index] = diagnostics.nis
            processed_nis_history[index] = diagnostics.integrity.processed_nis
            covariance_scale_history[index] = (
                diagnostics.integrity.measurement_covariance_scale
            )
            integrity_status_history[index] = diagnostics.integrity.status
            consecutive_anomalies = (
                consecutive_anomalies + 1
                if diagnostics.integrity.anomalous else 0
            )
            consecutive_anomaly_history[index] = consecutive_anomalies
            gate_history[index] = diagnostics.gated
            output_valid_history[index] = not diagnostics.skipped
            if diagnostics.skipped:
                rejected += 1
            else:
                accepted += 1
        else:
            state, covariance = predicted_state, predicted_covariance
            skipped += 1
            # Prediction-only output remains valid, matching the interface document.
            output_valid_history[index] = True
            integrity_status_history[index] = INTEGRITY_PREDICTION_ONLY
            consecutive_anomalies = 0

        state_history[index] = state
        covariance_history[index] = covariance

    acceleration_history = build_target_absolute_accel_history(
        state_history, chief_state_history_eci
    )
    return SingleModalHistory(
        state_history=state_history,
        covariance_history=covariance_history,
        acceleration_history=acceleration_history,
        nis_history=nis_history,
        gate_history=gate_history,
        valid_history=output_valid_history,
        statistics={"accepted": accepted, "rejected": rejected, "skipped": skipped},
        processed_nis_history=processed_nis_history,
        measurement_covariance_scale_history=covariance_scale_history,
        integrity_status_history=tuple(integrity_status_history),
        consecutive_anomaly_history=consecutive_anomaly_history,
    )


def compute_history_errors(
    history: SingleModalHistory,
    relative_truth_eci: Array,
    chief_state_history_eci: Array,
    timestamps: Array,
) -> dict[str, Array]:
    truth = np.asarray(relative_truth_eci, dtype=float)
    target_truth_velocity = chief_state_history_eci[:, 3:6] + truth[:, 3:6]
    target_reference_acceleration = numerical_diff_accel_from_velocity(
        timestamps, target_truth_velocity
    )
    return {
        "position_error": history.state_history[:, :3] - truth[:, :3],
        "velocity_error": history.state_history[:, 3:6] - truth[:, 3:6],
        "acceleration_error": history.acceleration_history - target_reference_acceleration,
    }
