from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


INTEGRITY_ACCEPTED = "ACCEPTED"
INTEGRITY_DOWNWEIGHTED = "DOWNWEIGHTED"
INTEGRITY_HARD_REJECTED = "HARD_REJECTED"
INTEGRITY_PREDICTION_ONLY = "PREDICTION_ONLY"

_VALID_STATUSES = {
    INTEGRITY_ACCEPTED,
    INTEGRITY_DOWNWEIGHTED,
    INTEGRITY_HARD_REJECTED,
    INTEGRITY_PREDICTION_ONLY,
}

INTEGRITY_MODE_NONE = "NONE"
INTEGRITY_MODE_LEGACY_FIXED_SOFT = "LEGACY_FIXED_SOFT"
INTEGRITY_MODE_PROPORTIONAL_INFLATION = "PROPORTIONAL_INFLATION"
INTEGRITY_MODE_HARD_GATE = "HARD_GATE"
INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE = (
    "PROPORTIONAL_INFLATION_WITH_HARD_GATE"
)
_VALID_MODES = {
    INTEGRITY_MODE_NONE,
    INTEGRITY_MODE_LEGACY_FIXED_SOFT,
    INTEGRITY_MODE_PROPORTIONAL_INFLATION,
    INTEGRITY_MODE_HARD_GATE,
    INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
}


@dataclass(frozen=True)
class MeasurementIntegrityPolicy:
    mode: str = INTEGRITY_MODE_NONE
    inflation_threshold: float | None = None
    maximum_covariance_scale: float = 1.0
    fixed_covariance_scale: float = 20.0
    hard_gate_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Unsupported measurement integrity mode: {self.mode}")
        if not np.isfinite(self.maximum_covariance_scale) or (
            self.maximum_covariance_scale < 1.0
        ):
            raise ValueError(
                "maximum_covariance_scale must be finite and at least one."
            )
        if not np.isfinite(self.fixed_covariance_scale) or (
            self.fixed_covariance_scale < 1.0
        ):
            raise ValueError(
                "fixed_covariance_scale must be finite and at least one."
            )
        if self.mode in {
            INTEGRITY_MODE_LEGACY_FIXED_SOFT,
            INTEGRITY_MODE_PROPORTIONAL_INFLATION,
            INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
        } and not _positive_finite(self.inflation_threshold):
            raise ValueError("Selected integrity mode requires inflation_threshold.")
        if self.mode in {
            INTEGRITY_MODE_HARD_GATE,
            INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
        } and not _positive_finite(self.hard_gate_threshold):
            raise ValueError("Selected integrity mode requires hard_gate_threshold.")


@dataclass(frozen=True)
class MeasurementIntegrityEvaluation:
    diagnostics: "MeasurementIntegrityDiagnostics"
    effective_measurement_covariance: np.ndarray
    innovation_covariance: np.ndarray
    skipped: bool


def evaluate_measurement_integrity(
    *, innovation: np.ndarray, predicted_measurement_covariance: np.ndarray,
    measurement_covariance: np.ndarray, policy: MeasurementIntegrityPolicy,
    regularization: float = 1e-9,
) -> MeasurementIntegrityEvaluation:
    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_measurement_covariance, dtype=float)
    covariance = np.asarray(measurement_covariance, dtype=float)
    identity = np.eye(innovation.size)
    innovation_covariance = predicted + covariance + regularization * identity
    raw_nis = float(
        innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation
    )
    scale = 1.0
    if (
        policy.mode == INTEGRITY_MODE_LEGACY_FIXED_SOFT
        and raw_nis > float(policy.inflation_threshold)
    ):
        scale = policy.fixed_covariance_scale
    elif policy.mode in {
        INTEGRITY_MODE_PROPORTIONAL_INFLATION,
        INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
    } and raw_nis > float(policy.inflation_threshold):
        scale = min(
            policy.maximum_covariance_scale,
            raw_nis / float(policy.inflation_threshold),
        )
    effective = covariance * scale
    if scale > 1.0:
        innovation_covariance = predicted + effective + regularization * identity
    processed_nis = float(
        innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation
    )
    skipped = bool(
        policy.mode in {
            INTEGRITY_MODE_HARD_GATE,
            INTEGRITY_MODE_PROPORTIONAL_WITH_HARD_GATE,
        }
        and processed_nis > float(policy.hard_gate_threshold)
    )
    status = (
        INTEGRITY_HARD_REJECTED if skipped
        else INTEGRITY_DOWNWEIGHTED if scale > 1.0
        else INTEGRITY_ACCEPTED
    )
    return MeasurementIntegrityEvaluation(
        diagnostics=MeasurementIntegrityDiagnostics(
            raw_nis=raw_nis, processed_nis=processed_nis,
            measurement_covariance_scale=scale, status=status,
        ),
        effective_measurement_covariance=effective,
        innovation_covariance=innovation_covariance,
        skipped=skipped,
    )


def _positive_finite(value: float | None) -> bool:
    return value is not None and np.isfinite(float(value)) and float(value) > 0.0


@dataclass(frozen=True)
class MeasurementIntegrityDiagnostics:
    """Estimator-independent diagnostics for one modality update."""

    raw_nis: float | None
    processed_nis: float | None
    measurement_covariance_scale: float
    status: str
    consecutive_anomaly_count: int = 0

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported measurement integrity status: {self.status}")
        if self.measurement_covariance_scale < 1.0:
            raise ValueError("measurement_covariance_scale must be at least one.")
        if self.consecutive_anomaly_count < 0:
            raise ValueError("consecutive_anomaly_count cannot be negative.")

    @property
    def accepted(self) -> bool:
        return self.status in {INTEGRITY_ACCEPTED, INTEGRITY_DOWNWEIGHTED}

    @property
    def anomalous(self) -> bool:
        return self.status in {INTEGRITY_DOWNWEIGHTED, INTEGRITY_HARD_REJECTED}

    def with_consecutive_anomaly_count(
        self, count: int,
    ) -> "MeasurementIntegrityDiagnostics":
        return replace(self, consecutive_anomaly_count=int(count))
