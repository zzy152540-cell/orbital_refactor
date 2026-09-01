from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from interfaces.data_objects import Observation

Array = np.ndarray
CANNMeasurementMode = Literal["PREPROCESSED", "PROPAGATED"]


@dataclass(frozen=True)
class CANNMeasurementProposal:
    """Boundary object between a CANN front end and the unchanged filter.

    The proposal contains only measurement-domain data.  It intentionally has
    no estimator state, covariance-state or CI-weight fields, so a CANN cannot
    mutate the filtering core through this interface.
    """

    timestamp: float
    observer_id: str
    target_id: str
    modality: str
    measurement: Array
    covariance: Array
    confidence: float
    frame: str
    valid_flag: bool
    mode: CANNMeasurementMode
    source_measurement_ids: tuple[str, ...]
    propagation_duration: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        measurement = np.asarray(self.measurement, dtype=float).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=float)
        if not np.isfinite(self.timestamp):
            raise ValueError("CANN proposal timestamp must be finite.")
        if not measurement.size or np.any(~np.isfinite(measurement)):
            raise ValueError("CANN proposal measurement must be finite and nonempty.")
        if covariance.shape != (measurement.size, measurement.size):
            raise ValueError("CANN proposal covariance has incompatible dimensions.")
        if np.any(~np.isfinite(covariance)) or not np.allclose(
            covariance, covariance.T,
        ):
            raise ValueError("CANN proposal covariance must be finite and symmetric.")
        if np.min(np.linalg.eigvalsh(covariance)) < 0.0:
            raise ValueError("CANN proposal covariance must be positive semidefinite.")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("CANN proposal confidence must lie in [0, 1].")
        if not self.source_measurement_ids:
            raise ValueError("CANN proposal must preserve source lineage.")
        if not np.isfinite(self.propagation_duration) or self.propagation_duration < 0.0:
            raise ValueError("CANN propagation duration must be finite and nonnegative.")
        if self.mode == "PREPROCESSED" and self.propagation_duration != 0.0:
            raise ValueError("Preprocessed measurements cannot claim propagation time.")

    def to_observation(self) -> Observation:
        self.validate()
        metadata = {
            "cann_mode": self.mode,
            "source_measurement_ids": self.source_measurement_ids,
            "propagation_duration": float(self.propagation_duration),
            "cann_diagnostics": dict(self.diagnostics),
        }
        return Observation(
            timestamp=float(self.timestamp), observer_id=self.observer_id,
            target_id=self.target_id, modality=self.modality,
            source_type=f"CANN_{self.mode}",
            measurement=np.asarray(self.measurement, dtype=float).copy(),
            covariance=np.asarray(self.covariance, dtype=float).copy(),
            confidence=float(self.confidence), frame=self.frame,
            valid_flag=bool(self.valid_flag), metadata=metadata,
        )


def preprocess_observation(
    source: Observation, *, measurement: Array,
    covariance: Array | None = None, diagnostics: dict[str, Any] | None = None,
) -> Observation:
    """Build a lineage-preserving CANN proposal from one real measurement."""
    source_id = str(source.metadata.get(
        "observation_id",
        f"{source.observer_id}:{source.target_id}:{source.modality}:"
        f"{float(source.timestamp):.12g}",
    ))
    proposal = CANNMeasurementProposal(
        timestamp=source.timestamp, observer_id=source.observer_id,
        target_id=source.target_id, modality=source.modality,
        measurement=np.asarray(measurement, dtype=float),
        covariance=(source.covariance if covariance is None else covariance),
        confidence=source.confidence, frame=source.frame,
        valid_flag=source.valid_flag, mode="PREPROCESSED",
        source_measurement_ids=(source_id,), diagnostics=diagnostics or {},
    )
    result = proposal.to_observation()
    result.metadata = {**source.metadata, **result.metadata}
    return result
