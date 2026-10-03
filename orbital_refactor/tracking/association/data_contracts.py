from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class UnlabeledObservationMessage:
    """Raw sensor detection that deliberately contains no external target ID."""

    message_id: str
    observer_id: str
    timestamp: float
    modality: str
    measurement: np.ndarray
    covariance: np.ndarray
    detection_group_id: str
    confidence: float = 1.0
    valid_flag: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        message_id = str(self.message_id).strip()
        observer_id = str(self.observer_id).strip()
        group_id = str(self.detection_group_id).strip()
        modality = str(self.modality).upper()
        if not message_id or not observer_id or not group_id:
            raise ValueError("Raw detection identifiers must be non-empty.")
        if modality not in {"RADAR", "OPTICAL", "INFRARED"}:
            raise ValueError("Raw detection modality is unsupported.")
        measurement = np.array(self.measurement, dtype=float, copy=True).reshape(-1)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        if measurement.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("Raw detection measurement and covariance must be 2-D.")
        if np.any(~np.isfinite(measurement)) or np.any(~np.isfinite(covariance)):
            raise ValueError("Raw detection values must be finite.")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("Raw detection covariance must be symmetric.")
        if np.linalg.eigvalsh(covariance).min() <= 0.0:
            raise ValueError("Raw detection covariance must be positive definite.")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Raw detection confidence must lie in [0, 1].")
        measurement.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "message_id", message_id)
        object.__setattr__(self, "observer_id", observer_id)
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "detection_group_id", group_id)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class UnlabeledIODObservation:
    """One target-originated measurement before a physical target is identified."""

    timestamp: float
    observer_id: str
    modality: str
    observer_state_eci: np.ndarray
    measurement: np.ndarray
    covariance: np.ndarray
    valid_flag: bool = True
    detection_group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        modality = str(self.modality).upper()
        if modality not in {"LOS", "RADAR"}:
            raise ValueError("Association modality must be LOS or RADAR.")
        state = np.array(self.observer_state_eci, dtype=float, copy=True).reshape(6)
        measurement = np.array(self.measurement, dtype=float, copy=True).reshape(-1)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        expected = 3 if modality == "LOS" else 2
        if measurement.shape != (expected,) or covariance.shape != (2, 2):
            raise ValueError(f"Invalid {modality} measurement or covariance shape.")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(measurement)):
            raise ValueError("Association observation values must be finite.")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("Association covariance must be symmetric.")
        if np.linalg.eigvalsh(covariance).min() <= 0.0:
            raise ValueError("Association covariance must be positive definite.")
        if modality == "LOS":
            norm = float(np.linalg.norm(measurement))
            if norm <= 0.0:
                raise ValueError("LOS measurement must be nonzero.")
            measurement /= norm
        elif measurement[0] <= 0.0:
            raise ValueError("RADAR range must be positive.")
        for value in (state, measurement, covariance):
            value.setflags(write=False)
        group = None if self.detection_group_id is None else str(self.detection_group_id).strip()
        if group == "":
            raise ValueError("detection_group_id must be non-empty when supplied.")
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "observer_id", str(self.observer_id))
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "observer_state_eci", state)
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "detection_group_id", group)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class AssociationMatch:
    observation_index: int
    target_id: str
    squared_mahalanobis: float


@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[AssociationMatch, ...]
    unassigned_observation_indices: tuple[int, ...]
    unobserved_target_ids: tuple[str, ...]

