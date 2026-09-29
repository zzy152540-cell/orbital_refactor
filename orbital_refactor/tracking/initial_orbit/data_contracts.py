from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class IODStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    POOR_GEOMETRY = "POOR_GEOMETRY"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass(frozen=True)
class IODObservation:
    timestamp: float
    observer_id: str
    target_id: str
    modality: str
    observer_state_eci: np.ndarray
    measurement: np.ndarray
    covariance: np.ndarray
    valid_flag: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        modality = str(self.modality).upper()
        if modality not in {"LOS", "RADAR"}:
            raise ValueError("IOD modality must be LOS or RADAR.")
        state = np.array(self.observer_state_eci, dtype=float, copy=True).reshape(6)
        measurement = np.array(self.measurement, dtype=float, copy=True).reshape(-1)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        expected = 3 if modality == "LOS" else 2
        covariance_dimension = 2
        if measurement.shape != (expected,) or np.any(~np.isfinite(measurement)):
            raise ValueError(f"{modality} measurement has an invalid shape or value.")
        if covariance.shape != (covariance_dimension, covariance_dimension):
            raise ValueError(f"{modality} covariance must have shape (2, 2).")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
            raise ValueError("IOD covariance must be symmetric.")
        if np.linalg.eigvalsh(covariance).min() <= 0.0:
            raise ValueError("IOD covariance must be positive definite.")
        if modality == "LOS":
            norm = float(np.linalg.norm(measurement))
            if norm <= 0.0:
                raise ValueError("LOS measurement must be nonzero.")
            measurement /= norm
        elif measurement[0] <= 0.0:
            raise ValueError("RADAR range must be positive.")
        for value in (state, measurement, covariance):
            value.setflags(write=False)
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "observer_id", str(self.observer_id))
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "observer_state_eci", state)
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class InitialOrbitEstimate:
    target_id: str
    track_id: str
    epoch: float
    state_eci: np.ndarray | None
    covariance_eci: np.ndarray | None
    status: IODStatus
    method: str
    converged: bool
    quality_score: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.state_eci is not None:
            state = np.array(self.state_eci, dtype=float, copy=True).reshape(6)
            state.setflags(write=False)
            object.__setattr__(self, "state_eci", state)
        if self.covariance_eci is not None:
            covariance = np.array(self.covariance_eci, dtype=float, copy=True).reshape(6, 6)
            covariance = 0.5 * (covariance + covariance.T)
            covariance.setflags(write=False)
            object.__setattr__(self, "covariance_eci", covariance)
        object.__setattr__(self, "status", IODStatus(self.status))
        object.__setattr__(self, "quality_score", float(np.clip(self.quality_score, 0.0, 1.0)))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
