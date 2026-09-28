from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .track_lifecycle import TrackLifecycle


Array = np.ndarray
ECI_FRAME = "J2000_ECI"


def _identifier(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must be a non-empty identifier.")
    return result


def _timestamp(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("timestamp must be finite.")
    return result


def _state(value: Array) -> Array:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError("state_eci must be a finite position-velocity vector with shape (6,).")
    result.setflags(write=False)
    return result


def _covariance(value: Array) -> Array:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (6, 6) or not np.all(np.isfinite(result)):
        raise ValueError("covariance_eci must be a finite matrix with shape (6, 6).")
    if not np.allclose(result, result.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance_eci must be symmetric.")
    if np.linalg.eigvalsh(result).min() < -1e-10:
        raise ValueError("covariance_eci must be positive semidefinite.")
    result.setflags(write=False)
    return result


def _frame(value: str) -> str:
    result = str(value).strip().upper()
    if result != ECI_FRAME:
        raise ValueError(f"Public target states must use frame {ECI_FRAME!r}.")
    return result


def _quality(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("quality_score must lie in [0, 1].")
    return result


@dataclass(frozen=True)
class TargetTrackKey:
    scene_id: str
    track_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _identifier(self.scene_id, "scene_id"))
        object.__setattr__(self, "track_id", _identifier(self.track_id, "track_id"))


@dataclass(frozen=True)
class ObserverState:
    observer_id: str
    timestamp: float
    state_eci: Array
    covariance_eci: Array
    frame: str = ECI_FRAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "observer_id", _identifier(self.observer_id, "observer_id"))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        object.__setattr__(self, "state_eci", _state(self.state_eci))
        object.__setattr__(self, "covariance_eci", _covariance(self.covariance_eci))
        object.__setattr__(self, "frame", _frame(self.frame))


@dataclass(frozen=True)
class TargetInitialState:
    target_id: str
    track_id: str
    timestamp: float
    state_eci: Array
    covariance_eci: Array
    frame: str = ECI_FRAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target_id"))
        object.__setattr__(self, "track_id", _identifier(self.track_id, "track_id"))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        object.__setattr__(self, "state_eci", _state(self.state_eci))
        object.__setattr__(self, "covariance_eci", _covariance(self.covariance_eci))
        object.__setattr__(self, "frame", _frame(self.frame))


@dataclass(frozen=True)
class LocalTargetEstimate:
    observer_id: str
    target_id: str
    track_id: str
    timestamp: float
    state_eci: Array
    covariance_eci: Array
    quality_score: float
    valid_flag: bool = True
    frame: str = ECI_FRAME
    modality_weights: Mapping[str, float] = field(default_factory=dict)
    used_measurement_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observer_id", _identifier(self.observer_id, "observer_id"))
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target_id"))
        object.__setattr__(self, "track_id", _identifier(self.track_id, "track_id"))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        object.__setattr__(self, "state_eci", _state(self.state_eci))
        object.__setattr__(self, "covariance_eci", _covariance(self.covariance_eci))
        object.__setattr__(self, "quality_score", _quality(self.quality_score))
        object.__setattr__(self, "frame", _frame(self.frame))
        object.__setattr__(self, "modality_weights", dict(self.modality_weights))
        object.__setattr__(self, "used_measurement_ids", tuple(map(str, self.used_measurement_ids)))


@dataclass(frozen=True)
class TargetNodeReport(LocalTargetEstimate):
    """One observer's absolute J2000 posterior for one physical target."""


@dataclass(frozen=True)
class GlobalTargetEstimate:
    target_id: str
    track_id: str
    timestamp: float
    state_eci: Array
    covariance_eci: Array
    contributing_observer_ids: tuple[str, ...]
    node_weights: Mapping[str, float]
    valid_flag: bool = True
    frame: str = ECI_FRAME

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target_id"))
        object.__setattr__(self, "track_id", _identifier(self.track_id, "track_id"))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        object.__setattr__(self, "state_eci", _state(self.state_eci))
        object.__setattr__(self, "covariance_eci", _covariance(self.covariance_eci))
        observers = tuple(_identifier(value, "contributing_observer_ids") for value in self.contributing_observer_ids)
        if len(set(observers)) != len(observers):
            raise ValueError("contributing_observer_ids must be unique.")
        object.__setattr__(self, "contributing_observer_ids", observers)
        object.__setattr__(self, "node_weights", dict(self.node_weights))
        object.__setattr__(self, "frame", _frame(self.frame))


@dataclass(frozen=True)
class TargetTrack:
    key: TargetTrackKey
    target_id: str | None
    lifecycle: TrackLifecycle
    estimate: GlobalTargetEstimate | TargetInitialState
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_id is not None:
            object.__setattr__(self, "target_id", _identifier(self.target_id, "target_id"))
            if self.estimate.target_id != self.target_id:
                raise ValueError("TargetTrack target_id must match its estimate target_id.")
        if self.estimate.track_id != self.key.track_id:
            raise ValueError("TargetTrack key.track_id must match its estimate track_id.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class MultiTargetInput:
    scene_id: str
    observer_states: Mapping[str, ObserverState]
    target_initial_states: Mapping[str, TargetInitialState]
    observations: tuple[Any, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _identifier(self.scene_id, "scene_id"))
        observers = dict(self.observer_states)
        targets = dict(self.target_initial_states)
        if any(key != value.observer_id for key, value in observers.items()):
            raise ValueError("observer_states keys must match ObserverState.observer_id.")
        if any(key != value.target_id for key, value in targets.items()):
            raise ValueError("target_initial_states keys must match TargetInitialState.target_id.")
        if set(observers) & set(targets):
            raise ValueError("Observer and target identifiers must belong to disjoint sets.")
        object.__setattr__(self, "observer_states", observers)
        object.__setattr__(self, "target_initial_states", targets)
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "config", dict(self.config))


@dataclass(frozen=True)
class MultiTargetOutput:
    scene_id: str
    timestamp: float
    estimates_by_target: Mapping[str, GlobalTargetEstimate]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scene_id", _identifier(self.scene_id, "scene_id"))
        object.__setattr__(self, "timestamp", _timestamp(self.timestamp))
        estimates = dict(self.estimates_by_target)
        if any(key != value.target_id for key, value in estimates.items()):
            raise ValueError("estimates_by_target keys must match GlobalTargetEstimate.target_id.")
        if any(not np.isclose(value.timestamp, self.timestamp) for value in estimates.values()):
            raise ValueError("Every global estimate must align with MultiTargetOutput.timestamp.")
        object.__setattr__(self, "estimates_by_target", estimates)
