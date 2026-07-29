from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class InitialState:
    target_id: str
    timestamp: float
    state_estimate: Array
    covariance: Array


@dataclass
class Observation:
    timestamp: float
    observer_id: str
    target_id: str
    modality: str
    source_type: str
    measurement: Array
    covariance: Array
    confidence: float
    frame: str
    valid_flag: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterSatelliteObservation:
    timestamp: float
    source_node_id: str
    target_node_id: str
    modality: str
    measurement: Array
    covariance: Array
    confidence: float
    valid_flag: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AbsolutePositionObservation:
    timestamp: float
    satellite_id: str
    measurement_eci: Array
    covariance: Array
    confidence: float
    valid_flag: bool
    source_type: str = "GNSS"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FleetStateMessage:
    source_node_id: str
    timestamp: float
    node_ids: tuple[str, ...]
    state_estimate: Array
    covariance: Array
    quality_score: float
    valid_flag: bool = True
    source_timestamp: float | None = None
    arrival_timestamp: float | None = None


@dataclass
class ModuleInput:
    initial_state: InitialState
    sensor_measurements: list[Observation]
    config: dict[str, Any]
    node_reports: list["NodeReport"] = field(default_factory=list)


@dataclass
class LocalEstimate:
    modality: str
    timestamp: float
    state_estimate: Array
    acceleration: Array
    covariance: Array
    quality_score: float
    valid_flag: bool
    node_id: str | None = None
    target_id: str | None = None


@dataclass
class SingleFusionResult:
    node_id: str
    target_id: str
    timestamp: float
    state_estimate: Array
    acceleration: Array
    covariance: Array
    modality_weights: dict[str, float]
    modality_valid_flags: dict[str, bool]
    confidence_level: float


@dataclass
class NodeReport:
    node_id: str
    target_id: str
    timestamp: float
    state_estimate: Array
    covariance: Array
    quality_score: float
    health_status: str
    communication_delay: float
    valid_flag: bool
    # Communication timing fields for asynchronous fusion.
    source_timestamp: float | None = None
    arrival_timestamp: float | None = None


@dataclass
class StateOutput:
    timestamp: float
    target_id: str
    position_estimate: Array
    velocity_estimate: Array
    acceleration_estimate: Array
    covariance: Array
    valid_flag: bool
    confidence_level: float


@dataclass
class FusionStatus:
    modality_weights: dict[str, float] = field(default_factory=dict)
    modality_valid_flags: dict[str, bool] = field(default_factory=dict)
    node_weights: dict[str, float] = field(default_factory=dict)
    active_nodes: list[str] = field(default_factory=list)
    lost_nodes: list[str] = field(default_factory=list)


@dataclass
class AbnormalEvent:
    timestamp: float
    event_type: str
    severity: str
    description: str
    node_id: str | None = None
    target_id: str | None = None
    modality: str | None = None


@dataclass
class RuntimeStatus:
    processing_time: float
    observation_count: int
    active_modality_count: int
    active_node_count: int
    status: str


@dataclass
class ModuleOutput:
    state_output: StateOutput
    fusion_status: FusionStatus
    abnormal_events: list[AbnormalEvent]
    runtime_status: RuntimeStatus
