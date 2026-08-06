from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CovarianceTransportEvent:
    timestamp: float
    state_estimate: Array
    error_transition: Array
    independent_process_noise: Array
    information_ids: tuple[str, ...] = ()


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
class StateMessage:
    """A state estimate sent by one node about one physical target."""

    source_node_id: str
    target_node_id: str
    timestamp: float
    state_estimate: Array
    covariance: Array
    quality_score: float
    valid_flag: bool = True
    source_timestamp: float | None = None
    arrival_timestamp: float | None = None
    information_ids: tuple[str, ...] = ()
    lineage_id: str | None = None
    reference_timestamp: float | None = None
    error_transition: Array | None = None
    accumulated_process_noise: Array | None = None
    reference_state_estimate: Array | None = None
    reference_covariance: Array | None = None
    transport_events: tuple[CovarianceTransportEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationMessage:
    """A communicable, directed relative observation."""

    message_id: str
    observer_id: str
    target_id: str
    timestamp: float
    modality: str
    measurement: Array
    covariance: Array
    frame: str = "ECI"
    confidence: float = 1.0
    valid_flag: bool = True
    source_timestamp: float | None = None
    arrival_timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    physical_observation_id: str | None = None

    @property
    def information_id(self) -> str:
        return (
            str(self.message_id)
            if self.physical_observation_id is None
            else str(self.physical_observation_id)
        )


@dataclass
class TargetEstimate:
    """One estimator's posterior for a specifically identified target."""

    estimator_node_id: str
    target_node_id: str
    timestamp: float
    state_estimate: Array
    covariance: Array
    quality_score: float
    valid_flag: bool = True
    information_ids: tuple[str, ...] = ()


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
    observation_id: str | None = None
    arrival_timestamp: float | None = None

    @property
    def information_id(self) -> str:
        return (
            str(self.observation_id)
            if self.observation_id is not None
            else f"{self.satellite_id}:absolute:{float(self.timestamp):.12g}"
        )


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
