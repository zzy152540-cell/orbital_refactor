"""Versioned, algorithm-independent data contract for visualization clients."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


Array = np.ndarray
VISUALIZATION_SCHEMA_VERSION = "v1.0"


def _readonly_array(value: Any, *, ndim: int | None = None) -> Array:
    result = np.array(value, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-D array, got shape {result.shape}.")
    if result.dtype.kind in "fc" and np.any(~np.isfinite(result)):
        raise ValueError("Visualization arrays must contain finite values.")
    result.setflags(write=False)
    return result


def _optional_array(value: Any | None, *, ndim: int | None = None) -> Array | None:
    return None if value is None else _readonly_array(value, ndim=ndim)


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class VisualNodeState:
    node_id: str
    estimate_state: Array
    covariance_diagonal: Array
    truth_state: Array | None = None
    covariance: Array | None = None
    valid: bool = True
    status: str = "OK"

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("Visual node IDs cannot be empty.")
        estimate = _readonly_array(self.estimate_state, ndim=1)
        covariance = _readonly_array(self.covariance_diagonal, ndim=1)
        truth = _optional_array(self.truth_state, ndim=1)
        full_covariance = _optional_array(self.covariance, ndim=2)
        if covariance.shape != estimate.shape:
            raise ValueError("Covariance diagonal must match the state dimension.")
        if truth is not None and truth.shape != estimate.shape:
            raise ValueError("Truth and estimate states must have the same shape.")
        if np.any(covariance < 0.0):
            raise ValueError("Covariance diagonal cannot be negative.")
        if full_covariance is not None:
            expected_shape = (estimate.size, estimate.size)
            if full_covariance.shape != expected_shape:
                raise ValueError("Full covariance has the wrong shape.")
            if not np.allclose(np.diag(full_covariance), covariance):
                raise ValueError("Full covariance diagonal is inconsistent.")
        object.__setattr__(self, "estimate_state", estimate)
        object.__setattr__(self, "covariance_diagonal", covariance)
        object.__setattr__(self, "truth_state", truth)
        object.__setattr__(self, "covariance", full_covariance)


@dataclass(frozen=True)
class VisualEdge:
    source_node_id: str
    target_node_id: str
    edge_type: str
    status: str = "ACTIVE"
    modality: str | None = None
    directed: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_node_id or not self.target_node_id:
            raise ValueError("Visual edge node IDs cannot be empty.")
        if self.source_node_id == self.target_node_id:
            raise ValueError("A visual edge requires two different nodes.")
        if not self.edge_type:
            raise ValueError("Visual edge type cannot be empty.")
        object.__setattr__(self, "metrics", _metadata(self.metrics))


@dataclass(frozen=True)
class VisualObservation:
    observer_id: str
    target_id: str
    modality: str
    measurement: Array
    covariance_diagonal: Array
    visible: bool
    valid: bool
    processing_status: str
    frame: str
    nis: float | None = None
    raw_sensor_data: Array | None = None
    raw_data_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observer_id == self.target_id:
            raise ValueError("A visual relative observation requires two nodes.")
        if not self.modality or not self.frame:
            raise ValueError("Observation modality and frame cannot be empty.")
        measurement = _readonly_array(self.measurement, ndim=1)
        covariance = _readonly_array(self.covariance_diagonal, ndim=1)
        if measurement.shape != covariance.shape:
            raise ValueError("Measurement covariance diagonal has the wrong shape.")
        if np.any(covariance < 0.0):
            raise ValueError("Measurement covariance diagonal cannot be negative.")
        if self.nis is not None and (not np.isfinite(self.nis) or self.nis < 0.0):
            raise ValueError("NIS must be finite and nonnegative when provided.")
        raw = _optional_array(self.raw_sensor_data)
        if (raw is None) != (self.raw_data_kind is None):
            raise ValueError("Raw sensor data and its kind must be provided together.")
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "covariance_diagonal", covariance)
        object.__setattr__(self, "raw_sensor_data", raw)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class VisualNavigationState:
    node_id: str
    absolute_navigation_available: bool
    last_absolute_navigation_timestamp: float | None = None
    anchor_age: float | None = None
    orbital_phase: float | None = None
    radial_along_track: Array | None = None
    reference_frame: str = "RT"
    status: str = "OK"
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("Navigation node ID cannot be empty.")
        for value in (
            self.last_absolute_navigation_timestamp,
            self.anchor_age,
            self.orbital_phase,
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError("Navigation scalars must be finite when provided.")
        if self.anchor_age is not None and self.anchor_age < 0.0:
            raise ValueError("Anchor age cannot be negative.")
        rt = _optional_array(self.radial_along_track, ndim=1)
        if rt is not None and rt.shape != (2,):
            raise ValueError("Radial/along-track state must contain two values.")
        object.__setattr__(self, "radial_along_track", rt)
        object.__setattr__(self, "metrics", _metadata(self.metrics))


@dataclass(frozen=True)
class VisualCANNSnapshot:
    node_id: str
    representation: str
    available: bool
    activity: Array | None = None
    decoded_value: Array | None = None
    concentration: float | None = None
    bump_width: float | None = None
    anchor_age: float | None = None
    cue_applied: bool = False
    status: str = "OK"
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id or not self.representation:
            raise ValueError("CANN node and representation cannot be empty.")
        activity = _optional_array(self.activity)
        decoded = _optional_array(self.decoded_value, ndim=1)
        if not self.available and (activity is not None or decoded is not None):
            raise ValueError("Unavailable CANN snapshots cannot contain activity.")
        for value in (self.concentration, self.bump_width, self.anchor_age):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError("CANN quality scalars must be finite and nonnegative.")
        object.__setattr__(self, "activity", activity)
        object.__setattr__(self, "decoded_value", decoded)
        object.__setattr__(self, "metrics", _metadata(self.metrics))


@dataclass(frozen=True)
class VisualDiagnosticEvent:
    timestamp: float
    event_type: str
    severity: str
    description: str
    node_id: str | None = None
    target_id: str | None = None
    modality: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("Visual event timestamp must be finite and nonnegative.")
        if not self.event_type or not self.severity or not self.description:
            raise ValueError("Visual event type, severity and description are required.")


@dataclass(frozen=True)
class VisualizationFrame:
    """One synchronized, read-only display snapshot.

    This object intentionally has no callback or mutable algorithm reference.
    It may contain truth for evaluation, so it must never be passed back to an
    estimator, topology policy, or CANN update path.
    """

    scenario_id: str
    run_id: str
    timestamp: float
    epoch_index: int
    nodes: tuple[VisualNodeState, ...]
    edges: tuple[VisualEdge, ...] = ()
    observations: tuple[VisualObservation, ...] = ()
    navigation: tuple[VisualNavigationState, ...] = ()
    cann: tuple[VisualCANNSnapshot, ...] = ()
    events: tuple[VisualDiagnosticEvent, ...] = ()
    schema_version: str = VISUALIZATION_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != VISUALIZATION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported visualization schema: {self.schema_version}."
            )
        if not self.scenario_id or not self.run_id:
            raise ValueError("Scenario and run IDs cannot be empty.")
        if not np.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("Frame timestamp must be finite and nonnegative.")
        if self.epoch_index < 0:
            raise ValueError("Frame epoch index cannot be negative.")
        node_ids = tuple(node.node_id for node in self.nodes)
        if not node_ids or len(node_ids) != len(set(node_ids)):
            raise ValueError("A visualization frame requires unique nodes.")
        known = set(node_ids)
        for edge in self.edges:
            if {edge.source_node_id, edge.target_node_id} - known:
                raise ValueError("Visual edge references an unknown node.")
        for observation in self.observations:
            if {observation.observer_id, observation.target_id} - known:
                raise ValueError("Visual observation references an unknown node.")
        for item in (*self.navigation, *self.cann):
            if item.node_id not in known:
                raise ValueError("Visual auxiliary state references an unknown node.")
        for event in self.events:
            if not np.isclose(event.timestamp, self.timestamp, atol=1e-9, rtol=0.0):
                raise ValueError("Frame events must match the frame timestamp.")
            if event.node_id is not None and event.node_id not in known:
                raise ValueError("Visual event references an unknown node.")
            if event.target_id is not None and event.target_id not in known:
                raise ValueError("Visual event references an unknown target.")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "navigation", tuple(self.navigation))
        object.__setattr__(self, "cann", tuple(self.cann))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


_VISUAL_TYPES = {
    item.__name__: item for item in (
        VisualNodeState,
        VisualEdge,
        VisualObservation,
        VisualNavigationState,
        VisualCANNSnapshot,
        VisualDiagnosticEvent,
        VisualizationFrame,
    )
}


def visualization_frame_to_dict(frame: VisualizationFrame) -> dict[str, Any]:
    """Convert one frame to a portable, JSON-compatible representation."""
    if not isinstance(frame, VisualizationFrame):
        raise TypeError("Expected a VisualizationFrame.")
    return _encode_value(frame)


def visualization_frame_from_dict(payload: Mapping[str, Any]) -> VisualizationFrame:
    """Restore and revalidate a frame created by ``visualization_frame_to_dict``."""
    result = _decode_value(dict(payload))
    if not isinstance(result, VisualizationFrame):
        raise ValueError("Payload does not contain a VisualizationFrame.")
    return result


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__array__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            **{
                item.name: _encode_value(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Visualization metadata keys must be strings.")
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported visualization value: {type(value).__name__}.")


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode_value(item) for item in value)
    if not isinstance(value, dict):
        return value
    if value.get("__array__") is True:
        required = {"__array__", "dtype", "shape", "data"}
        if set(value) != required:
            raise ValueError("Malformed visualization array payload.")
        result = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        expected_shape = tuple(int(item) for item in value["shape"])
        if result.shape != expected_shape:
            raise ValueError("Visualization array shape does not match payload.")
        return result
    type_name = value.get("__type__")
    if type_name is not None:
        selected = _VISUAL_TYPES.get(str(type_name))
        if selected is None:
            raise ValueError(f"Unknown visualization payload type: {type_name}.")
        return selected(**{
            key: _decode_value(item)
            for key, item in value.items() if key != "__type__"
        })
    return {key: _decode_value(item) for key, item in value.items()}
