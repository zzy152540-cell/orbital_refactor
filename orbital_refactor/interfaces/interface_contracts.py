"""Validation helpers for the public state-awareness data contract.

The validators are intentionally separate from the dataclasses: constructing a
message remains backwards compatible, while public processing entry points can
reject malformed data before it reaches an estimator.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from interfaces.data_objects import (
    InitialState, ModuleInput, NodeReport, Observation, RawSensorFrame,
)


MODULE_CONFIG_SCHEMA_VERSION = "v1.0"

__all__ = [
    "MODULE_CONFIG_SCHEMA_VERSION",
    "Modality",
    "ReferenceFrame",
    "RuntimeState",
    "Severity",
    "InterfaceErrorCode",
    "InterfaceValidationError",
    "canonical_modality",
    "validate_initial_state",
    "validate_observation",
    "validate_raw_sensor_frame",
    "validate_node_report",
    "validate_module_config",
    "validate_module_input",
]


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Modality(StringEnum):
    RADAR = "RADAR"
    INFRARED = "INFRARED"
    OPTICAL = "OPTICAL"
    ABSOLUTE_POSITION = "ABSOLUTE_POSITION"
    LEARNING = "LEARNING"


class ReferenceFrame(StringEnum):
    ECI = "ECI"
    ECI_RELATIVE = "ECI_RELATIVE"
    SPRI = "SPRI"
    BODY = "BODY"
    BODY_AZ_EL = "BODY_AZ_EL"
    CAMERA = "CAMERA"
    CAMERA_NORMALIZED = "CAMERA_NORMALIZED"


class RuntimeState(StringEnum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    PREDICTION_ONLY = "PREDICTION_ONLY"
    FAILED = "FAILED"


class Severity(StringEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class InterfaceErrorCode(StringEnum):
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    INVALID_STATE_SHAPE = "INVALID_STATE_SHAPE"
    INVALID_MEASUREMENT_SHAPE = "INVALID_MEASUREMENT_SHAPE"
    INVALID_RAW_SENSOR_DATA = "INVALID_RAW_SENSOR_DATA"
    INVALID_COVARIANCE = "INVALID_COVARIANCE"
    INVALID_CONFIDENCE = "INVALID_CONFIDENCE"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    INCOMPATIBLE_FRAME = "INCOMPATIBLE_FRAME"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    INVALID_CONFIG = "INVALID_CONFIG"


class InterfaceValidationError(ValueError):
    """Machine-readable public-interface validation failure."""

    def __init__(self, code: InterfaceErrorCode, message: str, *, field: str | None = None):
        self.code = code
        self.field = field
        prefix = f"{code.value}"
        if field is not None:
            prefix += f"[{field}]"
        super().__init__(f"{prefix}: {message}")


_MODALITY_ALIASES = {
    "RAD": Modality.RADAR.value,
    "IR": Modality.INFRARED.value,
    "OPT": Modality.OPTICAL.value,
    "TRADITIONAL_OPTICAL": Modality.OPTICAL.value,
    "NN": Modality.LEARNING.value,
    "LEARNING_OPTICAL": Modality.LEARNING.value,
}

_ALLOWED_FRAMES = {
    Modality.RADAR.value: {"ECI", "ECI_RELATIVE", "SPRI"},
    Modality.INFRARED.value: {"BODY", "BODY_AZ_EL", "SPRI"},
    Modality.OPTICAL.value: {
        "BODY", "CAMERA", "CAMERA_NORMALIZED", "SPRI", "ECI",
    },
    Modality.LEARNING.value: {"ECI", "ECI_RELATIVE", "SPRI"},
    Modality.ABSOLUTE_POSITION.value: {"ECI"},
}


def canonical_modality(value: str) -> str:
    candidate = str(value).strip().upper()
    candidate = _MODALITY_ALIASES.get(candidate, candidate)
    if candidate not in _ALLOWED_FRAMES:
        raise InterfaceValidationError(
            InterfaceErrorCode.UNSUPPORTED_MODALITY,
            f"unsupported modality {value!r}", field="modality",
        )
    return candidate


def validate_initial_state(initial: InitialState) -> None:
    _require_identifier(initial.target_id, "initial_state.target_id")
    _require_timestamp(initial.timestamp, "initial_state.timestamp")
    state = np.asarray(initial.state_estimate, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_STATE_SHAPE,
            "expected a finite ECI position-velocity vector with shape (6,)",
            field="initial_state.state_estimate",
        )
    _validate_covariance(initial.covariance, 6, "initial_state.covariance")


def validate_observation(observation: Observation) -> None:
    _require_timestamp(observation.timestamp, "observation.timestamp")
    _require_identifier(observation.observer_id, "observation.observer_id")
    _require_identifier(observation.target_id, "observation.target_id")
    modality = canonical_modality(observation.modality)
    frame = str(observation.frame).strip().upper()
    if frame not in _ALLOWED_FRAMES[modality]:
        raise InterfaceValidationError(
            InterfaceErrorCode.INCOMPATIBLE_FRAME,
            f"frame {observation.frame!r} is not valid for {modality}",
            field="observation.frame",
        )
    measurement = np.asarray(observation.measurement, dtype=float)
    if measurement.ndim != 1 or measurement.size == 0 or not np.all(np.isfinite(measurement)):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_MEASUREMENT_SHAPE,
            "measurement must be a non-empty finite one-dimensional vector",
            field="observation.measurement",
        )
    expected = _expected_measurement_dimension(modality, observation.metadata)
    if expected is not None and measurement.size != expected:
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_MEASUREMENT_SHAPE,
            f"{modality} expects {expected} components, received {measurement.size}",
            field="observation.measurement",
        )
    _validate_covariance(
        observation.covariance, measurement.size, "observation.covariance"
    )
    confidence = float(observation.confidence)
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_CONFIDENCE,
            "confidence must be finite and lie in [0, 1]",
            field="observation.confidence",
        )


def validate_raw_sensor_frame(frame: RawSensorFrame) -> None:
    _require_timestamp(frame.timestamp, "raw_sensor_frame.timestamp")
    _require_identifier(frame.observer_id, "raw_sensor_frame.observer_id")
    _require_identifier(frame.target_id, "raw_sensor_frame.target_id")
    modality = canonical_modality(frame.modality)
    expected_kind = {
        Modality.RADAR.value: "RANGE_DOPPLER_POWER_MAP",
        Modality.INFRARED.value: "INFRARED_POINT_SOURCE_IMAGE",
        Modality.OPTICAL.value: "OPTICAL_POINT_SOURCE_IMAGE",
    }.get(modality)
    if expected_kind is None or str(frame.data_kind) != expected_kind:
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_RAW_SENSOR_DATA,
            f"{modality} raw data requires data_kind {expected_kind!r}",
            field="raw_sensor_frame.data_kind",
        )
    data = np.asarray(frame.data, dtype=float)
    if data.ndim != 2 or min(data.shape) < 2 or not np.all(np.isfinite(data)):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_RAW_SENSOR_DATA,
            "raw sensor data must be a finite two-dimensional matrix",
            field="raw_sensor_frame.data",
        )
    required_axes = (
        {"range_m", "range_rate_mps"}
        if modality == Modality.RADAR.value else {"pixel_x", "pixel_y"}
    )
    if not required_axes.issubset(frame.axes):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_RAW_SENSOR_DATA,
            f"missing axes {sorted(required_axes - set(frame.axes))}",
            field="raw_sensor_frame.axes",
        )
    expected_lengths = {
        "pixel_x": data.shape[1], "pixel_y": data.shape[0],
        "range_m": data.shape[1], "range_rate_mps": data.shape[0],
    }
    for name in required_axes:
        axis = np.asarray(frame.axes[name], dtype=float)
        if axis.shape != (expected_lengths[name],) or not np.all(np.isfinite(axis)):
            raise InterfaceValidationError(
                InterfaceErrorCode.INVALID_RAW_SENSOR_DATA,
                f"axis {name!r} has an incompatible shape or non-finite values",
                field=f"raw_sensor_frame.axes.{name}",
            )
def validate_node_report(report: NodeReport, *, target_id: str | None = None) -> None:
    _require_identifier(report.node_id, "node_report.node_id")
    _require_identifier(report.target_id, "node_report.target_id")
    _require_timestamp(report.timestamp, "node_report.timestamp")
    for name in ("source_timestamp", "arrival_timestamp"):
        value = getattr(report, name)
        if value is not None:
            _require_timestamp(value, f"node_report.{name}")
    if target_id is not None and str(report.target_id) != str(target_id):
        raise InterfaceValidationError(
            InterfaceErrorCode.TARGET_MISMATCH,
            f"report target {report.target_id!r} differs from {target_id!r}",
            field="node_report.target_id",
        )
    state = np.asarray(report.state_estimate, dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_STATE_SHAPE,
            "expected a finite state vector with shape (6,)",
            field="node_report.state_estimate",
        )
    _validate_covariance(report.covariance, 6, "node_report.covariance")
    quality = float(report.quality_score)
    if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_CONFIDENCE,
            "quality_score must be finite and lie in [0, 1]",
            field="node_report.quality_score",
        )
    delay = float(report.communication_delay)
    if not np.isfinite(delay) or delay < 0.0:
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_TIMESTAMP,
            "communication_delay must be finite and non-negative",
            field="node_report.communication_delay",
        )


def validate_module_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_CONFIG, "config must be a dictionary", field="config"
        )
    version = config.get("schema_version")
    if version is not None and version != MODULE_CONFIG_SCHEMA_VERSION:
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_CONFIG,
            f"expected schema_version {MODULE_CONFIG_SCHEMA_VERSION}, received {version!r}",
            field="config.schema_version",
        )


def validate_module_input(module_input: ModuleInput) -> None:
    if not isinstance(module_input, ModuleInput):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_CONFIG,
            "public entry point requires a ModuleInput instance", field="module_input",
        )
    validate_initial_state(module_input.initial_state)
    validate_module_config(module_input.config)
    for observation in module_input.sensor_measurements:
        validate_observation(observation)
    target_id = str(module_input.initial_state.target_id)
    for report in module_input.node_reports:
        validate_node_report(report, target_id=target_id)


def _expected_measurement_dimension(modality: str, metadata: dict[str, Any]) -> int | None:
    measurement_type = str(metadata.get("measurement_type", "")).upper()
    if modality in {Modality.RADAR.value, Modality.INFRARED.value}:
        return 2
    if modality == Modality.ABSOLUTE_POSITION.value:
        return 3
    if modality in {Modality.OPTICAL.value, Modality.LEARNING.value}:
        if measurement_type in {"RELATIVE_POSITION_ECI", "POSITION_ECI"}:
            return 3
        if measurement_type in {"RELATIVE_STATE_ECI", "STATE_ECI"}:
            return 6
        if measurement_type in {"NORMALIZED_IMAGE_COORDINATES", "IMAGE_COORDINATES"}:
            return 2
        # Legacy optical/learning inputs predate explicit measurement_type.
        return None
    return None


def _require_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_IDENTIFIER,
            "identifier must be a non-empty string", field=field,
        )


def _require_timestamp(value: float, field: str) -> None:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_TIMESTAMP, "timestamp must be numeric", field=field,
        ) from exc
    if not np.isfinite(timestamp):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_TIMESTAMP, "timestamp must be finite", field=field,
        )


def _validate_covariance(value: Any, dimension: int, field: str) -> None:
    covariance = np.asarray(value, dtype=float)
    if covariance.shape != (dimension, dimension) or not np.all(np.isfinite(covariance)):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_COVARIANCE,
            f"expected a finite matrix with shape ({dimension}, {dimension})", field=field,
        )
    if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12):
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_COVARIANCE, "matrix must be symmetric", field=field,
        )
    tolerance = max(1.0, float(np.linalg.norm(covariance, ord=2))) * 1e-10
    if float(np.min(np.linalg.eigvalsh(covariance))) < -tolerance:
        raise InterfaceValidationError(
            InterfaceErrorCode.INVALID_COVARIANCE,
            "matrix must be positive semidefinite", field=field,
        )
