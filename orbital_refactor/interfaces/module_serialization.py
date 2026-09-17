"""Versioned, pickle-free persistence for public module input and output objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from interfaces.data_objects import (
    AbnormalEvent,
    FusionStatus,
    InitialState,
    ModuleInput,
    ModuleOutput,
    NodeReport,
    Observation,
    RuntimeStatus,
    StateOutput,
)
from interfaces.interface_contracts import validate_module_input


MODULE_BUNDLE_SCHEMA_VERSION = "v1.0"
MODULE_BUNDLE_MANIFEST = "module_bundle.json"
MODULE_BUNDLE_ARRAYS = "arrays.npz"

__all__ = [
    "MODULE_BUNDLE_SCHEMA_VERSION",
    "save_module_input",
    "load_module_input",
    "save_module_output",
    "load_module_output",
]


def save_module_input(value: ModuleInput, destination: str | Path) -> Path:
    """Save a validated public request as a portable JSON+NPZ directory."""

    validate_module_input(value)
    payload = {
        "bundle_type": "ModuleInput",
        "initial_state": _encode_record(value.initial_state),
        "sensor_measurements": [_encode_record(item) for item in value.sensor_measurements],
        "config": value.config,
        "node_reports": [_encode_record(item) for item in value.node_reports],
    }
    return _save_bundle(payload, destination)


def load_module_input(source: str | Path) -> ModuleInput:
    """Load and validate a public request bundle."""

    payload = _load_bundle(source, expected_type="ModuleInput")
    initial = payload["initial_state"]
    value = ModuleInput(
        initial_state=InitialState(**initial),
        sensor_measurements=[Observation(**item) for item in payload["sensor_measurements"]],
        config=payload["config"],
        node_reports=[NodeReport(**item) for item in payload["node_reports"]],
    )
    validate_module_input(value)
    return value


def save_module_output(value: ModuleOutput, destination: str | Path) -> Path:
    """Save the stable public result object as a portable JSON+NPZ directory."""

    payload = {
        "bundle_type": "ModuleOutput",
        "state_output": _encode_record(value.state_output),
        "fusion_status": _encode_record(value.fusion_status),
        "abnormal_events": [_encode_record(item) for item in value.abnormal_events],
        "runtime_status": _encode_record(value.runtime_status),
    }
    return _save_bundle(payload, destination)


def load_module_output(source: str | Path) -> ModuleOutput:
    """Load a public result bundle without executing estimator code."""

    payload = _load_bundle(source, expected_type="ModuleOutput")
    return ModuleOutput(
        state_output=StateOutput(**payload["state_output"]),
        fusion_status=FusionStatus(**payload["fusion_status"]),
        abnormal_events=[AbnormalEvent(**item) for item in payload["abnormal_events"]],
        runtime_status=RuntimeStatus(**payload["runtime_status"]),
    )


def _encode_record(value: Any) -> dict[str, Any]:
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def _save_bundle(payload: dict[str, Any], destination: str | Path) -> Path:
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Module bundle destination already exists: {target}")
    target.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {}
    encoded = {
        "schema_version": MODULE_BUNDLE_SCHEMA_VERSION,
        **_encode_value(payload, arrays, [0]),
    }
    np.savez_compressed(target / MODULE_BUNDLE_ARRAYS, **arrays)
    (target / MODULE_BUNDLE_MANIFEST).write_text(
        json.dumps(encoded, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def _load_bundle(source: str | Path, *, expected_type: str) -> dict[str, Any]:
    source = Path(source)
    manifest_path = source / MODULE_BUNDLE_MANIFEST
    arrays_path = source / MODULE_BUNDLE_ARRAYS
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing module bundle manifest: {manifest_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"Missing module bundle array archive: {arrays_path}")
    encoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = encoded.pop("schema_version", None)
    if version != MODULE_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected module bundle schema {MODULE_BUNDLE_SCHEMA_VERSION}, received {version!r}."
        )
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload = _restore_value(encoded, arrays)
    if payload.get("bundle_type") != expected_type:
        raise ValueError(
            f"Expected bundle type {expected_type!r}, received {payload.get('bundle_type')!r}."
        )
    return payload


def _encode_value(value: Any, arrays: dict[str, np.ndarray], counter: list[int]):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Module serialization does not support object arrays.")
        key = f"array_{counter[0]:04d}"
        counter[0] += 1
        arrays[key] = np.array(value, copy=True)
        return {"__array_key__": key, "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(name): _encode_value(item, arrays, counter) for name, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(item, arrays, counter) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item, arrays, counter) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Module JSON fields must contain finite numbers.")
        return value
    raise TypeError(f"Unsupported module bundle value: {type(value).__name__}")


def _restore_value(value: Any, arrays: Mapping[str, np.ndarray]):
    if isinstance(value, dict) and set(value) == {"__array_key__", "dtype", "shape"}:
        key = str(value["__array_key__"])
        if key not in arrays:
            raise ValueError(f"Module manifest references missing array {key!r}.")
        array = np.asarray(arrays[key])
        if array.dtype != np.dtype(value["dtype"]) or array.shape != tuple(value["shape"]):
            raise ValueError(f"Module array {key!r} failed dtype/shape validation.")
        return np.array(array, copy=True)
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_restore_value(item, arrays) for item in value["__tuple__"])
    if isinstance(value, dict):
        return {name: _restore_value(item, arrays) for name, item in value.items()}
    if isinstance(value, list):
        return [_restore_value(item, arrays) for item in value]
    return value
