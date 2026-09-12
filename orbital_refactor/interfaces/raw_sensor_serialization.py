"""Versioned, pickle-free persistence for the public RawSensorFrame contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from interfaces.data_objects import RawSensorFrame
from interfaces.interface_contracts import validate_raw_sensor_frame


RAW_SENSOR_FRAME_SCHEMA_VERSION = "v1.0"
RAW_SENSOR_FRAME_MANIFEST = "raw_sensor_frame.json"
RAW_SENSOR_FRAME_ARRAYS = "arrays.npz"

__all__ = [
    "RAW_SENSOR_FRAME_SCHEMA_VERSION",
    "save_raw_sensor_frame",
    "load_raw_sensor_frame",
]


def save_raw_sensor_frame(frame: RawSensorFrame, destination: str | Path) -> Path:
    """Save one frame as human-readable JSON plus compressed numeric arrays."""

    validate_raw_sensor_frame(frame)
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Raw sensor frame destination already exists: {target}")
    target.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {}
    counter = [0]
    payload = {
        "schema_version": RAW_SENSOR_FRAME_SCHEMA_VERSION,
        "timestamp": float(frame.timestamp),
        "observer_id": str(frame.observer_id),
        "target_id": str(frame.target_id),
        "modality": str(frame.modality),
        "data_kind": str(frame.data_kind),
        "valid_flag": bool(frame.valid_flag),
        "data": _externalize_array(np.asarray(frame.data), arrays, counter),
        "axes": {
            str(name): _externalize_array(np.asarray(value), arrays, counter)
            for name, value in frame.axes.items()
        },
        "calibration": _encode_value(frame.calibration, arrays, counter),
        "metadata": _encode_value(frame.metadata, arrays, counter),
    }
    np.savez_compressed(target / RAW_SENSOR_FRAME_ARRAYS, **arrays)
    (target / RAW_SENSOR_FRAME_MANIFEST).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return target


def load_raw_sensor_frame(source: str | Path) -> RawSensorFrame:
    """Load, integrity-check and validate one saved raw sensor frame."""

    source = Path(source)
    manifest_path = source / RAW_SENSOR_FRAME_MANIFEST
    arrays_path = source / RAW_SENSOR_FRAME_ARRAYS
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing raw sensor manifest: {manifest_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"Missing raw sensor array archive: {arrays_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != RAW_SENSOR_FRAME_SCHEMA_VERSION:
        raise ValueError(
            f"Expected raw sensor schema {RAW_SENSOR_FRAME_SCHEMA_VERSION}, "
            f"received {version!r}."
        )
    required = {
        "timestamp", "observer_id", "target_id", "modality", "data_kind",
        "valid_flag", "data", "axes", "calibration", "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Raw sensor manifest is missing fields: {sorted(missing)}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    frame = RawSensorFrame(
        timestamp=float(payload["timestamp"]),
        observer_id=str(payload["observer_id"]),
        target_id=str(payload["target_id"]),
        modality=str(payload["modality"]),
        data=_restore_value(payload["data"], arrays),
        data_kind=str(payload["data_kind"]),
        axes={
            str(name): _restore_value(value, arrays)
            for name, value in payload["axes"].items()
        },
        calibration=_restore_value(payload["calibration"], arrays),
        valid_flag=bool(payload["valid_flag"]),
        metadata=_restore_value(payload["metadata"], arrays),
    )
    validate_raw_sensor_frame(frame)
    return frame


def _externalize_array(value, arrays, counter):
    key = f"array_{counter[0]:04d}"
    counter[0] += 1
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("Raw sensor serialization does not support object arrays.")
    arrays[key] = np.array(array, copy=True)
    return {
        "__array_key__": key,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _encode_value(value: Any, arrays, counter):
    if isinstance(value, np.ndarray):
        return _externalize_array(value, arrays, counter)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(name): _encode_value(item, arrays, counter)
            for name, item in value.items()
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(item, arrays, counter) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item, arrays, counter) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Raw sensor JSON metadata must contain finite numbers.")
        return value
    raise TypeError(f"Unsupported raw sensor metadata type: {type(value).__name__}")


def _restore_value(value: Any, arrays: Mapping[str, np.ndarray]):
    if isinstance(value, dict) and set(value) == {"__array_key__", "dtype", "shape"}:
        key = str(value["__array_key__"])
        if key not in arrays:
            raise ValueError(f"Raw sensor manifest references missing array {key!r}.")
        array = np.asarray(arrays[key])
        expected_dtype = np.dtype(value["dtype"])
        expected_shape = tuple(int(item) for item in value["shape"])
        if array.dtype != expected_dtype or array.shape != expected_shape:
            raise ValueError(f"Raw sensor array {key!r} failed dtype/shape validation.")
        return np.array(array, copy=True)
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_restore_value(item, arrays) for item in value["__tuple__"])
    if isinstance(value, dict):
        return {name: _restore_value(item, arrays) for name, item in value.items()}
    if isinstance(value, list):
        return [_restore_value(item, arrays) for item in value]
    return value
