import json

import numpy as np
import pytest

from interfaces.data_objects import RawSensorFrame
from interfaces.raw_sensor_serialization import (
    RAW_SENSOR_FRAME_MANIFEST,
    load_raw_sensor_frame,
    save_raw_sensor_frame,
)


def _frame(dtype=np.float32):
    return RawSensorFrame(
        timestamp=12.5, observer_id="sat_a", target_id="sat_b",
        modality="RADAR", data=np.arange(20, dtype=dtype).reshape(4, 5),
        data_kind="RANGE_DOPPLER_POWER_MAP",
        axes={
            "range_m": np.linspace(900.0, 1100.0, 5),
            "range_rate_mps": np.linspace(-1.0, 1.0, 4),
        },
        calibration={
            "principal_xy": [2.0, 1.5], "bin_sizes": (50.0, 2.0 / 3.0),
        },
        metadata={
            "label": "仿真雷达帧",
            "nested": {"ideal": np.array([1000.0, 0.2])},
        },
    )


def test_raw_sensor_frame_json_npz_round_trip_is_exact(tmp_path):
    original = _frame()
    destination = save_raw_sensor_frame(original, tmp_path / "raw_frame")
    restored = load_raw_sensor_frame(destination)

    assert restored.timestamp == original.timestamp
    assert restored.observer_id == original.observer_id
    assert restored.target_id == original.target_id
    assert restored.modality == original.modality
    assert restored.data_kind == original.data_kind
    assert restored.data.dtype == original.data.dtype
    np.testing.assert_array_equal(restored.data, original.data)
    assert tuple(restored.calibration["bin_sizes"]) == (50.0, 2.0 / 3.0)
    np.testing.assert_array_equal(
        restored.metadata["nested"]["ideal"],
        original.metadata["nested"]["ideal"],
    )
    assert not restored.data.flags.writeable


def test_raw_sensor_frame_writer_does_not_overwrite(tmp_path):
    destination = save_raw_sensor_frame(_frame(), tmp_path / "raw_frame")
    with pytest.raises(FileExistsError):
        save_raw_sensor_frame(_frame(), destination)


def test_raw_sensor_frame_reader_rejects_schema_and_archive_tampering(tmp_path):
    destination = save_raw_sensor_frame(_frame(), tmp_path / "raw_frame")
    manifest_path = destination / RAW_SENSOR_FRAME_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "v2.0"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_raw_sensor_frame(destination)

    payload["schema_version"] = "v1.0"
    payload["data"]["shape"] = [99, 99]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dtype/shape"):
        load_raw_sensor_frame(destination)
