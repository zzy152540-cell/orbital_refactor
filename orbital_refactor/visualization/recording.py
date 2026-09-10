"""Safe directory-based recording for synchronized visualization frames."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from visualization.data_contract import (
    VISUALIZATION_SCHEMA_VERSION,
    VisualizationFrame,
    visualization_frame_from_dict,
    visualization_frame_to_dict,
)


RECORDING_FORMAT_VERSION = "v1.0"


@dataclass(frozen=True)
class VisualizationRecordingManifest:
    format_version: str
    frame_schema_version: str
    scenario_id: str
    run_id: str
    frame_count: int
    start_timestamp: float
    end_timestamp: float


class VisualizationRecordingWriter:
    """Write frames without pickle and without retaining an entire run in RAM."""

    def __init__(self, destination: str | Path) -> None:
        self.destination = Path(destination)
        self._frames_path = self.destination / "frames"
        self._arrays_path = self.destination / "arrays"
        self._closed = False
        self._count = 0
        self._scenario_id: str | None = None
        self._run_id: str | None = None
        self._start_timestamp: float | None = None
        self._end_timestamp: float | None = None

        if self.destination.exists():
            raise FileExistsError(
                f"Visualization recording already exists: {self.destination}"
            )
        self._frames_path.mkdir(parents=True)
        self._arrays_path.mkdir()

    def append(self, frame: VisualizationFrame) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed visualization recording.")
        if self._scenario_id is None:
            self._scenario_id = frame.scenario_id
            self._run_id = frame.run_id
            self._start_timestamp = frame.timestamp
        elif (
            frame.scenario_id != self._scenario_id
            or frame.run_id != self._run_id
        ):
            raise ValueError("All recording frames must belong to one run.")
        if frame.epoch_index != self._count:
            raise ValueError("Recording epoch indices must be contiguous from zero.")
        if self._end_timestamp is not None and frame.timestamp <= self._end_timestamp:
            raise ValueError("Recording timestamps must be strictly increasing.")

        encoded = visualization_frame_to_dict(frame)
        arrays: dict[str, np.ndarray] = {}
        externalized = _externalize_arrays(
            encoded, arrays=arrays, counter=[0],
        )
        np.savez_compressed(
            self._arrays_path / f"{self._count:08d}.npz", **arrays,
        )
        frame_path = self._frames_path / f"{self._count:08d}.json"
        frame_path.write_text(
            json.dumps(externalized, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        self._end_timestamp = frame.timestamp
        self._count += 1

    def close(self) -> VisualizationRecordingManifest:
        if self._closed:
            return self._read_written_manifest()
        if self._count == 0:
            raise ValueError("Cannot close an empty visualization recording.")
        manifest = VisualizationRecordingManifest(
            format_version=RECORDING_FORMAT_VERSION,
            frame_schema_version=VISUALIZATION_SCHEMA_VERSION,
            scenario_id=str(self._scenario_id), run_id=str(self._run_id),
            frame_count=self._count,
            start_timestamp=float(self._start_timestamp),
            end_timestamp=float(self._end_timestamp),
        )
        (self.destination / "manifest.json").write_text(
            json.dumps(manifest.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._closed = True
        return manifest

    def __enter__(self) -> "VisualizationRecordingWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()

    def _read_written_manifest(self) -> VisualizationRecordingManifest:
        payload = json.loads(
            (self.destination / "manifest.json").read_text(encoding="utf-8")
        )
        return VisualizationRecordingManifest(**payload)


class VisualizationRecordingReader:
    """Random-access reader for a completed visualization recording."""

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source)
        manifest_path = self.source / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing recording manifest: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest = VisualizationRecordingManifest(**payload)
        if self.manifest.format_version != RECORDING_FORMAT_VERSION:
            raise ValueError("Unsupported visualization recording format.")
        if self.manifest.frame_schema_version != VISUALIZATION_SCHEMA_VERSION:
            raise ValueError("Unsupported visualization frame schema.")

    def __len__(self) -> int:
        return self.manifest.frame_count

    def __getitem__(self, index: int) -> VisualizationFrame:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Visualization frame index out of range.")
        frame_path = self.source / "frames" / f"{index:08d}.json"
        payload = json.loads(frame_path.read_text(encoding="utf-8"))
        archive_path = self.source / "arrays" / f"{index:08d}.npz"
        with np.load(archive_path, allow_pickle=False) as arrays:
            restored = _restore_arrays(payload, arrays=arrays)
        return visualization_frame_from_dict(restored)

    def __iter__(self) -> Iterator[VisualizationFrame]:
        for index in range(len(self)):
            yield self[index]


def _externalize_arrays(
    value: Any, *, arrays: dict[str, np.ndarray], counter: list[int],
) -> Any:
    if isinstance(value, dict) and value.get("__array__") is True:
        array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        expected_shape = tuple(int(item) for item in value["shape"])
        if array.shape != expected_shape:
            raise ValueError("Encoded visualization array has an invalid shape.")
        name = f"array_{counter[0]:04d}"
        counter[0] += 1
        arrays[name] = array
        return {"__array_key__": name}
    if isinstance(value, dict):
        return {
            key: _externalize_arrays(
                item, arrays=arrays, counter=counter,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _externalize_arrays(
                item, arrays=arrays, counter=counter,
            )
            for item in value
        ]
    return value


def _restore_arrays(value: Any, *, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(value, dict) and set(value) == {"__array_key__"}:
        key = str(value["__array_key__"])
        if key not in arrays:
            raise ValueError("Visualization frame references a missing array.")
        array = np.asarray(arrays[key])
        return {
            "__array__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tolist(),
        }
    if isinstance(value, dict):
        return {
            key: _restore_arrays(item, arrays=arrays)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_arrays(item, arrays=arrays) for item in value]
    return value
