"""TRAJECTORY-2.0 export adapter for estimator visualization frames."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from visualization.data_contract import VisualizationFrame
from visualization.recording import VisualizationRecordingReader


TRAJECTORY_SCHEMA_VERSION = "TRAJECTORY-2.0"


def export_estimate_trajectory_recording(
    recording: str | Path,
    destination: str | Path,
    *,
    scene_id: str,
    time_base_id: str,
    dataset_id: str,
    start_time_ms: int,
    satellite_id_by_node: Mapping[str, int] | None = None,
) -> Path:
    """Export one completed replay as a standard one-second trajectory file."""

    reader = VisualizationRecordingReader(recording)
    return export_estimate_trajectory_frames(
        reader, destination,
        scene_id=scene_id,
        time_base_id=time_base_id,
        dataset_id=dataset_id,
        start_time_ms=start_time_ms,
        satellite_id_by_node=satellite_id_by_node,
    )


def export_estimate_trajectory_frames(
    frames: Iterable[VisualizationFrame],
    destination: str | Path,
    *,
    scene_id: str,
    time_base_id: str,
    dataset_id: str,
    start_time_ms: int,
    satellite_id_by_node: Mapping[str, int] | None = None,
) -> Path:
    """Write estimated position/velocity using the TRAJECTORY-2.0 structure."""

    frames = tuple(frames)
    _validate_export_arguments(
        frames, scene_id=scene_id, time_base_id=time_base_id,
        dataset_id=dataset_id, start_time_ms=start_time_ms,
    )
    node_ids = tuple(sorted(node.node_id for node in frames[0].nodes))
    numeric_ids = _numeric_satellite_ids(node_ids, satellite_id_by_node)
    initial_timestamp = float(frames[0].timestamp)
    encoded_frames = []
    for index, frame in enumerate(frames):
        expected_timestamp = initial_timestamp + index
        if not np.isclose(frame.timestamp, expected_timestamp, atol=1e-9, rtol=0.0):
            raise ValueError(
                "TRAJECTORY-2.0 export requires contiguous one-second frames."
            )
        nodes = {node.node_id: node for node in frame.nodes}
        if tuple(sorted(nodes)) != node_ids:
            raise ValueError("Every trajectory frame must contain identical nodes.")
        time_ms = int(start_time_ms) + index * 1000
        encoded_frames.append({
            "frameIndex": index,
            "time": _rfc3339_milliseconds(time_ms),
            "timeMs": time_ms,
            "satelliteList": [
                _satellite_payload(nodes[node_id], numeric_ids[node_id])
                for node_id in node_ids
            ],
        })
    end_time_ms = int(start_time_ms) + (len(frames) - 1) * 1000
    payload = {
        "schemaVersion": TRAJECTORY_SCHEMA_VERSION,
        "sceneId": str(scene_id),
        "timeBaseId": str(time_base_id),
        "datasetId": str(dataset_id),
        "startTime": _rfc3339_milliseconds(int(start_time_ms)),
        "startTimeMs": int(start_time_ms),
        "endTime": _rfc3339_milliseconds(end_time_ms),
        "endTimeMs": end_time_ms,
        "durationMs": end_time_ms - int(start_time_ms),
        "sampleIntervalMs": 1000,
        "frameCount": len(frames),
        "timeScale": "UTC",
        "frame": "J2000",
        "unit": "m",
        "unitV": "m/s",
        "frames": encoded_frames,
    }
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Trajectory output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return target


def _validate_export_arguments(
    frames, *, scene_id, time_base_id, dataset_id, start_time_ms,
) -> None:
    if not frames:
        raise ValueError("At least one visualization frame is required.")
    if not all(str(value).strip() for value in (scene_id, time_base_id, dataset_id)):
        raise ValueError("Trajectory identifiers cannot be empty.")
    if isinstance(start_time_ms, bool) or not isinstance(start_time_ms, int):
        raise TypeError("start_time_ms must be an integer Unix epoch millisecond value.")


def _numeric_satellite_ids(node_ids, provided):
    if provided is None:
        return {node_id: index for index, node_id in enumerate(node_ids, start=1)}
    if set(provided) != set(node_ids):
        raise ValueError("satellite_id_by_node must cover every trajectory node.")
    values = {node: int(provided[node]) for node in node_ids}
    if len(set(values.values())) != len(values) or any(value < 0 for value in values.values()):
        raise ValueError("Numeric satellite IDs must be unique and nonnegative.")
    return values


def _satellite_payload(node, numeric_id):
    state = np.asarray(node.estimate_state, dtype=float).reshape(-1)
    if state.size < 6 or np.any(~np.isfinite(state[:6])):
        raise ValueError("Every estimate must contain six finite position/velocity values.")
    return {
        "satObj": node.node_id,
        "satId": int(numeric_id),
        "position": {
            name: float(value)
            for name, value in zip(("x", "y", "z", "vx", "vy", "vz"), state[:6])
        },
    }


def _rfc3339_milliseconds(epoch_ms: int) -> str:
    instant = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")
