"""TRAJECTORY-2.0 export adapter for estimator visualization frames."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from visualization.data_contract import VisualizationFrame
from visualization.recording import VisualizationRecordingReader


TRAJECTORY_SCHEMA_VERSION = "TRAJECTORY-2.0"


def export_absolute_state_histories(
    timestamps: Iterable[float],
    state_history_by_satellite: Mapping[str, np.ndarray],
    destination: str | Path,
    *,
    scene_id: str,
    time_base_id: str,
    dataset_id: str,
    start_time_ms: int,
    satellite_id_by_node: Mapping[str, int] | None = None,
    data_kind: str = "ESTIMATED",
) -> Path:
    """Export absolute J2000 filter histories as a TRAJECTORY-2.0 data set."""

    times = np.asarray(tuple(timestamps), dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(~np.isfinite(times)):
        raise ValueError("timestamps must be a non-empty finite one-dimensional sequence.")
    if not np.allclose(np.diff(times), 1.0, atol=1e-9, rtol=0.0):
        raise ValueError("TRAJECTORY-2.0 export requires contiguous one-second frames.")
    histories = {
        str(node_id): np.asarray(history, dtype=float)
        for node_id, history in state_history_by_satellite.items()
    }
    node_ids = tuple(sorted(histories))
    if not node_ids:
        raise ValueError("At least one satellite state history is required.")
    for node_id in node_ids:
        history = histories[node_id]
        if history.shape != (times.size, 6):
            raise ValueError(
                f"State history for {node_id} must have shape ({times.size}, 6)."
            )
        if np.any(~np.isfinite(history)):
            raise ValueError(f"State history for {node_id} contains non-finite values.")
    _validate_identifiers_and_time(
        scene_id=scene_id, time_base_id=time_base_id,
        dataset_id=dataset_id, start_time_ms=start_time_ms,
    )
    numeric_ids = _numeric_satellite_ids(node_ids, satellite_id_by_node)
    encoded_frames = []
    for index in range(times.size):
        time_ms = int(start_time_ms) + index * 1000
        encoded_frames.append({
            "frameIndex": index,
            "time": _rfc3339_milliseconds(time_ms),
            "timeMs": time_ms,
            "satelliteList": [
                _state_payload(node_id, numeric_ids[node_id], histories[node_id][index])
                for node_id in node_ids
            ],
        })
    return _write_trajectory_payload(
        destination,
        scene_id=scene_id,
        time_base_id=time_base_id,
        dataset_id=dataset_id,
        start_time_ms=start_time_ms,
        frames=encoded_frames,
        data_kind=data_kind,
    )


def export_single_filter_history(
    history: Any,
    observer_state_history_eci: np.ndarray,
    destination: str | Path,
    *,
    target_id: str,
    scene_id: str,
    time_base_id: str,
    dataset_id: str,
    start_time_ms: int,
    satellite_id: int | None = None,
) -> Path:
    """Convert a relative single-target filter history to absolute J2000 output."""

    relative = getattr(history, "fused_state_history", None)
    if relative is None:
        relative = getattr(history, "state_history", None)
    if relative is None or not hasattr(history, "timestamps"):
        raise TypeError("history must expose timestamps and a supported state history.")
    relative = np.asarray(relative, dtype=float)
    observer = np.asarray(observer_state_history_eci, dtype=float)
    if observer.shape != relative.shape or relative.ndim != 2 or relative.shape[1] != 6:
        raise ValueError("Observer and relative histories must have matching shape (N, 6).")
    id_map = None if satellite_id is None else {str(target_id): int(satellite_id)}
    return export_absolute_state_histories(
        history.timestamps,
        {str(target_id): observer + relative},
        destination,
        scene_id=scene_id,
        time_base_id=time_base_id,
        dataset_id=dataset_id,
        start_time_ms=start_time_ms,
        satellite_id_by_node=id_map,
        data_kind="ESTIMATED",
    )


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
    return _write_trajectory_payload(
        destination,
        scene_id=scene_id,
        time_base_id=time_base_id,
        dataset_id=dataset_id,
        start_time_ms=start_time_ms,
        frames=encoded_frames,
        data_kind="ESTIMATED",
    )


def _validate_export_arguments(
    frames, *, scene_id, time_base_id, dataset_id, start_time_ms,
) -> None:
    if not frames:
        raise ValueError("At least one visualization frame is required.")
    _validate_identifiers_and_time(
        scene_id=scene_id, time_base_id=time_base_id,
        dataset_id=dataset_id, start_time_ms=start_time_ms,
    )


def _validate_identifiers_and_time(
    *, scene_id, time_base_id, dataset_id, start_time_ms,
) -> None:
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
    return _state_payload(node.node_id, numeric_id, node.estimate_state)


def _state_payload(node_id, numeric_id, state):
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.size < 6 or np.any(~np.isfinite(state[:6])):
        raise ValueError("Every estimate must contain six finite position/velocity values.")
    return {
        "satObj": str(node_id),
        "satId": int(numeric_id),
        "position": {
            name: float(value)
            for name, value in zip(("x", "y", "z", "vx", "vy", "vz"), state[:6])
        },
    }


def _write_trajectory_payload(
    destination, *, scene_id, time_base_id, dataset_id, start_time_ms,
    frames, data_kind,
):
    end_time_ms = int(start_time_ms) + (len(frames) - 1) * 1000
    payload = {
        "schemaVersion": TRAJECTORY_SCHEMA_VERSION,
        "sceneId": str(scene_id),
        "timeBaseId": str(time_base_id),
        "datasetId": str(dataset_id),
        "dataKind": str(data_kind),
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
        "frames": frames,
    }
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Trajectory output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return target


def _rfc3339_milliseconds(epoch_ms: int) -> str:
    instant = datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")
