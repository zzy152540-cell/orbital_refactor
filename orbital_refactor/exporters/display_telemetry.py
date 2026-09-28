"""Generate display-only estimation metrics aligned with TRAJECTORY-2.0."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


DISPLAY_TELEMETRY_SCHEMA_VERSION = "DISPLAY-TELEMETRY-1.0"
DISPLAY_TELEMETRY_MESSAGE_TYPE = "DISPLAY_TELEMETRY"
DISPLAY_TELEMETRY_CONTENT_TYPE = "VISUALIZATION"
_STATE_FIELDS = ("x", "y", "z", "vx", "vy", "vz")
_JSON_NUMBER = r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
_NUMERIC_ARRAY_PATTERN = re.compile(
    rf"\[\s*({_JSON_NUMBER}(?:\s*,\s*{_JSON_NUMBER})*)\s*\]"
)


def load_display_telemetry(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_display_telemetry(payload)
    return payload


def validate_display_telemetry(
    payload: Mapping[str, Any], trajectory: Mapping[str, Any] | None = None,
) -> None:
    if payload.get("schemaVersion") != DISPLAY_TELEMETRY_SCHEMA_VERSION:
        raise ValueError("Unexpected display telemetry schemaVersion.")
    if payload.get("messageType") != DISPLAY_TELEMETRY_MESSAGE_TYPE:
        raise ValueError("Unexpected display telemetry messageType.")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != payload.get("frameCount"):
        raise ValueError("Display telemetry frameCount does not match frames.")
    for index, frame in enumerate(frames):
        if frame.get("frameIndex") != index:
            raise ValueError("Display telemetry frame indices must be contiguous.")
    if trajectory is not None:
        for field in ("sceneId", "timeBaseId"):
            if payload.get(field) != trajectory.get(field):
                raise ValueError(f"Display telemetry {field} does not match trajectory.")
        trajectory_frames = trajectory.get("frames")
        trajectory_frame_count = trajectory.get(
            "frameCount",
            len(trajectory_frames) if isinstance(trajectory_frames, list) else None,
        )
        if len(frames) != trajectory_frame_count:
            raise ValueError("Display telemetry and trajectory frame counts differ.")
        for telemetry_frame, trajectory_frame in zip(frames, trajectory_frames):
            if (
                telemetry_frame.get("frameIndex") != trajectory_frame.get("frameIndex")
                or telemetry_frame.get("timeMs") != trajectory_frame.get("timeMs")
            ):
                raise ValueError("Display telemetry is not aligned with trajectory.")


def load_project_truth_histories(
    functional_path: str | Path,
    background_path: str | Path,
) -> list[dict[str, np.ndarray]]:
    """Load the two truth layouts used by the 31-satellite integration case."""

    functional = json.loads(Path(functional_path).read_text(encoding="utf-8"))
    background = json.loads(Path(background_path).read_text(encoding="utf-8"))
    if not isinstance(functional, list) or not functional:
        raise ValueError("Functional truth must contain a non-empty frame array.")
    satellites = background.get("satellites") if isinstance(background, dict) else None
    if not isinstance(satellites, list):
        raise ValueError("Background truth must contain satellites.")

    histories: list[dict[str, np.ndarray]] = []
    for frame in functional:
        items = frame.get("satelliteList") if isinstance(frame, dict) else None
        if not isinstance(items, list):
            raise ValueError("Functional truth frame is missing satelliteList.")
        histories.append({
            str(item["satObj"]): _state_from_mapping(item["position"])
            for item in items
        })
    for satellite in satellites:
        name = str(satellite.get("name", "")).strip()
        points = satellite.get("points")
        if not name or not isinstance(points, list) or len(points) != len(histories):
            raise ValueError("Background truth histories must match functional frames.")
        for index, point in enumerate(points):
            if name in histories[index]:
                raise ValueError(f"Duplicate truth satellite: {name}")
            histories[index][name] = _state_from_mapping(point)
    return histories


def build_display_telemetry(
    trajectory: Mapping[str, Any],
    truth_by_frame: Sequence[Mapping[str, np.ndarray]],
    *,
    selected_satellite: str | None = None,
    metric_purpose: str = "DEMO_VALIDATION",
    accuracy_claim_allowed: bool = False,
    cann_by_frame: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Calculate fleet/per-satellite errors; selected_satellite is deprecated."""

    frames = trajectory.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Trajectory must contain a non-empty frames array.")
    if len(truth_by_frame) != len(frames):
        raise ValueError("Truth and trajectory frame counts must match.")
    if cann_by_frame is not None and len(cann_by_frame) != len(frames):
        raise ValueError("CANN and trajectory frame counts must match.")
    telemetry_frames = []
    expected_nodes: set[str] | None = None
    expected_satellite_ids: dict[str, int] | None = None
    for index, (frame, truth) in enumerate(zip(frames, truth_by_frame)):
        satellite_ids = {
            str(item["satObj"]): int(item["satId"])
            for item in frame["satelliteList"]
        }
        estimates = {
            str(item["satObj"]): _state_from_mapping(item["position"])
            for item in frame["satelliteList"]
        }
        nodes = set(estimates)
        if expected_nodes is None:
            expected_nodes = nodes
            expected_satellite_ids = satellite_ids
        elif nodes != expected_nodes:
            raise ValueError("Trajectory satellite set changes between frames.")
        elif satellite_ids != expected_satellite_ids:
            raise ValueError("Trajectory satellite IDs change between frames.")
        if nodes != set(truth):
            missing = sorted(nodes - set(truth))
            unexpected = sorted(set(truth) - nodes)
            raise ValueError(
                f"Truth satellite set mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )
        errors = np.stack([estimates[node] - truth[node] for node in sorted(nodes)])
        position_norms = np.linalg.norm(errors[:, :3], axis=1)
        velocity_norms = np.linalg.norm(errors[:, 3:], axis=1)
        satellite_errors = []
        for node in sorted(nodes):
            node_error = estimates[node] - truth[node]
            satellite_errors.append({
                "satObj": node,
                "satId": satellite_ids[node],
                "positionErrorM": float(np.linalg.norm(node_error[:3])),
                "velocityErrorMps": float(np.linalg.norm(node_error[3:])),
            })
        telemetry_frame = {
            "frameIndex": int(frame["frameIndex"]),
            "time": str(frame["time"]),
            "timeMs": int(frame["timeMs"]),
            "estimation": {
                "fleetPositionRmseM": float(np.sqrt(np.mean(position_norms ** 2))),
                "fleetVelocityRmseMps": float(np.sqrt(np.mean(velocity_norms ** 2))),
                "satelliteErrors": satellite_errors,
                "metricPurpose": str(metric_purpose),
                "accuracyClaimAllowed": bool(accuracy_claim_allowed),
            },
        }
        if cann_by_frame is not None:
            cann_items = []
            for item in cann_by_frame[index]:
                cann_item = dict(item)
                node = str(cann_item["satObj"])
                if node not in satellite_ids:
                    raise ValueError("CANN satellite is absent from the trajectory.")
                if "satId" in cann_item and int(cann_item["satId"]) != satellite_ids[node]:
                    raise ValueError("CANN satId does not match the trajectory.")
                cann_item["satId"] = satellite_ids[node]
                cann_items.append(cann_item)
            telemetry_frame["cannBySatellite"] = cann_items
        telemetry_frames.append(telemetry_frame)

    return {
        "schemaVersion": DISPLAY_TELEMETRY_SCHEMA_VERSION,
        "messageType": DISPLAY_TELEMETRY_MESSAGE_TYPE,
        "type": DISPLAY_TELEMETRY_CONTENT_TYPE,
        "sceneId": str(trajectory["sceneId"]),
        "timeBaseId": str(trajectory["timeBaseId"]),
        "sourceDatasetId": str(trajectory["datasetId"]),
        "frameCount": len(telemetry_frames),
        "sampleIntervalMs": int(trajectory["sampleIntervalMs"]),
        "frames": telemetry_frames,
    }


def export_display_telemetry(
    trajectory_path: str | Path,
    functional_truth_path: str | Path,
    background_truth_path: str | Path,
    destination: str | Path,
    *,
    selected_satellite: str | None = None,
    include_cann: bool = True,
) -> Path:
    trajectory = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    truth = load_project_truth_histories(functional_truth_path, background_truth_path)
    cann = (
        build_cann_display_frames(trajectory)
        if include_cann else None
    )
    payload = build_display_telemetry(
        trajectory, truth, selected_satellite=selected_satellite,
        cann_by_frame=cann,
    )
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Display telemetry output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_display_telemetry(payload), encoding="utf-8")
    return target


def dumps_display_telemetry(payload: Mapping[str, Any]) -> str:
    """Pretty-print objects while keeping pure numeric arrays on one line."""

    formatted = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False,
    )
    return _NUMERIC_ARRAY_PATTERN.sub(
        lambda match: "[" + re.sub(r"\s+", "", match.group(1)) + "]",
        formatted,
    )


def display_telemetry_datagram(
    telemetry: Mapping[str, Any], frame_index: int,
) -> dict[str, Any]:
    """Build the independent UDP message paired with a trajectory frame."""

    frames = telemetry.get("frames")
    if not isinstance(frames, list) or not 0 <= int(frame_index) < len(frames):
        raise IndexError("Display telemetry frame index is out of range.")
    frame = frames[int(frame_index)]
    if int(frame["frameIndex"]) != int(frame_index):
        raise ValueError("Display telemetry frame indices are not contiguous.")
    message = {
        "schemaVersion": telemetry["schemaVersion"],
        "messageType": telemetry["messageType"],
        "type": DISPLAY_TELEMETRY_CONTENT_TYPE,
        "sceneId": telemetry["sceneId"],
        "timeBaseId": telemetry["timeBaseId"],
        "sourceDatasetId": telemetry["sourceDatasetId"],
        "frameIndex": frame["frameIndex"],
        "time": frame["time"],
        "timeMs": frame["timeMs"],
        "estimation": frame["estimation"],
    }
    if "cannBySatellite" in frame:
        message["cannBySatellite"] = frame["cannBySatellite"]
    return message


def build_cann_display_frames(
    trajectory: Mapping[str, Any],
) -> list[list[dict[str, Any]]]:
    """Run the passive CANNs for every satellite using the GUI's 3x3 map."""

    from brain_inspired.navigation_place_cells import (
        NavigationPlaceCellConfig,
        NavigationPlaceCellEncoder,
    )
    from brain_inspired.orbital_direction_runner import run_orbital_direction_states
    from brain_inspired.orbital_direction_state import OrbitalDirectionConfig
    from brain_inspired.orbital_phase_adapter import OrbitalPlaneFrame
    from brain_inspired.orbital_rt_grid_runner import run_orbital_rt_grid_states
    from brain_inspired.orbital_rt_grid_state import OrbitalRTGridConfig
    from brain_inspired.ring_cann import RingCANNConfig
    from orbital_core.dynamics import rk4_step_absolute

    frames = trajectory.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Trajectory must contain frames for CANN generation.")
    times = np.asarray([
        (int(frame["timeMs"]) - int(frames[0]["timeMs"])) / 1000.0
        for frame in frames
    ], dtype=float)
    nodes = sorted(str(item["satObj"]) for item in frames[0]["satelliteList"])
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("Trajectory satellite identifiers must be unique.")
    satellite_ids = {
        str(item["satObj"]): int(item["satId"])
        for item in frames[0]["satelliteList"]
    }
    posterior: dict[str, np.ndarray] = {}
    for node in nodes:
        values = []
        for frame in frames:
            selected = [
                item for item in frame["satelliteList"]
                if str(item["satObj"]) == node
            ]
            if len(selected) != 1:
                raise ValueError(
                    "Every satellite must occur once in every trajectory frame."
                )
            values.append(_state_from_mapping(selected[0]["position"]))
        posterior[node] = np.asarray(values)

    direction = run_orbital_direction_states(
        timestamps=times,
        posterior_state_history_by_node=posterior,
        frame_by_node={
            node: OrbitalPlaneFrame.from_ascending_node_of_state_eci(
                posterior[node][0]
            )
            for node in nodes
        },
        config=OrbitalDirectionConfig(
            ring=RingCANNConfig(num_neurons=90, internal_dt=0.002),
        ),
        retain_activity=True,
    )
    references = {}
    for node in nodes:
        reference = [posterior[node][0].copy()]
        for index in range(1, times.size):
            reference.append(rk4_step_absolute(
                reference[-1], float(times[index] - times[index - 1]),
            ))
        references[node] = np.asarray(reference)
    rt = run_orbital_rt_grid_states(
        timestamps=times,
        posterior_state_history_by_node=posterior,
        reference_state_history_by_node=references,
        config=OrbitalRTGridConfig(),
        retain_activity=False,
    )
    place_config = NavigationPlaceCellConfig.display_3x3()
    place_encoder = NavigationPlaceCellEncoder(place_config)

    result = []
    for index in range(times.size):
        items = []
        for node in nodes:
            direction_item = direction[node]
            rt_item = rt[node]
            place = place_encoder.encode(
                phase=direction_item.anchored_phase[index],
                radial_position=rt_item.anchored_rt[index, 0],
                along_track_position=rt_item.anchored_rt[index, 1],
            )
            heatmap = place.activity.reshape(
                place_config.phase_cell_count,
                len(place_config.radial_centers_m),
                len(place_config.along_track_centers_m),
            ).sum(axis=0)
            saturated = bool(
                rt_item.saturated_at_boundary[index]
                or place.boundary_saturated
            )
            items.append({
                "satObj": node,
                "satId": satellite_ids[node],
                "activityEncoding": "UINT8_NORMALIZED",
                "directionRingNeuronCount": 90,
                "directionPhaseType": "ABSOLUTE_ARGUMENT_OF_LATITUDE",
                "directionZeroReference": "ASCENDING_NODE",
                "directionRing": _quantize_activity(
                    direction_item.neural_activity[index]
                ),
                "directionDegrees": float(
                    np.degrees(direction_item.anchored_phase[index]) % 360.0
                ),
                "directionConcentration": float(
                    direction_item.bump_concentration[index]
                ),
                "directionBumpWidth": float(direction_item.bump_width[index]),
                "positionHeatmapShape": [3, 3],
                "positionHeatmapRadialCentersM": [
                    float(value) for value in place_config.radial_centers_m
                ],
                "positionHeatmapAlongTrackCentersM": [
                    float(value) for value in place_config.along_track_centers_m
                ],
                "positionHeatmap": _quantize_activity(heatmap),
                "decodedRadialAlongTrackM": [
                    float(value) for value in rt_item.anchored_rt[index]
                ],
                "valid": bool(
                    direction_item.valid[index]
                    and rt_item.valid[index]
                    and place.valid
                ),
                "status": "BOUNDARY_SATURATED" if saturated else "OK",
            })
        result.append(items)
    return result


def encode_display_telemetry_datagram(
    message: Mapping[str, Any], *, maximum_bytes: int = 65_506,
) -> bytes:
    if message.get("schemaVersion") != DISPLAY_TELEMETRY_SCHEMA_VERSION:
        raise ValueError("Unexpected display telemetry schemaVersion.")
    if message.get("messageType") != DISPLAY_TELEMETRY_MESSAGE_TYPE:
        raise ValueError("Unexpected display telemetry messageType.")
    if message.get("type") != DISPLAY_TELEMETRY_CONTENT_TYPE:
        raise ValueError("Unexpected display telemetry type.")
    encoded = json.dumps(
        message, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > int(maximum_bytes):
        raise ValueError("Display telemetry UDP payload exceeds the configured limit.")
    return encoded


def _state_from_mapping(payload: Mapping[str, Any]) -> np.ndarray:
    try:
        state = np.asarray([payload[name] for name in _STATE_FIELDS], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("State must contain x, y, z, vx, vy, and vz.") from exc
    if state.shape != (6,) or np.any(~np.isfinite(state)):
        raise ValueError("State values must be six finite numbers.")
    return state


def _positive_activity(values: Any) -> np.ndarray:
    activity = np.asarray(values, dtype=float)
    activity = np.maximum(activity - float(np.min(activity)), 0.0)
    if float(np.max(activity)) <= np.finfo(float).tiny:
        return np.ones_like(activity)
    return activity


def _quantize_activity(values: Any) -> list[int]:
    activity = _positive_activity(values)
    normalized = activity / float(np.max(activity))
    return np.rint(
        np.clip(normalized, 0.0, 1.0) * 255.0
    ).astype(int).reshape(-1).tolist()
