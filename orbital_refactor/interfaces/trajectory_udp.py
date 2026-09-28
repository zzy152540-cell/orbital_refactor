"""UDP transport for individual TRAJECTORY-2.0 estimate frames."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
from typing import Any, Mapping


TRAJECTORY_UDP_MESSAGE_TYPE = "TRAJECTORY_FRAME"
TRAJECTORY_UDP_CONTENT_TYPE = "TRAJECTORY"
MAX_UDP_PAYLOAD_BYTES = 65_506


def load_trajectory_time_series(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_trajectory_time_series(payload)
    return payload


def validate_trajectory_time_series(payload: Mapping[str, Any]) -> None:
    required = {
        "schemaVersion", "sceneId", "timeBaseId", "datasetId",
        "startTimeMs", "endTimeMs", "sampleIntervalMs", "frameCount",
        "timeScale", "frame", "unit", "unitV", "frames",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Trajectory payload is missing fields: {sorted(missing)}")
    if payload["schemaVersion"] != "TRAJECTORY-2.0":
        raise ValueError("UDP trajectory transport requires TRAJECTORY-2.0.")
    frames = payload["frames"]
    if not isinstance(frames, list) or len(frames) != payload["frameCount"]:
        raise ValueError("Trajectory frameCount does not match frames.")
    if int(payload["sampleIntervalMs"]) != 1000:
        raise ValueError("Initial UDP integration requires a one-second time step.")
    for index, frame in enumerate(frames):
        if frame.get("frameIndex") != index:
            raise ValueError("Trajectory frame indices must be contiguous from zero.")
        expected = int(payload["startTimeMs"]) + index * 1000
        if frame.get("timeMs") != expected:
            raise ValueError("Trajectory frame times do not match the time base.")


def validate_scene_activation_for_trajectory(
    payload: Mapping[str, Any], trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate external initial conditions for prepared trajectory replay."""

    if not isinstance(payload, Mapping):
        raise TypeError("activation body must be a JSON object")
    scene_id = _required_activation_text(payload, "sceneId")
    if scene_id != str(trajectory["sceneId"]):
        raise ValueError("activation sceneId does not match prepared trajectory")
    start_ms = _parse_activation_time(_required_activation_text(payload, "startTime"))
    end_ms = _parse_activation_time(_required_activation_text(payload, "endTime"))
    if start_ms != int(trajectory["startTimeMs"]):
        raise ValueError("activation startTime does not match prepared trajectory")
    if end_ms != int(trajectory["endTimeMs"]):
        raise ValueError("activation endTime does not match prepared trajectory")

    satellites = payload.get("satellites")
    if not isinstance(satellites, list) or not satellites:
        raise ValueError("activation satellites must be a nonempty array")
    if payload.get("satelliteCount") != len(satellites):
        raise ValueError("activation satelliteCount does not match satellites length")
    expected = {
        str(item["satObj"]) for item in trajectory["frames"][0]["satelliteList"]
    }
    received: list[str] = []
    numeric_fields = (
        "semiMajorAxis", "eccentricity", "inclination", "raan",
        "argPerigee", "meanAnomaly",
    )
    for index, satellite in enumerate(satellites):
        if not isinstance(satellite, Mapping):
            raise ValueError(f"satellites[{index}] must be a JSON object")
        node_id = str(satellite.get("assetName") or satellite.get("name") or "").strip()
        if not node_id:
            raise ValueError(f"satellites[{index}] is missing assetName/name")
        received.append(node_id)
        for field in numeric_fields:
            try:
                value = float(satellite[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"satellite {node_id} has invalid {field}") from exc
            if not math.isfinite(value):
                raise ValueError(f"satellite {node_id} has non-finite {field}")
        eccentricity = float(satellite["eccentricity"])
        if float(satellite["semiMajorAxis"]) <= 0.0:
            raise ValueError(f"satellite {node_id} semiMajorAxis must be positive")
        if not 0.0 <= eccentricity < 1.0:
            raise ValueError(f"satellite {node_id} eccentricity must be in [0, 1)")
        _required_activation_text(satellite, "epoch")
    if len(set(received)) != len(received):
        raise ValueError("activation satellite names must be unique")
    received_set = set(received)
    if received_set != expected:
        missing = sorted(expected - received_set)
        unexpected = sorted(received_set - expected)
        raise ValueError(
            f"activation satellite set mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return {
        "activationMode": "PRECOMPUTED_FILTER_REPLAY",
        "acceptedSatelliteCount": len(received),
        "initialConditionsAccepted": True,
    }


def _required_activation_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"activation is missing {name}")
    return value.strip()


def _parse_activation_time(value: str) -> int:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(round(parsed.timestamp() * 1000.0))


def trajectory_frame_datagram(
    trajectory: Mapping[str, Any], frame_index: int,
) -> dict[str, Any]:
    """Build one self-describing UDP message from a trajectory frame."""

    validate_trajectory_time_series(trajectory)
    frames = trajectory["frames"]
    if not 0 <= int(frame_index) < len(frames):
        raise IndexError("Trajectory frame index is out of range.")
    frame = frames[int(frame_index)]
    return {
        "schemaVersion": trajectory["schemaVersion"],
        "messageType": TRAJECTORY_UDP_MESSAGE_TYPE,
        "type": TRAJECTORY_UDP_CONTENT_TYPE,
        "sceneId": trajectory["sceneId"],
        "timeBaseId": trajectory["timeBaseId"],
        "datasetId": trajectory["datasetId"],
        "timeScale": trajectory["timeScale"],
        "frame": trajectory["frame"],
        "unit": trajectory["unit"],
        "unitV": trajectory["unitV"],
        "frameIndex": frame["frameIndex"],
        "time": frame["time"],
        "timeMs": frame["timeMs"],
        "satelliteList": frame["satelliteList"],
    }


def encode_trajectory_datagram(
    message: Mapping[str, Any], *, maximum_bytes: int = MAX_UDP_PAYLOAD_BYTES,
) -> bytes:
    if message.get("schemaVersion") != "TRAJECTORY-2.0":
        raise ValueError("Unexpected trajectory schemaVersion.")
    if message.get("messageType") != TRAJECTORY_UDP_MESSAGE_TYPE:
        raise ValueError("Unexpected trajectory UDP messageType.")
    if message.get("type") != TRAJECTORY_UDP_CONTENT_TYPE:
        raise ValueError("Unexpected trajectory UDP type.")
    encoded = json.dumps(
        message, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > int(maximum_bytes):
        raise ValueError(
            f"Trajectory UDP payload is {len(encoded)} bytes; limit is "
            f"{int(maximum_bytes)} bytes."
        )
    return encoded


def decode_trajectory_datagram(data: bytes) -> dict[str, Any]:
    try:
        message = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid UTF-8 trajectory UDP JSON.") from exc
    if not isinstance(message, dict):
        raise ValueError("Trajectory UDP message must be a JSON object.")
    if message.get("schemaVersion") != "TRAJECTORY-2.0":
        raise ValueError("Unexpected trajectory schemaVersion.")
    if message.get("messageType") != TRAJECTORY_UDP_MESSAGE_TYPE:
        raise ValueError("Unexpected trajectory UDP messageType.")
    if message.get("type") != TRAJECTORY_UDP_CONTENT_TYPE:
        raise ValueError("Unexpected trajectory UDP type.")
    for name in (
        "sceneId", "timeBaseId", "datasetId", "frameIndex", "timeMs",
        "satelliteList",
    ):
        if name not in message:
            raise ValueError(f"Trajectory UDP message is missing {name}.")
    return message


class TrajectoryUdpPublisher:
    """Small configurable IPv4 UDP sender; it owns no estimator state."""

    def __init__(
        self, host: str, port: int, *, maximum_bytes: int = MAX_UDP_PAYLOAD_BYTES,
        sock: socket.socket | None = None,
    ) -> None:
        if not str(host).strip():
            raise ValueError("UDP destination host cannot be empty.")
        if not 1 <= int(port) <= 65_535:
            raise ValueError("UDP destination port is out of range.")
        self.destination = (str(host), int(port))
        self.maximum_bytes = int(maximum_bytes)
        self._socket = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._owns_socket = sock is None

    def send_frame(self, trajectory: Mapping[str, Any], frame_index: int) -> int:
        message = trajectory_frame_datagram(trajectory, frame_index)
        encoded = encode_trajectory_datagram(
            message, maximum_bytes=self.maximum_bytes,
        )
        return self._socket.sendto(encoded, self.destination)

    def send_display_telemetry_frame(
        self, telemetry: Mapping[str, Any], frame_index: int,
    ) -> int:
        """Send one optional display message on the same UDP port."""

        from exporters.display_telemetry import (
            display_telemetry_datagram,
            encode_display_telemetry_datagram,
        )

        message = display_telemetry_datagram(telemetry, frame_index)
        encoded = encode_display_telemetry_datagram(
            message, maximum_bytes=self.maximum_bytes,
        )
        return self._socket.sendto(encoded, self.destination)

    def close(self) -> None:
        if self._owns_socket:
            self._socket.close()

    def __enter__(self) -> "TrajectoryUdpPublisher":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
