"""Online target-free IOD/tracking runtime with UDP-ready frame output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import threading
import time
from typing import Any, Mapping

import numpy as np

from brain_inspired import OnlineMultiTargetBrainSidecar
from exporters.display_telemetry import (
    DISPLAY_TELEMETRY_CONTENT_TYPE,
    DISPLAY_TELEMETRY_MESSAGE_TYPE,
    DISPLAY_TELEMETRY_SCHEMA_VERSION,
)
from tracking.association import (
    OnlineMultiTargetTracker,
    unlabeled_iod_observations_from_messages,
)
from tracking.online_json import parse_online_json_frame

from .scene_control import SceneClockController, SceneControlError
from .trajectory_udp import (
    TRAJECTORY_UDP_CONTENT_TYPE,
    TRAJECTORY_UDP_MESSAGE_TYPE,
    TrajectoryUdpPublisher,
)


ONLINE_MEASUREMENT_FRAME_PATH = "/api/external/scene/measurement-frame"


@dataclass(frozen=True)
class OnlineMultiTargetRuntimeFrame:
    frame_index: int
    timestamp: float
    time_ms: int
    trajectory_message: Mapping[str, Any]
    display_message: Mapping[str, Any]
    initialized_target_ids: tuple[str, ...]
    retired_target_ids: tuple[str, ...]


class OnlineMultiTargetRuntime:
    """Consume one target-free sensor frame and produce two UDP messages."""

    def __init__(
        self,
        *,
        scene_id: str,
        time_base_id: str,
        dataset_id: str,
        start_time_ms: int,
        sample_interval_seconds: float = 1.0,
        tracker: OnlineMultiTargetTracker | None = None,
        brain_sidecar: OnlineMultiTargetBrainSidecar | None = None,
    ) -> None:
        if not all(str(value).strip() for value in (
            scene_id, time_base_id, dataset_id,
        )):
            raise ValueError("Online runtime identifiers cannot be empty.")
        interval = float(sample_interval_seconds)
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("sample_interval_seconds must be positive.")
        self.scene_id = str(scene_id)
        self.time_base_id = str(time_base_id)
        self.dataset_id = str(dataset_id)
        self.start_time_ms = int(start_time_ms)
        self.sample_interval_seconds = interval
        self.tracker = tracker or OnlineMultiTargetTracker(scene_id=self.scene_id)
        if self.tracker.scene_id != self.scene_id:
            raise ValueError("Tracker scene ID must match the online runtime.")
        self.brain_sidecar = brain_sidecar or OnlineMultiTargetBrainSidecar()
        self._frame_index = 0
        self._last_timestamp: float | None = None
        self._source_start_timestamp: float | None = None
        self._satellite_id_by_target: dict[str, int] = {}

    def process_frame(self, frame: Mapping[str, Any]) -> OnlineMultiTargetRuntimeFrame:
        supplied_index = frame.get("frameIndex")
        if supplied_index is not None and int(supplied_index) != self._frame_index:
            raise ValueError("Online frameIndex must be contiguous from zero.")
        timestamp, messages, states, quaternions = parse_online_json_frame(frame)
        if self._source_start_timestamp is None:
            self._source_start_timestamp = timestamp
        if self._last_timestamp is not None:
            elapsed = timestamp - self._last_timestamp
            if not np.isclose(
                elapsed, self.sample_interval_seconds, rtol=0.0, atol=1.0e-9,
            ):
                raise ValueError("Online frames must follow the configured interval.")
        state_lookup = {
            (observer_id, timestamp): state
            for observer_id, state in states.items()
        }
        observations = unlabeled_iod_observations_from_messages(
            messages, observer_state_by_epoch=state_lookup,
        )
        update = self.tracker.step(
            timestamp=timestamp,
            observations=observations,
            observer_states_eci=states,
            q_eci2pri_by_observer=quaternions,
        )
        for target_id in update.retired_target_ids:
            self.brain_sidecar.remove(target_id)
        active = {
            target_id: track.estimate
            for target_id, track in update.tracks_by_target.items()
            if target_id not in set(update.retired_target_ids)
        }
        brain = self.brain_sidecar.update(
            timestamp=timestamp,
            state_by_target={
                target_id: estimate.state_eci
                for target_id, estimate in active.items()
            },
            covariance_by_target={
                target_id: estimate.covariance_eci
                for target_id, estimate in active.items()
            },
        ) if active else {}
        for target_id in sorted(active):
            self._satellite_id_by_target.setdefault(
                target_id, len(self._satellite_id_by_target) + 1,
            )
        elapsed_seconds = timestamp - self._source_start_timestamp
        time_ms = self.start_time_ms + int(round(elapsed_seconds * 1_000.0))
        trajectory = self._trajectory_message(
            update, active, time_ms=time_ms,
        )
        display = self._display_message(
            update, active, brain, time_ms=time_ms,
        )
        result = OnlineMultiTargetRuntimeFrame(
            frame_index=self._frame_index,
            timestamp=timestamp,
            time_ms=time_ms,
            trajectory_message=trajectory,
            display_message=display,
            initialized_target_ids=tuple(sorted(update.initialized_states_by_target)),
            retired_target_ids=tuple(update.retired_target_ids),
        )
        self._frame_index += 1
        self._last_timestamp = timestamp
        return result

    @staticmethod
    def publish(
        frame: OnlineMultiTargetRuntimeFrame,
        publisher: TrajectoryUdpPublisher,
    ) -> tuple[int, int]:
        return (
            publisher.send_trajectory_message(frame.trajectory_message),
            publisher.send_display_telemetry_message(frame.display_message),
        )

    def _trajectory_message(self, update, active, *, time_ms):
        return {
            "schemaVersion": "TRAJECTORY-2.0",
            "messageType": TRAJECTORY_UDP_MESSAGE_TYPE,
            "type": TRAJECTORY_UDP_CONTENT_TYPE,
            "sceneId": self.scene_id,
            "timeBaseId": self.time_base_id,
            "datasetId": self.dataset_id,
            "timeScale": "UTC",
            "frame": "J2000",
            "unit": "m",
            "unitV": "m/s",
            "frameIndex": self._frame_index,
            "time": _utc_text(time_ms),
            "timeMs": time_ms,
            "satelliteList": [
                {
                    "satObj": target_id,
                    "satId": self._satellite_id_by_target[target_id],
                    "position": {
                        name: float(value)
                        for name, value in zip(
                            ("x", "y", "z", "vx", "vy", "vz"),
                            active[target_id].state_eci,
                        )
                    },
                }
                for target_id in sorted(active)
            ],
        }

    def _display_message(self, update, active, brain, *, time_ms):
        return {
            "schemaVersion": DISPLAY_TELEMETRY_SCHEMA_VERSION,
            "messageType": DISPLAY_TELEMETRY_MESSAGE_TYPE,
            "type": DISPLAY_TELEMETRY_CONTENT_TYPE,
            "sceneId": self.scene_id,
            "timeBaseId": self.time_base_id,
            "sourceDatasetId": self.dataset_id,
            "frameIndex": self._frame_index,
            "time": _utc_text(time_ms),
            "timeMs": time_ms,
            "estimation": {
                "metricPurpose": "ONLINE_RUNTIME_DIAGNOSTICS",
                "accuracyClaimAllowed": False,
                "activeTargetCount": len(active),
                "initializedTargetIds": sorted(update.initialized_states_by_target),
                "retiredTargetIds": list(update.retired_target_ids),
                "associationMatchCount": len(update.association.matches),
                "ambiguousDetectionCount": len(
                    update.association.ambiguous_observation_indices
                ),
                "cannPreprocessedObservationCount": (
                    update.cann_preprocessed_observation_count
                ),
                "cannMeasurementPreprocessingMs": float(
                    update.cann_preprocessing_seconds * 1_000.0
                ),
            },
            "cannBySatellite": [
                _brain_item(
                    target_id,
                    self._satellite_id_by_target[target_id],
                    brain[target_id],
                )
                for target_id in sorted(brain)
            ],
        }


class OnlineMultiTargetRuntimeEndpoint:
    """Idempotent HTTP-to-runtime adapter gated by the scene clock."""

    def __init__(
        self,
        runtime: OnlineMultiTargetRuntime,
        controller: SceneClockController,
        publisher: TrajectoryUdpPublisher,
    ) -> None:
        if runtime.scene_id != controller.config.scene_id:
            raise ValueError("Runtime and scene controller must use the same sceneId.")
        self.runtime = runtime
        self.controller = controller
        self.publisher = publisher
        self._lock = threading.RLock()
        self._last_frame_index: int | None = None
        self._last_fingerprint: str | None = None
        self._last_response: dict[str, Any] | None = None

    @property
    def post_routes(self):
        return {ONLINE_MEASUREMENT_FRAME_PATH: self.handle}

    def reset(self, runtime: OnlineMultiTargetRuntime) -> None:
        """Install a fresh algorithm runtime for the next activated run."""

        if runtime.scene_id != self.controller.config.scene_id:
            raise ValueError("Runtime and scene controller must use the same sceneId.")
        with self._lock:
            self.runtime = runtime
            self._last_frame_index = None
            self._last_fingerprint = None
            self._last_response = None

    def handle(self, payload: Mapping[str, Any]):
        with self._lock:
            clock = self.controller.snapshot()
            if clock["clockState"] != "RUNNING":
                raise SceneControlError(
                    409, "measurement frames are accepted only while RUNNING",
                )
            try:
                frame_index = int(payload["frameIndex"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SceneControlError(400, "missing or invalid frameIndex") from exc
            fingerprint = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            )
            if frame_index == self._last_frame_index:
                if fingerprint != self._last_fingerprint:
                    raise SceneControlError(
                        409, "frameIndex was reused with different content",
                    )
                return 200, {**self._last_response, "applied": False}
            try:
                processing_started = time.perf_counter()
                output = self.runtime.process_frame(payload)
                processing_ms = 1_000.0 * (
                    time.perf_counter() - processing_started
                )
                publication_started = time.perf_counter()
                trajectory_bytes, display_bytes = self.runtime.publish(
                    output, self.publisher,
                )
                publication_ms = 1_000.0 * (
                    time.perf_counter() - publication_started
                )
            except (TypeError, ValueError) as exc:
                raise SceneControlError(422, str(exc)) from exc
            except OSError as exc:
                self.controller.note_udp_send_failure(str(exc))
                raise SceneControlError(503, "UDP publication failed") from exc
            self.controller.note_udp_send_success()
            response = {
                "protocolVersion": "MULTITARGET-ONLINE-1.0",
                "code": 200,
                "message": "measurement frame processed",
                "sceneId": self.runtime.scene_id,
                "frameIndex": output.frame_index,
                "timeMs": output.time_ms,
                "activeTargetCount": len(
                    output.trajectory_message["satelliteList"]
                ),
                "initializedTargetIds": list(output.initialized_target_ids),
                "retiredTargetIds": list(output.retired_target_ids),
                "trajectoryBytes": int(trajectory_bytes),
                "displayTelemetryBytes": int(display_bytes),
                "algorithmProcessingMs": float(processing_ms),
                "udpPublicationMs": float(publication_ms),
                "withinSampleInterval": bool(
                    processing_ms + publication_ms
                    <= self.runtime.sample_interval_seconds * 1_000.0
                ),
                "applied": True,
            }
            self._last_frame_index = frame_index
            self._last_fingerprint = fingerprint
            self._last_response = dict(response)
            return 200, response

def _brain_item(target_id, satellite_id, snapshot):
    return {
        "satObj": target_id,
        "satId": int(satellite_id),
        "activityEncoding": "UINT8_NORMALIZED",
        "brainStateSource": "ONLINE_COOPERATIVE_TARGET_POSTERIOR",
        "directionRingNeuronCount": int(snapshot.direction_ring.size),
        "directionPhaseType": "ABSOLUTE_ARGUMENT_OF_LATITUDE",
        "directionZeroReference": "ASCENDING_NODE",
        "directionRing": _quantize(snapshot.direction_ring),
        "directionDegrees": float(
            np.degrees(snapshot.decoded_phase_rad) % 360.0
        ),
        "positionHeatmapShape": [3, 3],
        "positionHeatmapRadialCentersM": [-2_000.0, 0.0, 2_000.0],
        "positionHeatmapAlongTrackCentersM": [-2_000.0, 0.0, 2_000.0],
        "positionHeatmap": _quantize(snapshot.place_heatmap_3x3),
        "decodedRadialAlongTrackM": [
            float(value) for value in snapshot.decoded_rt_m
        ],
        "valid": bool(snapshot.valid),
        "status": "OK" if snapshot.valid else "INVALID",
    }


def _quantize(values):
    activity = np.asarray(values, dtype=float)
    activity = np.maximum(activity - float(np.min(activity)), 0.0)
    maximum = float(np.max(activity))
    if maximum <= np.finfo(float).tiny:
        activity = np.ones_like(activity)
        maximum = 1.0
    return np.rint(np.clip(activity / maximum, 0.0, 1.0) * 255.0).astype(
        int,
    ).reshape(-1).tolist()


def _utc_text(time_ms):
    value = datetime.fromtimestamp(time_ms / 1_000.0, tz=timezone.utc)
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
