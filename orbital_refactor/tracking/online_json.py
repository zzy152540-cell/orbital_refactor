from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .association import (
    OnlineMultiTargetTracker,
    UnlabeledObservationMessage,
    unlabeled_iod_observations_from_messages,
)


PROTOCOL_VERSION = "MULTITARGET-ONLINE-1.0"


def run_online_json_replay(payload: dict) -> dict:
    """Replay canonical one-epoch JSON frames through the persistent tracker."""

    if str(payload.get("protocolVersion")) != PROTOCOL_VERSION:
        raise ValueError(f"protocolVersion must be {PROTOCOL_VERSION!r}.")
    scene_id = _required_text(payload, "sceneId")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty array.")
    tracker = OnlineMultiTargetTracker(scene_id=scene_id)
    outputs = []
    last_timestamp = None
    for frame_index, frame in enumerate(frames):
        timestamp, messages, states, quaternions = parse_online_json_frame(frame)
        if last_timestamp is not None and timestamp <= last_timestamp:
            raise ValueError("Frame timestamps must be strictly increasing.")
        state_lookup = {
            (observer_id, timestamp): state
            for observer_id, state in states.items()
        }
        observations = unlabeled_iod_observations_from_messages(
            messages, observer_state_by_epoch=state_lookup,
        )
        update = tracker.step(
            timestamp=timestamp,
            observations=observations,
            observer_states_eci=states,
            q_eci2pri_by_observer=quaternions,
        )
        outputs.append(online_update_to_json(update, frame_index=frame_index))
        last_timestamp = timestamp
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "type": "MULTITARGET_ESTIMATION_REPLAY",
        "sceneId": scene_id,
        "frameCount": len(outputs),
        "frames": outputs,
    }


def parse_online_json_frame(frame):
    if not isinstance(frame, dict):
        raise TypeError("Each frame must be a JSON object.")
    timestamp = float(frame["timestamp"])
    observer_values = frame.get("observers")
    if not isinstance(observer_values, list) or not observer_values:
        raise ValueError("Each frame requires at least one observer.")
    states = {}
    quaternions = {}
    for value in observer_values:
        observer_id = _required_text(value, "observerId")
        if observer_id in states:
            raise ValueError(f"Duplicate observerId {observer_id!r} in one frame.")
        states[observer_id] = np.asarray(value["stateEci"], dtype=float).reshape(6)
        quaternions[observer_id] = np.asarray(
            value["qEci2PriWxyz"], dtype=float,
        ).reshape(4)
    messages = []
    for value in frame.get("detections", []):
        observer_id = _required_text(value, "observerId")
        if observer_id not in states:
            raise ValueError(f"Detection references unknown observer {observer_id!r}.")
        messages.append(UnlabeledObservationMessage(
            message_id=_required_text(value, "messageId"),
            observer_id=observer_id,
            timestamp=timestamp,
            modality=_required_text(value, "modality"),
            measurement=value["measurement"],
            covariance=value["covariance"],
            detection_group_id=_required_text(value, "detectionGroupId"),
            confidence=float(value.get("confidence", 1.0)),
            valid_flag=bool(value.get("valid", True)),
            metadata=dict(value.get("metadata", {})),
        ))
    return timestamp, tuple(messages), states, quaternions


def online_update_to_json(update, *, frame_index):
    targets = []
    for target_id, track in sorted(update.tracks_by_target.items()):
        estimate = track.estimate
        covariance = np.asarray(estimate.covariance_eci, dtype=float)
        maneuver = update.maneuver_by_target.get(target_id)
        targets.append({
            "targetId": target_id,
            "trackId": estimate.track_id,
            "lifecycle": track.lifecycle.value,
            "frame": "J2000_ECI",
            "x": float(estimate.state_eci[0]),
            "y": float(estimate.state_eci[1]),
            "z": float(estimate.state_eci[2]),
            "vx": float(estimate.state_eci[3]),
            "vy": float(estimate.state_eci[4]),
            "vz": float(estimate.state_eci[5]),
            "covarianceDiagonal": np.diag(covariance).tolist(),
            "contributingObserverIds": list(
                getattr(estimate, "contributing_observer_ids", ()),
            ),
            "maneuver": (
                None
                if maneuver is None
                else {
                    "suspected": maneuver.suspected,
                    "normalizedInnovationSquared": (
                        maneuver.normalized_innovation_squared
                    ),
                    "thresholdExceeded": maneuver.threshold_exceeded,
                    "processNoiseScale": maneuver.process_noise_scale,
                }
            ),
        })
    return {
        "type": "MULTITARGET_ESTIMATION_FRAME",
        "frameIndex": int(frame_index),
        "timestamp": float(update.timestamp),
        "association": {
            "matchCount": len(update.association.matches),
            "unassignedDetectionCount": len(
                update.association.unassigned_observation_indices
            ),
            "ambiguousDetectionCount": len(
                update.association.ambiguous_observation_indices
            ),
            "unobservedTargetIds": list(update.association.unobserved_target_ids),
        },
        "initializedTargetIds": sorted(update.initialized_states_by_target),
        "targets": targets,
    }


def run_online_json_replay_file(input_path, output_path):
    source = Path(input_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = run_online_json_replay(payload)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return result


def _required_text(value, key):
    result = str(value.get(key, "")).strip()
    if not result:
        raise ValueError(f"{key} must be a non-empty string.")
    return result

