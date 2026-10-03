import json

import numpy as np

from orbital_core.coordinates import build_rtn_quaternion
from orbital_core.dynamics import propagate_absolute_orbit
from orbital_core.measurements import measure_relative_range, measure_relative_range_rate
from tests.multitarget.test_measurement_association import _state
from tracking.online_json import PROTOCOL_VERSION, run_online_json_replay, run_online_json_replay_file


def _payload():
    times = np.arange(0.0, 41.0, 10.0)
    targets = [
        propagate_absolute_orbit(_state(0.5), times),
        propagate_absolute_orbit(_state(3.5), times),
    ]
    observers = {
        "observer-1": propagate_absolute_orbit(_state(-1.0, 54.7, 698e3), times),
        "observer-2": propagate_absolute_orbit(_state(-0.5, 54.9, 702e3), times),
    }
    frames = []
    for epoch_index, timestamp in enumerate(times):
        observer_values = []
        detections = []
        for observer_id, history in observers.items():
            observer = history[epoch_index]
            observer_values.append({
                "observerId": observer_id,
                "stateEci": observer.tolist(),
                "qEci2PriWxyz": build_rtn_quaternion(observer).tolist(),
            })
            for target_index, target_history in enumerate(targets):
                target = target_history[epoch_index]
                relative = target[:3] - observer[:3]
                group = f"{observer_id}-{epoch_index}-{target_index}"
                detections.extend((
                    {
                        "messageId": f"radar-{group}",
                        "observerId": observer_id,
                        "modality": "RADAR",
                        "detectionGroupId": group,
                        "measurement": [
                            measure_relative_range(observer, target),
                            measure_relative_range_rate(observer, target),
                        ],
                        "covariance": np.diag([10.0, 0.05]) ** 2,
                    },
                    {
                        "messageId": f"los-{group}",
                        "observerId": observer_id,
                        "modality": "LOS",
                        "detectionGroupId": group,
                        "measurement": (relative / np.linalg.norm(relative)).tolist(),
                        "covariance": (np.diag([2e-5, 2e-5]) ** 2).tolist(),
                        "metadata": {"sourceModality": "INFRARED"},
                    },
                ))
        # Convert NumPy radar covariance to ordinary JSON-compatible lists.
        for detection in detections:
            if isinstance(detection["covariance"], np.ndarray):
                detection["covariance"] = detection["covariance"].tolist()
        frames.append({
            "frameIndex": epoch_index,
            "timestamp": timestamp,
            "observers": observer_values,
            "detections": detections,
        })
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "type": "MULTITARGET_SENSOR_REPLAY",
        "sceneId": "json-replay-two-targets",
        "frames": frames,
    }


def test_json_replay_runs_target_free_frames_to_stable_output_contract():
    result = run_online_json_replay(_payload())

    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["frameCount"] == 5
    assert result["frames"][0]["targets"] == []
    assert len(result["frames"][2]["initializedTargetIds"]) == 2
    assert len(result["frames"][-1]["targets"]) == 2
    for target in result["frames"][-1]["targets"]:
        assert target["lifecycle"] == "TRACKING"
        assert target["frame"] == "J2000_ECI"
        assert all(key in target for key in ("x", "y", "z", "vx", "vy", "vz"))
        assert "maneuver" in target


def test_json_replay_file_writes_parseable_result(tmp_path):
    source = tmp_path / "input.json"
    destination = tmp_path / "output.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    result = run_online_json_replay_file(source, destination)

    assert destination.exists()
    assert json.loads(destination.read_text(encoding="utf-8")) == result

