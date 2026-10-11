import json
import socket
import threading
from urllib.request import Request, urlopen

import numpy as np

from brain_inspired.cann_measurement_adapter import preprocess_observation
from exporters.display_telemetry import encode_display_telemetry_datagram
import pytest

from interfaces.online_multitarget_runtime import (
    ONLINE_MEASUREMENT_FRAME_PATH,
    OnlineMultiTargetRuntime,
    OnlineMultiTargetRuntimeEndpoint,
)
from interfaces.scene_control import (
    SceneClockConfig,
    SceneClockController,
    SceneControlError,
    create_scene_control_server,
)
from interfaces.trajectory_udp import (
    TrajectoryUdpPublisher,
    decode_trajectory_datagram,
)
from tests.multitarget.test_online_json_replay import _payload
from tracking.association.online_tracker import OnlineMultiTargetTracker


class _RecordingPreprocessor:
    def __init__(self):
        self.calls = 0
        self.processed = []

    def process(self, observations, timestamps):
        assert np.asarray(timestamps).shape == (1,)
        self.calls += 1
        result = [
            preprocess_observation(
                item,
                measurement=item.measurement,
                diagnostics={"testPreprocessor": True},
            )
            for item in observations
        ]
        self.processed.extend(result)
        return result


class _RecordingPublisher:
    def __init__(self):
        self.trajectory = []
        self.display = []

    def send_trajectory_message(self, message):
        self.trajectory.append(message)
        return 101

    def send_display_telemetry_message(self, message):
        self.display.append(message)
        return 202


def _runtime():
    return OnlineMultiTargetRuntime(
        scene_id="json-replay-two-targets",
        time_base_id="TB-ONLINE",
        dataset_id="ONLINE-IOD",
        start_time_ms=1_789_704_000_000,
        sample_interval_seconds=10.0,
    )


def test_online_runtime_runs_iod_tracking_and_brain_output_per_frame():
    runtime = _runtime()
    outputs = [
        runtime.process_frame(frame) for frame in _payload()["frames"]
    ]

    assert outputs[0].trajectory_message["satelliteList"] == []
    assert len(outputs[2].initialized_target_ids) == 2
    assert len(outputs[-1].trajectory_message["satelliteList"]) == 2
    assert len(outputs[-1].display_message["cannBySatellite"]) == 2
    for target in outputs[-1].trajectory_message["satelliteList"]:
        assert list(target["position"]) == ["x", "y", "z", "vx", "vy", "vz"]
    for brain in outputs[-1].display_message["cannBySatellite"]:
        assert brain["directionRingNeuronCount"] == 180
        assert len(brain["directionRing"]) == 180
        assert brain["positionHeatmapShape"] == [3, 3]
        assert len(brain["positionHeatmap"]) == 9
        assert brain["valid"]
    assert outputs[-1].time_ms == 1_789_704_040_000
    assert encode_display_telemetry_datagram(outputs[-1].display_message)


def test_online_runtime_publishes_trajectory_and_display_on_one_udp_port():
    runtime = _runtime()
    output = None
    for frame in _payload()["frames"]:
        output = runtime.process_frame(frame)

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    try:
        host, port = receiver.getsockname()
        with TrajectoryUdpPublisher(host, port) as publisher:
            sent = runtime.publish(output, publisher)
        first = receiver.recvfrom(65_535)[0]
        second = receiver.recvfrom(65_535)[0]
    finally:
        receiver.close()

    assert sent == (len(first), len(second))
    assert decode_trajectory_datagram(first)["frameIndex"] == 4
    assert b'"messageType":"DISPLAY_TELEMETRY"' in second


def test_online_runtime_rejects_a_noncontiguous_clock():
    runtime = _runtime()
    frames = _payload()["frames"]
    runtime.process_frame(frames[0])
    frames[1]["timestamp"] = 11.0

    try:
        runtime.process_frame(frames[1])
    except ValueError as error:
        assert "configured interval" in str(error)
    else:
        raise AssertionError("Expected online clock interval validation failure.")


def test_online_runtime_uses_persistent_per_stream_cann_preprocessors():
    processors = {}

    def factory(observer_id, target_id):
        processor = _RecordingPreprocessor()
        processors[(observer_id, target_id)] = processor
        return processor

    tracker = OnlineMultiTargetTracker(
        scene_id="json-replay-two-targets",
        measurement_preprocessor_factory=factory,
    )
    runtime = OnlineMultiTargetRuntime(
        scene_id="json-replay-two-targets",
        time_base_id="TB-ONLINE",
        dataset_id="ONLINE-IOD",
        start_time_ms=1_789_704_000_000,
        sample_interval_seconds=10.0,
        tracker=tracker,
    )

    outputs = [runtime.process_frame(frame) for frame in _payload()["frames"]]

    assert processors
    assert all(processor.calls > 0 for processor in processors.values())
    assert outputs[-1].display_message["estimation"][
        "cannPreprocessedObservationCount"
    ] > 0
    assert all(
        observation.metadata.get("cann_diagnostics", {}).get("testPreprocessor")
        for processor in processors.values()
        for observation in processor.processed
    )


def test_online_runtime_maps_absolute_source_time_to_configured_time_base():
    runtime = _runtime()
    frame = dict(_payload()["frames"][0])
    frame["timestamp"] = 1_789_704_000.0

    output = runtime.process_frame(frame)

    assert output.time_ms == 1_789_704_000_000


def test_online_endpoint_requires_running_and_is_idempotent():
    runtime = _runtime()
    controller = SceneClockController(SceneClockConfig(
        "json-replay-two-targets", "run-online", "TB-ONLINE",
        1_789_704_000_000, 1_789_704_100_000,
        sample_interval_ms=10_000,
    ))
    publisher = _RecordingPublisher()
    endpoint = OnlineMultiTargetRuntimeEndpoint(runtime, controller, publisher)
    frame = _payload()["frames"][0]

    assert list(endpoint.post_routes) == [ONLINE_MEASUREMENT_FRAME_PATH]
    with pytest.raises(SceneControlError) as error:
        endpoint.handle(frame)
    assert error.value.status == 409

    controller.set_run_state({
        "protocolVersion": "SCENE-CTRL-2.0",
        "source": "main-display",
        "sceneId": "json-replay-two-targets",
        "runId": "run-online",
        "commandId": "start-online",
        "action": "START",
    })
    status, first = endpoint.handle(frame)
    repeated_status, repeated = endpoint.handle(frame)

    assert status == repeated_status == 200
    assert first["applied"] is True
    assert repeated["applied"] is False
    assert first["trajectoryBytes"] == 101
    assert first["displayTelemetryBytes"] == 202
    assert first["algorithmProcessingMs"] >= 0.0
    assert first["udpPublicationMs"] >= 0.0
    assert first["withinSampleInterval"] is True
    assert len(publisher.trajectory) == len(publisher.display) == 1


def test_online_endpoint_rejects_changed_reuse_of_frame_index():
    runtime = _runtime()
    controller = SceneClockController(SceneClockConfig(
        "json-replay-two-targets", "run-online", "TB-ONLINE",
        1_789_704_000_000, 1_789_704_100_000,
        sample_interval_ms=10_000,
    ))
    controller.set_run_state({
        "protocolVersion": "SCENE-CTRL-2.0",
        "source": "main-display",
        "sceneId": "json-replay-two-targets",
        "runId": "run-online",
        "commandId": "start-online",
        "action": "START",
    })
    endpoint = OnlineMultiTargetRuntimeEndpoint(
        runtime, controller, _RecordingPublisher(),
    )
    frame = _payload()["frames"][0]
    endpoint.handle(frame)
    changed = dict(frame)
    changed["timestamp"] = 1.0

    with pytest.raises(SceneControlError) as error:
        endpoint.handle(changed)
    assert error.value.status == 409


def test_online_endpoint_reset_starts_a_fresh_frame_sequence():
    controller = SceneClockController(
        SceneClockConfig(
            "json-replay-two-targets", "run-online", "TB-ONLINE",
            1_789_704_000_000, 1_789_704_100_000,
            sample_interval_ms=10_000,
        ),
        require_activation=True,
    )
    publisher = _RecordingPublisher()
    endpoint = OnlineMultiTargetRuntimeEndpoint(_runtime(), controller, publisher)
    command = {
        "protocolVersion": "SCENE-CTRL-2.0",
        "source": "main-display",
        "sceneId": "json-replay-two-targets",
        "runId": "run-online",
        "commandId": "start-online",
        "action": "START",
    }
    controller.activate({"sceneId": "json-replay-two-targets", "round": 1})
    controller.set_run_state(command)
    endpoint.handle(_payload()["frames"][0])
    controller.stop({
        **command, "commandId": "stop-online", "reason": "OPERATOR_STOP",
    })
    endpoint.reset(_runtime())
    controller.rearm_for_activation()
    controller.activate({"sceneId": "json-replay-two-targets", "round": 2})
    controller.set_run_state({**command, "commandId": "start-online-2"})

    _, response = endpoint.handle(_payload()["frames"][0])

    assert response["frameIndex"] == 0
    assert len(publisher.trajectory) == 2


def test_http_measurement_frame_runs_algorithm_and_publishes_udp():
    runtime = _runtime()
    controller = SceneClockController(SceneClockConfig(
        "json-replay-two-targets", "run-online", "TB-ONLINE",
        1_789_704_000_000, 1_789_704_100_000,
        sample_interval_ms=10_000,
    ))
    controller.set_run_state({
        "protocolVersion": "SCENE-CTRL-2.0",
        "source": "main-display",
        "sceneId": "json-replay-two-targets",
        "runId": "run-online",
        "commandId": "start-online",
        "action": "START",
    })
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    host, port = receiver.getsockname()
    publisher = TrajectoryUdpPublisher(host, port)
    endpoint = OnlineMultiTargetRuntimeEndpoint(runtime, controller, publisher)
    server = create_scene_control_server(
        controller, "127.0.0.1", 0, post_routes=endpoint.post_routes,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}"
            + ONLINE_MEASUREMENT_FRAME_PATH,
            data=json.dumps(_payload()["frames"][0]).encode("utf-8"),
            headers={"Content-Type": "application/json;charset=UTF-8"},
            method="POST",
        )
        with urlopen(request, timeout=2) as http_response:
            response = json.load(http_response)
        datagrams = [receiver.recvfrom(65_535)[0] for _ in range(2)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        publisher.close()
        receiver.close()

    assert response["applied"] is True
    assert response["activeTargetCount"] == 0
    assert decode_trajectory_datagram(datagrams[0])["frameIndex"] == 0
    assert b'"messageType":"DISPLAY_TELEMETRY"' in datagrams[1]
