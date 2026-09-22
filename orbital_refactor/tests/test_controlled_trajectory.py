import json
import socket
import threading
from urllib.request import Request, urlopen

from interfaces.controlled_trajectory import ControlledTrajectoryStreamer
from interfaces.scene_control import (
    SceneClockConfig,
    SceneClockController,
    create_scene_control_server,
)
from interfaces.trajectory_udp import TrajectoryUdpPublisher, decode_trajectory_datagram


class FakeMonotonic:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class RecordingPublisher:
    def __init__(self):
        self.indices = []

    def send_frame(self, trajectory, frame_index):
        self.indices.append(frame_index)
        return 100 + frame_index


def _trajectory():
    return {
        "schemaVersion": "TRAJECTORY-2.0",
        "sceneId": "scene-a", "timeBaseId": "TB-A", "datasetId": "EST-A",
        "startTimeMs": 1_000, "endTimeMs": 4_000,
        "sampleIntervalMs": 1_000, "frameCount": 4,
        "timeScale": "UTC", "frame": "J2000", "unit": "m", "unitV": "m/s",
        "frames": [
            {"frameIndex": i, "timeMs": 1_000 + i * 1_000,
             "time": f"frame-{i}", "satelliteList": []}
            for i in range(4)
        ],
    }


def _command(command_id, **extra):
    return {
        "protocolVersion": "SCENE-CTRL-2.0", "source": "main-display",
        "sceneId": "scene-a", "runId": "run-a", "commandId": command_id,
        **extra,
    }


def _post(base_url, path, payload):
    request = Request(
        base_url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json;charset=UTF-8"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        return json.load(response)


def test_streamer_waits_for_start_and_sends_every_due_frame_once():
    monotonic = FakeMonotonic()
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 4_000),
        monotonic=monotonic,
    )
    publisher = RecordingPublisher()
    streamer = ControlledTrajectoryStreamer(controller, _trajectory(), publisher)
    assert streamer.step().sent_frames == 0

    controller.set_run_state(_command("start", action="START"))
    assert streamer.step().sent_frames == 1
    monotonic.value += 2.1
    result = streamer.step()
    assert result.sent_frames == 2
    assert (result.first_frame_index, result.last_frame_index) == (1, 2)
    assert publisher.indices == [0, 1, 2]
    assert streamer.step().sent_frames == 0


def test_pause_freezes_output_and_resume_finishes_without_duplicates():
    monotonic = FakeMonotonic()
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 4_000),
        monotonic=monotonic,
    )
    publisher = RecordingPublisher()
    streamer = ControlledTrajectoryStreamer(controller, _trajectory(), publisher)
    controller.set_run_state(_command("start", action="START"))
    streamer.step()
    controller.set_run_state(_command("pause", action="PAUSE"))
    monotonic.value += 20.0
    assert streamer.step().sent_frames == 0

    controller.set_run_state(_command("resume", action="START"))
    monotonic.value += 4.0
    result = streamer.step()
    assert result.clock_state == "FINISHED"
    assert result.complete
    assert publisher.indices == [0, 1, 2, 3]


def test_stop_cancels_unsent_frames():
    monotonic = FakeMonotonic()
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 4_000),
        monotonic=monotonic,
    )
    publisher = RecordingPublisher()
    streamer = ControlledTrajectoryStreamer(controller, _trajectory(), publisher)
    controller.stop(_command("stop", reason="OPERATOR_STOP"))
    result = streamer.step()
    assert result.clock_state == "STOPPED"
    assert result.sent_frames == 0


def test_streamer_rejects_time_base_mismatch():
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-other", 1_000, 4_000),
    )
    try:
        ControlledTrajectoryStreamer(controller, _trajectory(), RecordingPublisher())
    except ValueError as error:
        assert "timeBaseId" in str(error)
    else:
        raise AssertionError("time-base mismatch was accepted")


def test_http_start_drives_real_udp_frames_over_loopback():
    monotonic = FakeMonotonic()
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 4_000),
        monotonic=monotonic,
    )
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    server = create_scene_control_server(controller, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = receiver.getsockname()
        with TrajectoryUdpPublisher(host, port) as publisher:
            streamer = ControlledTrajectoryStreamer(
                controller, _trajectory(), publisher,
            )
            body = json.dumps(_command("http-start", action="START")).encode()
            request = Request(
                f"http://127.0.0.1:{server.server_address[1]}"
                "/api/external/scene/run-state",
                data=body,
                headers={"Content-Type": "application/json;charset=UTF-8"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                assert json.load(response)["clockState"] == "RUNNING"
            monotonic.value += 2.1
            assert streamer.step().sent_frames == 3

        messages = [
            decode_trajectory_datagram(receiver.recvfrom(65_535)[0])
            for _ in range(3)
        ]
        assert [message["frameIndex"] for message in messages] == [0, 1, 2]
        assert [message["timeMs"] for message in messages] == [1_000, 2_000, 3_000]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        receiver.close()


def test_http_pause_resume_speed_and_stop_control_udp_output_end_to_end():
    monotonic = FakeMonotonic()
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 4_000),
        monotonic=monotonic,
    )
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    server = create_scene_control_server(controller, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        host, port = receiver.getsockname()
        with TrajectoryUdpPublisher(host, port) as publisher:
            streamer = ControlledTrajectoryStreamer(
                controller, _trajectory(), publisher,
            )
            assert streamer.step().sent_frames == 0

            started = _post(
                base, "/api/external/scene/run-state",
                _command("start-e2e", action="START"),
            )
            assert started["clockState"] == "RUNNING"
            assert streamer.step().sent_frames == 1

            monotonic.value += 1.0
            assert streamer.step().sent_frames == 1
            paused = _post(
                base, "/api/external/scene/run-state",
                _command("pause-e2e", action="PAUSE"),
            )
            assert paused["clockState"] == "PAUSED"
            monotonic.value += 30.0
            assert streamer.step().sent_frames == 0

            resumed = _post(
                base, "/api/external/scene/run-state",
                _command("resume-e2e", action="START"),
            )
            assert resumed["clockState"] == "RUNNING"
            faster = _post(
                base, "/api/external/scene/speed",
                _command("speed-e2e", speed=2),
            )
            assert faster["speed"] == 2
            monotonic.value += 0.5
            assert streamer.step().sent_frames == 1

            stopped = _post(
                base, "/api/external/scene/stop",
                _command("stop-e2e", reason="OPERATOR_STOP"),
            )
            assert stopped["clockState"] == "STOPPED"
            monotonic.value += 30.0
            assert streamer.step().sent_frames == 0

        messages = [
            decode_trajectory_datagram(receiver.recvfrom(65_535)[0])
            for _ in range(3)
        ]
        assert [message["frameIndex"] for message in messages] == [0, 1, 2]
        assert [message["timeMs"] for message in messages] == [1_000, 2_000, 3_000]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        receiver.close()
