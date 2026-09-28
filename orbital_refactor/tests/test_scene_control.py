import json
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from interfaces.scene_control import (
    SceneClockConfig,
    SceneClockController,
    SceneControlError,
    create_scene_control_server,
)


class FakeMonotonic:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


def _controller(end=12_000):
    clock = FakeMonotonic()
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, end),
        monotonic=clock,
    )
    return controller, clock


def _command(command_id, **extra):
    return {
        "protocolVersion": "SCENE-CTRL-2.0",
        "source": "main-display",
        "sceneId": "scene-a",
        "runId": "run-a",
        "commandId": command_id,
        **extra,
    }


def test_run_pause_speed_stop_and_idempotency():
    controller, clock = _controller()
    status, started = controller.set_run_state(_command("c1", action="START"))
    assert status == 200 and started["clockState"] == "RUNNING"
    assert started["clockRevision"] == 2

    clock.value += 2.0
    _, faster = controller.set_speed(_command("c2", speed=2))
    assert faster["timeMs"] == 3_000
    assert faster["clockRevision"] == 3

    _, replayed = controller.set_speed(_command("c2", speed=2))
    assert replayed == faster

    clock.value += 1.0
    _, paused = controller.set_run_state(_command("c3", action="PAUSE"))
    assert paused["timeMs"] == 5_000
    assert paused["clockState"] == "PAUSED"

    _, stopped = controller.stop(_command("c4", reason="USER_EXIT"))
    assert stopped["clockState"] == "STOPPED"
    with pytest.raises(SceneControlError) as error:
        controller.set_run_state(_command("c5", action="START"))
    assert error.value.status == 409


def test_command_id_cannot_be_reused_for_different_content():
    controller, _clock = _controller()
    controller.set_speed(_command("same", speed=2))
    with pytest.raises(SceneControlError) as error:
        controller.set_speed(_command("same", speed=5))
    assert error.value.status == 409


def test_running_clock_finishes_at_end_time():
    controller, clock = _controller(end=2_000)
    controller.set_run_state(_command("start", action="START"))
    clock.value += 10.0
    _, status = controller.status({
        "protocolVersion": "SCENE-CTRL-2.0", "source": "main-display",
        "sceneId": "scene-a",
    })
    assert status["clockState"] == "FINISHED"
    assert status["timeMs"] == 2_000


def test_required_activation_blocks_start_until_initial_conditions_are_accepted():
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 12_000),
        require_activation=True,
        activation_validator=lambda payload: {
            "acceptedSatelliteCount": payload["satelliteCount"],
            "initialConditionsAccepted": True,
        },
    )
    assert controller.snapshot()["clockState"] == "WAITING_ACTIVATION"
    with pytest.raises(SceneControlError) as error:
        controller.set_run_state(_command("early-start", action="START"))
    assert error.value.status == 409

    payload = {"sceneId": "scene-a", "satelliteCount": 31}
    _, activated = controller.activate(payload)
    assert activated["clockState"] == "READY"
    assert activated["acceptedSatelliteCount"] == 31
    assert controller.activation_payload == payload
    _, replayed = controller.activate(payload)
    assert replayed["applied"] is False

    _, started = controller.set_run_state(_command("start", action="START"))
    assert started["clockState"] == "RUNNING"


def test_stop_can_cancel_a_runtime_waiting_for_activation():
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 12_000),
        require_activation=True,
    )
    _, stopped = controller.stop(_command("cancel", reason="USER_EXIT"))
    assert stopped["clockState"] == "STOPPED"


def test_stopped_runtime_can_rearm_and_accept_a_new_activation():
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 12_000),
        require_activation=True,
        activation_validator=lambda payload: {
            "acceptedSatelliteCount": payload["satelliteCount"],
        },
    )
    first = {"sceneId": "scene-a", "satelliteCount": 31, "round": 1}
    controller.activate(first)
    controller.set_run_state(_command("start", action="START"))
    controller.stop(_command("stop", reason="OPERATOR_STOP"))

    waiting = controller.rearm_for_activation()

    assert waiting["clockState"] == "WAITING_ACTIVATION"
    assert waiting["timeMs"] == 1_000
    assert waiting["speed"] == 1
    assert waiting["appliedCommandId"] is None
    assert "acceptedSatelliteCount" not in waiting
    assert controller.activation_payload is None

    second = {"sceneId": "scene-a", "satelliteCount": 31, "round": 2}
    _, activated = controller.activate(second)
    assert activated["clockState"] == "READY"
    assert controller.activation_payload == second
    # Command IDs are scoped to one run and may be reused in the next round.
    _, restarted = controller.set_run_state(_command("start", action="START"))
    assert restarted["clockState"] == "RUNNING"


def test_rearm_rejects_an_active_run():
    controller, _clock = _controller()
    with pytest.raises(SceneControlError) as error:
        controller.rearm_for_activation()
    assert error.value.status == 409


def test_http_activation_endpoint_accepts_original_scene_body():
    controller = SceneClockController(
        SceneClockConfig("scene-a", "run-a", "TB-A", 1_000, 12_000),
        require_activation=True,
        activation_validator=lambda payload: {
            "acceptedSatelliteCount": payload["satelliteCount"],
        },
    )
    server = create_scene_control_server(controller, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}"
            "/api/external/scene/activate",
            data=json.dumps({"sceneId": "scene-a", "satelliteCount": 31}).encode(),
            headers={"Content-Type": "application/json;charset=UTF-8"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            activated = json.load(response)
        assert activated["clockState"] == "READY"
        assert activated["acceptedSatelliteCount"] == 31
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_receiver_accepts_commands_and_status_queries():
    controller, _clock = _controller()
    server = create_scene_control_server(controller, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = Request(
            base + "/api/external/scene/run-state",
            data=json.dumps(_command("http-start", action="START")).encode(),
            headers={"Content-Type": "application/json;charset=UTF-8"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            started = json.load(response)
        assert started["clockState"] == "RUNNING"

        query = urlencode({
            "protocolVersion": "SCENE-CTRL-2.0", "source": "main-display",
            "sceneId": "scene-a", "runId": "run-a",
        })
        with urlopen(base + "/api/external/scene/status?" + query, timeout=2) as response:
            status = json.load(response)
        assert status["applied"] is None
        assert status["appliedCommandId"] == "http-start"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_receiver_rejects_bad_token():
    controller, _clock = _controller()
    server = create_scene_control_server(
        controller, "127.0.0.1", 0, auth_token="secret",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/api/external/scene/status"
        with pytest.raises(HTTPError) as error:
            urlopen(url, timeout=2)
        assert error.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_runtime_diagnostics_are_separate_from_standard_status():
    controller, _clock = _controller()
    original_status_keys = set(controller.snapshot())
    controller.set_run_state(_command("start", action="START"))
    assert controller.note_udp_send_failure("network unavailable") == 1
    assert controller.note_udp_send_failure("network unavailable") == 2
    controller.fail_current_run("UDP_CONSECUTIVE_SEND_FAILURE", "network unavailable")
    controller.rearm_for_activation()

    _, diagnostics = controller.runtime_diagnostics()

    assert diagnostics["serviceState"] == "LISTENING"
    assert diagnostics["lastRunStatus"] == "FAILED"
    assert diagnostics["lastErrorCode"] == "UDP_CONSECUTIVE_SEND_FAILURE"
    assert diagnostics["lastErrorMessage"] == "network unavailable"
    assert diagnostics["lastErrorTimeMs"] is not None
    assert diagnostics["consecutiveUdpFailures"] == 2
    assert set(controller.snapshot()) == original_status_keys


def test_http_runtime_diagnostics_endpoint_does_not_require_status_query():
    controller, _clock = _controller()
    server = create_scene_control_server(controller, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = (
            f"http://127.0.0.1:{server.server_address[1]}"
            "/api/external/scene/runtime-diagnostics"
        )
        with urlopen(url, timeout=2) as response:
            diagnostics = json.load(response)
        assert diagnostics["lastRunStatus"] == "NEVER_RUN"
        assert diagnostics["lastErrorCode"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
