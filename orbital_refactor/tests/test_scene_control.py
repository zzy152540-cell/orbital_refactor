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
