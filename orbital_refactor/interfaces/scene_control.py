"""SCENE-CTRL-2.0 HTTP command receiver and authoritative scene clock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlparse


PROTOCOL_VERSION = "SCENE-CTRL-2.0"
ALLOWED_SPEEDS = frozenset({1, 2, 5, 10, 30, 60, 120, 300})
TERMINAL_STATES = frozenset({"FINISHED", "STOPPED"})


def _utc_text(time_ms: int) -> str:
    value = datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class SceneClockConfig:
    scene_id: str
    run_id: str
    time_base_id: str
    start_time_ms: int
    end_time_ms: int
    sample_interval_ms: int = 1000
    speed: int = 1

    def __post_init__(self) -> None:
        if not self.scene_id or not self.run_id or not self.time_base_id:
            raise ValueError("Scene, run, and time-base identifiers are required.")
        if self.scene_id == self.run_id:
            raise ValueError("sceneId and runId must be different.")
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("endTimeMs must not precede startTimeMs.")
        if self.sample_interval_ms <= 0:
            raise ValueError("sampleIntervalMs must be positive.")
        if self.speed not in ALLOWED_SPEEDS:
            raise ValueError("Unsupported initial scene speed.")

    @property
    def frame_count(self) -> int:
        return (self.end_time_ms - self.start_time_ms) // self.sample_interval_ms + 1


class SceneControlError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = message


class SceneClockController:
    """Thread-safe command state machine with a monotonic wall-clock anchor."""

    def __init__(
        self,
        config: SceneClockConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._state = "READY"
        self._speed = config.speed
        self._time_ms = config.start_time_ms
        self._anchor_monotonic = monotonic()
        self._revision = 1
        self._applied_command_id: str | None = None
        self._commands: dict[str, tuple[str, int, dict[str, Any]]] = {}

    def _settle(self) -> None:
        if self._state != "RUNNING":
            return
        now = self._monotonic()
        elapsed_ms = max(0.0, now - self._anchor_monotonic) * 1000.0
        self._time_ms = min(
            self.config.end_time_ms,
            int(self._time_ms + elapsed_ms * self._speed),
        )
        self._anchor_monotonic = now
        if self._time_ms >= self.config.end_time_ms:
            self._time_ms = self.config.end_time_ms
            self._state = "FINISHED"
            self._revision += 1

    def snapshot(
        self,
        *,
        code: int = 200,
        message: str = "success",
        applied: bool | None = None,
        command_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._settle()
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "code": code,
                "message": message,
                "sceneId": self.config.scene_id,
                "runId": self.config.run_id,
                "timeBaseId": self.config.time_base_id,
                "clockState": self._state,
                "timeMs": self._time_ms,
                "time": _utc_text(self._time_ms),
                "speed": self._speed,
                "clockRevision": self._revision,
                "appliedCommandId": (
                    command_id if command_id is not None else self._applied_command_id
                ),
                "applied": applied,
                "startTimeMs": self.config.start_time_ms,
                "endTimeMs": self.config.end_time_ms,
                "sampleIntervalMs": self.config.sample_interval_ms,
                "frameCount": self.config.frame_count,
            }

    def _validate_identity(self, request: Mapping[str, Any]) -> None:
        required = ("protocolVersion", "source", "sceneId", "runId")
        if any(not isinstance(request.get(name), str) or not request[name] for name in required):
            raise SceneControlError(400, "missing or invalid common field")
        if request["protocolVersion"] != PROTOCOL_VERSION:
            raise SceneControlError(400, "unsupported protocolVersion")
        if request["sceneId"] != self.config.scene_id:
            raise SceneControlError(404, "sceneId does not exist")
        if request["runId"] != self.config.run_id:
            raise SceneControlError(404, "runId does not exist")

    def _command(
        self,
        request: Mapping[str, Any],
        command_type: str,
        apply: Callable[[], bool],
    ) -> tuple[int, dict[str, Any]]:
        with self._lock:
            self._validate_identity(request)
            command_id = request.get("commandId")
            if not isinstance(command_id, str) or not command_id:
                raise SceneControlError(400, "missing or invalid commandId")
            fingerprint = json.dumps(request, sort_keys=True, separators=(",", ":"))
            previous = self._commands.get(command_id)
            if previous is not None:
                old_fingerprint, status, response = previous
                if old_fingerprint != fingerprint:
                    raise SceneControlError(409, "commandId was used for another command")
                return status, dict(response)
            self._settle()
            if self._state in TERMINAL_STATES:
                raise SceneControlError(409, "runId is stale or state is terminal")
            changed = apply()
            if changed:
                self._revision += 1
            self._applied_command_id = command_id
            response = self.snapshot(applied=changed, command_id=command_id)
            self._commands[command_id] = (fingerprint, 200, dict(response))
            return 200, response

    def set_run_state(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        action = request.get("action")
        if action not in {"START", "PAUSE"}:
            raise SceneControlError(422, "action must be START or PAUSE")

        def apply() -> bool:
            target = "RUNNING" if action == "START" else (
                "PAUSED" if self._state == "RUNNING" else self._state
            )
            if target == self._state:
                return False
            self._state = target
            self._anchor_monotonic = self._monotonic()
            return True

        return self._command(request, "run-state", apply)

    def set_speed(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        speed = request.get("speed")
        if isinstance(speed, bool) or speed not in ALLOWED_SPEEDS:
            raise SceneControlError(422, "speed is outside the allowed set")

        def apply() -> bool:
            if speed == self._speed:
                return False
            self._speed = int(speed)
            self._anchor_monotonic = self._monotonic()
            return True

        return self._command(request, "speed", apply)

    def stop(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if request.get("reason") not in {
            "USER_EXIT", "OPERATOR_STOP", "SYSTEM_SHUTDOWN",
        }:
            raise SceneControlError(422, "unsupported stop reason")

        def apply() -> bool:
            self._state = "STOPPED"
            return True

        return self._command(request, "stop", apply)

    def status(self, query: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        normalized = {
            name: value[0] if isinstance(value, list) and value else value
            for name, value in query.items()
        }
        normalized.setdefault("runId", self.config.run_id)
        self._validate_identity(normalized)
        return 200, self.snapshot(applied=None)

    def error_response(self, error: SceneControlError) -> dict[str, Any]:
        return self.snapshot(code=error.status, message=error.message, applied=False)


def create_scene_control_server(
    controller: SceneClockController,
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    """Create, but do not start, the SCENE-CTRL HTTP server."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "SceneControl/2.0"

        def _write(self, status: int, payload: Mapping[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json;charset=UTF-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            return auth_token is None or self.headers.get("X-Auth-Token") == auth_token

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SceneControlError(400, "invalid UTF-8 JSON body") from exc
            if not isinstance(payload, dict):
                raise SceneControlError(400, "request body must be a JSON object")
            return payload

        def _dispatch(self, method: str) -> None:
            if not self._authorized():
                self._write(401, {"protocolVersion": PROTOCOL_VERSION, "code": 401,
                                  "message": "unauthorized"})
                return
            parsed = urlparse(self.path)
            try:
                if method == "GET" and parsed.path == "/api/external/scene/status":
                    status, response = controller.status(parse_qs(parsed.query))
                elif method == "POST":
                    body = self._body()
                    routes = {
                        "/api/external/scene/run-state": controller.set_run_state,
                        "/api/external/scene/speed": controller.set_speed,
                        "/api/external/scene/stop": controller.stop,
                    }
                    if parsed.path not in routes:
                        raise SceneControlError(404, "endpoint does not exist")
                    status, response = routes[parsed.path](body)
                else:
                    raise SceneControlError(404, "endpoint does not exist")
            except SceneControlError as exc:
                status, response = exc.status, controller.error_response(exc)
            except Exception:
                error = SceneControlError(500, "internal scene-control error")
                status, response = 500, controller.error_response(error)
            self._write(status, response)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, int(port)), Handler)
