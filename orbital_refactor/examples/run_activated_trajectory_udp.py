"""Activate a prepared trajectory runtime from one deployment config file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time

from exporters.display_telemetry import (
    load_display_telemetry,
    validate_display_telemetry,
)
from interfaces.controlled_trajectory import ControlledTrajectoryStreamer
from interfaces.scene_control import (
    ALLOWED_SPEEDS,
    SceneClockConfig,
    SceneClockController,
    create_scene_control_server,
)
from interfaces.trajectory_udp import (
    TrajectoryUdpPublisher,
    load_trajectory_time_series,
    validate_scene_activation_for_trajectory,
)


def load_runtime_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Runtime config must be a JSON object.")
    required = ("trajectory", "udpHost", "udpPort", "controlHost", "controlPort", "runId")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Runtime config is missing fields: {missing}")
    trajectory = Path(str(payload["trajectory"]))
    if not trajectory.is_absolute():
        trajectory = config_path.parent / trajectory
    telemetry_value = payload.get("displayTelemetry")
    display_telemetry = None
    if telemetry_value:
        display_telemetry = Path(str(telemetry_value))
        if not display_telemetry.is_absolute():
            display_telemetry = config_path.parent / display_telemetry
    speed = int(payload.get("speed", 1))
    if speed not in ALLOWED_SPEEDS:
        raise ValueError(f"speed must be one of {sorted(ALLOWED_SPEEDS)}")
    udp_port = int(payload["udpPort"])
    control_port = int(payload["controlPort"])
    if not 1 <= udp_port <= 65_535 or not 1 <= control_port <= 65_535:
        raise ValueError("UDP and control ports must be in 1..65535.")
    poll_ms = float(payload.get("pollMs", 10.0))
    if poll_ms <= 0.0:
        raise ValueError("pollMs must be positive.")
    return {
        "trajectory": trajectory,
        "display_telemetry": display_telemetry,
        "udp_host": str(payload["udpHost"]),
        "udp_port": udp_port,
        "control_host": str(payload["controlHost"]),
        "control_port": control_port,
        "run_id": str(payload["runId"]),
        "speed": speed,
        "poll_ms": poll_ms,
        "auth_token": payload.get("authToken"),
    }


def run_activated_runtime(config_path: str | Path) -> int:
    config = load_runtime_config(config_path)
    trajectory = load_trajectory_time_series(config["trajectory"])
    display_telemetry = None
    if config["display_telemetry"] is not None:
        display_telemetry = load_display_telemetry(config["display_telemetry"])
        validate_display_telemetry(display_telemetry, trajectory)
    clock_config = SceneClockConfig(
        scene_id=str(trajectory["sceneId"]),
        run_id=config["run_id"],
        time_base_id=str(trajectory["timeBaseId"]),
        start_time_ms=int(trajectory["startTimeMs"]),
        end_time_ms=int(trajectory["endTimeMs"]),
        sample_interval_ms=int(trajectory["sampleIntervalMs"]),
        speed=config["speed"],
    )
    controller = SceneClockController(
        clock_config,
        require_activation=True,
        activation_validator=lambda payload: validate_scene_activation_for_trajectory(
            payload, trajectory,
        ),
    )
    server = create_scene_control_server(
        controller, config["control_host"], config["control_port"],
        auth_token=config["auth_token"],
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    listening = controller.snapshot(applied=None)
    listening.update({
        "event": "SCENE_RUNTIME_LISTENING",
        "processId": os.getpid(),
        "controlHost": config["control_host"],
        "controlPort": server.server_address[1],
        "udpHost": config["udp_host"],
        "udpPort": config["udp_port"],
        "trajectory": str(config["trajectory"]),
        "activationEndpoint": "/api/external/scene/activate",
    })
    print(json.dumps(listening, ensure_ascii=False, separators=(",", ":")), flush=True)

    exit_code = 0
    terminal_state = "WAITING_ACTIVATION"
    maximum_udp_failures = 5
    try:
        with TrajectoryUdpPublisher(config["udp_host"], config["udp_port"]) as publisher:
            streamer = ControlledTrajectoryStreamer(controller, trajectory, publisher)
            ready_emitted = False
            while True:
                try:
                    result = streamer.step()
                    if (
                        display_telemetry is not None
                        and result.last_frame_index is not None
                    ):
                        publisher.send_display_telemetry_frame(
                            display_telemetry, result.last_frame_index,
                        )
                    if result.sent_frames:
                        controller.note_udp_send_success()
                except OSError as exc:
                    failures = controller.note_udp_send_failure(str(exc))
                    print(json.dumps({
                        "event": "SCENE_RUNTIME_UDP_WARNING",
                        "sceneId": clock_config.scene_id,
                        "runId": clock_config.run_id,
                        "consecutiveFailures": failures,
                        "message": str(exc),
                    }, ensure_ascii=False, separators=(",", ":")), flush=True)
                    if failures < maximum_udp_failures:
                        time.sleep(config["poll_ms"] / 1000.0)
                        continue
                    controller.fail_current_run(
                        "UDP_CONSECUTIVE_SEND_FAILURE",
                        f"UDP transmission failed {failures} consecutive times: {exc}",
                    )
                    print(json.dumps({
                        "event": "SCENE_RUNTIME_RUN_ERROR",
                        "sceneId": clock_config.scene_id,
                        "runId": clock_config.run_id,
                        "errorCode": "UDP_CONSECUTIVE_SEND_FAILURE",
                        "message": str(exc),
                    }, ensure_ascii=False, separators=(",", ":")), flush=True)
                    controller.rearm_for_activation()
                    streamer = ControlledTrajectoryStreamer(
                        controller, trajectory, publisher,
                    )
                    ready_emitted = False
                    terminal_state = "WAITING_ACTIVATION"
                    continue
                if result.clock_state == "READY" and not ready_emitted:
                    ready = controller.snapshot(applied=None)
                    ready.update({
                        "event": "SCENE_RUNTIME_READY",
                        "processId": os.getpid(),
                        "controlHost": config["control_host"],
                        "controlPort": server.server_address[1],
                        "udpHost": config["udp_host"],
                        "udpPort": config["udp_port"],
                    })
                    print(json.dumps(
                        ready, ensure_ascii=False, separators=(",", ":"),
                    ), flush=True)
                    ready_emitted = True
                if result.clock_state == "STOPPED":
                    print(json.dumps({
                        "event": "SCENE_RUNTIME_RUN_ENDED",
                        "sceneId": clock_config.scene_id,
                        "runId": clock_config.run_id,
                        "clockState": "STOPPED",
                    }, ensure_ascii=False, separators=(",", ":")), flush=True)
                    controller.rearm_for_activation()
                    streamer = ControlledTrajectoryStreamer(
                        controller, trajectory, publisher,
                    )
                    ready_emitted = False
                    terminal_state = "WAITING_ACTIVATION"
                if result.clock_state == "FINISHED" and result.complete:
                    print(json.dumps({
                        "event": "SCENE_RUNTIME_RUN_ENDED",
                        "sceneId": clock_config.scene_id,
                        "runId": clock_config.run_id,
                        "clockState": "FINISHED",
                    }, ensure_ascii=False, separators=(",", ":")), flush=True)
                    controller.rearm_for_activation()
                    streamer = ControlledTrajectoryStreamer(
                        controller, trajectory, publisher,
                    )
                    ready_emitted = False
                    terminal_state = "WAITING_ACTIVATION"
                time.sleep(config["poll_ms"] / 1000.0)
    except KeyboardInterrupt:
        terminal_state = "INTERRUPTED"
        exit_code = 130
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
    print(json.dumps({
        "event": "SCENE_RUNTIME_EXIT",
        "sceneId": clock_config.scene_id,
        "runId": clock_config.run_id,
        "clockState": terminal_state,
        "exitCode": exit_code,
    }, ensure_ascii=False, separators=(",", ":")), flush=True)
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(run_activated_runtime(args.config))
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({
            "event": "SCENE_RUNTIME_ERROR",
            "errorType": type(exc).__name__,
            "message": str(exc),
        }, ensure_ascii=False, separators=(",", ":")), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
