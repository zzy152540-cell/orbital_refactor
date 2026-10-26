"""Run reproducible file, HTTP, and UDP public-interface checks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import socket
import threading
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

import numpy as np

from examples.run_cross_platform_smoke import run_smoke
from examples.run_standard_interface import build_demo_input
from interfaces.interface_contracts import (
    InterfaceValidationError,
    validate_initial_state,
)
from interfaces.scene_control import (
    SceneClockConfig,
    SceneClockController,
    create_scene_control_server,
)
from interfaces.trajectory_udp import (
    TrajectoryUdpPublisher,
    decode_trajectory_datagram,
    encode_trajectory_datagram,
    trajectory_frame_datagram,
)


def run_interface_conformance() -> dict:
    checks: list[dict] = []

    def check(name, callback):
        try:
            details = callback()
        except Exception as exc:
            checks.append({
                "name": name, "passed": False,
                "errorType": type(exc).__name__, "message": str(exc),
            })
        else:
            checks.append({"name": name, "passed": True, "details": details})

    check("PUBLIC_JSON_NPZ_ROUND_TRIP", run_smoke)
    check("INVALID_INPUT_REJECTION", _invalid_input_rejection)
    check("SCENE_CONTROL_HTTP_LOOPBACK", _http_loopback)
    check("TRAJECTORY_UDP_LOOPBACK", _udp_loopback)
    return {
        "schemaVersion": "interface-conformance-acceptance.v1",
        "transportScope": ["JSON+NPZ", "HTTP", "IPv4-UDP"],
        "checks": checks,
        "passedCount": sum(bool(item["passed"]) for item in checks),
        "checkCount": len(checks),
        "overallPassed": bool(checks and all(item["passed"] for item in checks)),
        "externalToolsRequired": False,
        "externalToolsNote": (
            "Postman and Packet Sender may be used for manual demonstration, "
            "but are not part of the automatic verdict."
        ),
    }


def _invalid_input_rejection() -> dict:
    initial = build_demo_input().initial_state
    covariance = np.array(initial.covariance, copy=True)
    covariance[0, 1] += 1.0
    try:
        validate_initial_state(replace(initial, covariance=covariance))
    except InterfaceValidationError as exc:
        return exc.to_dict()
    raise AssertionError("Invalid asymmetric covariance was accepted.")


def _http_loopback() -> dict:
    controller = SceneClockController(SceneClockConfig(
        "interface-scene", "interface-run", "interface-time", 1_000, 2_000,
    ))
    server = create_scene_control_server(controller, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        query = urlencode({
            "protocolVersion": "SCENE-CTRL-2.0",
            "source": "acceptance-diagnostic",
            "sceneId": "interface-scene", "runId": "interface-run",
        })
        url = (
            f"http://127.0.0.1:{server.server_address[1]}"
            f"/api/external/scene/status?{query}"
        )
        # This endpoint is deliberately bound to loopback.  Do not let an
        # inherited HTTP(S)_PROXY route the acceptance request to a corporate
        # or VM proxy; the check is for the local scene-control interface.
        direct_opener = build_opener(ProxyHandler({}))
        with direct_opener.open(url, timeout=2.0) as response:
            payload = json.load(response)
        if payload.get("clockState") != "READY":
            raise AssertionError("Unexpected initial scene-control state.")
        return {
            "protocolVersion": payload.get("protocolVersion"),
            "clockState": payload.get("clockState"),
            "httpStatus": 200,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _udp_loopback() -> dict:
    trajectory = {
        "schemaVersion": "TRAJECTORY-2.0",
        "sceneId": "interface-scene", "timeBaseId": "interface-time",
        "datasetId": "interface-estimate", "startTimeMs": 1_000,
        "endTimeMs": 1_000, "sampleIntervalMs": 1_000, "frameCount": 1,
        "timeScale": "UTC", "frame": "J2000", "unit": "m",
        "unitV": "m/s", "frames": [{
            "frameIndex": 0, "time": "1970-01-01T00:00:01.000Z",
            "timeMs": 1_000, "satelliteList": [],
        }],
    }
    expected = trajectory_frame_datagram(trajectory, 0)
    if decode_trajectory_datagram(encode_trajectory_datagram(expected)) != expected:
        raise AssertionError("In-memory UDP codec round trip changed the payload.")
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    try:
        host, port = receiver.getsockname()
        with TrajectoryUdpPublisher(host, port) as publisher:
            sent = publisher.send_frame(trajectory, 0)
        data, source = receiver.recvfrom(65_535)
    finally:
        receiver.close()
    decoded = decode_trajectory_datagram(data)
    if decoded != expected:
        raise AssertionError("Received UDP payload differs from the sent frame.")
    return {
        "destination": f"127.0.0.1:{port}", "sourceHost": source[0],
        "payloadBytes": sent, "frameIndex": decoded["frameIndex"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_interface_conformance()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(args.output)
    # A failed acceptance verdict is data, not a worker-process crash.  The GUI
    # must still be able to load the report and display FAIL.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
