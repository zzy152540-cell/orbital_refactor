import socket

import pytest

from interfaces.trajectory_udp import (
    TrajectoryUdpPublisher,
    decode_trajectory_datagram,
    encode_trajectory_datagram,
    trajectory_frame_datagram,
    validate_scene_activation_for_trajectory,
)


def _trajectory():
    return {
        "schemaVersion": "TRAJECTORY-2.0",
        "sceneId": "scene-a", "timeBaseId": "TB-A", "datasetId": "EST-A",
        "startTimeMs": 1000, "endTimeMs": 2000,
        "sampleIntervalMs": 1000, "frameCount": 2,
        "timeScale": "UTC", "frame": "J2000", "unit": "m", "unitV": "m/s",
        "frames": [
            {"frameIndex": 0, "time": "1970-01-01T00:00:01.000Z",
             "timeMs": 1000, "satelliteList": []},
            {"frameIndex": 1, "time": "1970-01-01T00:00:02.000Z",
             "timeMs": 2000, "satelliteList": []},
        ],
    }


def test_trajectory_datagram_round_trip():
    message = trajectory_frame_datagram(_trajectory(), 1)
    decoded = decode_trajectory_datagram(encode_trajectory_datagram(message))
    assert decoded == message
    assert decoded["frameIndex"] == 1
    assert decoded["type"] == "TRAJECTORY"


def test_scene_activation_validates_initial_elements_against_prepared_trajectory():
    trajectory = _trajectory()
    for frame in trajectory["frames"]:
        frame["satelliteList"] = [{"satObj": "sat-a", "satId": "1"}]
    activation = {
        "source": "main-display", "sceneId": "scene-a", "name": "demo",
        "startTime": "1970-01-01 00:00:01.000",
        "endTime": "1970-01-01 00:00:02.000",
        "satelliteCount": 1,
        "satellites": [{
            "name": "sat-a", "satelliteId": "1", "norad": "1",
            "semiMajorAxis": "7000", "eccentricity": "0.001",
            "inclination": "30", "raan": "40", "argPerigee": "50",
            "meanAnomaly": "60", "epoch": "26001.0",
            "tleLine1": "", "tleLine2": "",
        }],
    }

    details = validate_scene_activation_for_trajectory(activation, trajectory)

    assert details == {
        "activationMode": "PRECOMPUTED_FILTER_REPLAY",
        "acceptedSatelliteCount": 1,
        "initialConditionsAccepted": True,
    }
    activation["satellites"][0]["name"] = "wrong-satellite"
    with pytest.raises(ValueError, match="satellite set mismatch"):
        validate_scene_activation_for_trajectory(activation, trajectory)


def test_trajectory_datagram_rejects_configured_size_limit():
    message = trajectory_frame_datagram(_trajectory(), 0)
    with pytest.raises(ValueError, match="limit"):
        encode_trajectory_datagram(message, maximum_bytes=10)


def test_udp_publisher_sends_one_decodable_frame_over_loopback():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    try:
        host, port = receiver.getsockname()
        with TrajectoryUdpPublisher(host, port) as publisher:
            byte_count = publisher.send_frame(_trajectory(), 0)
        data, _source = receiver.recvfrom(65_535)
    finally:
        receiver.close()

    assert byte_count == len(data)
    assert decode_trajectory_datagram(data)["timeMs"] == 1000


def test_udp_publisher_sends_trajectory_then_display_telemetry_on_same_port():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    trajectory = _trajectory()
    telemetry = {
        "schemaVersion": "DISPLAY-TELEMETRY-1.0",
        "messageType": "DISPLAY_TELEMETRY", "sceneId": "scene-a",
        "timeBaseId": "TB-A", "sourceDatasetId": "EST-A", "frameCount": 2,
        "frames": [
            {"frameIndex": i, "time": f"frame-{i}", "timeMs": 1000 + i * 1000,
             "estimation": {"fleetPositionRmseM": 1.0}}
            for i in range(2)
        ],
    }
    try:
        host, port = receiver.getsockname()
        with TrajectoryUdpPublisher(host, port) as publisher:
            publisher.send_frame(trajectory, 0)
            publisher.send_display_telemetry_frame(telemetry, 0)
        first = receiver.recvfrom(65_535)[0]
        second = receiver.recvfrom(65_535)[0]
    finally:
        receiver.close()
    assert b'"messageType":"TRAJECTORY_FRAME"' in first
    assert b'"messageType":"DISPLAY_TELEMETRY"' in second
    assert b'"type":"TRAJECTORY"' in first
    assert b'"type":"VISUALIZATION"' in second
