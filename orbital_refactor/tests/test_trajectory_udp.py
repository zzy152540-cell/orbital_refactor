import socket

import pytest

from interfaces.trajectory_udp import (
    TrajectoryUdpPublisher,
    decode_trajectory_datagram,
    encode_trajectory_datagram,
    trajectory_frame_datagram,
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
