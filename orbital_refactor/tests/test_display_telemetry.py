import numpy as np
import pytest

from exporters.display_telemetry import (
    build_cann_display_frames,
    build_display_telemetry,
    display_telemetry_datagram,
    dumps_display_telemetry,
    encode_display_telemetry_datagram,
    validate_display_telemetry,
)


def _trajectory():
    return {
        "sceneId": "scene-a", "timeBaseId": "TB-A", "datasetId": "EST-A",
        "sampleIntervalMs": 1000,
        "frames": [{
            "frameIndex": 0, "time": "2026-09-18 04:00:00.000", "timeMs": 1000,
            "satelliteList": [
                {"satObj": "a", "satId": 1, "position": {
                    "x": 3, "y": 4, "z": 0, "vx": 0, "vy": 0, "vz": 2,
                }},
                {"satObj": "b", "satId": 2, "position": {
                    "x": 0, "y": 0, "z": 0, "vx": 0, "vy": 0, "vz": 0,
                }},
            ],
        }],
    }


def test_display_metrics_are_aligned_and_do_not_modify_trajectory_schema():
    trajectory = _trajectory()
    truth = [{"a": np.zeros(6), "b": np.zeros(6)}]

    payload = build_display_telemetry(
        trajectory, truth, selected_satellite="a",
    )

    frame = payload["frames"][0]
    assert payload["schemaVersion"] == "DISPLAY-TELEMETRY-1.0"
    assert payload["messageType"] == "DISPLAY_TELEMETRY"
    assert payload["type"] == "VISUALIZATION"
    assert frame["frameIndex"] == 0
    assert frame["timeMs"] == 1000
    assert frame["estimation"]["fleetPositionRmseM"] == pytest.approx(
        np.sqrt(25.0 / 2.0)
    )
    assert frame["estimation"]["fleetVelocityRmseMps"] == pytest.approx(
        np.sqrt(4.0 / 2.0)
    )
    assert "selectedSatellite" not in frame["estimation"]
    assert frame["estimation"]["satelliteErrors"] == [
        {"satObj": "a", "satId": 1,
         "positionErrorM": 5.0, "velocityErrorMps": 2.0},
        {"satObj": "b", "satId": 2,
         "positionErrorM": 0.0, "velocityErrorMps": 0.0},
    ]
    assert frame["estimation"]["accuracyClaimAllowed"] is False
    assert "estimation" not in trajectory["frames"][0]
    assert list(trajectory["frames"][0]["satelliteList"][0]["position"]) == [
        "x", "y", "z", "vx", "vy", "vz",
    ]


def test_display_datagram_contains_only_one_aligned_frame():
    payload = build_display_telemetry(
        _trajectory(), [{"a": np.zeros(6), "b": np.zeros(6)}],
        selected_satellite="a",
    )
    message = display_telemetry_datagram(payload, 0)
    assert message["messageType"] == "DISPLAY_TELEMETRY"
    assert message["type"] == "VISUALIZATION"
    assert message["frameIndex"] == 0
    assert "frames" not in message
    assert "selectedSatellite" not in message["estimation"]
    encoded = encode_display_telemetry_datagram(message)
    assert b'"messageType":"DISPLAY_TELEMETRY"' in encoded
    assert b'"type":"VISUALIZATION"' in encoded


def test_display_metrics_reject_truth_satellite_mismatch():
    with pytest.raises(ValueError, match="satellite set mismatch"):
        build_display_telemetry(
            _trajectory(), [{"a": np.zeros(6)}], selected_satellite="a",
        )


def test_display_json_keeps_numeric_arrays_on_one_line():
    text = dumps_display_telemetry({
        "directionRing": [255, 252, 244],
        "positionHeatmap": [0, 1, 255],
        "items": [{"satObj": "a"}, {"satObj": "b"}],
    })
    assert '"directionRing": [255,252,244]' in text
    assert '"positionHeatmap": [0,1,255]' in text
    assert '"items": [' in text


def test_display_telemetry_alignment_validation_rejects_wrong_time():
    trajectory = _trajectory()
    payload = build_display_telemetry(
        trajectory, [{"a": np.zeros(6), "b": np.zeros(6)}],
        selected_satellite="a",
    )
    payload["frames"][0]["timeMs"] += 1
    with pytest.raises(ValueError, match="not aligned"):
        validate_display_telemetry(payload, trajectory)


def test_cann_display_uses_real_ring_and_gui_3_by_3_place_cells_for_all_nodes():
    trajectory = _trajectory()
    second = {
        "frameIndex": 1, "time": "2026-09-18 04:00:01.000", "timeMs": 2000,
        "satelliteList": [
            {"satObj": "a", "satId": 1, "position": {}},
            {"satObj": "b", "satId": 2, "position": {}},
        ],
    }
    for frame, angle in ((trajectory["frames"][0], 0.0), (second, 0.001)):
        for item_index, radius in enumerate((7.0e6, 7.1e6)):
            item_angle = angle + 0.1 * item_index
            frame["satelliteList"][item_index]["position"] = {
                "x": radius * np.cos(item_angle),
                "y": radius * np.sin(item_angle), "z": 0,
                "vx": -7500.0 * np.sin(item_angle),
                "vy": 7500.0 * np.cos(item_angle), "vz": 0,
            }
    trajectory["frames"].append(second)

    cann = build_cann_display_frames(trajectory)

    assert len(cann) == 2
    assert [item["satObj"] for item in cann[0]] == ["a", "b"]
    assert [item["satId"] for item in cann[0]] == [1, 2]
    assert cann[0][0]["directionDegrees"] == pytest.approx(0.0, abs=0.1)
    assert cann[0][1]["directionDegrees"] == pytest.approx(
        np.degrees(0.1), abs=0.1,
    )
    for item in cann[0]:
        assert len(item["directionRing"]) == 90
        assert item["positionHeatmapShape"] == [3, 3]
        assert item["positionHeatmapRadialCentersM"] == [-2000.0, 0.0, 2000.0]
        assert item["positionHeatmapAlongTrackCentersM"] == [-2000.0, 0.0, 2000.0]
        assert item["directionPhaseType"] == "ABSOLUTE_ARGUMENT_OF_LATITUDE"
        assert item["directionZeroReference"] == "ASCENDING_NODE"
        assert len(item["positionHeatmap"]) == 9
        assert min(item["directionRing"]) >= 0
        assert max(item["directionRing"]) == 255
        assert max(item["positionHeatmap"]) == 255
        assert item["activityEncoding"] == "UINT8_NORMALIZED"


def test_display_datagram_includes_optional_cann_without_touching_trajectory():
    trajectory = _trajectory()
    cann = [[{"satObj": "a", "directionRing": [0, 255],
              "positionHeatmapShape": [3, 3], "positionHeatmap": [0] * 9}]]
    payload = build_display_telemetry(
        trajectory, [{"a": np.zeros(6), "b": np.zeros(6)}],
        selected_satellite="a", cann_by_frame=cann,
    )
    message = display_telemetry_datagram(payload, 0)
    assert message["cannBySatellite"][0]["satObj"] == "a"
    assert message["cannBySatellite"][0]["directionRing"] == [0, 255]
    assert message["cannBySatellite"][0]["satId"] == 1
    assert "cannBySatellite" not in trajectory["frames"][0]
