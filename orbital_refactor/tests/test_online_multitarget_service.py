import pytest

from examples.run_online_multitarget_service import (
    _activation_validator,
    build_parser,
)
from interfaces.scene_control import SceneClockConfig
from tests.test_external_scene_input import _payload


def _config():
    return SceneClockConfig(
        "scene-1", "run-online", "TB-ONLINE",
        1_789_704_000_000, 1_789_704_002_000,
    )


def test_online_service_activation_validates_structured_initial_conditions():
    result = _activation_validator(_config())(_payload())

    assert result["activationMode"] == "ONLINE_TARGET_FREE_IOD_TRACKING"
    assert result["initialConditionsAccepted"] is True
    assert result["acceptedSatelliteCount"] == 1
    assert result["observerStateSource"] == "ONLINE_MEASUREMENT_FRAMES"


def test_online_service_activation_rejects_a_different_time_base():
    payload = _payload()
    payload["endTime"] = "2026-09-18 04:00:03"

    with pytest.raises(ValueError, match="endTime"):
        _activation_validator(_config())(payload)


def test_online_service_cann_is_opt_in_with_realtime_default():
    parser = build_parser()
    base = [
        "--scene-id", "scene-1",
        "--run-id", "run-1",
        "--time-base-id", "TB-1",
        "--dataset-id", "dataset-1",
        "--udp-host", "127.0.0.1",
        "--start-time", "2026-09-18 04:00:00.000",
        "--end-time", "2026-09-18 04:00:02.000",
    ]

    disabled = parser.parse_args(base)
    enabled = parser.parse_args([
        *base, "--enable-measurement-cann", "--cann-profile", "research",
    ])

    assert disabled.enable_measurement_cann is False
    assert disabled.cann_profile == "realtime"
    assert enabled.enable_measurement_cann is True
    assert enabled.cann_profile == "research"
