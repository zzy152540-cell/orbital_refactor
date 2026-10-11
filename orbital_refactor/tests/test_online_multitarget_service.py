import pytest

from examples.run_online_multitarget_service import _activation_validator
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
