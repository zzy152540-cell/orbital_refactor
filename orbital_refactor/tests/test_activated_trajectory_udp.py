import json

import pytest

from examples.run_activated_trajectory_udp import load_runtime_config


def test_activation_config_resolves_trajectory_relative_to_config(tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({
        "trajectory": "data/estimated.json",
        "udpHost": "127.0.0.1", "udpPort": 5005,
        "controlHost": "0.0.0.0", "controlPort": 18080,
        "runId": "run-a", "speed": 10,
    }), encoding="utf-8")

    config = load_runtime_config(config_path)

    assert config["trajectory"] == tmp_path / "data" / "estimated.json"
    assert config["display_telemetry"] is None
    assert config["speed"] == 10
    assert config["poll_ms"] == 10.0


def test_activation_config_rejects_unsupported_speed(tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({
        "trajectory": "estimated.json",
        "udpHost": "127.0.0.1", "udpPort": 5005,
        "controlHost": "0.0.0.0", "controlPort": 18080,
        "runId": "run-a", "speed": 3,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="speed"):
        load_runtime_config(config_path)


def test_activation_config_resolves_optional_display_telemetry(tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({
        "trajectory": "data/estimated.json",
        "displayTelemetry": "data/display_telemetry.json",
        "udpHost": "127.0.0.1", "udpPort": 5010,
        "controlHost": "0.0.0.0", "controlPort": 8089,
        "runId": "run-a",
    }), encoding="utf-8")
    config = load_runtime_config(config_path)
    assert config["display_telemetry"] == tmp_path / "data" / "display_telemetry.json"
