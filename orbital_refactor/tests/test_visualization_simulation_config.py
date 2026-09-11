import pytest
from pathlib import Path

import experiments.run_v15_visualization_simulation as simulation_runner
from experiments.visualization_simulation_config import (
    load_visualization_simulation_config, next_available_recording_path,
    visualization_simulation_config_from_dict,
)


def test_visualization_simulation_config_loads_and_converts_units():
    config = visualization_simulation_config_from_dict({
        "schema_version": "v1.0", "total_satellites": 10,
        "plane_count": 5, "duration_s": 20.0, "dt_s": 2.0,
        "altitude_km": 800.0, "inclination_deg": 60.0,
        "maximum_range_km": 5000.0, "communication_profile": "mild",
        "output": "results/example", "open_after_run": False,
    })
    arguments = config.generator_arguments
    assert arguments["total_satellites"] == 10
    assert arguments["altitude"] == 800e3
    assert arguments["maximum_range"] == 5000e3
    assert not config.open_after_run


def test_visualization_simulation_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown visualization"):
        visualization_simulation_config_from_dict({
        "schema_version": "v1.0", "unsupported": True,
        })


def test_visualization_simulation_config_rejects_invalid_walker_partition():
    with pytest.raises(ValueError, match="must divide"):
        visualization_simulation_config_from_dict({
            "schema_version": "v1.0", "total_satellites": 10,
            "plane_count": 3,
        })


def test_configured_runner_forwards_validated_arguments_without_opening(
    monkeypatch,
):
    config = visualization_simulation_config_from_dict({
        "schema_version": "v1.0", "duration_s": 8.0,
        "output": "results/configured-test", "open_after_run": False,
    })
    captured = {}

    def fake_generate(output, **arguments):
        captured.update(arguments)
        return output

    monkeypatch.setattr(
        simulation_runner, "generate_dynamic_walker_recording", fake_generate,
    )
    monkeypatch.setattr(
        simulation_runner, "load_visualization_simulation_config",
        lambda _path: config,
    )
    output = simulation_runner.run_configured_visualization_simulation("unused.json")
    assert output.as_posix() == "results/configured-test"
    assert captured["duration"] == 8.0
    assert captured["total_satellites"] == 20


def test_repository_example_configuration_is_valid():
    config = load_visualization_simulation_config(
        "configs/visualization_walker20.json"
    )
    assert config.total_satellites == 20


def test_next_available_recording_path_uses_stable_numeric_suffix(monkeypatch):
    occupied = {"run", "run_001"}
    monkeypatch.setattr(Path, "exists", lambda self: self.name in occupied)
    assert next_available_recording_path("results/run").as_posix() == (
        "results/run_002"
    )
